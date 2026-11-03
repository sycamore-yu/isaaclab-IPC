"""AC-MPC actor with IPC's linear jerk model and differentiable hard-constraint QP."""

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

from isaaclab_ipc.corridor_storage import expand_corridors
from isaaclab_ipc.execution import brake_jerk, next_braking
from isaaclab_ipc.mpc import MPC, Config

from .qp import DifferentiableQP


def flatness_torch(acceleration, jerk, rotation, yaw_target, thrust_limits=(0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7), rate_limit=None):
    acc = acceleration + acceleration.new_tensor([0, 0, 9.81])
    magnitude = acc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    z = acc / magnitude
    heading = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
    y = torch.linalg.cross(z, heading)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = torch.linalg.cross(y, z)
    hw = (jerk - (z * jerk).sum(-1, keepdim=True) * z) / magnitude
    error = (yaw_target - yaw + torch.pi) % (2 * torch.pi) - torch.pi
    thrust = (acc * rotation[:, :, 2]).sum(-1).clamp(*thrust_limits)
    rates = torch.stack((-(hw * y).sum(-1), (hw * x).sum(-1), error * 0.5 * z[:, 2]), -1)
    if rate_limit is not None:
        rates = rates * (rate_limit / rates.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp_max(1)
    return torch.cat((thrust[:, None], rates), -1)


class MPCActor(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(64, 64),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        qp_iterations=4096,
        planner_params=None,
        thrust_limits=(0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7),
        rate_limit=None,
    ):
        if output_dim != 4 or obs_normalization:
            raise ValueError("MPCActor needs four physical actions and fixed feature scaling")
        distribution_cfg = distribution_cfg or dict(
            class_name="GaussianDistribution", init_std=0.03, std_type="log", learn_std=False
        )
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.mlp = MLP(self.obs_dim, 7, hidden_dims, activation)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
        self.thrust_limits, self.rate_limit = thrust_limits, rate_limit
        m = MPC(Config(planner_params) if planner_params else Config.load("avoid"))
        for name in ("M", "C", "q", "A0", "lower", "upper", "jlow", "jhigh", "a", "b"):
            self.register_buffer(name, torch.as_tensor(getattr(m, name), dtype=torch.float64))
        groups = []
        for axis in range(3):
            mask = np.zeros(135)
            for step in range(15):
                mask[step * 9 + axis * 3 : step * 9 + axis * 3 + 3] = 1
            groups.append(m.q * mask)
        qgroups = np.stack(groups)
        hgroups = np.einsum("si,ks,sj->kij", m.C, qgroups, m.C)
        hjerk = m.H - hgroups.sum(0)
        self.register_buffer("qgroups", torch.tensor(qgroups, dtype=torch.float64))
        self.register_buffer("hgroups", torch.tensor(np.concatenate((hgroups, hjerk[None])), dtype=torch.float64))
        bias = m.C[m.pi].T @ (m.q[m.pi, None] * np.tile(np.eye(3), (15, 1)))
        self.register_buffer("linear_bias", torch.tensor(bias, dtype=torch.float64))
        self.solver = DifferentiableQP(max_iterations=qp_iterations)

    def problem(self, obs, costs):
        state = obs["mpc_state"].double()
        state = torch.cat((state[:, :3], state[:, 3:].clamp(self.lower, self.upper)), -1)
        offset = state @ self.M.T
        error = offset - obs["reference"].double().flatten(1)
        weights = torch.exp(np.log(2) * costs[:, :4].double().tanh())
        h = torch.einsum("bk,kij->bij", weights, self.hgroups)
        q = weights[:, :3] @ self.qgroups
        g = (q * error) @ self.C - 0.2 * costs[:, 4:].double().tanh() @ self.linear_bias.T
        planes, face_mask = expand_corridors(obs)
        planes = planes.double()
        normals = planes[..., :3]
        positions = self.C.reshape(15, 9, 45)[:, :3]
        sfc_a = torch.einsum("btki,tij->btkj", normals, positions).flatten(1, 2)
        sfc_upper = -planes[..., 3] - (normals * offset.reshape(-1, 15, 9)[:, :, None, :3]).sum(-1)
        mask = face_mask.bool().flatten(1)
        sfc_upper = sfc_upper.flatten(1).masked_fill(~mask, torch.inf)
        acceleration = offset.reshape(-1, 15, 9)[:, :, 6:].flatten(1)
        velocity = offset.reshape(-1, 15, 9)[:, :, 3:6].flatten(1)
        lo = torch.cat(
            (
                self.jlow.expand(len(state), -1),
                self.lower[3:].repeat(15) - acceleration,
                self.lower[:3].repeat(15) - velocity,
                torch.full_like(sfc_upper, -torch.inf),
            ),
            -1,
        )
        hi = torch.cat(
            (
                self.jhigh.expand(len(state), -1),
                self.upper[3:].repeat(15) - acceleration,
                self.upper[:3].repeat(15) - velocity,
                sfc_upper,
            ),
            -1,
        )
        a = torch.cat((self.A0.expand(len(state), -1, -1), sfc_a), 1)
        return h, g, a, lo, hi, state

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        costs = self.mlp(self.get_latent(obs))
        h, g, a, lo, hi, state = self.problem(obs, costs)
        valid = obs["sfc_valid"].flatten().bool()
        if valid.all():
            solution, solved, primal, dual, iterations = self.solver(h, g, a, lo, hi)
        else:
            solution = torch.zeros_like(g)
            solved = torch.zeros_like(valid)
            primal = torch.full_like(g[:, 0], torch.inf)
            dual = primal.clone()
            iterations = 0
            if valid.any():
                x, ok, rp, rd, iterations = self.solver(h[valid], g[valid], a[valid], lo[valid], hi[valid])
                solution[valid], solved[valid], primal[valid], dual[valid] = x, ok, rp, rd
        braking = next_braking(
            obs["braking"].flatten().bool(), solved, state[:, 3:6].norm(dim=-1), state[:, 6:].norm(dim=-1)
        )
        backup = brake_jerk(state, self.lower, self.upper, self.jlow, self.jhigh, 0.1)
        jerk = torch.where(braking[:, None], backup, solution[:, :3])
        predicted = state @ self.a.T + jerk @ self.b.T
        rotation = obs["rotation"].reshape(-1, 3, 3).double()
        yaw = obs["yaw_target"].flatten().double()
        mean = flatness_torch(predicted[:, 6:], jerk, rotation, yaw, self.thrust_limits, self.rate_limit).float()
        backup_state = state @ self.a.T + backup @ self.b.T
        self._backup = flatness_torch(backup_state[:, 6:], backup, rotation, yaw, self.thrust_limits, self.rate_limit).float().detach()
        self._braking = braking
        self.distribution.update(mean)
        self.last_result = dict(
            solution=solution.detach(),
            solved=solved,
            braking=braking,
            state=state.detach(),
            predicted=predicted.detach(),
            jerk=jerk.detach(),
            primal=primal,
            dual=dual,
            iterations=iterations,
            mean=mean.detach(),
            sfc_valid=valid,
        )
        if stochastic_output:
            return torch.where(braking[:, None], self._backup, self.distribution.sample())
        return mean

    def get_output_log_prob(self, outputs):
        # The common measure includes the deterministic braking atom and continuous actions.
        at_backup = (outputs == self._backup).all(-1)
        continuous = self.distribution.log_prob(outputs)
        impossible = torch.full_like(continuous, -1e9)
        return torch.where(
            self._braking, torch.where(at_backup, 0.0, impossible), torch.where(at_backup, impossible, continuous)
        )

    @property
    def output_entropy(self):
        return torch.where(self._braking, 0.0, self.distribution.entropy)

    @property
    def output_distribution_params(self):
        return (*self.distribution.params, self._braking[:, None].float())

    def get_kl_divergence(self, old_params, new_params):
        old_brake, new_brake = old_params[2].squeeze(-1).bool(), new_params[2].squeeze(-1).bool()
        kl = self.distribution.kl_divergence(old_params[:2], new_params[:2])
        return torch.where(old_brake != new_brake, torch.inf, torch.where(old_brake, 0.0, kl))

    def as_jit(self):
        raise NotImplementedError("Use the saved MPCActor checkpoint; JIT cannot export the QP solver")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Use the saved MPCActor checkpoint; ONNX cannot export the QP solver")
