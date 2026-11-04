"""AC-MPC cost policy using the native TurboMPC jerk solver."""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

from isaaclab_ipc.execution import brake_jerk, next_braking, safe_climb_jerk
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.turbompc.ipc import IPCQPBuilder, make_ipc_solver


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
        qp_iterations=16384,
        compile_kernels=True,
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
        self.safe_climb_enabled = bool(
            planner_params and planner_params.get("frontend", {}).get("safe_climb", False)
        )
        cfg = Config(planner_params) if planner_params else Config.load("avoid")
        m = MPC(cfg)
        for name in ("lower", "upper", "jlow", "jhigh", "a", "b"):
            self.register_buffer(name, torch.as_tensor(getattr(m, name), dtype=torch.float64))
        self.ipc_problem = IPCQPBuilder(cfg)
        self.qp_iterations = int(qp_iterations)
        self.compile_kernels = bool(compile_kernels)
        self._native_solver = None
        self.last_native_result = None
        self.native_solve_calls = 0
        self.fixed_cost = False

    @property
    def native_solver(self):
        if self._native_solver is None:
            self._native_solver = make_ipc_solver(self.ipc_problem, self.qp_iterations, self.compile_kernels)
        return self._native_solver

    def _solve_context(self, obs, costs):
        """Solve this immutable observation without changing the online Planner."""
        state = obs["mpc_state"].double()
        state = torch.cat((state[:, :3], state[:, 3:].clamp(self.lower, self.upper)), -1)
        valid = obs["sfc_valid"].flatten().bool()
        solution = costs.sum(-1, keepdim=True).double().expand(-1, 45) * 0
        solved = torch.zeros_like(valid)
        primal = state.new_full((len(state),), torch.inf)
        dual = primal.clone()
        iterations = 0
        self.last_native_result = None
        if valid.any():
            selected = {name: value[valid] for name, value in obs.items()}
            data, guess = self.ipc_problem(selected, costs[valid])
            result = self.native_solver.solve(data, guess)
            self.last_native_result = result
            self.native_solve_calls += 1
            solution[valid] = result.controls[:, :-1].flatten(1)
            solved[valid] = (result.status == 0) & (result.admm.status == 0)
            primal[valid] = result.errors[:, 1:].amax(-1)
            dual[valid] = result.errors[:, 0]
            iterations = int(result.admm.iterations.max())
        return solution, solved, primal, dual, iterations, state

    def problem_snapshot(self, obs, costs):
        """Expose the actual problem for failed-solve diagnostics."""
        from dataclasses import fields

        data, guess = self.ipc_problem(obs, costs)
        return {**{field.name: getattr(data, field.name) for field in fields(data)}, "initial_guess": guess}

    def close(self):
        if self._native_solver is not None:
            self._native_solver.close()
            self._native_solver = None
        self.last_native_result = None

    def solver_metadata(self):
        return {"backend": "turbompc-cuda", "dynamics": "ipc-jerk", "horizon": 15,
                "prediction_dt": 0.1, "qp_iterations": self.qp_iterations,
                "native_solve_calls": self.native_solve_calls}

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        costs = (obs["mpc_state"].new_zeros((len(obs["mpc_state"]), 7))
                 if self.fixed_cost else self.mlp(self.get_latent(obs)))
        solution, solved, primal, dual, iterations, state = self._solve_context(obs, costs)
        valid = obs["sfc_valid"].flatten().bool()
        braking = next_braking(
            obs["braking"].flatten().bool(), solved, state[:, 3:6].norm(dim=-1), state[:, 6:].norm(dim=-1)
        )
        backup = brake_jerk(state, self.lower, self.upper, self.jlow, self.jhigh, 0.1)
        climb = torch.zeros_like(valid)
        if self.safe_climb_enabled:
            climb = obs["climb_active"].flatten().bool()
            vertical = safe_climb_jerk(
                state, obs["climb_target"][:, 2].double(), self.lower, self.upper,
                self.jlow, self.jhigh, 0.1,
            )
            backup = torch.where(climb[:, None], vertical, backup)
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
            climb_active=climb,
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
