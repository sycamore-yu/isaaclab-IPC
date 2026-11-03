"""IPC's published jerk QP in physical multiple-shooting coordinates."""

import math

import torch
from torch import nn

from isaaclab_ipc.corridor_storage import expand_corridors
from isaaclab_ipc.mpc import Config

from .dynamics import LinearDynamics
from .solver import MPCInputs


class IPCQPBuilder(nn.Module):
    """Build the existing IPC problem without running a planner or a solver.

    ``obs`` holds batched ``mpc_state``, ``reference``, ``planes``, and
    ``plane_mask`` tensors, plus optional compact ``corridor_index``. Planes
    already use the storage order from ``pack_corridors`` and the convention
    ``[A, -b]``. Schedule slot k constrains x[k+1]. No face is normalized,
    removed, or shrunk here. The caller routes invalid ``sfc_valid`` samples.

    Seven costs have exactly MPCActor's meaning: four bounded positive group
    multipliers and three bounded linear position biases. Zero costs recover
    the fixed IPC objective. The independent terminal dummy control has unit
    quadratic cost, no linear cost or constraint, and unique optimum zero.

    Buffers remain float64. Move this module to the desired device before
    calling ``dynamics()`` and constructing its solver.
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = Config.load("avoid") if cfg is None else cfg
        p = cfg.params["mpc"]
        self.horizon = int(p["horizon"])
        self.dt = float(p["step"])
        if self.horizon < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("IPC needs a positive horizon and prediction step")

        eye = torch.eye(3, dtype=torch.float64)
        a, b = torch.eye(9, dtype=torch.float64), torch.zeros((9, 3), dtype=torch.float64)
        a[:3, 3:6], a[:3, 6:9] = self.dt * eye, self.dt**2 / 2 * eye
        a[3:6, 3:6] -= torch.diag(torch.tensor([p["D_" + axis] for axis in "xyz"], dtype=torch.float64)) * self.dt
        a[3:6, 6:9] = self.dt * eye
        b[:3], b[3:6], b[6:] = self.dt**3 / 6 * eye, self.dt**2 / 2 * eye, self.dt * eye
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("M", torch.cat([torch.linalg.matrix_power(a, k + 1) for k in range(self.horizon)]))

        q = torch.tensor([p["R_p"], p["R_v"], p["R_a"]], dtype=torch.float64).repeat_interleave(3)
        q = q.expand(self.horizon, -1).clone()
        q[-1] = torch.tensor([p["R_pN"], p["R_vN"], p["R_aN"]], dtype=torch.float64).repeat_interleave(3)
        continuity = torch.full((self.horizon,), 2.0, dtype=torch.float64)
        continuity[[0, -1]] = 1.0
        # OSQP-Eigen 0.7 drops the original lower-only continuity cross terms.
        # Keep the effective published Hessian, including both endpoint weights.
        r = (float(p["R_u"]) + continuity * float(p["R_u_con"]))[:, None].expand(-1, 3).clone()
        self.register_buffer("state_weights", q)
        self.register_buffer("jerk_weights", r)
        for name, groups, side in (
            ("lower", ("vx", "vy", "vz", "ax", "ay", "az"), "min"),
            ("upper", ("vx", "vy", "vz", "ax", "ay", "az"), "max"),
            ("jerk_lower", ("ux", "uy", "uz"), "min"),
            ("jerk_upper", ("ux", "uy", "uz"), "max"),
        ):
            self.register_buffer(name, torch.tensor([p[f"{key}_{side}"] for key in groups], dtype=torch.float64))
        if not all(torch.isfinite(value).all() for value in self.buffers()):
            raise ValueError("IPC numerical parameters must be finite")
        if (q < 0).any() or (r <= 0).any():
            raise ValueError("IPC state weights must be nonnegative and effective jerk weights positive")
        if (self.lower > self.upper).any() or (self.jerk_lower > self.jerk_upper).any():
            raise ValueError("IPC bounds must be ordered")

    def dynamics(self):
        """Return the exact discrete IPC map using this module's current buffers."""
        return LinearDynamics(torch.cat((self.a, self.b), -1), nx=9)

    def _context(self, obs, costs):
        state, reference = obs["mpc_state"], obs["reference"]
        planes, mask = obs["planes"], obs["plane_mask"]
        if state.ndim != 2 or state.shape[1] != 9 or reference.shape != (len(state), self.horizon, 9):
            raise ValueError("Expected mpc_state[B,9] and reference[B,horizon,9]")
        if (
            planes.ndim != 4
            or planes.shape[0] != len(state)
            or planes.shape[-1] != 4
            or mask.shape != planes.shape[:-1]
        ):
            raise ValueError("Expected planes[B,time_or_corridor,faces,4] and matching plane_mask")
        if mask.dtype is not torch.bool:
            raise ValueError("plane_mask must be boolean")
        tensors = [state, reference, planes, mask]
        if "corridor_index" in obs:
            indices = obs["corridor_index"]
            if indices.shape != (len(state), self.horizon) or indices.dtype != torch.int64 or planes.shape[1] != 2:
                raise ValueError("Compact SFC requires two blocks and int64 corridor_index[B,horizon]")
            if ((indices < -1) | (indices >= 2)).any():
                raise ValueError("corridor_index must be -1, 0 or 1")
            tensors.append(indices)
        elif planes.shape[1] != self.horizon:
            raise ValueError("Dense SFC requires one plane block per prediction step")
        if costs is None:
            costs = state.new_zeros((len(state), 7), dtype=torch.float64)
        if costs.shape != (len(state), 7) or not costs.is_floating_point():
            raise ValueError("Expected seven floating-point costs per sample")
        if self.a.dtype != torch.float64 or any(value.device != self.a.device for value in [*tensors, costs]):
            raise ValueError("IPCQPBuilder needs float64 buffers and all inputs on the same device")
        planes, mask = expand_corridors(obs)
        # Padding carries no inequality. Ignore its stored values before any
        # arithmetic, including inactive NaN placeholders from external storage.
        planes = planes.double().masked_fill(~mask[..., None], 0)
        state, reference, costs = state.double(), reference.double(), costs.double()
        if not all(torch.isfinite(value).all() for value in (state, reference, costs, planes)):
            raise ValueError("State, reference, costs and active SFC planes must be finite")
        state = torch.cat((state[:, :3], state[:, 3:].clamp(self.lower, self.upper)), -1)
        return state, reference, planes, mask, costs

    def forward(self, obs, costs=None):
        """Return ``(MPCInputs, zero_jerk_guess)`` with owned physical QP arrays."""
        state, reference, planes, mask, costs = self._context(obs, costs)
        batch, horizon = len(state), self.horizon
        scales = torch.exp(math.log(2) * costs[:, :4].tanh())
        state_weights = self.state_weights[None] * scales[:, None, :3].repeat_interleave(3, -1)
        jerk_weights = self.jerk_weights[None] * scales[:, 3, None, None]
        diagonal = torch.cat(
            (
                torch.cat((state.new_zeros((batch, 1, 9)), state_weights), 1),
                torch.cat((jerk_weights, state.new_ones((batch, 1, 3))), 1),
            ),
            -1,
        )
        bias = 0.2 * costs[:, 4:].tanh()
        state_q = -state_weights * reference
        state_q = state_q - torch.cat(
            (self.state_weights[None, :, :3] * bias[:, None], torch.zeros_like(state_q[..., 3:])), -1
        )
        q = torch.cat(
            (torch.cat((state.new_zeros((batch, 1, 9)), state_q), 1), state.new_zeros((batch, horizon + 1, 3))), -1
        )

        faces = planes.shape[-2]
        g = state.new_zeros((batch, horizon + 1, 9 + faces, 12))
        lower = state.new_full((batch, horizon + 1, 9 + faces), -torch.inf)
        upper = state.new_full((batch, horizon + 1, 9 + faces), torch.inf)
        g[:, 1:, :6, 3:9] = torch.eye(6, dtype=state.dtype, device=state.device)
        lower[:, 1:, :6], upper[:, 1:, :6] = self.lower, self.upper
        g[:, :-1, 6:9, 9:12] = torch.eye(3, dtype=state.dtype, device=state.device)
        lower[:, :-1, 6:9], upper[:, :-1, 6:9] = self.jerk_lower, self.jerk_upper
        g[:, 1:, 9:, :3] = planes[..., :3]
        upper[:, 1:, 9:] = (-planes[..., 3]).masked_fill(~mask, torch.inf)
        inputs = MPCInputs(
            state, torch.diag_embed(diagonal), state.new_zeros((batch, horizon, 12, 12)), q, g, lower, upper
        )
        guess = state.new_zeros((batch, horizon + 1, 12))
        guess[:, 0, :9] = state
        guess[:, 1:, :9] = (state @ self.M.T).reshape(batch, horizon, 9)
        return inputs, guess
