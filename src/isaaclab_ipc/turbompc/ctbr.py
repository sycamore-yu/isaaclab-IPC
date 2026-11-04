"""Physical CTBR tracking costs and complete IPC corridor constraints.

This is a new seven-output cost contract, not a compatible interpretation of
the IPC jerk actor's checkpoint. The learned groups are position, velocity,
attitude, and input, followed by three linear position biases. The canonical
author model is retained; costs, horizon, integration interval and conservative
rate bounds are explicit adaptations to this project's Newton/controller.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from isaaclab_ipc.corridor_storage import expand_corridors

from .quadrotor import AUTHOR_BODY_RATE_LIMITS, CTBRDynamics, controls_to_actions
from .solver import MPCInputs


@dataclass(frozen=True)
class CTBRConfig:
    horizon: int = 15
    dt: float = 0.1
    mass: float = 0.75
    gravity: float = 9.81
    position_weight: float = 10.0
    velocity_weight: float = 2.0
    attitude_weight: float = 4.0
    input_weights: tuple[float, ...] = (0.2, 0.1, 0.1, 0.2)
    terminal_state_factor: float = 5.0
    position_bias_scale: float = 0.2
    # Author limits are [10,10,4]. These narrower limits are a local controller
    # choice: at hover, each axis has torque margin with gain 45 and I=0.012.
    # They do not guarantee saturation-free simultaneous commands at all thrusts.
    body_rate_limits: tuple[float, ...] = (1.5, 1.5, 0.1)
    specific_thrust_limits: tuple[float, ...] = (0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
    velocity_lower: tuple[float, ...] = (-10.0, -10.0, -10.0)
    velocity_upper: tuple[float, ...] = (10.0, 10.0, 10.0)
    acceleration_lower: tuple[float, ...] = (-20.0, -20.0, -10.0)
    acceleration_upper: tuple[float, ...] = (20.0, 20.0, 20.0)

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("CTBR horizon must be a positive integer")
        for name in (
            "dt",
            "mass",
            "gravity",
            "position_weight",
            "velocity_weight",
            "attitude_weight",
            "terminal_state_factor",
            "position_bias_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name, size in (
            ("input_weights", 4),
            ("body_rate_limits", 3),
            ("specific_thrust_limits", 2),
            ("velocity_lower", 3),
            ("velocity_upper", 3),
            ("acceleration_lower", 3),
            ("acceleration_upper", 3),
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, values)
        if min(self.input_weights) <= 0 or min(self.body_rate_limits) <= 0:
            raise ValueError("Input weights and body-rate limits must be positive")
        if any(value > cap for value, cap in zip(self.body_rate_limits, AUTHOR_BODY_RATE_LIMITS, strict=True)):
            raise ValueError("Body-rate limits exceed the declared author control range")
        if not 0 <= self.specific_thrust_limits[0] < self.gravity < self.specific_thrust_limits[1]:
            raise ValueError("Nonnegative thrust bounds must contain hover in their strict interior")
        for kind in ("velocity", "acceleration"):
            if any(
                lo >= hi for lo, hi in zip(getattr(self, f"{kind}_lower"), getattr(self, f"{kind}_upper"), strict=True)
            ):
                raise ValueError(f"{kind} bounds must be strictly ordered")


class CTBRQPBuilder(nn.Module):
    """Build [B,H+1,14] multiple-shooting data without consulting a Planner.

    Inputs: ``ctbr_state[B,10]``, ``ctbr_reference[B,H,10]``, and the same
    ``planes/plane_mask[/corridor_index]`` snapshot as IPCQPBuilder. Slot k's
    SFC constrains x[k+1]. Velocity is bounded at x[1:H+1]; total force and
    body rates are bounded only at u[0:H]. True acceleration constraints use
    those same H physical controls, via a separate fixed PathConstraints map.

    u[H] is an independent dummy, with unit Hessian and optimum hover. It has
    no physical input/path constraints and does not enter the executed plan.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = CTBRConfig() if config is None else config
        if not isinstance(self.config, CTBRConfig):
            raise TypeError("config must be CTBRConfig")
        cfg = self.config
        self.horizon = cfg.horizon
        self.nx, self.nu = 10, 4
        state_weights = torch.tensor(
            [cfg.position_weight] * 3 + [cfg.attitude_weight] * 4 + [cfg.velocity_weight] * 3, dtype=torch.float64
        )
        schedule = state_weights.expand(cfg.horizon, -1).clone()
        schedule[-1] *= cfg.terminal_state_factor
        self.register_buffer("state_weights", schedule)
        self.register_buffer("input_weights", torch.tensor(cfg.input_weights, dtype=torch.float64))
        self.register_buffer(
            "hover_control", torch.tensor([cfg.mass * cfg.gravity, 0.0, 0.0, 0.0], dtype=torch.float64)
        )
        lower = torch.tensor(
            [cfg.mass * cfg.specific_thrust_limits[0], *(-value for value in cfg.body_rate_limits)], dtype=torch.float64
        )
        upper = torch.tensor([cfg.mass * cfg.specific_thrust_limits[1], *cfg.body_rate_limits], dtype=torch.float64)
        self.register_buffer("control_lower", lower)
        self.register_buffer("control_upper", upper)
        self.register_buffer("action_lower", controls_to_actions(lower, cfg.mass))
        self.register_buffer("action_upper", controls_to_actions(upper, cfg.mass))
        for name in ("velocity_lower", "velocity_upper", "acceleration_lower", "acceleration_upper"):
            self.register_buffer(name, torch.tensor(getattr(cfg, name), dtype=torch.float64))
        stage_mask = torch.ones((cfg.horizon + 1, 3), dtype=torch.bool)
        stage_mask[-1] = False
        self.register_buffer("acceleration_stage_mask", stage_mask)

    def get_extra_state(self):
        """Checkpoint metadata makes equal-width IPC and CTBR heads distinct."""
        return {
            "model_type": "ctbr",
            "cost_semantics_version": 1,
            "cost_groups": ["position", "velocity", "attitude", "input", "bias_x", "bias_y", "bias_z"],
            "state_order": "p_world,q_wxyz,v_world",
            "control_order": "force_N,omega_body_rad_s",
            "config": asdict(self.config),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("CTBR checkpoint cost semantics or physical configuration do not match this task")

    def dynamics(self):
        cfg = self.config
        return CTBRDynamics(dt=cfg.dt, mass=cfg.mass, gravity=cfg.gravity)

    def path_constraints(self):
        from .constraints import PathConstraints

        return PathConstraints(
            self.dynamics().acceleration,
            lower=self.acceleration_lower,
            upper=self.acceleration_upper,
            stage_mask=self.acceleration_stage_mask,
        )

    def _context(self, obs, costs):
        state, reference = obs["ctbr_state"], obs["ctbr_reference"]
        planes, mask = obs["planes"], obs["plane_mask"]
        if (
            state.ndim != 2
            or state.shape[-1] != 10
            or len(state) < 1
            or reference.shape != (len(state), self.horizon, 10)
        ):
            raise ValueError("Expected ctbr_state[B,10] and ctbr_reference[B,H,10]")
        if not state.is_floating_point() or not reference.is_floating_point() or not planes.is_floating_point():
            raise ValueError("CTBR state, reference and planes must be floating point")
        if (
            planes.ndim != 4
            or planes.shape[0] != len(state)
            or planes.shape[-1] != 4
            or mask.shape != planes.shape[:-1]
            or mask.dtype != torch.bool
        ):
            raise ValueError("Expected planes[B,time_or_corridor,faces,4] and boolean plane_mask")
        tensors = [state, reference, planes, mask]
        if "corridor_index" in obs:
            index = obs["corridor_index"]
            if index.shape != (len(state), self.horizon) or index.dtype != torch.int64 or planes.shape[1] != 2:
                raise ValueError("Compact SFC requires two blocks and int64 corridor_index[B,H]")
            if ((index < -1) | (index >= 2)).any():
                raise ValueError("corridor_index must be -1, 0 or 1")
            tensors.append(index)
        elif planes.shape[1] != self.horizon:
            raise ValueError("Dense SFC requires one plane block per prediction step")
        if costs is None:
            costs = state.new_zeros((len(state), 7), dtype=torch.float64)
        if costs.shape != (len(state), 7) or not costs.is_floating_point():
            raise ValueError("Expected seven floating-point CTBR costs per sample")
        if self.hover_control.dtype != torch.float64 or any(
            value.device != self.hover_control.device for value in [*tensors, costs]
        ):
            raise ValueError("CTBRQPBuilder needs float64 buffers and all inputs on their device")
        planes, mask = expand_corridors(obs)
        planes = planes.double().masked_fill(~mask[..., None], 0)
        state, reference, costs = state.double(), reference.double(), costs.double()
        if not all(torch.isfinite(value).all() for value in (state, reference, costs, planes)):
            raise ValueError("CTBR state, reference, costs and active SFC planes must be finite")
        if any(
            ((quaternion.norm(dim=-1) - 1).abs() > 1e-4).any() for quaternion in (state[:, 3:7], reference[..., 3:7])
        ):
            raise ValueError("External CTBR state/reference quaternions must be near unit and nonzero")
        # Choose the same continuous quaternion hemisphere as the initial state.
        # This discrete context choice is independent of learned cost parameters.
        aligned, previous = [], state[:, 3:7]
        for quaternion in reference[..., 3:7].unbind(1):
            quaternion = torch.where((quaternion * previous).sum(-1, keepdim=True) < 0, -quaternion, quaternion)
            aligned.append(quaternion)
            previous = quaternion
        reference = torch.cat((reference[..., :3], torch.stack(aligned, 1), reference[..., 7:10]), -1)
        return state, reference, planes, mask, costs

    def forward(self, obs, costs=None):
        state, reference, planes, mask, costs = self._context(obs, costs)
        batch, horizon = len(state), self.horizon
        scales = torch.exp(math.log(2) * costs[:, :4].tanh())
        state_scale = torch.cat(
            (scales[:, 0:1].expand(-1, 3), scales[:, 2:3].expand(-1, 4), scales[:, 1:2].expand(-1, 3)), -1
        )
        state_weights = self.state_weights[None] * state_scale[:, None]
        input_weights = (self.input_weights[None] * scales[:, 3:4])[:, None].expand(-1, horizon, -1)
        diagonal = torch.cat(
            (
                torch.cat((state.new_zeros((batch, 1, 10)), state_weights), 1),
                torch.cat((input_weights, state.new_ones((batch, 1, 4))), 1),
            ),
            -1,
        )
        # A bounded *linear cost bias*: equivalent reference displacement is
        # position_bias_scale*tanh(cost[4:])/scale_position, as in IPC.
        bias = self.config.position_bias_scale * costs[:, 4:].tanh()
        state_q = -state_weights * reference
        state_q = state_q - torch.cat(
            (self.state_weights[None, :, :3] * bias[:, None], torch.zeros_like(state_q[..., 3:])), -1
        )
        control_q = -input_weights * self.hover_control
        q = torch.cat(
            (
                torch.cat((state.new_zeros((batch, 1, 10)), state_q), 1),
                torch.cat((control_q, -self.hover_control.expand(batch, 1, -1)), 1),
            ),
            -1,
        )
        faces = planes.shape[-2]
        g = state.new_zeros((batch, horizon + 1, 7 + faces, 14))
        lower = state.new_full((batch, horizon + 1, 7 + faces), -torch.inf)
        upper = state.new_full((batch, horizon + 1, 7 + faces), torch.inf)
        g[:, 1:, :3, 7:10] = torch.eye(3, dtype=state.dtype, device=state.device)
        lower[:, 1:, :3], upper[:, 1:, :3] = self.velocity_lower, self.velocity_upper
        g[:, :-1, 3:7, 10:14] = torch.eye(4, dtype=state.dtype, device=state.device)
        lower[:, :-1, 3:7], upper[:, :-1, 3:7] = self.control_lower, self.control_upper
        g[:, 1:, 7:, :3] = planes[..., :3]
        upper[:, 1:, 7:] = (-planes[..., 3]).masked_fill(~mask, torch.inf)
        inputs = MPCInputs(
            state, torch.diag_embed(diagonal), state.new_zeros((batch, horizon, 14, 14)), q, g, lower, upper
        )
        model = self.dynamics()
        controls = self.hover_control.expand(batch, -1)
        rollout = [state]
        for _ in range(horizon):
            rollout.append(model(torch.cat((rollout[-1], controls), -1)))
        guess = torch.cat((torch.stack(rollout, 1), self.hover_control.expand(batch, horizon + 1, -1)), -1)
        return inputs, guess
