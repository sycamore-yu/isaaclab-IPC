"""Per-node AC-MPC quadratic costs and hard CTBR/SFC constraints.

N is the number of cost nodes, including the measured initial state. There are
N-1 physical transitions. Every node owns 14 diagonal and 14 linear parameters,
in [position, quaternion WXYZ, velocity, force N, body rates] order. The final
control is costed and bounded, as in the author implementation, but is not run.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .quadrotor import AUTHOR_BODY_RATE_LIMITS, CTBRDynamics, controls_to_actions
from .solver import MPCInputs


@dataclass(frozen=True)
class CTBRConfig:
    cost_nodes: int = 5
    dt: float = 0.1
    mass: float = 0.75
    gravity: float = 9.81
    position_weight: float = 10.0
    velocity_weight: float = 2.0
    attitude_weight: float = 4.0
    input_weights: tuple[float, ...] = (0.2, 1.0, 1.0, 2.0)
    # Physical residual scales, not a shared seven-parameter cost head.
    linear_scales: tuple[float, ...] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5)
    diagonal_floor: float = 1e-4
    body_rate_limits: tuple[float, ...] = (3.0, 3.0, 1.5)
    specific_thrust_limits: tuple[float, ...] = (0.0, 3.0 * 9.81)
    velocity_lower: tuple[float, ...] = (-10.0, -10.0, -10.0)
    velocity_upper: tuple[float, ...] = (10.0, 10.0, 10.0)
    acceleration_lower: tuple[float, ...] = (-20.0, -20.0, -10.0)
    acceleration_upper: tuple[float, ...] = (20.0, 20.0, 20.0)

    @property
    def horizon(self):
        """Number of physical transitions, distinct from the cost-node count."""
        return self.cost_nodes - 1

    @property
    def cost_dim(self):
        return 28 * self.cost_nodes

    def __post_init__(self):
        if isinstance(self.cost_nodes, bool) or not isinstance(self.cost_nodes, int) or self.cost_nodes < 2:
            raise ValueError("CTBR needs at least two cost nodes, including the initial state")
        for name in ("dt", "mass", "gravity", "position_weight", "velocity_weight", "attitude_weight", "diagonal_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.diagonal_floor > 1:
            raise ValueError("diagonal_floor must not exceed the unit baseline cost")
        for name, size in (
            ("input_weights", 4), ("linear_scales", 14), ("body_rate_limits", 3), ("specific_thrust_limits", 2),
            ("velocity_lower", 3), ("velocity_upper", 3), ("acceleration_lower", 3), ("acceleration_upper", 3),
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, values)
        if min(self.input_weights) <= 0 or min(self.linear_scales) <= 0 or min(self.body_rate_limits) <= 0:
            raise ValueError("Cost weights, residual scales and body-rate limits must be positive")
        if any(value > cap for value, cap in zip(self.body_rate_limits, AUTHOR_BODY_RATE_LIMITS, strict=True)):
            raise ValueError("Body-rate limits exceed the author control range")
        if not 0 <= self.specific_thrust_limits[0] < self.gravity < self.specific_thrust_limits[1]:
            raise ValueError("Nonnegative thrust bounds must contain hover")
        for kind in ("velocity", "acceleration"):
            if any(lo >= hi for lo, hi in zip(getattr(self, f"{kind}_lower"), getattr(self, f"{kind}_upper"), strict=True)):
                raise ValueError(f"{kind} bounds must be strictly ordered")


class CTBRQPBuilder(nn.Module):
    """Build a frozen [B,N,14] shooting problem; never consult an online planner.

    The learned cost is 0.5*e.T*diag(d)*e + linear.T*e, e=z-reference.
    Consequently q_absolute=linear-d*reference. Zero logits recover the fixed
    tracking cost exactly. N*14 diagonal logits precede N*14 linear logits.

    SFC slot k constrains x[k]. The measured x[0] is fixed by equality and its
    SFC membership is a frontend input contract, not a redundant KKT row.
    """

    def __init__(self, config=None, *, compile_kernels=False):
        super().__init__()
        self.config = CTBRConfig() if config is None else config
        if not isinstance(self.config, CTBRConfig):
            raise TypeError("config must be CTBRConfig")
        cfg = self.config
        self.horizon, self.cost_nodes = cfg.horizon, cfg.cost_nodes
        self.nx, self.nu = 10, 4
        weights = [cfg.position_weight] * 3 + [cfg.attitude_weight] * 4 + [cfg.velocity_weight] * 3
        self.register_buffer("base_diagonal", torch.tensor([*weights, *cfg.input_weights], dtype=torch.float64))
        self.register_buffer("linear_scales", torch.tensor(cfg.linear_scales, dtype=torch.float64))
        self.register_buffer("hover_control", torch.tensor([cfg.mass * cfg.gravity, 0.0, 0.0, 0.0], dtype=torch.float64))
        lower = torch.tensor([cfg.mass * cfg.specific_thrust_limits[0], *(-v for v in cfg.body_rate_limits)], dtype=torch.float64)
        upper = torch.tensor([cfg.mass * cfg.specific_thrust_limits[1], *cfg.body_rate_limits], dtype=torch.float64)
        self.register_buffer("control_lower", lower)
        self.register_buffer("control_upper", upper)
        self.register_buffer("action_lower", controls_to_actions(lower, cfg.mass))
        self.register_buffer("action_upper", controls_to_actions(upper, cfg.mass))
        for name in ("velocity_lower", "velocity_upper", "acceleration_lower", "acceleration_upper"):
            self.register_buffer(name, torch.tensor(getattr(cfg, name), dtype=torch.float64))
        stage_mask = torch.ones((cfg.cost_nodes, 3), dtype=torch.bool)
        stage_mask[-1] = False
        self.register_buffer("acceleration_stage_mask", stage_mask)
        self._model = CTBRDynamics(dt=cfg.dt, mass=cfg.mass, gravity=cfg.gravity)
        self._rollout = self._model.rollout
        if compile_kernels:
            self.compile(fullgraph=True, dynamic=False)

    def get_extra_state(self):
        return {
            "model_type": "ctbr",
            "cost_semantics_version": 2,
            "cost_layout": "all_node_diagonal14_then_all_node_linear14",
            "state_order": "p_world,q_wxyz,v_world",
            "control_order": "force_N,omega_body_rad_s",
            "cost_coordinates": "reference_centered",
            "config": asdict(self.config),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("CTBR checkpoint cost semantics or physics do not match this task")

    def dynamics(self):
        return self._model

    def path_constraints(self):
        from .constraints import PathConstraints

        return PathConstraints(
            self._model.acceleration,
            lower=self.acceleration_lower,
            upper=self.acceleration_upper,
            stage_mask=self.acceleration_stage_mask,
        )

    def map_costs(self, reference, costs=None):
        """Return diagonal D and absolute q for [B,N,14] references."""
        batch, nodes = reference.shape[:2]
        if reference.shape != (batch, self.cost_nodes, 14):
            raise ValueError("Expected reference[B,cost_nodes,14]")
        if costs is None:
            diagonal = self.base_diagonal.expand(batch, nodes, -1)
            linear = torch.zeros_like(reference)
        else:
            if costs.shape != (batch, self.config.cost_dim):
                raise ValueError("Expected 28*cost_nodes logits per sample")
            raw_diagonal, raw_linear = costs.double().split(14 * nodes, dim=-1)
            raw_diagonal = raw_diagonal.reshape(batch, nodes, 14)
            raw_linear = raw_linear.reshape(batch, nodes, 14)
            # Centered sigmoid matches the author's positive diagonal/bounded
            # linear parameterization without the common 100000 objective scale.
            floor = self.config.diagonal_floor
            diagonal = self.base_diagonal * (floor + (1.0 - floor) * (2.0 * raw_diagonal.sigmoid()))
            linear = self.base_diagonal * self.linear_scales * (2.0 * raw_linear.sigmoid() - 1.0)
        return diagonal, linear - diagonal * reference

    def forward(self, obs, costs=None):
        state, reference = obs["ctbr_state"].double(), obs["ctbr_reference"].double()
        planes, mask = obs["planes"].double(), obs["plane_mask"]
        batch, nodes = len(state), self.cost_nodes
        if state.shape != (batch, 10) or reference.shape != (batch, nodes, 10):
            raise ValueError("Expected ctbr_state[B,10], ctbr_reference[B,cost_nodes,10]")
        if planes.ndim != 4 or planes.shape[:2] != (batch, nodes) or planes.shape[-1] != 4:
            raise ValueError("Expected dense planes[B,cost_nodes,faces,4]")
        if mask.shape != planes.shape[:-1] or mask.dtype != torch.bool:
            raise ValueError("plane_mask must match dense planes with boolean dtype")
        if state.device != self.hover_control.device or self.hover_control.dtype != torch.float64:
            raise ValueError("CTBRQPBuilder must use float64 buffers on the input device")
        # The hemisphere choice is frozen with observations, independent of cost.
        sign = torch.where((reference[..., 3:7] * state[:, None, 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        reference = torch.cat((reference[..., :3], reference[..., 3:7] * sign, reference[..., 7:]), -1)
        control_reference = obs.get("ctbr_control_reference", self.hover_control.expand(batch, nodes, -1)).double()
        if control_reference.shape != (batch, nodes, 4):
            raise ValueError("Expected ctbr_control_reference[B,cost_nodes,4]")
        diagonal, q = self.map_costs(torch.cat((reference, control_reference), -1), costs)
        faces = planes.shape[-2]
        g = state.new_zeros((batch, nodes, 7 + faces, 14))
        lower = state.new_full((batch, nodes, 7 + faces), -torch.inf)
        upper = state.new_full((batch, nodes, 7 + faces), torch.inf)
        g[:, 1:, :3, 7:10] = torch.eye(3, dtype=state.dtype, device=state.device)
        lower[:, 1:, :3], upper[:, 1:, :3] = self.velocity_lower, self.velocity_upper
        g[:, :, 3:7, 10:14] = torch.eye(4, dtype=state.dtype, device=state.device)
        lower[:, :, 3:7], upper[:, :, 3:7] = self.control_lower, self.control_upper
        active_planes = planes.masked_fill(~mask[..., None], 0)
        g[:, 1:, 7:, :3] = active_planes[:, 1:, :, :3]
        upper[:, 1:, 7:] = (-active_planes[:, 1:, :, 3]).masked_fill(~mask[:, 1:], torch.inf)
        data = MPCInputs(
            state, torch.diag_embed(diagonal), state.new_zeros((batch, nodes - 1, 14, 14)), q, g, lower, upper
        )
        controls = obs.get("ctbr_warm_start", self.hover_control.expand(batch, nodes - 1, -1)).double()
        if controls.shape != (batch, nodes - 1, 4):
            raise ValueError("Expected ctbr_warm_start[B,cost_nodes-1,4]")
        controls = controls.clamp(self.control_lower, self.control_upper)
        states = self._rollout(state, controls)
        terminal = control_reference[:, -1:].clamp(self.control_lower, self.control_upper)
        guess = torch.cat((states, torch.cat((controls, terminal), 1)), -1)
        return data, guess
