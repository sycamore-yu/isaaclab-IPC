"""11D CTBR + arc-length progress using the existing quadratic MPC backend.

The local path model and SFC assignment are frozen with each observation.
All physical state/control costs retain the existing 28N network contract.
"""

from dataclasses import asdict, dataclass

import torch

from .ctbr import CTBRQPBuilder
from .dynamics import AutoDiffDynamics
from .quadrotor import CTBRDynamics
from .solver import MPCInputs


@dataclass(frozen=True)
class ProgressConfig:
    reward: float = 12.0
    lag_weight: float = 50.0
    rate_weight: float = 0.1
    rate_change_weight: float = 0.2
    contour_weight: float = 25.0
    velocity_consistency_weight: float = 10.0
    max_speed: float = 4.0
    lateral_acceleration: float = 3.5
    braking_acceleration: float = 4.0


class ProgressDynamics(AutoDiffDynamics):
    """x=[p,q,v,s], u=[F,omega,nu]; the physical map is unchanged."""

    def __init__(self, *, dt=.1, mass=.75, gravity=9.81):
        self.physical = CTBRDynamics(dt=dt, mass=mass, gravity=gravity)
        self.dt, self.nu = dt, 5
        super().__init__(self.step, nx=11)

    def step(self, z):
        physical_z = torch.cat((z[..., :10], z[..., 11:15]), -1)
        return torch.cat((self.physical(physical_z), z[..., 10:11] + self.dt * z[..., 15:16]), -1)

    def linearize(self, z):
        physical_z = torch.cat((z[..., :10], z[..., 11:15]), -1)
        physical_next, physical_jac = self.physical.linearize(physical_z)
        jacobian = z.new_zeros((*z.shape[:-1], 11, 16))
        jacobian[..., :10, :10] = physical_jac[..., :10, :10]
        jacobian[..., :10, 11:15] = physical_jac[..., :10, 10:14]
        jacobian[..., 10, 10] = 1.
        jacobian[..., 10, 15] = self.dt
        return torch.cat((physical_next, z[..., 10:11] + self.dt * z[..., 15:16]), -1), jacobian

    def lagrangian_hessian(self, z, multipliers):
        physical_z = torch.cat((z[..., :10], z[..., 11:15]), -1)
        physical_hessian = self.physical.lagrangian_hessian(physical_z, multipliers[..., :10])
        hessian = z.new_zeros((*z.shape[:-1], 16, 16))
        hessian[..., :10, :10] = physical_hessian[..., :10, :10]
        hessian[..., :10, 11:15] = physical_hessian[..., :10, 10:14]
        hessian[..., 11:15, :10] = physical_hessian[..., 10:14, :10]
        hessian[..., 11:15, 11:15] = physical_hessian[..., 10:14, 10:14]
        return hessian

    def rollout(self, initial_state, controls):
        state, states = initial_state, [initial_state]
        for control in controls.unbind(1):
            state = self.step(torch.cat((state, control), -1))
            states.append(state)
        return torch.stack(states, 1)


class ProgressQPBuilder(CTBRQPBuilder):
    """A local contour/lag quadratic cost with physical and virtual controls."""

    def __init__(self, config=None, *, progress_config=None, compile_kernels=False):
        # Compile only after selecting the enlarged physical/virtual model.
        super().__init__(config, compile_kernels=False)
        self.progress_config = (ProgressConfig(**progress_config) if isinstance(progress_config, dict)
                                else progress_config or ProgressConfig())
        self.nx, self.nu = 11, 5
        self._model = ProgressDynamics(dt=self.config.dt, mass=self.config.mass, gravity=self.config.gravity)
        self._rollout = self._model.rollout
        if compile_kernels:
            self.compile(fullgraph=True, dynamic=False)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'model_type': 'ctbr-progress',
                'progress': asdict(self.progress_config)}

    def forward(self, obs, costs=None):
        pc = self.progress_config
        state = obs['ctbr_state'].double()
        reference = obs['ctbr_reference'].double()
        geometry = obs['progress_reference'].double()
        planes, mask = obs['planes'].double(), obs['plane_mask']
        batch, nodes = len(state), self.cost_nodes
        arc, tangent = geometry[..., 0], geometry[..., 1:4]
        s_lower, s_upper = geometry[..., 4], geometry[..., 5]

        # e14 = M*z16 - c. The matrix contains position/progress and
        # velocity/virtual-speed couplings rather than a timed speed target.
        transform = state.new_zeros((batch, nodes, 14, 16))
        transform[..., :10, :10] = torch.eye(10, dtype=state.dtype, device=state.device)
        transform[..., 10:14, 11:15] = torch.eye(4, dtype=state.dtype, device=state.device)
        transform[..., :3, 10] = -tangent
        transform[..., 7:10, 15] = -tangent
        center = torch.cat((reference[..., :3] - tangent * arc[..., None],
                            reference[..., 3:7], state.new_zeros((batch, nodes, 3)),
                            self.hover_control.expand(batch, nodes, 4)), -1)
        diagonal, linear = self.map_costs(torch.zeros_like(center), costs)
        transposed = transform.transpose(-1, -2)
        d = transposed @ (diagonal[..., :, None] * transform)
        q = (transposed @ (linear - diagonal * center)[..., None]).squeeze(-1)

        # The spatial curve is the reference, not a timed velocity command.
        # Penalize lateral position and actual velocity that diverges from
        # the optimized virtual progress rate.
        lateral = torch.eye(3, dtype=state.dtype, device=state.device) - (
            tangent[..., :, None] * tangent[..., None, :]
        )
        d[..., :3, :3] += pc.contour_weight * lateral
        q[..., :3] -= pc.contour_weight * (lateral @ reference[..., :3, None]).squeeze(-1)
        velocity_map = transform[..., 7:10, :]
        d += pc.velocity_consistency_weight * (velocity_map.transpose(-1, -2) @ velocity_map)

        lag = state.new_zeros((batch, nodes, 16))
        lag[..., :3], lag[..., 10] = tangent, -1.
        lag_center = (reference[..., :3] * tangent).sum(-1) - arc
        d = d + pc.lag_weight * lag[..., :, None] * lag[..., None, :]
        q = q - pc.lag_weight * lag * lag_center[..., None]
        d[..., 15, 15] += pc.rate_weight
        q[:, -1, 10] -= pc.reward
        e = state.new_zeros((batch, nodes - 1, 16, 16))
        d[:, 1:, 15, 15] += pc.rate_change_weight
        d[:, :-1, 15, 15] += pc.rate_change_weight
        e[..., 15, 15] = -pc.rate_change_weight

        faces = planes.shape[-2]
        g = state.new_zeros((batch, nodes, 9 + faces, 16))
        lower = state.new_full((batch, nodes, 9 + faces), -torch.inf)
        upper = state.new_full((batch, nodes, 9 + faces), torch.inf)
        g[:, 1:, :3, 7:10] = torch.eye(3, dtype=state.dtype, device=state.device)
        lower[:, 1:, :3], upper[:, 1:, :3] = self.velocity_lower, self.velocity_upper
        g[:, :, 3:7, 11:15] = torch.eye(4, dtype=state.dtype, device=state.device)
        lower[:, :, 3:7], upper[:, :, 3:7] = self.control_lower, self.control_upper
        g[:, :, 7, 15] = 1.
        lower[:, :, 7] = 0.
        upper[:, :, 7] = geometry[..., 7].clamp(min=0., max=pc.max_speed)
        g[:, 1:, 8, 10] = 1.
        lower[:, 1:, 8], upper[:, 1:, 8] = s_lower[:, 1:], s_upper[:, 1:]
        active_planes = planes.masked_fill(~mask[..., None], 0.)
        g[:, 1:, 9:, :3] = active_planes[:, 1:, :, :3]
        upper[:, 1:, 9:] = (-active_planes[:, 1:, :, 3]).masked_fill(~mask[:, 1:], torch.inf)
        initial_state = torch.cat((state, arc[:, :1]), -1)
        data = MPCInputs(initial_state, d, e, q, g, lower, upper)

        physical_controls = obs['ctbr_warm_start'].double().clamp(self.control_lower, self.control_upper)
        rates = torch.minimum(geometry[:, :-1, 6:7].clamp_min(0.),
                              upper[:, :-1, 7:8])
        controls = torch.cat((physical_controls, rates), -1)
        states = self._rollout(initial_state, controls)
        terminal = torch.cat((self.hover_control.expand(batch, 1, 4), rates[:, -1:]), -1)
        guess = torch.cat((states, torch.cat((controls, terminal), 1)), -1)
        return data, guess
