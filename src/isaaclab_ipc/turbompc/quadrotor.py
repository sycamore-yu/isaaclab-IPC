"""CA-AC-MPC's discrete collective-thrust/body-rate prediction model.

Canonical reference: ``prisma-lab/CA-AC-MPC`` at
``14aa349bb1f509c529bf3662c7feb72db68f1cb7``,
``differentialMPCPerformance/dynamics/drone_ts.py:DroneDynamics.forward``.
This independent implementation uses position/velocity Euler steps and a
quaternion exponential. It deliberately preserves raw Hamilton rotation away
from unit quaternions and the unconditional ``norm + 1e-8`` normalization.
Those details affect derivatives at SQP iterates. The author's traced function
and Euler-approximate analytic Jacobian are not the reference for this map.

The exponential uses the analytic sinc limit near zero body rate. This small
numerical change removes the undefined second derivative of ``norm(omega)``
at hover; it differs slightly from the author's ``norm(omega) + 1e-8`` formula.
The prediction model does not replace Newton's rigid-body/motor integration.
"""

import math

import torch

from .dynamics import AutoDiffDynamics

AUTHOR_COMMIT = "14aa349bb1f509c529bf3662c7feb72db68f1cb7"
AUTHOR_SOURCE_SHA256 = "090d99a39f2cfccf98ae872486f01fbe6a0976c2a3bf499550b7b34379759246"
AUTHOR_BODY_RATE_LIMITS = (10.0, 10.0, 4.0)
QUATERNION_NORMALIZATION_EPS = 1e-8


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _floating(value, width, name):
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(f"{name} must be a tensor with final dimension {width}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def _physical_pose(pose_xyzw, world_twist):
    _floating(pose_xyzw, 7, "pose_xyzw")
    _floating(world_twist, 6, "world_twist")
    if pose_xyzw.shape[:-1] != world_twist.shape[:-1] or pose_xyzw.device != world_twist.device:
        raise ValueError("Pose and world twist must have matching batches and device")
    return pose_xyzw.double(), world_twist.double()


def state_from_isaac(pose_xyzw, world_twist, env_origins):
    """Return [local-world position, WXYZ attitude, world linear velocity].

    Isaac Lab 3 pose tensors use XYZW. Translation by the environment origin
    keeps the same world axes as the IPC map and SFC. Velocity is never rotated
    into the body frame. Quaternion components are reordered without silently
    projecting or clipping the measured state.
    """
    pose, twist = _physical_pose(pose_xyzw, world_twist)
    _floating(env_origins, 3, "env_origins")
    if env_origins.shape != pose.shape[:-1] + (3,) or env_origins.device != pose.device:
        raise ValueError("Environment origins must match the pose batch and device")
    return torch.cat((pose[..., :3] - env_origins.double(), pose[..., [6, 3, 4, 5]], twist[..., :3]), -1)


def body_rates_from_isaac(pose_xyzw, world_twist):
    """Rotate measured world angular velocity into body axes, in rad/s.

    The measured rate is an observation/diagnostic, not an eleventh CTBR state.
    Pose quaternions are normalized for this coordinate transform only.
    """
    pose, twist = _physical_pose(pose_xyzw, world_twist)
    quaternion = pose[..., 3:7]
    norm = quaternion.norm(dim=-1, keepdim=True)
    quaternion = quaternion / norm
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    omega = twist[..., 3:]
    cross = torch.linalg.cross(vector, omega)
    return omega - 2 * scalar * cross + 2 * torch.linalg.cross(vector, cross)


def reference_from_ipc(reference_pva, yaw_target):
    """Map IPC [p,v,a] references to upright-yaw CTBR [p,q_wxyz,v].

    ``reference_pva`` is [B,H,9]; yaw may be [B], [B,1], or [B,H].
    Acceleration is not a CTBR state. The optimizer selects tilt and thrust;
    this conversion does not use flatness to generate the physical action.
    """
    _floating(reference_pva, 9, "reference_pva")
    if reference_pva.ndim != 3 or not isinstance(yaw_target, torch.Tensor) or not yaw_target.is_floating_point():
        raise ValueError("Expected reference_pva[B,H,9] and floating-point yaw_target")
    batch, horizon = reference_pva.shape[:2]
    if yaw_target.shape == (batch,):
        yaw_target = yaw_target[:, None]
    if yaw_target.shape not in ((batch, 1), (batch, horizon)) or yaw_target.device != reference_pva.device:
        raise ValueError("Yaw must have shape [B], [B,1], or [B,H] on the reference device")
    if not torch.isfinite(reference_pva).all() or not torch.isfinite(yaw_target).all():
        raise ValueError("Reference and yaw must be finite")
    reference = reference_pva.double()
    half_yaw = 0.5 * yaw_target.double().expand(batch, horizon)
    zero = torch.zeros_like(half_yaw)
    quaternion = torch.stack((half_yaw.cos(), zero, zero, half_yaw.sin()), -1)
    return torch.cat((reference[..., :3], quaternion, reference[..., 3:6]), -1)


def controls_to_actions(controls, mass=0.75):
    """Convert total force [N] to the task's specific thrust [m/s^2]."""
    _floating(controls, 4, "controls")
    mass = _positive(mass, "mass")
    return torch.cat((controls[..., :1] / mass, controls[..., 1:]), -1)


def reference_from_pva(reference_pva, yaw_target, gravity=9.81):
    """Convert a world PVA reference to [p,q_wxyz,v] using thrust and heading.

    The body z axis follows ``a + gravity * e_z``. Its body x/y axes follow
    the requested yaw heading. At zero required thrust, orientation is free;
    the convention is upright yaw. At a heading/thrust singularity, use the
    perpendicular horizontal heading. These choices are tensor operations,
    so the normal reference path never reads a CUDA scalar on the host.
    """
    _floating(reference_pva, 9, "reference_pva")
    if reference_pva.ndim != 3 or not isinstance(yaw_target, torch.Tensor) or not yaw_target.is_floating_point():
        raise ValueError("Expected reference_pva[B,N,9] and floating-point yaw_target")
    batch, horizon = reference_pva.shape[:2]
    if yaw_target.shape == (batch,):
        yaw_target = yaw_target[:, None]
    if yaw_target.shape not in ((batch, 1), (batch, horizon)) or yaw_target.device != reference_pva.device:
        raise ValueError("Yaw must have shape [B], [B,1], or [B,N] on the reference device")
    reference = reference_pva.double()
    yaw = yaw_target.double().expand(batch, horizon)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    world_z = torch.stack((zero, zero, one), -1)
    thrust = reference[..., 6:9] + _positive(gravity, "gravity") * world_z
    norm = thrust.norm(dim=-1, keepdim=True)
    body_z = torch.where(norm > 1e-12, thrust / norm.clamp_min(1e-12), world_z)
    heading = torch.stack((yaw.cos(), yaw.sin(), zero), -1)
    body_y = torch.linalg.cross(body_z, heading)
    alternate = torch.stack((-yaw.sin(), yaw.cos(), zero), -1)
    body_y = torch.where(
        body_y.norm(dim=-1, keepdim=True) > 1e-12,
        body_y,
        torch.linalg.cross(body_z, alternate),
    )
    body_y = body_y / body_y.norm(dim=-1, keepdim=True)
    body_x = torch.linalg.cross(body_y, body_z)
    rotation = torch.stack((body_x, body_y, body_z), -1)
    r00, r01, r02 = rotation[..., 0, :].unbind(-1)
    r10, r11, r12 = rotation[..., 1, :].unbind(-1)
    r20, r21, r22 = rotation[..., 2, :].unbind(-1)
    # Four equivalent quaternion formulas; choose the largest denominator.
    # This remains well-conditioned at rotations of pi, unlike trace-only code.
    squared = torch.stack((1 + r00 + r11 + r22, 1 + r00 - r11 - r22, 1 - r00 + r11 - r22, 1 - r00 - r11 + r22), -1)
    roots = squared.clamp_min(0).sqrt()
    candidates = torch.stack(
        (
            torch.stack((squared[..., 0], r21 - r12, r02 - r20, r10 - r01), -1),
            torch.stack((r21 - r12, squared[..., 1], r10 + r01, r02 + r20), -1),
            torch.stack((r02 - r20, r10 + r01, squared[..., 2], r12 + r21), -1),
            torch.stack((r10 - r01, r20 + r02, r21 + r12, squared[..., 3]), -1),
        ),
        -2,
    ) / (2 * roots.clamp_min(1e-12)[..., :, None])
    index = roots.argmax(-1)[..., None, None].expand(batch, horizon, 1, 4)
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    return torch.cat((reference[..., :3], quaternion, reference[..., 3:6]), -1)


class CTBRDynamics(AutoDiffDynamics):
    """State [p(3),q_wxyz(4),v_world(3)]; input [F_N,omega_body(3)]."""

    def __init__(self, *, dt=0.1, mass=0.75, gravity=9.81):
        self.dt = _positive(dt, "dt")
        self.mass = _positive(mass, "mass")
        self.gravity = _positive(gravity, "gravity")
        self.nu = 4
        super().__init__(self.step, nx=10)

    def acceleration(self, z):
        """True world acceleration, including off-unit SQP quaternion semantics."""
        w, x, y, zz = z[..., 3:7].unbind(-1)
        # Raw q * [0,0,0,F] * conjugate(q), not the unit-only 1-2*x*x-2*y*y.
        body_z = torch.stack((2 * (x * zz + w * y), 2 * (y * zz - w * x), w * w - x * x - y * y + zz * zz), -1)
        return body_z * (z[..., 10:11] / self.mass) + z.new_tensor([0.0, 0.0, -self.gravity])

    def __call__(self, z):
        return self.step(z)

    def step(self, z):
        """One or more [x,u] vectors -> next x, entirely on the input device."""
        quaternion, omega = z[..., 3:7], z[..., 11:14]
        angle2 = omega.square().sum(-1) * (0.5 * self.dt) ** 2
        # Both branches have finite derivatives at zero. The inactive sqrt
        # branch must be protected as torch.where evaluates both expressions.
        angle = angle2.clamp_min(1e-16).sqrt()
        small = angle2 < 1e-8
        cosine = torch.where(small, 1 - angle2 / 2 + angle2.square() / 24, angle.cos())
        sinc = torch.where(small, 1 - angle2 / 6 + angle2.square() / 120, angle.sin() / angle)
        increment = omega * (0.5 * self.dt * sinc[..., None])
        scalar = quaternion[..., 0] * cosine - (quaternion[..., 1:] * increment).sum(-1)
        vector = (
            quaternion[..., :1] * increment
            + cosine[..., None] * quaternion[..., 1:]
            + torch.linalg.cross(quaternion[..., 1:], increment)
        )
        next_quaternion = torch.cat((scalar[..., None], vector), -1)
        # This is the author's unconditional discrete update, not merely a
        # boundary check. Keep its exact derivative in the implicit Hessian.
        next_quaternion = next_quaternion / (next_quaternion.norm(dim=-1, keepdim=True) + QUATERNION_NORMALIZATION_EPS)
        return torch.cat(
            (z[..., :3] + self.dt * z[..., 7:10], next_quaternion, z[..., 7:10] + self.dt * self.acceleration(z)), -1
        )

    def rollout(self, initial_state, controls):
        """Integrate [B,N,4] controls from measured [B,10], returning [B,N+1,10].

        Time is sequential; all B environments stay batched on the GPU. The
        fixed-N loop can be compiled with fullgraph=True and dynamic=False.
        No predicted state from a previous control cycle is reused as x0.
        """
        if initial_state.ndim != 2 or initial_state.shape[-1] != self.nx:
            raise ValueError("Expected initial_state[B,10]")
        if controls.ndim != 3 or controls.shape[0] != len(initial_state) or controls.shape[-1] != self.nu:
            raise ValueError("Expected controls[B,N,4]")
        states, state = [initial_state], initial_state
        for control in controls.unbind(1):
            state = self.step(torch.cat((state, control), -1))
            states.append(state)
        return torch.stack(states, 1)

    def controls_to_actions(self, controls):
        return controls_to_actions(controls, self.mass)
