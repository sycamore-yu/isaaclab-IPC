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
    if not torch.isfinite(pose_xyzw).all() or not torch.isfinite(world_twist).all():
        raise ValueError("Physical pose and twist must be finite")
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
    if not torch.isfinite(env_origins).all():
        raise ValueError("Environment origins must be finite")
    return torch.cat((pose[..., :3] - env_origins.double(), pose[..., [6, 3, 4, 5]], twist[..., :3]), -1)


def body_rates_from_isaac(pose_xyzw, world_twist):
    """Rotate measured world angular velocity into body axes, in rad/s.

    The measured rate is an observation/diagnostic, not an eleventh CTBR state.
    Pose quaternions are normalized for this coordinate transform only.
    """
    pose, twist = _physical_pose(pose_xyzw, world_twist)
    quaternion = pose[..., 3:7]
    norm = quaternion.norm(dim=-1, keepdim=True)
    if (norm <= 1e-12).any():
        raise ValueError("Physical quaternion must be nonzero")
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

    def step(self, z):
        """One vector [x,u] -> next x; batching/AD come from AutoDiffDynamics."""
        quaternion, omega = z[3:7], z[11:14]
        angle2 = omega.square().sum() * (0.5 * self.dt) ** 2
        # Both branches have finite derivatives at zero. The inactive sqrt
        # branch must be protected as torch.where evaluates both expressions.
        angle = angle2.clamp_min(1e-16).sqrt()
        small = angle2 < 1e-8
        cosine = torch.where(small, 1 - angle2 / 2 + angle2.square() / 24, angle.cos())
        sinc = torch.where(small, 1 - angle2 / 6 + angle2.square() / 120, angle.sin() / angle)
        increment = omega * (0.5 * self.dt * sinc)
        scalar = quaternion[0] * cosine - (quaternion[1:] * increment).sum()
        vector = quaternion[0] * increment + cosine * quaternion[1:] + torch.linalg.cross(quaternion[1:], increment)
        next_quaternion = torch.cat((scalar.reshape(1), vector))
        # This is the author's unconditional discrete update, not merely a
        # boundary check. Keep its exact derivative in the implicit Hessian.
        next_quaternion = next_quaternion / (next_quaternion.norm() + QUATERNION_NORMALIZATION_EPS)
        return torch.cat((z[:3] + self.dt * z[7:10], next_quaternion, z[7:10] + self.dt * self.acceleration(z)))

    def controls_to_actions(self, controls):
        return controls_to_actions(controls, self.mass)
