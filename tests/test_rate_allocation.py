"""Physical collective/RP/yaw priority against an independent motor-force LP."""

import numpy as np
import pytest
from scipy.optimize import linprog
import warp as wp

from isaaclab_ipc.assets.quadrotor import INERTIA, MASS, MAX_THRUST, QUADROTOR_CFG
from isaaclab_ipc.assets.rate_control import rate_targets


@wp.kernel
def previous_uniform_targets(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    ids: wp.array(dtype=wp.int32),
    command: wp.array2d(dtype=wp.float32),
    inverse: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    mass: float,
    inertia: float,
    max_thrust: float,
):
    """Prior production kernel, only for the bitwise feasible-path regression."""
    i = wp.tid()
    q = wp.transform_get_rotation(body_q[ids[i]])
    twist = body_qd[ids[i]]
    omega = wp.quat_rotate_inv(q, wp.vec3(twist[3], twist[4], twist[5]))
    desired = wp.vec3(command[i, 1], command[i, 2], command[i, 3])
    torque = inertia * 45.0 * (desired - omega)
    collective = wp.clamp(command[i, 0] * mass, 0.0, 4.0 * max_thrust)
    baseline = collective * 0.25
    scale = 1.0
    for j in range(4):
        delta = inverse[j, 1] * torque[0] + inverse[j, 2] * torque[1] + inverse[j, 3] * torque[2]
        if delta > 0.0:
            scale = wp.min(scale, (max_thrust - baseline) / delta)
        elif delta < 0.0:
            scale = wp.min(scale, baseline / (-delta))
    for j in range(4):
        delta = inverse[j, 1] * torque[0] + inverse[j, 2] * torque[1] + inverse[j, 3] * torque[2]
        target[i, j] = wp.clamp(baseline + wp.clamp(scale, 0.0, 1.0) * delta, 0.0, max_thrust)


def motor_force_lp(allocation, collective, torque, maximum):
    """Optimize motor forces and RP scale; no affine yaw interval formulas."""
    equality = np.zeros((3, 5))
    equality[:, :4] = allocation[:3]
    equality[1:, 4] = -torque[:2]
    rhs = [collective, 0.0, 0.0]
    bounds = [(0.0, maximum)] * 4 + [(0.0, 1.0)]
    first = linprog([0, 0, 0, 0, -1], A_eq=equality, b_eq=rhs, bounds=bounds, method="highs")
    assert first.success, first.message
    scale = first.x[4]
    equality = np.vstack((equality, [0, 0, 0, 0, 1]))
    rhs = [*rhs, scale]
    yaw = np.r_[allocation[3], 0.0]
    lower = linprog(yaw, A_eq=equality, b_eq=rhs, bounds=bounds, method="highs")
    upper = linprog(-yaw, A_eq=equality, b_eq=rhs, bounds=bounds, method="highs")
    assert lower.success and upper.success
    return scale, float(np.clip(torque[2], lower.fun, -upper.fun))


def launch(kernel, command, omega, inverse, device):
    count = len(command)
    pose = np.zeros((count, 7), dtype=np.float32)
    pose[:, 6] = 1.0
    twist = np.zeros((count, 6), dtype=np.float32)
    twist[:, 3:] = omega
    target = wp.empty((count, 4), dtype=wp.float32, device=device)
    wp.launch(kernel, dim=count, inputs=[
        wp.array(pose, dtype=wp.transform, device=device),
        wp.array(twist, dtype=wp.spatial_vector, device=device),
        wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(command, dtype=wp.float32, device=device),
        wp.array(inverse, dtype=wp.float32, device=device), target,
        MASS, INERTIA, MAX_THRUST,
    ], device=device)
    return target.numpy()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_newton_priority_matches_independent_motor_force_lp(device):
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    allocation = np.asarray(QUADROTOR_CFG.allocation_matrix, dtype=np.float32)[[2, 3, 4, 5]]
    inverse = np.linalg.inv(allocation).astype(np.float32)
    rng = np.random.default_rng(1091)
    count = 32
    command = np.zeros((count, 4), dtype=np.float32)
    command[:, 0] = rng.uniform(0.0, 4.0 * MAX_THRUST / MASS, count)
    command[:, 1:] = rng.uniform(-3.0, 3.0, (count, 3))
    omega = rng.uniform(-2.0, 2.0, (count, 3)).astype(np.float32)
    # Both collective endpoints, zero torque, feasible small requests, yaw-only,
    # and the directly measured Dense V19 endpoint.
    command[:8, 0] = [0, 4 * MAX_THRUST / MASS, 9.81, 9.81, 9.81, 9.81, 9.81, 9.81]
    command[:8, 1:] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [.01, -.01, .01],
                       [0, 0, 3], [3, 3, 0], [3, -3, 1.5], [-3, 3, -1.5]]
    omega[:8] = 0
    command[8] = [5.861665964126587 / MASS, 3, 3, 1.4688522815704346]
    omega[8] = [-.1414310336112976, .07296504825353622, .22537021338939667]
    # Nonzero torques at both collective limits must still yield no torque.
    command[9:11, 0] = [0, 4 * MAX_THRUST / MASS]
    actual = launch(rate_targets, command, omega, inverse, device)
    assert np.isfinite(actual).all()
    assert actual.min() >= 0 and actual.max() <= np.float32(MAX_THRUST)
    torque = np.float32(INERTIA) * np.float32(45.0) * (command[:, 1:] - omega)
    collective = np.clip(command[:, 0] * np.float32(MASS), 0.0, np.float32(4 * MAX_THRUST))
    wrench = actual.astype(float) @ allocation.astype(float).T
    np.testing.assert_allclose(wrench[:, 0], collective, atol=8e-6, rtol=0)
    for row in range(count):
        scale, yaw = motor_force_lp(allocation.astype(float), float(collective[row]),
                                    torque[row].astype(float), float(np.float32(MAX_THRUST)))
        np.testing.assert_allclose(wrench[row, 1:3], scale * torque[row, :2], atol=1e-5, rtol=0)
        np.testing.assert_allclose(wrench[row, 3], yaw, atol=1e-5, rtol=0)
    # Preserve the old feasible path bitwise, independently of LP tolerances.
    previous = launch(previous_uniform_targets, command[:4], omega[:4], inverse, device)
    np.testing.assert_array_equal(actual[:4], previous)
    dense_rp_scale = np.dot(wrench[8, 1:3], torque[8, :2]) / np.dot(torque[8, :2], torque[8, :2])
    assert dense_rp_scale == pytest.approx(.47628164598332473, abs=2e-6)
    assert wrench[8, 3] < 0  # Maximal RP may require yaw opposite to its request.

