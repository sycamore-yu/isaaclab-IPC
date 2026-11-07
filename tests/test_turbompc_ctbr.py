"""Frozen author/NumPy CTBR physics contracts, unchanged by the 28N actor."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_ipc.turbompc.quadrotor import (
    AUTHOR_COMMIT,
    AUTHOR_SOURCE_SHA256,
    CTBRDynamics,
    body_rates_from_isaac,
    controls_to_actions,
    reference_from_ipc,
    state_from_isaac,
)


def hamilton(left, right):
    a, b, c, d = left
    e, f, g, h = right
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ]
    )


def numpy_step(z, *, dt=0.1, mass=0.75, gravity=9.81):
    """Independent Hamilton-product oracle, including off-unit rotation."""
    q, u = z[3:7], z[10:]
    half_rotation = 0.5 * dt * u[1:]
    angle = np.linalg.norm(half_rotation)
    increment = np.r_[np.cos(angle), np.sinc(angle / np.pi) * half_rotation]
    next_q = hamilton(q, increment)
    next_q /= np.linalg.norm(next_q) + 1e-8
    force_world = hamilton(hamilton(q, np.array([0.0, 0.0, 0.0, u[0]])), q * [1, -1, -1, -1])[1:]
    acceleration = force_world / mass - [0, 0, gravity]
    return np.r_[z[:3] + dt * z[7:10], next_q, z[7:10] + dt * acceleration]


def test_author_forward_fixture_is_pinned_and_matches_at_author_parameters():
    path = Path(__file__).parent / "data" / "ca_acmpc_forward_reference.npz"
    provenance = json.loads(path.with_suffix(".json").read_text())
    assert provenance["commit"] == AUTHOR_COMMIT
    assert provenance["source_sha256"] == AUTHOR_SOURCE_SHA256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with np.load(path) as fixture:
        # The source stores gravity as float32 even for a float64 model input.
        model = CTBRDynamics(
            dt=float(fixture["dt"]), mass=float(fixture["mass"]), gravity=float(-fixture["gravity"][2])
        )
        z = torch.from_numpy(np.concatenate((fixture["states"], fixture["controls"]), -1))
        value = model(z).numpy()
        np.testing.assert_allclose(value, fixture["next_states"], atol=2e-10, rtol=0)
        assert np.count_nonzero(np.linalg.norm(fixture["controls"][:, 1:], axis=1) == 0) >= 1


@pytest.mark.parametrize("rate", [(0.0, 0.0, 0.0), (1e-9, -2e-9, 3e-9), (0.8, -0.5, 0.08)])
def test_exact_map_jacobian_and_lagrangian_hessian_at_hover_and_rotation(rate):
    rng = np.random.default_rng(83)
    z_np = rng.normal(size=14)
    z_np[3:7] /= np.linalg.norm(z_np[3:7])
    z_np[3:7] *= 1.07  # Off-manifold SQP iterates need the raw Hamilton semantics.
    z_np[10:] = [7.3, *rate]
    z = torch.from_numpy(z_np)
    model = CTBRDynamics()
    value, jacobian = model.linearize(z)
    np.testing.assert_allclose(value.numpy(), numpy_step(z_np), atol=3e-15, rtol=3e-15)
    h = 1e-6
    differences = [(numpy_step(z_np + h * column) - numpy_step(z_np - h * column)) / (2 * h) for column in np.eye(14)]
    np.testing.assert_allclose(jacobian.numpy(), np.array(differences).T, atol=3e-9, rtol=3e-8)
    multipliers = torch.from_numpy(rng.normal(size=10))
    hessian = model.lagrangian_hessian(z, multipliers)
    direction = torch.from_numpy(rng.normal(size=14))
    h = 1e-5
    jp = model.linearize(z + h * direction)[1]
    jm = model.linearize(z - h * direction)[1]
    expected_hvp = (jp.T @ multipliers - jm.T @ multipliers) / (2 * h)
    assert hessian.isfinite().all()
    torch.testing.assert_close(hessian @ direction, expected_hvp, atol=3e-8, rtol=3e-7)
    torch.testing.assert_close(hessian, hessian.T, atol=1e-14, rtol=1e-14)


def test_raw_quaternion_rotation_scale_sign_and_unconditional_normalization():
    model = CTBRDynamics()
    z = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 7.3575, 0.0, 0.0, 0.0], dtype=torch.float64)
    value = model(z)
    assert value[3] == 1 / (1 + 1e-8)
    raw = z.clone()
    raw[3:7] *= 1.2
    gravity = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
    torch.testing.assert_close(model.acceleration(raw) + gravity, 1.2**2 * (model.acceleration(z) + gravity))
    flipped = raw.clone()
    flipped[3:7] *= -1
    expected = model(raw)
    expected[3:7] *= -1
    torch.testing.assert_close(model(flipped), expected, atol=0, rtol=0)
    batch = torch.stack((z, raw, flipped)).expand(2, -1, -1)
    assert model(batch).shape == (2, 3, 10)
    assert model.acceleration(batch).shape == (2, 3, 3)


def test_isaac_pose_world_velocity_body_rates_and_specific_thrust_contract():
    half = math.sqrt(0.5)
    pose = torch.tensor([[5.0, 6.0, 7.0, 0.0, 0.0, half, half]], dtype=torch.float64)
    twist = torch.tensor([[1.0, 2.0, 3.0, 0.0, 2.0, 0.3]], dtype=torch.float64)
    origins = torch.tensor([[4.0, 4.0, 4.0]], dtype=torch.float64)
    state = state_from_isaac(pose, twist, origins)
    expected = torch.tensor([[1.0, 2.0, 3.0, half, 0.0, 0.0, half, 1.0, 2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(state, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        body_rates_from_isaac(pose, twist), torch.tensor([[2.0, 0.0, 0.3]], dtype=torch.float64), atol=1e-15, rtol=0
    )
    pose[:, 3:] *= 1.01
    assert torch.linalg.vector_norm(state_from_isaac(pose, twist, origins)[0, 3:7]) > 1
    control = torch.tensor([[7.3575, 0.2, -0.3, 0.04]], dtype=torch.float64)
    torch.testing.assert_close(
        controls_to_actions(control), torch.tensor([[9.81, 0.2, -0.3, 0.04]], dtype=torch.float64)
    )


@pytest.mark.parametrize("layout", ["batch", "constant", "time"])
def test_ipc_reference_preserves_world_pv_and_uses_explicit_yaw_quaternion(layout):
    reference = torch.arange(2 * 5 * 9, dtype=torch.float64).reshape(2, 5, 9) / 10
    yaw = torch.tensor([0.3, -0.8], dtype=torch.float64)
    if layout == "constant":
        yaw = yaw[:, None]
    elif layout == "time":
        yaw = yaw[:, None].expand(-1, 5)
    converted = reference_from_ipc(reference, yaw)
    torch.testing.assert_close(converted[..., :3], reference[..., :3])
    torch.testing.assert_close(converted[..., 7:], reference[..., 3:6])
    assert not converted[..., 4:6].any()
    torch.testing.assert_close(converted[:, 0, 3], torch.cos(torch.tensor([0.15, -0.4], dtype=torch.float64)))
    torch.testing.assert_close(converted[..., 3:7].norm(dim=-1), torch.ones((2, 5), dtype=torch.float64))
