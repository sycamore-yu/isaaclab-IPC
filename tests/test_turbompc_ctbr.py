"""CPU contracts for the physical CTBR map and its seven-cost/SFC adapter."""

import hashlib
import io
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
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


def context(horizon=5, capacity=4):
    first = np.array([[1, 0, 0, -0.5], [2, 0, 0, -1.0], [0, 1, 0, -0.8], [0, -1, 0, -0.8]])
    second = np.array([[-1, 0, 0, -0.5], [0, 0, 1, -2.0], [0, 0, -1, 0.2]])
    schedule = [np.empty((0, 4))] + [first] * ((horizon - 1) // 2)
    schedule += [second] * (horizon - len(schedule))
    state = torch.tensor([[0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 12.0, -11.0, 0.2]], dtype=torch.float64)
    reference = state[:, None].expand(-1, horizon, -1).clone()
    reference[0, :, :3] = torch.from_numpy(np.linspace([0.1, -0.2, 1], [1.0, 0.4, 1.2], horizon))
    reference[:, :, 7:] = torch.tensor([0.3, -0.1, 0.2])
    planes = torch.zeros((1, horizon, capacity, 4), dtype=torch.float64)
    mask = torch.zeros((1, horizon, capacity), dtype=torch.bool)
    for k, faces in enumerate(schedule):
        planes[0, k, : len(faces)] = torch.from_numpy(faces[::-1].copy())
        mask[0, k, : len(faces)] = True
    return {"ctbr_state": state, "ctbr_reference": reference, "planes": planes, "plane_mask": mask}, schedule


def physical_cost(z, reference, cost, cfg):
    """Reference objective in physical groups, independent of MPC matrix assembly."""
    scales = 2.0 ** np.tanh(cost[:4])
    bias = cfg.position_bias_scale * np.tanh(cost[4:])
    total = 0.0
    for k in range(cfg.horizon):
        state = z[k + 1, :10]
        factor = cfg.terminal_state_factor if k == cfg.horizon - 1 else 1.0
        total += 0.5 * factor * cfg.position_weight * scales[0] * np.sum((state[:3] - reference[k, :3]) ** 2)
        total += 0.5 * factor * cfg.velocity_weight * scales[1] * np.sum((state[7:10] - reference[k, 7:10]) ** 2)
        total += 0.5 * factor * cfg.attitude_weight * scales[2] * np.sum((state[3:7] - reference[k, 3:7]) ** 2)
        total -= factor * cfg.position_weight * np.dot(bias, state[:3])
        difference = z[k, 10:] - np.array([cfg.mass * cfg.gravity, 0, 0, 0])
        total += 0.5 * scales[3] * np.dot(cfg.input_weights, difference**2)
    dummy = z[-1, 10:] - np.array([cfg.mass * cfg.gravity, 0, 0, 0])
    return total + 0.5 * np.dot(dummy, dummy)


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


def test_builder_objective_and_seven_cost_gradients_match_physical_groups():
    cfg = CTBRConfig(horizon=5)
    builder = CTBRQPBuilder(cfg)
    obs, _ = context()
    costs = torch.tensor([[0.2, -0.3, 0.4, -0.1, 0.15, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    inputs, guess = builder(obs, costs)
    rng = np.random.default_rng(918)
    left = guess + torch.from_numpy(rng.normal(size=guess.shape))
    right = guess + torch.from_numpy(rng.normal(size=guess.shape))

    def matrix_cost(z):
        return (0.5 * z * (inputs.D @ z[..., None]).squeeze(-1) + inputs.q * z).sum()

    actual = matrix_cost(left) - matrix_cost(right)
    reference = obs["ctbr_reference"][0].numpy()

    def independent(cost):
        return physical_cost(left[0].numpy(), reference, cost, cfg) - physical_cost(
            right[0].numpy(), reference, cost, cfg
        )

    expected = independent(costs.detach()[0].numpy())
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-10, rel=1e-12)
    gradient = torch.autograd.grad(actual, costs)[0][0].numpy()
    h = 1e-5
    c = costs.detach()[0].numpy()
    fd = np.array([(independent(c + h * e) - independent(c - h * e)) / (2 * h) for e in np.eye(7)])
    np.testing.assert_allclose(gradient, fd, atol=1e-7, rtol=1e-7)
    assert not inputs.E.any()
    torch.testing.assert_close(inputs.D[0, -1, 10:, 10:], torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(inputs.q[0, -1, 10:], -builder.hover_control)
    assert not inputs.G[0, -1, :, 10:].any()


def test_hover_initial_guess_obeys_exact_nonlinear_dynamics_without_clipping_state():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, _ = context()
    inputs, guess = builder(obs)
    torch.testing.assert_close(inputs.initial_state, obs["ctbr_state"], atol=0, rtol=0)
    assert inputs.initial_state[0, 7] > builder.velocity_upper[0]
    expected = obs["ctbr_state"][0].numpy().copy()
    np.testing.assert_allclose(guess[0, 0, :10].numpy(), expected, atol=0, rtol=0)
    for k in range(5):
        expected = numpy_step(np.r_[expected, builder.hover_control.numpy()])
        np.testing.assert_allclose(guess[0, k + 1, :10].numpy(), expected, atol=3e-14, rtol=1e-14)
    torch.testing.assert_close(guess[..., 10:], builder.hover_control.expand(1, 6, -1))
    torch.testing.assert_close(builder.action_lower[0] * builder.config.mass, builder.control_lower[0])
    torch.testing.assert_close(builder.action_upper[0] * builder.config.mass, builder.control_upper[0])


def test_all_sfc_faces_and_time_slots_are_lossless_for_dense_and_compact_storage():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, schedule = context()
    dense, _ = builder(obs)
    packed = dict(obs)
    for key, value in zip(("planes", "plane_mask", "corridor_index"), pack_corridors(schedule, 4), strict=True):
        packed[key] = torch.from_numpy(value)[None]
    compact, _ = builder(packed)
    for name in dense.__dataclass_fields__:
        torch.testing.assert_close(getattr(dense, name), getattr(compact, name), atol=0, rtol=0)
    assert not dense.G[0, :2, 7:].any()
    for k, faces in enumerate(schedule):
        np.testing.assert_array_equal(dense.G[0, k + 1, 7 : 7 + len(faces), :3].numpy(), faces[::-1, :3])
        np.testing.assert_array_equal(dense.upper[0, k + 1, 7 : 7 + len(faces)].numpy(), -faces[::-1, 3])
    assert torch.isneginf(dense.lower[:, 0, :3]).all()
    assert torch.isposinf(dense.upper[:, 0, :3]).all()
    assert torch.isneginf(dense.lower[:, -1, 3:7]).all()
    assert torch.isposinf(dense.upper[:, -1, 3:7]).all()


def test_inactive_nan_padding_and_faces_beyond_64_are_preserved():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, _ = context()
    base, _ = builder(obs)
    padded = dict(obs)
    padded["planes"] = torch.nn.functional.pad(obs["planes"], (0, 0, 0, 65), value=float("nan"))
    padded["plane_mask"] = torch.nn.functional.pad(obs["plane_mask"], (0, 65))
    expanded, _ = builder(padded)
    torch.testing.assert_close(expanded.G[:, :, :11], base.G, atol=0, rtol=0)
    assert not expanded.G[:, :, 11:].any()
    assert torch.isposinf(expanded.upper[:, :, 11:]).all()
    padded["planes"][0, -1, 68] = torch.tensor([3.0, 0.0, 0.0, -0.9])
    padded["plane_mask"][0, -1, 68] = True
    full, _ = builder(padded)
    torch.testing.assert_close(full.G[0, 5, 75, :3], torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))
    assert full.upper[0, 5, 75] == -padded["planes"][0, -1, 68, 3]


def test_reference_quaternion_sign_changes_are_aligned_to_initial_hemisphere():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, _ = context()
    base, _ = builder(obs)
    signed = {key: value.clone() for key, value in obs.items()}
    signed["ctbr_reference"][:, ::2, 3:7] *= -1
    aligned, _ = builder(signed)
    torch.testing.assert_close(base.q, aligned.q, atol=0, rtol=0)
    signed["ctbr_state"][:, 3:7] *= -1
    negative, _ = builder(signed)
    expected = base.q.clone()
    expected[..., 3:7] *= -1
    torch.testing.assert_close(negative.q, expected, atol=0, rtol=0)


def test_acceleration_path_has_true_hamilton_value_and_no_terminal_dummy_constraint():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, _ = context()
    _, guess = builder(obs)
    path = builder.path_constraints()
    guess = guess.clone()
    guess[0, -1, 10] = 1000
    expected = torch.stack([(torch.from_numpy(numpy_step(row.numpy()))[7:] - row[7:10]) / 0.1 for row in guess[0]])
    torch.testing.assert_close(path(guess)[0], expected, atol=3e-14, rtol=1e-14)
    assert (path(guess)[0, -1] > builder.acceleration_upper).any()
    assert not path.violations(guess)[0, -1].any()
    assert not path.normal_cone_residual(guess, torch.ones_like(path(guess)))[0, -1].any()
    assert builder.acceleration_stage_mask[:-1].all() and not builder.acceleration_stage_mask[-1].any()


def test_checkpoint_roundtrip_and_semantics_or_physical_mismatch_rejection():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    stream = io.BytesIO()
    torch.save(builder.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    CTBRQPBuilder(builder.config).load_state_dict(saved)
    assert saved["_extra_state"]["config"] == asdict(builder.config)
    assert saved["_extra_state"]["cost_groups"] == [
        "position",
        "velocity",
        "attitude",
        "input",
        "bias_x",
        "bias_y",
        "bias_z",
    ]
    with pytest.raises(RuntimeError, match="physical configuration"):
        CTBRQPBuilder(replace(builder.config, mass=0.752)).load_state_dict(saved)
    saved["_extra_state"]["cost_semantics_version"] = 0
    with pytest.raises(RuntimeError, match="cost semantics"):
        CTBRQPBuilder(builder.config).load_state_dict(saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass", 0.0),
        ("dt", float("nan")),
        ("horizon", 0),
        ("body_rate_limits", (11.0, 1.0, 1.0)),
        ("specific_thrust_limits", (10.0, 12.0)),
        ("input_weights", (0.2, 0.0, 0.1, 0.2)),
    ],
)
def test_invalid_physical_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        CTBRConfig(**{field: value})


def test_invalid_external_state_geometry_and_schedule_are_rejected():
    builder = CTBRQPBuilder(CTBRConfig(horizon=5))
    obs, schedule = context()
    obs["ctbr_state"][:, 3:7] *= 0.8
    with pytest.raises(ValueError, match="near unit"):
        builder(obs)
    obs, _ = context()
    obs["planes"][0, 1, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="active SFC"):
        builder(obs)
    for key, value in zip(("planes", "plane_mask", "corridor_index"), pack_corridors(schedule, 4), strict=True):
        obs[key] = torch.from_numpy(value)[None]
    obs["corridor_index"][0, 0] = -2
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        builder(obs)
