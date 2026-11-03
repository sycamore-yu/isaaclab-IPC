"""CPU witnesses, reference ownership, and the recorded infeasible navigation state."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_ipc.ctbr_reference import (
    REFERENCE_DIAGNOSTIC_FIELDS,
    REFERENCE_GOVERNOR_VERSION,
    CTBRReferenceConfig,
    ReferenceReason,
    _backup,
    _ctbr_step,
    _horizontal_radius,
    _policy_table,
    adapt_ctbr_reference,
)
from isaaclab_ipc.frontend import PlanningContext
from isaaclab_ipc.turbompc.ctbr import CTBRConfig
from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics


def box(xmax=10.0, *, xmin=-1.0, ymin=-3.0, ymax=3.0):
    return np.array(
        [[1.0, 0, 0, -xmax], [-1, 0, 0, xmin], [0, 1, 0, -ymax], [0, -1, 0, ymin], [0, 0, 1, -3], [0, 0, -1, 0]]
    )


def straight_context(xmax=10.0, speed=2.0):
    path = np.column_stack((np.linspace(0, 8, 801), np.zeros(801), np.ones(801)))
    indices = np.minimum(np.arange(15) * int(speed * 10), int((xmax - 0.05) * 100))
    state = np.array([0.01, 0, 1, 1, 0, 0, 0, 0.3, 0.02, 0.01])
    measured = np.r_[state[:3], state[7:], np.zeros(3)]
    reference = np.zeros((15, 9))
    reference[:, :3] = path[indices]
    reference[:, 3:6] = np.diff(np.vstack((state[:3], reference[:, :3])), axis=0) / 0.1
    return PlanningContext(measured, reference, [box(xmax)] * 15, True), state, path


def assert_original_witness(witness, model, faces):
    """Independently replay the reported controls through the production CPU map."""
    assert witness["certified"] and witness["reason"] == "verified_local_brake"
    states, controls = witness["states"], witness["controls"]
    assert len(controls) >= model.horizon and len(states) == len(controls) + 1
    dynamics = CTBRDynamics(dt=model.dt, mass=model.mass, gravity=model.gravity)
    joined = torch.from_numpy(np.concatenate((states[:-1], controls), axis=1))
    expected = torch.stack([dynamics.step(z) for z in joined]).numpy()
    acceleration = dynamics.acceleration(joined).numpy()
    np.testing.assert_allclose(states[1:], expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(witness["accelerations"], acceleration, atol=2e-14, rtol=2e-14)
    assert (states[:, :3] @ faces[:, :3].T + faces[:, 3] <= 0).all()
    assert (states[1:, 7:] >= model.velocity_lower).all()
    assert (states[1:, 7:] <= model.velocity_upper).all()
    assert (acceleration >= model.acceleration_lower).all()
    assert (acceleration <= model.acceleration_upper).all()
    assert (controls[:, 0] >= model.mass * model.specific_thrust_limits[0]).all()
    assert (controls[:, 0] <= model.mass * model.specific_thrust_limits[1]).all()
    assert (np.abs(controls[:, 1:]) <= model.body_rate_limits).all()
    assert np.linalg.norm(states[-1, 7:]) <= CTBRReferenceConfig().stop_speed
    assert np.linalg.norm(acceleration[-1]) <= CTBRReferenceConfig().stop_acceleration
    assert witness["max_plane_violation"] == witness["max_physical_violation"] == 0


@pytest.mark.parametrize("rate_scale,q_scale", [(0, 1), (1e-7, 1), (1.3, 1), (1.3, 0.97), (2.0, 1.00001)])
def test_numpy_discrete_map_preserves_canonical_raw_quaternion_semantics(rate_scale, q_scale):
    model, rng = CTBRConfig(), np.random.default_rng(203)
    dynamics = CTBRDynamics(dt=model.dt, mass=model.mass, gravity=model.gravity)
    for _ in range(8):
        state = rng.normal(size=10)
        state[3:7] *= q_scale / np.linalg.norm(state[3:7])
        control = np.r_[rng.uniform(1, 9), rate_scale * rng.normal(size=3)]
        expected = dynamics.step(torch.from_numpy(np.r_[state, control])).numpy()
        np.testing.assert_allclose(_ctbr_step(state, control, model), expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("speed", [2.0, 4.0])
def test_open_geometry_identity_has_complete_actual_and_distinct_virtual_witnesses(speed):
    context, state, path = straight_context(speed=speed)
    # V2's wide-open identity has large clearance in every horizontal direction.
    context.corridors = [box(10, xmin=-10, ymin=-10, ymax=10)] * 15
    original = [a.tobytes() for a in (context.state, context.reference, state, path, context.corridors[0])]
    for value in (context.state, context.reference, state, path, context.corridors[0]):
        value.flags.writeable = False
    adapted, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    assert adapted is context and diagnostics["reason_code"] == ReferenceReason.IDENTITY_CERTIFIED
    assert diagnostics["version"] == REFERENCE_GOVERNOR_VERSION == 2
    assert diagnostics["reference_scale"] == 1 and diagnostics["actual_backup_matches_returned_schedule"]
    assert diagnostics["actual_backup"]["state_kind"] == "measured"
    assert diagnostics["selected_backup"]["state_kind"] == "virtual_path_speed_at_measured_pose"
    np.testing.assert_array_equal(diagnostics["actual_backup"]["states"][0], state)
    np.testing.assert_array_equal(diagnostics["selected_backup"]["states"][0, 8:], state[8:])
    assert diagnostics["selected_backup"]["states"][0, 7] == pytest.approx(speed)
    np.testing.assert_array_equal(diagnostics["forward_tangent_xy"], [1.0, 0.0])
    # The first nearest reference is behind the vehicle and is preserved exactly.
    assert adapted.reference[0, 0] < state[0] and adapted.reference[0, 3] < 0
    for name in ("actual_backup", "selected_backup"):
        assert_original_witness(diagnostics[name], CTBRConfig(), context.corridors[0])
    selected = diagnostics["selected_backup"]
    assert not diagnostics["actual_backup"]["horizontal_reserve_required"]
    assert selected["horizontal_reserve_required"]
    assert selected["horizontal_max_excursion"] <= selected["horizontal_clearances"][selected["corridor"]]
    assert original == [a.tobytes() for a in (context.state, context.reference, state, path, context.corridors[0])]
    assert np.isfinite([diagnostics[k] for k in REFERENCE_DIAGNOSTIC_FIELDS]).all()


def test_limited_geometry_selects_reported_grid_and_owns_all_adapted_arrays():
    context, state, path = straight_context(xmax=0.5)
    faces = np.vstack((context.corridors[0], context.corridors[0][[0, 3]]))
    context.corridors = [faces] * 15
    state_before, reference_before, faces_before = context.state.copy(), context.reference.copy(), faces.copy()
    adapted, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    assert adapted is not context and adapted.valid == context.valid
    assert diagnostics["reason_code"] == ReferenceReason.ADAPTED_LOCAL_BACKUP
    assert diagnostics["actual_backup_certified"] and not diagnostics["nominal_backup_certified"]
    assert 0 < diagnostics["selected_speed"] < diagnostics["nominal_speed"]
    attempts = diagnostics["evaluated_speeds"]
    assert all(not item["certified"] for item in attempts[:-1]) and attempts[-1]["certified"]
    assert attempts[-1]["speed"] == diagnostics["selected_speed"]
    assert max(np.diff(adapted.reference[:, 0])) <= diagnostics["selected_speed"] * 0.1
    np.testing.assert_array_equal(adapted.reference[0, :3], context.reference[0, :3])
    np.testing.assert_array_equal(adapted.state, state_before)
    assert not np.shares_memory(adapted.state, context.state)
    assert not np.shares_memory(adapted.reference, context.reference)
    for output in adapted.corridors:
        np.testing.assert_array_equal(output, faces_before)
        assert not np.shares_memory(output, faces)
    assert_original_witness(diagnostics["selected_backup"], CTBRConfig(), faces_before)
    adapted.state[0] += 1
    adapted.reference[0, 0] += 1
    adapted.corridors[0][0, 3] += 1
    np.testing.assert_array_equal(context.state, state_before)
    np.testing.assert_array_equal(context.reference, reference_before)
    np.testing.assert_array_equal(faces, faces_before)


def test_adapted_float32_reference_retains_checked_float64_path_points():
    context, state, path = straight_context(xmax=0.5)
    path = path.astype(np.float32)
    context.reference = context.reference.astype(np.float32)
    original = context.reference.copy()
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    assert diagnostics["applied"] and result.reference.dtype == np.float64
    expected = diagnostics["selected_speed"] * CTBRConfig().dt
    assert result.reference[1, 0] == expected
    assert result.reference[1, 0] != np.float64(np.float32(expected))
    np.testing.assert_array_equal(context.reference, original)


def test_corner_retiming_uses_original_path_and_keeps_complete_two_polytope_schedule():
    context, state, _ = straight_context(xmax=0.5)
    east = np.column_stack((np.linspace(0, 0.3, 31), np.zeros(31), np.ones(31)))
    north = np.column_stack((np.full(280, 0.3), np.linspace(0.01, 2.8, 280), np.ones(280)))
    path = np.vstack((east, north))
    old, new = box(0.5, ymax=0.3), box(0.5, xmin=0.2)
    old = np.vstack((old, old[[1]]))
    new = np.vstack((new, new[[0, 0]]))
    reference = context.reference.copy()
    reference[:, :3] = path[np.arange(15) * 20]
    context = replace(context, reference=reference, corridors=[old] * 3 + [new] * 12)
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    assert diagnostics["applied"] and diagnostics["selected_backup_certified"]
    positions = result.reference[:, :3]
    # Resampled positions remain on the east-then-north path, not its skipped-corner chord.
    assert ((positions[:, 1] == 0) | (positions[:, 0] == 0.3)).all()
    assert ((positions[:, 0] == 0.3) & (positions[:, 1] > 0) & (positions[:, 1] < 0.1)).any()
    indices = diagnostics["adapted_corridor_indices"]
    assert (np.diff(indices) >= 0).all() and set(indices) == {0, 1}
    for point, planes, index in zip(positions, result.corridors, indices, strict=True):
        np.testing.assert_array_equal(planes, [old, new][index])
        assert not np.shares_memory(planes, [old, new][index])
        assert (planes[:, :3] @ point + planes[:, 3] <= 0).all()
    # Local actual brake stays in old; this is explicitly not a witness for the returned switch.
    assert diagnostics["actual_backup_certified"]
    assert not diagnostics["actual_backup_matches_returned_schedule"]


def test_float32_origin_subtraction_is_compared_at_observation_precision_only():
    context, state, path = straight_context()
    pose, origin = np.float32(8.8765), np.float32(-1.2345)
    measured_position = np.float32(pose - origin)
    exact_position = np.float64(pose) - np.float64(origin)
    shift = exact_position - state[0]
    path[:, 0] += shift
    context.reference[:, 0] += shift
    state[0] = exact_position
    context.state = context.state.astype(np.float32)
    context.state[0] = measured_position
    state[7:] = context.state[3:6].astype(np.float64)
    for face in context.corridors[0]:
        face[3] -= face[0] * shift
    original = state.copy()
    assert measured_position != exact_position and np.float32(exact_position) == measured_position
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    assert result is context and diagnostics["actual_backup_certified"]
    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(diagnostics["actual_backup"]["states"][0], original)
    inconsistent = state.copy()
    inconsistent[0] = np.nextafter(measured_position, np.float32(np.inf), dtype=np.float32)
    with pytest.raises(ValueError, match="share the measured"):
        adapt_ctbr_reference(context, inconsistent, CTBRConfig(), path=path, nearest=0)


def test_saved_infeasible_nav_state_is_not_certified_or_rewritten():
    source = Path(__file__).with_name("data") / "ctbr_reference_nav_failure_v2.npz"
    metadata = json.loads(source.with_suffix(".json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    assert metadata["source_sha256"] == "65599edcb1b3759766794c8229a5322914cf4bb7c65471b2fc89708a18618d0f"
    with np.load(source) as saved:
        state = saved["ctbr_state"].copy()
        planes = [saved["old_planes"].copy(), saved["new_planes"].copy()]
        context = PlanningContext(
            saved["mpc_state"].copy(), saved["reference"].copy(), [planes[i] for i in saved["corridor_index"]], True
        )
    original = state.copy(), context.state.copy(), context.reference.copy()
    # Original Planner.path was not saved. No invented path is needed to check actual braking.
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig())
    assert result is context and result.valid
    assert diagnostics["reason_code"] == ReferenceReason.ACTUAL_BACKUP_NOT_CERTIFIED
    assert not diagnostics["actual_backup_certified"] and not diagnostics["selected_backup_certified"]
    assert not diagnostics["actual_backup_matches_returned_schedule"]
    assert diagnostics["evaluated_speeds"] == []
    assert diagnostics["actual_backup"]["reason"] in ("left_every_single_polytope", "physical_bound_or_nonfinite")
    backup = diagnostics["actual_backup"]
    assert not backup["horizontal_reserve_required"]
    assert backup["horizontal_max_excursion"] == np.linalg.norm(backup["states"][:, :2] - state[:2], axis=1).max()
    np.testing.assert_array_equal(state, original[0])
    np.testing.assert_array_equal(context.state, original[1])
    np.testing.assert_array_equal(context.reference, original[2])
    assert np.isfinite([diagnostics[k] for k in REFERENCE_DIAGNOSTIC_FIELDS]).all()


@pytest.mark.parametrize("mode", ["invalid", "no_path", "short_budget"])
def test_unavailable_certificate_preserves_geometry_and_identity(mode):
    context, state, path = straight_context()
    options, kwargs = CTBRReferenceConfig(), {"path": path, "nearest": 0}
    if mode == "invalid":
        context = replace(context, valid=False)
        expected = ReferenceReason.INVALID_CONTEXT
    elif mode == "no_path":
        kwargs = {}
        expected = ReferenceReason.NO_FORWARD_PATH
    else:
        options = CTBRReferenceConfig(max_backup_steps=2)
        expected = ReferenceReason.ACTUAL_BACKUP_NOT_CERTIFIED
    before = context.state.tobytes(), context.reference.tobytes(), context.corridors[0].tobytes()
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), options=options, **kwargs)
    assert result is context and diagnostics["reason_code"] == expected
    assert not diagnostics["applied"] and result.valid == context.valid
    assert before == (context.state.tobytes(), context.reference.tobytes(), context.corridors[0].tobytes())


def test_horizontal_disc_uses_all_faces_at_actual_height_and_ignores_caps():
    position = np.array([1.0, 2.0, 3.0])
    planes = np.array([[3.0, 4, 0, -17], [0, -2, 0, 1], [1, 0, 2, -8], [2, 0, 4, -16], [0, 0, 1, -3.0000001]])
    # Horizontal distances are 6/5, 3/2, 1, 1; the close ceiling imposes no horizontal cap.
    assert _horizontal_radius(planes, position) == 1.0
    assert _horizontal_radius(planes * np.array([2, 3, 0.5, 7, 9])[:, None], position) == 1.0
    circle = position + np.column_stack(
        (np.cos(np.linspace(0, 2 * np.pi, 33)), np.sin(np.linspace(0, 2 * np.pi, 33)), np.zeros(33))
    )
    assert (circle @ planes[:, :3].T + planes[:, 3] <= 0).all()
    assert ((position + [1.01, 0, 0]) @ planes[:, :3].T + planes[:, 3] > 0).any()


def test_reference_reserve_does_not_reclassify_actual_full_face_backup():
    context, state, path = straight_context()
    state[7] = context.state[3] = 2.0
    context.corridors = [box(10, xmin=-10, ymin=-0.3, ymax=0.3)] * 15
    result, diagnostics = adapt_ctbr_reference(context, state, CTBRConfig(), path=path, nearest=0)
    actual, virtual = diagnostics["actual_backup"], diagnostics["selected_backup"]
    assert result is not context and diagnostics["actual_backup_certified"]
    assert not actual["horizontal_reserve_required"] and actual["horizontal_clearances"] is None
    assert actual["horizontal_max_excursion"] > 0.3
    assert virtual["horizontal_reserve_required"] and virtual["horizontal_max_excursion"] <= 0.3
    assert 0 < diagnostics["selected_speed"] < diagnostics["nominal_speed"]
    assert diagnostics["evaluated_speeds"][0]["reason"] == "horizontal_reserve_not_verified"
    np.testing.assert_array_equal(actual["states"][0], state)
    assert_original_witness(actual, CTBRConfig(), context.corridors[0])
    assert_original_witness(virtual, CTBRConfig(), context.corridors[0])


def test_same_polytope_must_provide_the_radius_and_the_entire_three_dimensional_witness():
    model, options = CTBRConfig(), CTBRReferenceConfig()
    table = _policy_table(model, options)
    state = np.array([0.0, 0, 1, 1, 0, 0, 0, 2.0, 0, 0.1])
    wide_but_low = box(10, xmin=-10, ymin=-10, ymax=10)
    wide_but_low[4, 3] = -1.005
    narrow_but_high = box(10, xmin=-10, ymin=-0.3, ymax=0.3)
    original = _backup(state, [wide_but_low, narrow_but_high], model, options, table)
    checked = _backup(state, [wide_but_low, narrow_but_high], model, options, table, require_horizontal_reserve=True)
    assert original["certified"] and original["corridor"] == 1
    assert not checked["certified"] and checked["reason"] == "horizontal_reserve_not_verified"
    assert checked["horizontal_clearances"].tolist() == [10.0, 0.3]
    assert np.linalg.norm(checked["states"][1, :2] - state[:2]) < 0.3
    assert checked["horizontal_max_excursion"] > 0.3 and len(checked["states"]) > 2
    # The large-radius polytope fails at z1: borrowing that radius would be incorrect.
    assert checked["states"][1, 2] > 1.005
    assert checked["horizontal_max_excursion"] == np.linalg.norm(checked["states"][:, :2] - state[:2], axis=1).max()
    high_and_wide = box(10, xmin=-10, ymin=-10, ymax=10)
    accepted = _backup(state, [narrow_but_high, high_and_wide], model, options, table, require_horizontal_reserve=True)
    assert accepted["certified"] and accepted["corridor"] == 1
    assert_original_witness(accepted, model, high_and_wide)


def test_pure_vertical_caps_give_no_horizontal_cap_but_still_constrain_the_full_trajectory():
    model, options = CTBRConfig(), CTBRReferenceConfig()
    table = _policy_table(model, options)
    state = np.array([0.0, 0, 1, 1, 0, 0, 0, 4.0, 0, 0])
    caps = np.array([[0.0, 0, 1, -1.000001], [0, 0, -1, 0.999999]])
    result = _backup(state, [caps], model, options, table, require_horizontal_reserve=True)
    assert result["certified"] and np.isinf(result["horizontal_clearances"][0])
    assert result["horizontal_max_excursion"] > 1
    assert_original_witness(result, model, caps)
    state[9] = 0.1
    failed = _backup(state, [caps], model, options, table, require_horizontal_reserve=True)
    assert not failed["certified"] and failed["reason"] == "left_every_single_polytope"
    assert failed["horizontal_max_excursion"] == np.linalg.norm(failed["states"][:, :2] - state[:2], axis=1).max()
