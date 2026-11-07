"""Actual native CTBR boundary metadata, hold semantics and failure classes."""

import json

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
)


def configuration(**overrides):
    values = dict(
        bounds_lower=(-2.0, -2.0, 0.0), bounds_upper=(4.0, 2.0, 3.0),
        resolution=0.2, robot_radius=0.0, max_velocity=3.0,
        thrust_min=0.0, nominal_thrust_min=2.0, thrust_max=22.0725,
        safety_margin=0.0, tilt_max=np.pi / 2, use_cuda=False,
    )
    values.update(overrides)
    return PathCoverConfig(**values)


def native(config=None):
    module = pytest.importorskip("isaaclab_ipc._pathcover_native")
    assert hasattr(module.PlanResult, "initial_boundary"), "load the unpublished V27 candidate first"
    return module.NativeFrontend((config or configuration()).native_options())


def test_native_free_force_preserves_pv_axis_and_json_boundary():
    planner = native()
    position, velocity = np.array([-1.0, 0.0, 1.5]), np.array([.4, 0.0, .1])
    acceleration = np.array([.3, .2, -1.0])
    axis = np.array([.1, .1, 1.0])
    axis /= np.linalg.norm(axis)
    result = planner.plan(position, velocity, acceleration, [2.0, 0.0, 1.5], initial_thrust_axis=axis)
    boundary = result.initial_boundary
    assert boundary["mode"] == "ctbr_collective"
    assert result.optimizer_status >= 0
    coefficients = np.stack(result.coefficients)
    np.testing.assert_allclose(coefficients[0, :, 5], position, rtol=0, atol=1e-12)
    np.testing.assert_allclose(coefficients[0, :, 4], velocity, rtol=0, atol=1e-12)
    reference_a = 2 * coefficients[0, :, 3]
    np.testing.assert_allclose(boundary["reference_acceleration"], reference_a, rtol=0, atol=0)
    np.testing.assert_array_equal(boundary["measured_acceleration"], acceleration)
    np.testing.assert_array_equal(boundary["thrust_axis"], axis)
    assert np.linalg.norm(reference_a - acceleration) > 1e-4
    force = .75 * np.linalg.norm(reference_a + [0, 0, 9.81])
    assert 0 <= boundary["force_n"] <= 22.0725
    assert boundary["force_n"] == pytest.approx(force, abs=1e-12)
    np.testing.assert_allclose(reference_a + [0, 0, 9.81], axis * (force / .75), rtol=0, atol=1e-12)
    assert isinstance(json.loads(json.dumps(boundary))["reference_acceleration"], list)
    assert planner.failed_problem() is None

    fixed = planner.plan(position, velocity, acceleration, [2.0, 0.0, 1.5])
    assert fixed.initial_boundary["mode"] == "fixed_pva"
    assert fixed.initial_boundary["thrust_axis"] is None
    np.testing.assert_allclose(2 * np.stack(fixed.coefficients)[0, :, 3], acceleration, rtol=0, atol=1e-12)


def test_wrapper_hold_ignores_measured_acceleration_only_in_ctbr_mode():
    native()  # Assert the new binary, rather than silently testing an old one.
    cfg = configuration()
    planner = NativePathCoverFrontend(cfg)
    state = PathCoverState(np.array([-1.0, 0.0, 1.5]), np.zeros(3), np.array([.2, -.3, -1.0]))
    plan = planner.plan(None, state, state.position, initial_thrust_axis=[0, 0, 1])
    boundary = plan.metrics["initial_boundary"]
    assert boundary["mode"] == "ctbr_collective"
    np.testing.assert_array_equal(boundary["reference_acceleration"], np.zeros(3))
    np.testing.assert_array_equal(boundary["measured_acceleration"], state.acceleration)
    assert boundary["force_n"] == cfg.mass * cfg.gravity
    assert plan.metrics["reference_thrust_initial_n"] == boundary["force_n"]
    assert plan.metrics["thrust_initial_n"] != boundary["force_n"]
    json.dumps(plan.metrics)
    assert planner.failed_problem() is None


def test_native_zero_length_and_heading_domain_failures_retain_input_only():
    planner = native()
    position = np.array([-1.0, 0.0, 1.5])
    acceleration = np.array([.2, 0.0, -1.0])
    tilted = np.array([.1, 0, 1.0])
    tilted /= np.linalg.norm(tilted)
    with pytest.raises(PathCoverPlanningError, match="cannot hold"):
        planner.plan(position, np.zeros(3), acceleration, position, initial_thrust_axis=tilted)
    boundary = planner.failed_problem()["initial_boundary"]
    assert boundary["mode"] == "ctbr_collective"
    assert boundary["reference_acceleration"] is None and boundary["force_n"] is None
    np.testing.assert_array_equal(boundary["thrust_axis"], tilted)
    json.dumps(boundary)

    with pytest.raises(PathCoverPlanningError, match="upright heading chart"):
        planner.plan(position, np.zeros(3), acceleration, [2.0, 0, 1.5], initial_thrust_axis=[0, 0, -1])
    assert planner.failed_problem()["initial_boundary"]["force_n"] is None
    with pytest.raises(ValueError, match="finite unit vector"):
        planner.plan(position, np.zeros(3), acceleration, [2.0, 0, 1.5], initial_thrust_axis=[0, 0, 2])
    assert planner.failed_problem() is None

    unavailable_hover = native(configuration(thrust_max=5.0))
    with pytest.raises(PathCoverPlanningError, match="cannot hold"):
        unavailable_hover.plan(position, np.zeros(3), np.zeros(3), position, initial_thrust_axis=[0, 0, 1])


def test_native_optimizer_failure_keeps_raw_pva_and_unrealized_force_contract():
    planner = native(configuration(max_iterations=1))
    position = np.array([-1.0, 0.0, 1.5])
    velocity, acceleration = np.array([.4, .1, .1]), np.array([.3, .2, -1.0])
    with pytest.raises(PathCoverPlanningError, match="optimization failed"):
        planner.plan(position, velocity, acceleration, [2.0, 0, 1.5], initial_thrust_axis=[0, 0, 1])
    problem = planner.failed_problem()
    assert problem["stage"] == "optimize" and problem["optimizer_status"] < 0
    np.testing.assert_array_equal(problem["initial_pva"], np.column_stack((position, velocity, acceleration)))
    boundary = problem["initial_boundary"]
    assert boundary["mode"] == "ctbr_collective" and boundary["thrust_axis"] == (0, 0, 1)
    assert boundary["reference_acceleration"] is None and boundary["force_n"] is None
    assert boundary["force_lower_n"] == 0.0 and boundary["force_upper_n"] == 22.0725
    planner.reset()
    assert planner.failed_problem() is None
