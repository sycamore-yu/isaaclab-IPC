"""Native motion-tip API keeps successful warm plans and genuine failures explicit."""

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
)


def frontend():
    pytest.importorskip("isaaclab_ipc._pathcover_native")
    return NativePathCoverFrontend(PathCoverConfig(
        bounds_lower=(-2.0, -2.0, 0.0), bounds_upper=(4.0, 2.0, 3.0),
        resolution=0.1, robot_radius=0.0, max_velocity=3.0, use_cuda=False,
    ))


def test_native_safe_preferred_tip_preserves_plan_and_exposes_selected_seed():
    planner = frontend()
    state = PathCoverState(np.array([-1.0, 0.0, 1.5]), np.zeros(3), np.zeros(3))
    seed = np.array([state.position, [-0.9, 0.0, 1.5], [-0.8, 0.0, 1.5]])
    tips = np.array([[-0.8, 0.0, 1.0], [-0.8, 0.0, 2.0]])
    fixed = planner.plan(None, state, [3.0, 0.0, 1.5], initial_motion_seed=seed)
    chosen = planner.plan(None, state, [3.0, 0.0, 1.5], initial_motion_seed=seed,
                          initial_motion_tip_segment=tips)
    assert chosen.metrics["initial_motion_seed_adjusted"] is False
    np.testing.assert_array_equal(chosen.metrics["selected_initial_motion_seed"], seed)
    np.testing.assert_array_equal(chosen.path, fixed.path)
    np.testing.assert_array_equal(chosen.trajectory.coefficients, fixed.trajectory.coefficients)
    np.testing.assert_array_equal(chosen.trajectory.durations, fixed.trajectory.durations)
    for a, b in zip(chosen.corridors, fixed.corridors, strict=True):
        np.testing.assert_array_equal(a, b)


def test_native_motion_tip_invalid_shape_and_entire_segment_blocked_are_distinct():
    planner = frontend()
    state = PathCoverState(np.array([-1.0, 0.0, 1.5]), np.zeros(3), np.zeros(3))
    seed = np.array([state.position, [-0.9, 0.0, 1.5], [0.05, 0.05, 1.55]])
    tips = np.array([[0.05, 0.05, 1.51], [0.05, 0.05, 1.59]])
    with pytest.raises(ValueError, match="two finite points"):
        planner.plan(None, state, [3.0, 0.0, 1.5], initial_motion_seed=seed,
                     initial_motion_tip_segment=tips[:1])
    with pytest.raises(ValueError, match="three-point seed"):
        planner.plan(None, state, [3.0, 0.0, 1.5], initial_motion_tip_segment=tips)
    planner.update_cloud(seed[2:3], state.position)
    # A few-ULP LP endpoint sliver is still rejected by the original SAT gap;
    # both exact emptiness and numerical non-separation remain typed failures.
    with pytest.raises(PathCoverPlanningError, match="motion tip segment has no .*seed"):
        planner.plan(None, state, [3.0, 0.0, 1.5], initial_motion_seed=seed,
                     initial_motion_tip_segment=tips)
    failed = planner.failed_problem()
    np.testing.assert_array_equal(failed["initial_motion_seed"], seed)
    np.testing.assert_array_equal(failed["initial_motion_tip_segment"], tips)
