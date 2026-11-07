"""Actual extension contracts for explicit integral/continuous position objectives."""

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import PathCoverConfig, PathCoverPlanningError


def configuration(mode):
    return PathCoverConfig(
        bounds_lower=(-3, -3, 0), bounds_upper=(3, 3, 4),
        resolution=.25, robot_radius=.1, use_cuda=False,
        max_velocity=10, nominal_velocity=3, thrust_min=0,
        nominal_thrust_min=2.094972067039106, thrust_max=22.0725,
        tilt_max=np.pi / 2, safety_margin=0, position_penalty_aggregation=mode,
    )


def options(mode):
    return configuration(mode).native_options()


@pytest.mark.parametrize("mode", ["integral", "continuous_max"])
def test_native_position_mode_is_reported_for_hold_and_early_failure(mode):
    from isaaclab_ipc import _pathcover_native

    frontend = _pathcover_native.NativeFrontend(options(mode))
    position = np.array([0., 0., 1.])
    hold = frontend.plan(position, np.zeros(3), np.zeros(3), position)
    assert hold.position_penalty_aggregation == mode
    assert np.array_equal(np.asarray(hold.coefficients)[0, :, -1], position)
    with pytest.raises(PathCoverPlanningError, match="workspace"):
        frontend.plan(position, np.zeros(3), np.zeros(3), np.array([30., 0., 1.]))
    assert frontend.failed_problem()["position_penalty_aggregation"] == mode


def test_native_missing_mode_is_integral_and_unknown_mode_is_rejected():
    from isaaclab_ipc import _pathcover_native

    legacy = options("integral")
    legacy.pop("position_penalty_aggregation")
    frontend = _pathcover_native.NativeFrontend(legacy)
    position = np.array([0., 0., 1.])
    hold = frontend.plan(position, np.zeros(3), np.zeros(3), position)
    assert hold.position_penalty_aggregation == "integral"
    with pytest.raises(ValueError, match="position_penalty_aggregation"):
        _pathcover_native.NativeFrontend(dict(legacy, position_penalty_aggregation="piece_max"))


@pytest.mark.parametrize("axis", [None, np.array([0., 0., 1.])])
def test_native_continuous_mode_is_independent_of_initial_force(axis):
    from isaaclab_ipc.pathcover_native import NativePathCoverFrontend, PathCoverState

    frontend = NativePathCoverFrontend(configuration("continuous_max"))
    state = PathCoverState(np.array([0., 0., 1.]), np.zeros(3), np.zeros(3))
    plan = frontend.plan(np.empty((0, 3)), state, np.array([1., 0., 1.]),
                         initial_thrust_axis=axis)
    assert plan.metrics["position_penalty_aggregation"] == "continuous_max"
    assert plan.metrics["initial_boundary"]["mode"] == ("fixed_pva" if axis is None else "ctbr_collective")
    assert np.all(np.isfinite(plan.trajectory.coefficients))
