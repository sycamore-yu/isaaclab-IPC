"""SUPER P0/P1 geometry and control checks, using the recorded High-speed state."""

import numpy as np
import pytest
import torch

from isaaclab_ipc.execution import brake_jerk, safe_climb_jerk
from isaaclab_ipc.frontend import _prepare_context
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.planner import Planner, SafeClimbPending, voxel_centroids
from isaaclab_ipc.safe_recovery import vertical_segment_clear
from isaaclab_ipc.super_benchmark import planner_params


def test_safe_climb_reproduces_and_releases_high_speed_deadlock():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "build/super/high-speed/points.npy"
    if not source.exists():
        pytest.skip("Original SUPER map is not mounted")
    points = np.asarray(np.load(source, mmap_mode="r"))
    workspace = np.array([[-7.5, -55, -0.5], [7.5, 55, 5.0]])
    valid = ((points >= workspace[0]) & (points <= workspace[1])).all(1)
    cloud = voxel_centroids(points[valid], 0.1)

    params = planner_params("high_speed")
    planner = Planner(Config(params), [0, 50, 1.5])
    planner.cloud = cloud
    planner.grid.cloud_fast(cloud, 0.6)
    planner._grid_cloud, planner._grid_radius = cloud, 0.6
    state = np.array([-0.96008, -8.86022, 0.8337, 0, 0, 0, 0, 0, 0])
    assert not planner.grid.endpoint_free(state[:3])
    result = _prepare_context((planner, state, None))
    assert not result.valid
    assert planner.recovery_target is not None
    assert planner.recovery_target[2] >= 1.1
    assert planner.safe_climb_events == 1
    assert planner.frontend_failures == 0
    assert vertical_segment_clear(cloud, state[:3], planner.recovery_target, .25)
    second = _prepare_context((planner, state, None))
    assert not second.valid
    assert planner.safe_climb_events == 1

    recovered = state.copy()
    recovered[:3] = planner.recovery_target
    planner._continue_safe_climb(recovered)
    assert planner.recovery_target is None
    assert planner.replan

    baseline_params = planner_params("high_speed")
    baseline_params["frontend"]["safe_climb"] = False
    baseline = Planner(Config(baseline_params), [0, 50, 1.5])
    baseline.cloud = cloud
    baseline.grid.cloud_fast(cloud, 0.6)
    failed = _prepare_context((baseline, state, None))
    assert not failed.valid
    assert baseline.frontend_failures == 1
    assert baseline.recovery_target is None


def test_safe_climb_rejects_observed_overhead_obstacle():
    planner = Planner(Config(planner_params("high_speed")), [0, 50, 1.5])
    p = np.array([0.0, -50.0, 0.834])
    planner.cloud = np.array([[0.0, -50, 0.0], [0.02, -50, 1.05]])
    planner.grid.cloud_fast(planner.cloud, .6)
    assert planner._safe_vertical_target(p) is None


def test_observed_climb_path_rechecks_lateral_drift_and_map_changes():
    cloud = np.array([[0, -50, 0], [0.7, -50, 1.3]])
    start = np.array([0, -50, 0.834])
    target = np.array([0, -50, 1.284])
    assert vertical_segment_clear(cloud, start, target, 0.25)
    assert vertical_segment_clear(cloud, start + [0.03, 0, 0], target, 0.25)
    assert not vertical_segment_clear(np.vstack((cloud, [[0.1, -50, 1.05]])), start, target, 0.25)

    planner = Planner(Config(planner_params("high_speed")), [0, 50, 1.5])
    planner.cloud = cloud.copy()
    planner.grid.cloud_fast(planner.cloud, .6)
    planner.recovery_target = target.copy()
    planner._climb_cloud = object()
    state = np.array([0.03, -50, .95, 0, 0, 0, 0, 0, 0])
    with pytest.raises(SafeClimbPending):
        planner._continue_safe_climb(state)
    assert planner.recovery_target is not None
    planner.cloud = np.vstack((planner.cloud, [[0.05, -50, 1.1]]))
    planner._climb_cloud = object()
    with pytest.raises(ValueError, match="blocks the vertical recovery"):
        planner._continue_safe_climb(state)
    assert planner.recovery_target is None


def test_safe_climb_control_is_bounded_and_agrees_between_numpy_and_torch():
    mpc = MPC(Config(planner_params("high_speed")))
    state = np.array([-0.96, -8.86, .834, .2, .1, -.15, 0, 0, 0])
    jerk = safe_climb_jerk(state, 1.284, mpc.lower, mpc.upper, mpc.jlow, mpc.jhigh, .1)
    ordinary = brake_jerk(state, mpc.lower, mpc.upper, mpc.jlow, mpc.jhigh, .1)
    np.testing.assert_allclose(jerk[:2], ordinary[:2])
    assert jerk[2] > ordinary[2]
    assert np.all(jerk >= mpc.jlow[:3]) and np.all(jerk <= mpc.jhigh[:3])
    state_t = torch.tensor(state[None], dtype=torch.float64)
    result = safe_climb_jerk(
        state_t, torch.tensor([1.284]), torch.tensor(mpc.lower), torch.tensor(mpc.upper),
        torch.tensor(mpc.jlow), torch.tensor(mpc.jhigh), .1,
    )
    np.testing.assert_allclose(result.squeeze(0).numpy(), jerk, atol=1e-8)


def test_loong_prediction_reuse_is_guarded_by_map_and_goal():
    params = planner_params("high_speed")
    planner = Planner(Config(params), [0, 50, 1.5])
    state = np.array([0, -50, 1.5, 0, 3, 0, 0, 0, 0.0])
    planner.context(state)
    planner.consume_state(state, np.zeros(45), "solved", 0.0)
    assert planner._safe_prediction is not None
    future = planner._safe_prediction.copy()
    planner.replan = True
    next_state = state.copy()
    next_state[1] += 0.03
    planner.context(next_state)
    assert planner.reused_prediction_count == 1
    planner.goal[:] = [0, 50, 1.5]
    planner._safe_prediction_goal = planner.goal.copy()
    planner._safe_prediction = future.copy()
    planner.grid.cloud_fast(np.array([[0, -49.6, 1.5]]), .6)
    assert planner._reuse_safe_prefix(next_state[:3]) is None
    assert any(np.allclose(point, future[-1], atol=1e-4) for point in planner.path)

    planner.goal[:] = [0, -50, 1.5]
    planner.replan = True
    planner.context(next_state)
    assert planner.reused_prediction_count == 1
