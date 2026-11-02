import numpy as np
import pytest

from isaaclab_ipc import _geometry
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner, contains, voxel_centroids
from isaaclab_ipc.scenes import AvoidScene


def test_empty_corridor_and_voxel_centroids():
    planes = _geometry.corridor([0, 0, 1], [0, 0, 1], [], 0.3)
    assert contains(planes, [0, 0, 1])
    assert not contains(np.empty((0, 4)), [0, 0, 1])
    assert not contains(planes, [0, 0, 3.01])
    cloud = voxel_centroids([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [0.3, 0.3, 0.3]])
    np.testing.assert_allclose(cloud, [[0.05] * 3, [0.3] * 3], atol=1e-7)


def test_original_cubic_inflation_and_previous_solution_fallback():
    cfg = Config.load()
    cfg.params["recovery"] = "previous_sequence"
    planner = Planner(cfg, [8, 0, 1])
    planner.grid.cloud([[0, 0, 1]], 0.4)
    assert not planner.grid.free([0.3, 0.3, 1.3])
    assert planner.grid.free([1, 1, 1])
    state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0.0])
    problem = planner.mpc.problem(state, np.tile(state, (15, 1)), [[]] * 15)
    with pytest.raises(RuntimeError, match="Initial"):
        planner.consume(problem, None, "primal infeasible", 0.0)
    controls = np.arange(45).reshape(15, 3) * 0.01
    planner.consume(problem, controls, "solved", 0.0)
    x, j, ok = planner.consume(problem, None, "primal infeasible", 0.25)
    assert not ok
    np.testing.assert_array_equal(j, controls[2])
    np.testing.assert_allclose(x, planner.mpc.a @ state + planner.mpc.b @ controls[2])
    _, j, _ = planner.consume(problem, None, "primal infeasible", 10.0)
    np.testing.assert_array_equal(j, controls[-1])


def test_obstacle_first_event_then_periodic_retarget():
    scene = AvoidScene(1.0)
    assert len(scene.advance(0.0, [0, 0, 1], 0.0)) == 1
    scene.advance(0.05, [6, 0, 1], 3.0)
    assert scene.first and not scene.active and scene.y == 0.5
    scene.advance(0.101, [6, 0.1, 1], 3.0)
    assert scene.active
    assert scene.y == pytest.approx(0.1)
    assert scene.trigger_speed == 3.0


def test_real_brakes_and_replans_without_a_previous_solution():
    planner = Planner(Config.load(), [10, 0, 1])
    state = np.array([0, 0, 1, 3.0, 0, 0, 0, 0, 0])
    p = planner.mpc.problem(state, np.tile(state, (15, 1)), [[]] * 15)
    x, j, ok = planner.consume(p, None, "primal infeasible", 0.0)
    assert not ok and planner.braking and planner.replan
    assert planner.recovery_events == 1 and planner.brake_steps == 1 and planner.failures == 1
    assert j[0] < 0 and x[3] < state[3]
    planner.prepare(x)
    assert len(planner.path) > 1 and not planner.replan
    assert np.all(j >= planner.mpc.jlow[:3]) and np.all(j <= planner.mpc.jhigh[:3])
    planner.consume(p, np.zeros(45), "solved", 0.01)
    assert planner.braking
    stopped = state.copy()
    stopped[3:] = 0
    p = planner.mpc.problem(stopped, np.tile(stopped, (15, 1)), [[]] * 15)
    _, _, ok = planner.consume(p, np.zeros(45), "solved", 0.02)
    assert ok and not planner.braking and planner.replan


def test_execution_profiles():
    assert Config.load().params["recovery"] == "brake_replan"
    ideal = Config.load("avoid", profile="ideal")
    assert ideal.params["perfect_simu"] and ideal.params["recovery"] == "previous_sequence"
    with pytest.raises(ValueError):
        Config.load(profile="bad")
