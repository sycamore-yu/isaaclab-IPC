import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isaaclab_ipc import _geometry
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.super_benchmark import General360PatternCfg, general360_directions, load_case, planner_params


@pytest.mark.parametrize('name,speed,resolution,distance,rays', [
    ('high_speed', 25, .3, 70, 153600), ('dense', 8, .05, 15, 115200),
])
def test_pinned_super_parameters_and_real_scan_law(name, speed, resolution, distance, rays):
    case = load_case(name)
    assert case['start'] == [0, -50, 1.5]
    assert case['waypoints'] == [[0, 50, 1.5], [0, -50, 1.5]]
    assert case['waypoint_switch_distances'] == [1, 1]
    assert case['simulation']['sensing_horizon'] == distance
    params = planner_params(name)
    assert params['mpc']['vy_max'] == speed
    assert params['frontend']['safe_climb']
    assert params['frontend']['reuse_prediction'] == (name == 'high_speed')
    assert params['frontend']['resolution'] == resolution
    assert params['frontend']['size'] == [15, 110, 6]
    assert params['sfc_dis'] == .2
    cfg = General360PatternCfg(polar_resolution=case['simulation']['polar_resolution'])
    directions = general360_directions(cfg, 'cpu', phase=.5)
    assert directions.shape == (rays, 3)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(rays))
    height = math.ceil(178 / cfg.polar_resolution)
    rows = np.minimum(np.floor(np.arange(128) * height / 128 + .5 * height / 128 + .5), height - 1)
    expected_z = -np.sin(np.deg2rad((height - 1 - rows - height / 2) * cfg.polar_resolution))
    np.testing.assert_allclose(directions.reshape(128, -1, 3)[:, 0, 2], expected_z, atol=2e-7)
    assert not torch.equal(directions, general360_directions(cfg, 'cpu', phase=0))


def test_custom_grid_and_sfc_workspace_keep_original_defaults():
    grid = _geometry.Grid(.05, [-1, -2, -.5], [1, 2, 2.5])
    grid.center([0, -50, 1.5])
    grid.cloud([], .3)
    path = np.asarray(grid.search([0, -51, 2], [0, -49, 2]))
    np.testing.assert_allclose(path[[0, -1]], [[0, -51, 2], [0, -49, 2]])
    grid.height_limits(1, 3)
    assert not grid.free([0, -50, 3.5])
    assert grid.free([0, -50, 2])
    workspace = [[-7.5, -55, -.1], [7.5, 55, 3.5]]
    faces = _geometry.corridor([0, -50, 1.5], [0, -49, 1.5], [], .2, workspace)
    assert (faces[:, :3] @ np.array([0, -49, 1.5]) + faces[:, 3] <= 0).all()
    assert not (faces[:, :3] @ np.array([0, 56, 1.5]) + faces[:, 3] <= 0).all()
    # Old calls retain the original moving +/-10m workspace and z ceiling3.
    original = _geometry.corridor([20, 0, 1], [21, 0, 1], [], .3)
    assert (original[:, :3] @ [21, 0, 1] + original[:, 3] <= 0).all()
    with pytest.raises(ValueError):
        _geometry.Grid(0, [-1, -1, -1], [1, 1, 1])


def test_super_factories_propagate_sensor_and_actor_limits():
    from isaaclab_ipc.tasks.super_env import make_super_agent_cfg, make_super_cfg
    for name in ['high_speed', 'dense']:
        if not (Path(__file__).resolve().parents[1] / 'build/super' / name.replace('_', '-') / 'forest.usdc').exists():
            pytest.skip('SUPER map assets not built')
        case = load_case(name)
        cfg = make_super_cfg(name)
        agent = make_super_agent_cfg(name)
        assert cfg.collision_radius == .2
        assert cfg.scene.mid360.max_distance == case['simulation']['sensing_horizon']
        assert cfg.planner_params == agent.actor.planner_params
        assert cfg.thrust_limits == agent.actor.thrust_limits
        assert agent.actor.rate_limit == case['planner']['traj_opt']['boundary']['max_omg']
        assert cfg.initial_position == (0, -50, 1.5)
    # No benchmark parameters leak into existing IPC callers.
    assert 'frontend' not in Config.load('forest').params


def test_source_waypoints_switch_without_resetting_physics(monkeypatch):
    from types import SimpleNamespace

    from isaaclab_ipc.tasks.mpc_env import IPCMPCEnv
    from isaaclab_ipc.tasks.super_env import SuperMPCEnv

    env = object.__new__(SuperMPCEnv)
    env.cfg = SimpleNamespace(super_case='high_speed', sim=SimpleNamespace(device='cpu'))
    env.sim = SimpleNamespace(device='cpu')
    env.super_source = load_case('high_speed')
    env.waypoints = torch.tensor([[0., 50, 1.5], [0, -50, 1.5]])
    env.goal = env.waypoints[:1].clone()
    env.waypoint_ids = torch.zeros(1, dtype=torch.long)
    env.waypoint_times = [[]]
    env.previous_distance = torch.zeros(1)
    env.elapsed = torch.tensor([8.])
    env.hit = torch.zeros(1, dtype=torch.bool)
    env.reported = np.zeros(1, bool)
    env.scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
    env.planners = [SimpleNamespace(
        goal=np.array([0, 50, 1.5]), replan=False, safe_climb_events=0,
        safe_climb_steps=0, reused_prediction_count=0,
    )]
    env.peak_speed = torch.zeros(1)
    env.norm_violations = torch.zeros(1, 4, dtype=torch.long)
    env.last_mpc_result = {'jerk': torch.zeros(1, 3)}
    env.acceleration = lambda: torch.zeros(1, 3)
    env.records = []
    position = torch.tensor([[0., 49.5, 1.5, 0, 0, 0, 1]])
    env.body_state = lambda: (position, torch.zeros(1, 6))

    def parent_dones(self):
        success = bool((position[:, :3] - self.goal).norm() < 1)
        if success:
            self.records.append(dict(env_id=0, success=True))
        return torch.tensor([success]), torch.tensor([False])

    monkeypatch.setattr(IPCMPCEnv, '_get_dones', parent_dones)
    done, _ = env._get_dones()
    assert not done.any() and env.waypoint_ids.item() == 1
    assert env.planners[0].replan and env.waypoint_times == [[8.]]
    torch.testing.assert_close(env.goal, torch.tensor([[0., -50, 1.5]]))
    assert env.previous_distance.item() == 99.5
    position[0, 1] = -49.5
    env.elapsed[0] = 16
    done, _ = env._get_dones()
    assert done.all() and env.records[-1]['waypoints_reached'] == 2


def test_receding_local_goal_advances_toward_global_waypoint():
    from isaaclab_ipc.planner import Planner
    p = Planner(Config(planner_params('high_speed')), [0, 50, 1.5])
    state = np.array([0, -50., 1.5, 0, 0, 0, 0, 0, 0])
    p.context(state)
    first_end = p.path[-1].copy()
    assert first_end[1] == pytest.approx(-20)
    state[1] = -25
    p.context(state)
    assert p.path[-1, 1] > first_end[1] + 10
    np.testing.assert_array_equal(p.goal, [0, 50, 1.5])


@pytest.mark.parametrize('resolution,radius', [(.05, .3), (.05, .31), (.1, .4), (.3, .6), (.1, 0)])
def test_fast_inflation_preserves_every_occupied_cell(resolution, radius):
    rng = np.random.default_rng(42)
    points = rng.uniform([-1.1, -52.1, 0.9], [1.1, -47.9, 3.1], (400, 3))
    points = np.vstack((points, [[0, -50, 1.5], [1, -50, 2], [-1 + 1e-14, -50, 2]]))
    low, high = np.array([-1., -2, -.5]), np.array([1., 2, 2.5])
    old, fast = _geometry.Grid(resolution, low, high), _geometry.Grid(resolution, low, high)
    old.center([0, -50, 1.5])
    fast.center([0, -50, 1.5])
    old.cloud(points, radius)
    fast.cloud_fast(points, radius)
    assert old.occupancy_matches(fast)


def test_numpy_corridor_preserves_original_point_sequence_and_planes():
    points = np.random.default_rng(42).uniform([1, -2, .2], [3, 2, 3], (500, 3))
    workspace = [[-7.5, -55, -.1], [7.5, 55, 3.5]]
    expected = _geometry.corridor([0, 0, 1.5], [0, .1, 1.5], points, .2, workspace)
    actual = _geometry.corridor_numpy(np.array([0., 0, 1.5]), np.array([0., .1, 1.5]), points, .2, workspace)
    np.testing.assert_array_equal(actual, expected)
