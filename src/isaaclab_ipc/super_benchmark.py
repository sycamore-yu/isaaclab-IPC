"""Pinned SUPER scene parameters, adapted to IPC's linear hard-constraint MPC."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

from .mpc import Config


def load_case(name):
    path = Path(__file__).parent / 'config/super_benchmarks.yaml'
    return yaml.safe_load(path.read_text())['cases'][name]


def planner_params(name):
    case = load_case(name)
    source = case['planner']
    bounds, mapping = source['traj_opt']['boundary'], source['rog_map']
    params = Config.load('forest').params
    params['benchmark'] = name
    params['sfc_dis'] = source['super_planner']['robot_r']
    params['fsm']['ref_dis'] = round(bounds['max_vel'] * params['mpc']['step'] / params['fsm']['path_dis'])
    for prefix, limit in [('v', bounds['max_vel']), ('a', bounds['max_acc']), ('u', bounds['max_jerk'])]:
        for axis in 'xyz':
            params['mpc'][prefix + axis + '_min'] = -limit
            params['mpc'][prefix + axis + '_max'] = limit
    for mode in ['expand_fix', 'expand_dyn']:
        params['astar'][mode] = mapping['inflation_resolution'] * mapping['inflation_step']
    size = np.array(mapping['map_size'], dtype=float)
    center = np.array(mapping['fix_map_origin'], dtype=float)
    low, high = center - size / 2, center + size / 2
    low[2], high[2] = mapping['virtual_ground_height'], mapping['virtual_ceil_height']
    params['frontend'] = dict(
        resolution=mapping['resolution'], size=size.tolist(), center=center.tolist(),
        workspace=[low.tolist(), high.tolist()],
        cloud_leaf=min(case['simulation']['downsample_res'], mapping['resolution']),
        planning_horizon=source['super_planner']['planning_horizon'],
        receding_distance=source['super_planner']['receding_dis'],
    )
    return params


def general360_directions(cfg, device, phase=0.0):
    """MARSIM GENERAL_360: 128 rounded image rows with a uniform scan phase."""
    width = math.ceil(360 / cfg.polar_resolution)
    height = math.ceil(cfg.vertical_fov / cfg.polar_resolution)
    step = height / 128
    rows = torch.floor(torch.arange(128, device=device) * step + phase * step + .5).clamp(0, height - 1)
    # MARSIM flips pattern rows when reading the depth image; camera -Z is body +X.
    elevation = torch.deg2rad((height - 1 - rows - height / 2) * cfg.polar_resolution)
    azimuth = torch.deg2rad((torch.arange(width, device=device) - width / 2) * cfg.polar_resolution)
    e, a = torch.meshgrid(elevation, azimuth, indexing='ij')
    return torch.stack((e.cos() * a.cos(), -e.cos() * a.sin(), -e.sin()), -1).reshape(-1, 3)


def general360_pattern(cfg, device):
    directions = general360_directions(cfg, device)
    return torch.zeros_like(directions), directions


@configclass
class General360PatternCfg(PatternBaseCfg):
    func: Callable = general360_pattern
    polar_resolution: float = .3
    vertical_fov: float = 178.


def sensor_config(name):
    source = load_case(name)['simulation']
    return MultiMeshRayCasterCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base', spawn=None,
        pattern_cfg=General360PatternCfg(polar_resolution=source['polar_resolution'], vertical_fov=source['vertical_fov']),
        max_distance=source['sensing_horizon'], update_period=1 / source['sensing_rate'], ray_alignment='base',
        mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr='{ENV_REGEX_NS}/Forest', is_shared=True, track_mesh_transforms=False)],
    )


def reset_scan(env, env_ids):
    if len(env_ids):
        sensor = env.scene['mid360']
        phase = float(torch.rand((), generator=env.mid360_rng))
        sensor.ray_directions.torch[env_ids] = general360_directions(sensor.cfg.pattern_cfg, env.device, phase)


def advance_scan(env, env_ids):
    ticks = torch.round(env.elapsed * 100).long()
    reset_scan(env, (ticks % 10 == 0).nonzero().flatten())
