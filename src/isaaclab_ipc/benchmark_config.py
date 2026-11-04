"""Shared pure task configuration for Newton training and ROS evaluation."""

from pathlib import Path

import numpy as np
import yaml

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
        resolution=mapping['resolution'], size=size.tolist(), center=center.tolist(), gpu_incremental=True,
        safe_climb=True, reuse_prediction=(name == 'high_speed'),
        workspace=[low.tolist(), high.tolist()],
        cloud_leaf=min(case['simulation']['downsample_res'], mapping['resolution']),
        planning_horizon=source['super_planner']['planning_horizon'],
        receding_distance=source['super_planner']['receding_dis'],
    )
    return params
