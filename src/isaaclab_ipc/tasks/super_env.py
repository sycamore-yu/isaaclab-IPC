"""SUPER benchmark scenes with IPC/Newton control and the source waypoint mission."""

from pathlib import Path

import torch
from isaaclab.managers import EventTermCfg

from isaaclab_ipc.rl.config import MPCTrainCfg
from isaaclab_ipc.super_benchmark import advance_scan, load_case, planner_params, reset_scan, sensor_config

from .env import make_cfg
from .mpc_env import FEATURES, MAX_PLANES, IPCMPCEnv
from .rewards import NavigationRewardCfg


def make_super_cfg(name):
    folder = Path(__file__).resolve().parents[3] / 'build/super' / name.replace('_', '-')
    if not (folder / 'forest.usdc').is_file():
        raise FileNotFoundError(f'Build SUPER map first: {folder}; see docs/super-benchmarks.md')
    source = load_case(name)
    bounds = source['planner']['traj_opt']['boundary']
    cfg = make_cfg(scenario='forest', forest_usd=folder / 'forest.usdc')
    cfg.super_case = name
    cfg.planner_params = planner_params(name)
    cfg.reference_strides = (cfg.planner_params['fsm']['ref_dis'],)
    cfg.scene.mid360 = sensor_config(name)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 150.
    cfg.initial_position = tuple(source['start'])
    cfg.initial_yaw = source['simulation']['init_yaw']
    cfg.mission_start_delay = source['mission']['start_program_delay']
    cfg.goal_x = source['waypoints'][0][0]
    cfg.goal_tolerance = source['waypoint_switch_distances'][-1]
    cfg.collision_radius = source['planner']['super_planner']['robot_r']
    cfg.sensor_blind = source['simulation']['sensing_blind']
    cfg.sensor_return_max = source['simulation']['sensing_horizon'] - .1
    cfg.thrust_limits = (bounds['min_acc_thr'], bounds['max_acc_thr'])
    cfg.episode_length_s = 180.  # Evaluation budget; upstream waypoint mission has no timeout.
    cfg.reward = NavigationRewardCfg()
    cfg.observation_space = FEATURES
    cfg.max_planes = MAX_PLANES
    cfg.events.reset_scan = EventTermCfg(func=reset_scan, mode='reset')
    cfg.events.advance_scan = EventTermCfg(func=advance_scan, mode='ipc_control')
    return cfg


def make_high_speed_cfg():
    return make_super_cfg('high_speed')


def make_dense_cfg():
    return make_super_cfg('dense')


def make_super_agent_cfg(name):
    bounds = load_case(name)['planner']['traj_opt']['boundary']
    cfg = MPCTrainCfg()
    cfg.actor.planner_params = planner_params(name)
    cfg.actor.thrust_limits = (bounds['min_acc_thr'], bounds['max_acc_thr'])
    cfg.actor.rate_limit = bounds['max_omg']
    return cfg


def make_high_speed_agent_cfg():
    return make_super_agent_cfg('high_speed')


def make_dense_agent_cfg():
    return make_super_agent_cfg('dense')


class SuperMPCEnv(IPCMPCEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.super_source = load_case(cfg.super_case)
        self.waypoints = torch.tensor(self.super_source['waypoints'], device=self.device)
        self.waypoint_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal = self.waypoints[0].repeat(self.num_envs, 1)
        self.waypoint_times = [[] for _ in range(self.num_envs)]
        self.peak_speed = torch.zeros(self.num_envs, device=self.device)
        self.norm_violations = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)

    def _reset_idx(self, env_ids):
        ids = self.ids if env_ids is None else env_ids
        self.waypoint_ids[ids] = 0
        self.goal[ids] = self.waypoints[0]
        self.peak_speed[ids] = 0
        self.norm_violations[ids] = 0
        for i in ids.cpu().tolist():
            self.waypoint_times[i] = []
        super()._reset_idx(ids)
        start = torch.tensor(self.cfg.initial_position, device=self.device)
        self.previous_distance[ids] = (self.goal[ids] - start).norm(dim=-1)

    def _get_dones(self):
        pose, twist = self.body_state()
        position = pose[:, :3] - self.scene.env_origins
        ready = self.elapsed >= self.super_source['mission']['start_program_delay']
        thresholds = torch.tensor(self.super_source['waypoint_switch_distances'], device=self.device)[self.waypoint_ids]
        reached = (position - self.goal).norm(dim=-1) < thresholds
        for i in (reached & ready & ~self.hit & ~torch.as_tensor(self.reported, device=self.device)).nonzero().flatten().cpu().tolist():
            if self.waypoint_ids[i] < len(self.waypoints) - 1:
                self.waypoint_times[i].append(float(self.elapsed[i]))
                self.waypoint_ids[i] += 1
                self.goal[i] = self.waypoints[self.waypoint_ids[i]]
                self.planners[i].goal = self.goal[i].cpu().numpy().copy()
                self.planners[i].replan = True
                self.previous_distance[i] = (position[i] - self.goal[i]).norm()
        speed = twist[:, :3].norm(dim=-1)
        self.peak_speed = torch.maximum(self.peak_speed, speed)
        norms = torch.stack((speed, self.acceleration().norm(dim=-1),
                             self.last_mpc_result['jerk'].norm(dim=-1), twist[:, 3:].norm(dim=-1)), -1)
        bounds = self.super_source['planner']['traj_opt']['boundary']
        limits = norms.new_tensor([bounds[k] for k in ['max_vel', 'max_acc', 'max_jerk', 'max_omg']])
        self.norm_violations += (norms > limits + .02) & ~torch.as_tensor(self.reported, device=self.device)[:, None]
        first = len(self.records)
        done = super()._get_dones()
        for row in self.records[first:]:
            i = row['env_id']
            row.update(super_case=self.cfg.super_case, peak_actual_speed_mps=float(self.peak_speed[i]),
                       waypoint_times_s=self.waypoint_times[i].copy(),
                       waypoints_reached=len(self.waypoint_times[i]) + int(row['success']),
                       safe_climb_events=self.planners[i].safe_climb_events,
                       safe_climb_steps=self.planners[i].safe_climb_steps,
                       reused_prediction_count=self.planners[i].reused_prediction_count,
                       norm_violation_steps=dict(zip(['velocity', 'acceleration', 'jerk', 'angular_velocity'],
                                                     self.norm_violations[i].cpu().tolist(), strict=True)))
        return done
