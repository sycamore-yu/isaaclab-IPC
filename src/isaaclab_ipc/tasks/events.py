"""Official EventManager terms retaining IPC's reset and obstacle timing."""

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass


def advance_scene(env, env_ids):
    if env.cfg.scenario == "avoid":
        pose, _ = env.body_state()
        p = pose[:, :3] - env.scene.env_origins
        near = torch.abs(p[:, 0] - 6.9) <= env.cfg.obstacle_distance
        end = torch.round(env.elapsed * 1000).long()
        for offset in range(-9, 1):
            tick = end + offset
            valid = tick >= 0
            periodic = valid & (tick % 101 == 0)
            activate = periodic & near & ~env.obstacle_active
            env.target_y.copy_(torch.where(activate, p[:, 1], env.target_y))
            env.obstacle_y.copy_(torch.where(activate, p[:, 1] + 1.0, env.obstacle_y))
            env.obstacle_active |= activate
            env.obstacle_y.copy_(
                torch.where(
                    periodic & env.obstacle_active,
                    torch.maximum(env.obstacle_y - 1.0, env.target_y),
                    env.obstacle_y,
                )
            )
            first = valid & near & ~env.triggered
            env.obstacle_y.copy_(torch.where(first, torch.maximum(env.obstacle_y - 1.0, env.target_y), env.obstacle_y))
            env.triggered |= first
        obs = env.scene["obstacle"]
        obstacle_pose = obs.data.default_root_pose.torch.clone()
        obstacle_pose[:, :3] += env.scene.env_origins
        obstacle_pose[:, 1] = env.obstacle_y + env.scene.env_origins[:, 1]
        obs.write_root_pose_to_sim_index(root_pose=obstacle_pose, env_ids=env.ids)
    env.elapsed += env.step_dt


def reset_flight(env, env_ids):
    ids = env_ids.to(torch.int32)
    pose = env.robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([0, 0, 1.0], device=env.device)
    pose[:, 3:] = torch.tensor([0, 0, 0, 1.0], device=env.device)
    env.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids), 6), device=env.device), env_ids=ids)
    env.robot.reset_motor_thrust(torch.full((len(ids), 4), 0.75 * 9.81 / 4, device=env.device), ids)
    env.controller.command[ids] = torch.tensor([9.81, 0, 0, 0], device=env.device)
    env.elapsed[ids] = 0
    env.triggered[ids] = False
    env.obstacle_active[ids] = False
    env.obstacle_y[ids] = 1.5
    env.target_y[ids] = 0
    env.hit[ids] = False
    env.min_clearance[ids] = float("inf")
    env.previous_distance[ids] = env.cfg.goal_x
    if env.cfg.scenario == "avoid":
        obs = env.scene["obstacle"]
        obstacle_pose = obs.data.default_root_pose.torch[ids].clone()
        obstacle_pose[:, :3] += env.scene.env_origins[ids]
        obs.write_root_pose_to_sim_index(root_pose=obstacle_pose, env_ids=ids)


def reset_scan(env, env_ids):
    from isaaclab_ipc.mid360 import update_mid360_directions

    update_mid360_directions(env, env_ids)


def advance_scan(env, env_ids):
    from isaaclab_ipc.mid360 import update_mid360_directions

    ticks = torch.round(env.elapsed * 100).long()
    ids = (ticks % 10 == 0).nonzero().flatten()
    update_mid360_directions(env, ids)


@configclass
class IPCEventsCfg:
    reset_flight = EventTermCfg(func=reset_flight, mode="reset")
    reset_scan = EventTermCfg(func=reset_scan, mode="reset")
    advance_scene = EventTermCfg(func=advance_scene, mode="ipc_control")
    advance_scan = EventTermCfg(func=advance_scan, mode="ipc_control")
