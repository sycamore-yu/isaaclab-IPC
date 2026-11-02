"""Exercise native physical hover, body-rate response, and partial reset."""

import json

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry

from isaaclab_ipc.tasks.env import make_cfg

registered = load_cfg_from_registry("Isaac-IPC-Avoid-Direct", "env_cfg_entry_point")
assert "wall" in registered.scene.__dict__
cfg = make_cfg(num_envs=4, scenario="empty")
cfg.automatic_reset = False
with gym.make("Isaac-IPC-Avoid-Direct", cfg=cfg) as wrapped:
    env = wrapped.unwrapped
    obs, _ = env.reset(seed=0)
    action = torch.tensor([[9.81, 0, 0, 0]], device=env.device).expand(4, -1).clone()
    for _ in range(30):
        obs, rew, term, trunc, extra = env.step(action)
    hover = obs["policy"][:, :3].clone()
    assert torch.max(torch.abs(hover - torch.tensor([0, 0, 1.0], device=env.device))) < 0.03, hover
    action[0, 1] = 0.4
    for _ in range(10):
        obs, *_ = env.step(action)
    _, twist = env.body_state()
    rate = twist[0, 3].item()
    assert rate > 0.15, twist
    untouched = env.body_state()[0][1:].clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    torch.testing.assert_close(env.body_state()[0][1:], untouched)
    torch.testing.assert_close(env.acceleration()[0], torch.zeros(3, device=env.device))
    cfg.automatic_reset = True
    env.episode_length_buf[0] = env.max_episode_length - 1
    action[:, 1:] = 0
    obs, _, terminated, truncated, _ = env.step(action)
    assert truncated[0] and not truncated[1:].any()
    torch.testing.assert_close(obs["policy"][0, :3], torch.tensor([0, 0, 1.0], device=env.device))
    pose = env.body_state()[0][0:1].clone()
    pose[0, :3] = env.goal + env.scene.env_origins[0]
    env.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env.ids[:1])
    obs, _, terminated, truncated, extra = env.step(action)
    assert terminated[0] and extra["success"][0] and not extra["collision"][0]
    assert (extra["terminal_position"][0] - env.goal).norm() < 0.05
    torch.testing.assert_close(obs["policy"][0, :3], torch.tensor([0, 0, 1.0], device=env.device))
    print(
        json.dumps(
            {
                "hover": hover.cpu().tolist(),
                "roll_rate": rate,
                "partial_reset": True,
                "timeout_auto_reset": True,
                "goal_auto_reset": True,
                "registry": True,
            }
        )
    )
