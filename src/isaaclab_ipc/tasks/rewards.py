"""Task-level four-term navigation reward following AC-SMPC Eq. (27)."""

import torch
from isaaclab.utils import configclass


@configclass
class NavigationRewardCfg:
    progress: float = 0.1
    time: float = 0.01
    smooth: float = 0.0004
    goal: float = 2.0
    collision: float = 4.0
    failure: float = 2.0
    action_scale: tuple[float, ...] = (9.81, 1.0, 1.0, 1.0)


def navigation_reward(cfg, previous_distance, distance, action, previous_action, dt, success, collision, failure):
    scale = action.new_tensor(cfg.action_scale)
    terminal = cfg.goal * success.float() - cfg.collision * collision.float() - cfg.failure * failure.float()
    return torch.stack(
        (
            cfg.progress * (previous_distance - distance),
            torch.full_like(distance, -cfg.time * dt),
            -cfg.smooth * ((action - previous_action) / scale).square().sum(-1),
            terminal,
        ),
        dim=-1,
    )
