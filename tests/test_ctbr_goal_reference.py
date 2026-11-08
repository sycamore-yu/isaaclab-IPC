"""A0 time-sampled references share a frozen Newton/ROS observation contract."""

import math

import torch

from isaaclab_ipc.rl.ctbr_observation import goal_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig


def state(batch=1):
    value = torch.zeros((batch, 10), dtype=torch.float64)
    value[:, 3] = 1
    return value


def test_distant_goal_is_not_repeated_at_every_prediction_node():
    cfg = CTBRConfig()
    measured = state()
    measured[:, :3] = torch.tensor([0., -50., 1.5])
    measured[:, 7:] = torch.tensor([.1, -.2, .3])
    goal = torch.tensor([[0., 50., 1.5]], dtype=torch.float64)
    warm = torch.zeros((1, cfg.horizon, 4), dtype=torch.float64)
    warm[..., 0] = cfg.mass * cfg.gravity
    obs = goal_ctbr_observation(measured, torch.zeros((1, 3)), goal, ctbr_config=cfg,
                                warm_start_controls=warm)
    expected = measured[:, None, :3] + torch.tensor([0., 1., 0.]) * (torch.arange(5) * .3)[None, :, None]
    torch.testing.assert_close(obs["reference"][..., :3], expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(obs["reference"][..., 3:6], torch.tensor([0., 3., 0.], dtype=torch.float64).expand(1, 5, 3))
    torch.testing.assert_close(obs["ctbr_state"], measured, rtol=0, atol=0)
    torch.testing.assert_close(obs["goal"], goal, rtol=0, atol=0)
    torch.testing.assert_close(obs["policy"][0, :3], torch.tensor([0., 10., 0.]), rtol=0, atol=0)
    torch.testing.assert_close(obs["yaw_target"], torch.tensor([[math.pi / 2]], dtype=torch.float64))
    torch.testing.assert_close(obs["ctbr_warm_start"], warm, rtol=0, atol=0)
    assert obs["planes"].shape == (1, 5, 0, 4)


def test_partial_arrival_and_hold_keep_discrete_position_velocity_consistent():
    cfg = CTBRConfig()
    measured = state(3)
    goals = torch.tensor([[.35, 0., 0.], [0., 0., 0.], [1e-14, 0., 0.]], dtype=torch.float64)
    obs = goal_ctbr_observation(measured, torch.zeros((3, 3)), goals, ctbr_config=cfg)
    pva = obs["reference"]
    torch.testing.assert_close(pva[0, :, 0], torch.tensor([0., .3, .35, .35, .35], dtype=torch.float64))
    torch.testing.assert_close(pva[0, :, 3], torch.tensor([3., .5, 0., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(pva[:, 1:, :3] - pva[:, :-1, :3], cfg.dt * pva[:, :-1, 3:6], rtol=1e-14, atol=0)
    assert torch.equal(pva[..., 6:], torch.zeros_like(pva[..., 6:]))
    assert torch.equal(pva[1], torch.zeros_like(pva[1]))
    assert all(bool(value.isfinite().all()) for value in obs.values())


def test_ros_wrapper_and_frozen_context_preserve_shared_reference_contract():
    from isaaclab_ipc.ros_ctbr_policy import goal_observation

    cfg = CTBRConfig()
    measured = state()
    rates = torch.zeros((1, 3), dtype=torch.float64)
    goal = torch.tensor([[1., 2., 0.]], dtype=torch.float64)
    direct = goal_ctbr_observation(measured, rates, goal, ctbr_config=cfg, reference_speed=2.)
    ros = goal_observation(measured, rates, goal, cfg, reference_speed=2., max_planes=0)
    for name in direct:
        torch.testing.assert_close(ros[name], direct[name], rtol=0, atol=0)
    old_state = ros["ctbr_state"].clone()
    old_goal = ros["goal"].clone()
    old_reference = ros["reference"].clone()
    measured.fill_(4.)
    goal.fill_(5.)
    torch.testing.assert_close(ros["ctbr_state"], old_state, rtol=0, atol=0)
    torch.testing.assert_close(ros["goal"], old_goal, rtol=0, atol=0)
    torch.testing.assert_close(ros["reference"], old_reference, rtol=0, atol=0)
