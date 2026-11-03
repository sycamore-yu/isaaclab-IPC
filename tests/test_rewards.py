import numpy as np
import torch

from isaaclab_ipc.tasks.rewards import NavigationRewardCfg, navigation_reward


def test_four_terms_have_expected_units_and_exclusive_terminals():
    cfg = NavigationRewardCfg(progress=5.0, time=0.5, smooth=0.02, goal=50.0, collision=50.0, failure=50.0)
    action = torch.tensor([[9.81, 0, 0, 0.0], [19.62, 1, 0, 0], [9.81, 0, 0, 0]])
    previous = torch.tensor([[9.81, 0, 0, 0.0]]).expand(3, -1)
    terms = navigation_reward(
        cfg,
        torch.tensor([2.0, 2, 2]),
        torch.tensor([1.5, 2.5, 2]),
        action,
        previous,
        0.01,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([False, False, True]),
    )
    torch.testing.assert_close(terms[:, 0], torch.tensor([2.5, -2.5, 0]))
    torch.testing.assert_close(terms[:, 1], torch.full((3,), -0.005))
    torch.testing.assert_close(terms[:, 2], torch.tensor([0, -0.04, 0]))
    torch.testing.assert_close(terms[:, 3], torch.tensor([50.0, -50, -50]))
    assert terms[0].sum() > terms[1].sum()


def test_terminal_reward_is_paid_once_and_logged_by_component():
    from types import SimpleNamespace

    from isaaclab_ipc.tasks.mpc_env import IPCMPCEnv

    env = SimpleNamespace(
        body_state=lambda: (torch.tensor([[9.6, 0, 1, 0, 0, 0, 1.0], [2, 0, 1, 0, 0, 0, 1.0]]), None),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
        goal=torch.tensor([10.0, 0, 1]),
        hit=torch.tensor([False, True]),
        reward_active=torch.ones(2, dtype=torch.bool),
        extras={"success": torch.tensor([True, False])},
        command_violations=np.zeros(2, dtype=int),
        device="cpu",
        reset_terminated=torch.ones(2, dtype=torch.bool),
        reset_time_outs=torch.zeros(2, dtype=torch.bool),
        previous_distance=torch.tensor([0.5, 8.0]),
        controller=SimpleNamespace(command=torch.tensor([[9.81, 0, 0, 0.0]]).repeat(2, 1)),
        previous_action=torch.tensor([[9.81, 0, 0, 0.0]]).repeat(2, 1),
        step_dt=0.01,
        cfg=SimpleNamespace(
            reward=NavigationRewardCfg(progress=5.0, time=0.5, smooth=0.02, goal=50.0, collision=50.0, failure=50.0),
            log_dir=None,
        ),
        reward_sums=torch.zeros(2, 4),
        last_mpc_result={
            "solved": torch.ones(2, dtype=torch.bool),
            "braking": torch.zeros(2, dtype=torch.bool),
            "sfc_valid": torch.ones(2, dtype=torch.bool),
        },
        records=[{}, {}],
        episode_record_index=[0, 1],
        recorder_manager=SimpleNamespace(record_post_step=lambda: None),
    )
    rewards = IPCMPCEnv._get_rewards(env)
    assert rewards[0] > 50 and rewards[1] < -50
    assert set(env.records[0]["reward_terms"]) == {"progress", "time", "smooth", "terminal"}
    assert env.records[0]["reward_terms"]["terminal"] == 50
    assert env.records[1]["reward_terms"]["terminal"] == -50
    torch.testing.assert_close(IPCMPCEnv._get_rewards(env), torch.zeros(2))
