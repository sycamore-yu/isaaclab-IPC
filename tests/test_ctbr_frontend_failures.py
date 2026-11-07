"""Planning failures obey Isaac's reward-before-reset step order."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_ipc.pathcover_native import PathCoverPlanningError
from isaaclab_ipc.rl.ctbr_observation import make_ctbr_observation
from isaaclab_ipc.tasks.ctbr_env import CTBRMPCEnv, IPCEnv
from isaaclab_ipc.tasks.rewards import NavigationRewardCfg
from isaaclab_ipc.turbompc.ctbr import CTBRConfig


def environment(monkeypatch, automatic_reset=True):
    """Keep production step hooks; replace only physics and native planning."""
    env = CTBRMPCEnv.__new__(CTBRMPCEnv)
    env.sim = SimpleNamespace(device="cpu")
    env.scene = SimpleNamespace(num_envs=2, env_origins=torch.zeros((2, 3)))
    env.cfg = SimpleNamespace(
        automatic_reset=automatic_reset, sim=SimpleNamespace(dt=0.01), decimation=10,
        log_dir=None, scenario="forest", frontend="pathcover", reward=NavigationRewardCfg(),
    )
    env.ctbr_config = CTBRConfig(cost_nodes=2)
    env.ids = torch.arange(2)
    pose = torch.zeros((2, 7))
    pose[:, 2], pose[:, 6] = 1.0, 1.0
    pose[0, 0] = 0.1
    twist = torch.zeros((2, 6))
    env.body_state = lambda: (pose, twist)
    env.goal = torch.tensor([8.0, 0.0, 1.0])
    env.elapsed = torch.full((2,), 0.1)
    env.episode_length_buf = torch.ones(2, dtype=torch.long)
    env.reported = torch.zeros(2, dtype=torch.bool)
    env.reward_active = torch.ones(2, dtype=torch.bool)
    env.hit = torch.zeros_like(env.reported)
    env._reset_context = torch.zeros_like(env.reported)
    env._next_context_valid = torch.zeros_like(env.reported)
    env.frontend_failure = torch.zeros_like(env.reported)
    env._frontend_errors = {}
    env._next_context = None
    env.records, env.observation_times = [], []
    env.extras = {}
    env.episode_ids = np.ones(2, dtype=np.int64)
    env.episode_record_index = np.full(2, -1, dtype=np.int64)
    env.command_violations = torch.zeros(2, dtype=torch.long)
    env.qp_failures = torch.zeros(2, dtype=torch.long)
    env.qp_residual_max = torch.zeros(2)
    env.sfc_violation_max = torch.zeros(2)
    env.speed_max = torch.zeros(2)
    env.min_clearance = torch.ones(2)
    env.previous_distance = torch.full((2,), 8.0)
    env.controller = SimpleNamespace(command=torch.tensor([[9.81, 0.0, 0.0, 0.0]]).repeat(2, 1))
    env.previous_action = env.controller.command.clone()
    env.reward_sums = torch.zeros((2, 4))
    env._warm_controls = torch.zeros((2, 1, 4), dtype=torch.float64)
    env._warm_controls[..., 0] = 0.75 * 9.81
    env.last_mpc_result = {"solved": torch.ones(2, dtype=torch.bool)}
    env.recorder_manager = SimpleNamespace(record_post_step=lambda: None)
    reset_calls = []
    env.navigation = SimpleNamespace(reset=lambda ids: reset_calls.append(list(ids)))

    state = torch.zeros((2, 10), dtype=torch.float64)
    state[:, 2], state[:, 3] = 1.0, 1.0
    reference = torch.zeros((2, 2, 9), dtype=torch.float64)
    reference[..., 2] = 1.0
    planes = torch.zeros((2, 2, 1, 4), dtype=torch.float64)
    planes[..., 0, 0], planes[..., 0, 3] = 1.0, -0.05
    env.current_context = make_ctbr_observation(
        state, torch.zeros((2, 3)), env.goal.expand(2, -1), reference,
        torch.zeros((2, 1)), planes, torch.ones((2, 2, 1), dtype=torch.bool),
        ctbr_config=env.ctbr_config,
    )

    def physical_dones(instance):
        instance.extras.update(
            success=torch.zeros(2, dtype=torch.bool), collision=instance.hit.clone(),
            terminal_position=pose[:, :3].clone(),
        )
        return torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)

    def physical_reset(instance, ids):
        pose[ids, 0] = 0
        instance.elapsed[ids] = 0
        instance.episode_length_buf[ids] = 0

    monkeypatch.setattr(IPCEnv, "_get_dones", physical_dones)
    monkeypatch.setattr(IPCEnv, "_reset_idx", physical_reset)
    return env, reset_calls


def test_failed_row_terminates_before_reward_and_only_that_row_replans_after_reset(monkeypatch):
    env, resets = environment(monkeypatch)
    old = env.current_context
    frozen = {name: value.clone() for name, value in old.items()}
    calls = []

    def observe(ids=None, *, allow_failures=False):
        calls.append((ids.tolist(), allow_failures))
        result = {name: value[ids].clone() for name, value in old.items()}
        result["policy"].fill_(7 if allow_failures else 9)
        if allow_failures:
            result["sfc_valid"][0] = False
            return result, {0: PathCoverPlanningError("injected occupied start")}
        return result

    env._observe_pathcover = observe
    term, trunc = env._get_dones()
    assert term.tolist() == [True, False] and not trunc.any()
    assert env.records[0]["frontend_failure"]
    assert env.records[0]["frontend_failure_reason"] == "injected occupied start"
    # The measured violation is against the pre-action corridor, before replan.
    assert env.records[0]["maximum_sfc_violation_m"] == pytest.approx(0.05)
    env.reset_terminated, env.reset_time_outs = term, trunc
    env._get_rewards()
    assert env.extras["reward_terms"]["terminal"].tolist() == [-2.0, 0.0]
    assert env.records[0]["reward_terms"]["terminal"] == -2.0
    with pytest.raises(RuntimeError, match="Reset failed planning rows"):
        env._get_observations()

    env._reset_idx(torch.tensor([0]))
    result = env._get_observations()
    assert calls == [([0, 1], True), ([0], False)]
    assert resets == [[0]]
    assert env.episode_ids.tolist() == [2, 1]
    assert result["sfc_valid"].all()
    assert result["policy"][:, 0].tolist() == [9.0, 7.0]
    for name in old:
        torch.testing.assert_close(old[name], frozen[name], atol=0, rtol=0)
        assert result[name].data_ptr() != old[name].data_ptr()


@pytest.mark.parametrize("stage", ["initial", "reset", "fixed_evaluation"])
def test_initial_reset_and_fixed_evaluation_planning_failures_raise(monkeypatch, stage):
    env, _ = environment(monkeypatch, automatic_reset=stage != "fixed_evaluation")
    if stage == "reset":
        env._next_context = {name: value.clone() for name, value in env.current_context.items()}
        env._reset_context[0] = True
    calls = []

    def observe(ids=None, *, allow_failures=False):
        calls.append(allow_failures)
        raise PathCoverPlanningError("strict planning failure")

    env._observe_pathcover = observe
    if stage == "fixed_evaluation":
        env._get_dones()
        assert calls == []
    with pytest.raises(PathCoverPlanningError, match="strict planning failure"):
        env._get_observations()
    assert calls == [False]


def test_completed_evaluation_rows_return_measurement_without_replanning(monkeypatch):
    env, _ = environment(monkeypatch, automatic_reset=False)
    previous = env.current_context
    frozen = {name: value.clone() for name, value in previous.items()}
    env.reported[0] = True
    calls = []

    def observe(ids=None, *, allow_failures=False):
        calls.append((ids.tolist(), allow_failures))
        assert ids.tolist() == [1]
        return {name: value[ids].clone() for name, value in previous.items()}

    env._observe_pathcover = observe
    result = env._get_observations()
    assert calls == [([1], False)]
    assert result["ctbr_state"][0, 0] == pytest.approx(0.1)
    assert result["policy"][0, 0] == pytest.approx(0.79)
    assert result["policy"][0, 13] == pytest.approx(-0.01)
    for name in ("reference", "planes", "ctbr_reference", "ctbr_control_reference"):
        torch.testing.assert_close(result[name][0], frozen[name][0], atol=0, rtol=0)
    for name in previous:
        torch.testing.assert_close(previous[name], frozen[name], atol=0, rtol=0)
    env.reported[:] = True
    final = env._get_observations()
    assert calls == [([1], False)]
    torch.testing.assert_close(final["ctbr_state"][:, :3], env.body_state()[0][:, :3].double())
