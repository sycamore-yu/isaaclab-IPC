#!/usr/bin/env python3
"""Inject one expected planning failure into real Newton/PPO, using acmpc.py."""

import argparse
import json
import sys
from pathlib import Path

import torch
from acmpc import main as acmpc_main

from isaaclab_ipc.pathcover_native import NativePathCoverFrontend, PathCoverPlanningError
from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor
from isaaclab_ipc.tasks.ctbr_env import CTBRMPCEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"injected_failures": 0, "actor_batches_checked": 0, "steps": [], "resets": [], "passed": False}
    target_frontend = None
    original_plan = NativePathCoverFrontend.plan
    original_observe = CTBRMPCEnv._observe_pathcover
    original_reset = CTBRMPCEnv._reset_idx
    original_step = CTBRMPCEnv.step
    original_actor = CTBRMPCActor.forward

    def observe(env, env_ids=None, *, allow_failures=False):
        nonlocal target_frontend
        if allow_failures and target_frontend is None:
            target_frontend = env.navigation.frontends[0]
        return original_observe(env, env_ids, allow_failures=allow_failures)

    def plan(frontend, points_world, current_state, goal, **kwargs):
        if frontend is target_frontend and current_state.time > 0 and report["injected_failures"] == 0:
            report["injected_failures"] += 1
            raise PathCoverPlanningError("injected one-row planning failure")
        return original_plan(frontend, points_world, current_state, goal, **kwargs)

    def reset(env, env_ids):
        ids = env.ids if env_ids is None else env_ids
        report["resets"].append({"step": int(env.common_step_counter), "env_ids": ids.cpu().tolist()})
        return original_reset(env, env_ids)

    def step(env, actions):
        previous = env.current_context
        frozen = {name: value.clone() for name, value in previous.items()}
        episodes_before = env.episode_ids.copy()
        result = original_step(env, actions)
        observations, reward, terminated, truncated, extras = result
        for name, value in previous.items():
            torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
        assert bool(observations["sfc_valid"].all()), "Failed placeholder reached returned observations"
        record = {
            "terminated": terminated.cpu().tolist(), "truncated": truncated.cpu().tolist(),
            "reward": reward.cpu().tolist(), "terminal_reward": extras["reward_terms"]["terminal"].cpu().tolist(),
            "frontend_failure": extras["frontend_failure"].cpu().tolist(),
            "episode_ids_before": episodes_before.tolist(), "episode_ids_after": env.episode_ids.tolist(),
            "elapsed_after_s": env.elapsed.cpu().tolist(), "snapshot_unchanged": True,
            "valid_returned_context": True,
        }
        report["steps"].append(record)
        if len(report["steps"]) == 1:
            assert record["terminated"] == [True, False]
            assert record["truncated"] == [False, False]
            assert record["frontend_failure"] == [True, False]
            assert record["terminal_reward"] == [-2.0, 0.0]
            assert env.episode_ids.tolist() == [int(episodes_before[0] + 1), int(episodes_before[1])]
            torch.testing.assert_close(env.elapsed.cpu(), torch.tensor([0.0, 0.1]), atol=1e-7, rtol=0)
        return result

    def actor(module, obs, *positional, **keywords):
        assert bool(obs["sfc_valid"].all()), "Failed planning placeholder reached Actor/PPO"
        report["actor_batches_checked"] += 1
        return original_actor(module, obs, *positional, **keywords)

    NativePathCoverFrontend.plan = plan
    CTBRMPCEnv._observe_pathcover = observe
    CTBRMPCEnv._reset_idx = reset
    CTBRMPCEnv.step = step
    CTBRMPCActor.forward = actor
    sys.argv = [
        "acmpc.py", "train", "--audit-training", "--num-envs", "2", "--iterations", "1",
        "--task", "Isaac-ACMPC-PathCover-Forest-Direct", "--output", str(args.output),
        "agent.num_steps_per_env=4", "agent.algorithm.num_mini_batches=2",
        "agent.algorithm.num_learning_epochs=1",
    ]
    try:
        acmpc_main()
        metrics = json.loads((args.output / "metrics.json").read_text())
        audit = metrics["training_audit"]
        assert report["injected_failures"] == 1 and len(report["steps"]) == 4
        assert metrics["checkpoint_action_match"]
        assert metrics["cost_network_max_parameter_change"] > 0
        for phase in ("update", "reload"):
            assert audit["counts"][phase]["native_frontend_plan"] == 0
        report.update(
            passed=True, checkpoint_action_match=True,
            cost_network_max_parameter_change=metrics["cost_network_max_parameter_change"],
            frontend_calls_during_update=audit["counts"]["update"]["native_frontend_plan"],
            frontend_calls_during_reload=audit["counts"]["reload"]["native_frontend_plan"],
        )
    finally:
        NativePathCoverFrontend.plan = original_plan
        CTBRMPCEnv._observe_pathcover = original_observe
        CTBRMPCEnv._reset_idx = original_reset
        CTBRMPCEnv.step = original_step
        CTBRMPCActor.forward = original_actor
        if args.output.is_dir():
            (args.output / "failure-reset-check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
