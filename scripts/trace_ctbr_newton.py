#!/usr/bin/env python3
"""Trace fixed-cost Newton navigation through scripts/acmpc.py for diagnosis.

Tracing copies tensors to the host and is not a performance measurement.
The production environment, frontend, policy, solver, and evaluation loop run
unchanged. The first frontend input and the last observation are frozen for replay.
"""

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from acmpc import main as acmpc_main

from isaaclab_ipc.pathcover_native import NativePathCoverFrontend
from isaaclab_ipc.tasks.ctbr_env import CTBRMPCEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Isaac-ACMPC-PathCover-Avoid-Direct")
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    events = []
    last_observation = None
    plan_count = 0
    reset_generations = {}
    original_plan = NativePathCoverFrontend.plan
    original_reset = NativePathCoverFrontend.reset
    original_observe = CTBRMPCEnv._get_observations
    original_step = CTBRMPCEnv._pre_physics_step

    def plan(frontend, points_world, current_state, goal, **kwargs):
        nonlocal plan_count
        plan_count += 1
        frozen = dict(
            points_world=None if points_world is None else np.asarray(points_world).copy(),
            current_state=asdict(current_state), goal=np.asarray(goal).copy(),
            reset_generation=reset_generations.get(id(frontend), 0),
        )
        for name in ("sensor_origin", "initial_motion_seed", "initial_motion_tip_segment", "initial_thrust_axis"):
            if kwargs.get(name) is not None:
                frozen[name] = np.asarray(kwargs[name]).copy()
        inputs = args.output / "frontend-inputs"
        inputs.mkdir(exist_ok=True)
        torch.save(frozen, inputs / f"{plan_count:05d}.pt")
        if plan_count == 1:
            torch.save(frozen, args.output / "first-frontend-input.pt")
        try:
            result = original_plan(frontend, points_world, current_state, goal, **kwargs)
        except BaseException:
            frozen["gcopter_problem"] = frontend.failed_problem()
            frozen["geometry_obstacles"] = frontend.geometry_obstacles()
            frozen["frontend_config"] = asdict(frontend.config)
            torch.save(frozen, args.output / "failed-frontend-call.pt")
            raise
        frozen["frontend_plan"] = result
        if "initial_boundary" in result.metrics:
            frozen["initial_boundary"] = deepcopy(result.metrics["initial_boundary"])
        torch.save(frozen, inputs / f"{plan_count:05d}.pt")
        if plan_count == 1:
            torch.save(frozen, args.output / "first-frontend-input.pt")
        return result

    def reset(frontend):
        key = id(frontend)
        reset_generations[key] = reset_generations.get(key, 0) + 1
        return original_reset(frontend)

    def observe(env):
        nonlocal last_observation
        result = original_observe(env)
        last_observation = {name: value.detach().cpu().clone() for name, value in result.items()}
        if not events:
            torch.save(last_observation, args.output / "first-observation.pt")
        events.append(dict(
            kind="observation", elapsed_s=env.elapsed.cpu().tolist(),
            frontend_plan_calls=plan_count, reported=env.reported.cpu().tolist(),
            state=result["ctbr_state"].cpu().tolist(),
            measured_body_rates=result["body_rates"].cpu().tolist(),
            measured_acceleration_world=env.acceleration().cpu().tolist(),
            goal=result["goal"].cpu().tolist(),
            reference_pva=result["reference"].cpu().tolist(),
            reference_controls=result["ctbr_control_reference"].cpu().tolist(),
            active_faces=result["plane_mask"].sum(-1).cpu().tolist(),
        ))
        return result

    def step(env, actions):
        result = env._pending
        events.append(dict(
            kind="action", elapsed_s=env.elapsed.cpu().tolist(),
            action=actions.detach().cpu().tolist(),
            solved=result["solved"].cpu().tolist(),
            primal=result["primal"].cpu().tolist(), dual=result["dual"].cpu().tolist(),
            predicted=result["predicted"].cpu().tolist(),
            trajectory=result["trajectory"].cpu().tolist(),
        ))
        return original_step(env, actions)

    NativePathCoverFrontend.plan = plan
    NativePathCoverFrontend.reset = reset
    CTBRMPCEnv._get_observations = observe
    CTBRMPCEnv._pre_physics_step = step
    sys.argv = [
        "acmpc.py", "evaluate", "--fixed-cost", "--num-envs", "1",
        "--task", args.task, "--output", str(args.output),
        f"env.episode_length_s={args.duration_s}", *args.override,
    ]
    try:
        acmpc_main()
    finally:
        NativePathCoverFrontend.plan = original_plan
        NativePathCoverFrontend.reset = original_reset
        CTBRMPCEnv._get_observations = original_observe
        CTBRMPCEnv._pre_physics_step = original_step
        if args.output.is_dir():
            (args.output / "state-reference-trace.json").write_text(json.dumps(events, indent=2))
            if last_observation is not None:
                torch.save(last_observation, args.output / "last-observation.pt")


if __name__ == "__main__":
    main()
