"""Evaluate IPC through actual Newton/MJWarp motor forces, without training."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from isaaclab_ipc.execution import ExecutionProfile
from isaaclab_ipc.frontend import PlanningBatch
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner, flatness
from isaaclab_ipc.scenes import AvoidScene
from isaaclab_ipc.tasks.env import IPCEnv, make_cfg
from isaaclab_ipc.turbompc.ipc import FixedIPCSolver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=list(ExecutionProfile), default=ExecutionProfile.REAL.value)
    p.add_argument("--goal-x", type=float)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--planner-workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--forest", type=Path)
    p.add_argument("--max-steps", type=int, default=3500)
    a = p.parse_args()
    profile = ExecutionProfile(a.profile)
    try:
        profile.validate_evaluation(forest=a.forest, goal_x=a.goal_x, num_envs=a.num_envs)
    except ValueError as error:
        p.error(str(error))
    if not profile.uses_newton:
        from evaluate_ideal import run

        run(a.stride, a.distance, a.output)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    kind = "forest" if a.forest else "avoid"
    cfg = make_cfg(
        a.num_envs,
        device=a.device,
        distance=a.distance,
        scenario=kind,
        forest_usd=(a.forest / "forest.usdc").resolve() if a.forest else None,
    )
    cfg.automatic_reset = False
    if a.goal_x is not None:
        cfg.goal_x = a.goal_x
        cfg.episode_length_s = 90.0
    planners = [Planner(Config.load(kind, a.stride, profile=profile), [cfg.goal_x, 0, 1]) for _ in range(a.num_envs)]
    scenes = [AvoidScene(a.distance) for _ in planners]
    traces = [[] for _ in planners]
    done = np.zeros(a.num_envs, bool)
    records = []
    first_failure = np.zeros(a.num_envs, bool)
    timings = []
    violations = np.zeros(a.num_envs, int)
    min_clear = np.full(a.num_envs, np.inf)
    with IPCEnv(cfg) as env, PlanningBatch(a.planner_workers) as frontend, FixedIPCSolver(
        planners[0].cfg.params, device=a.device
    ) as solver:
        obs, _ = env.reset(seed=0)
        start = perf_counter()
        for step in range(a.max_steps):
            tick_start = perf_counter()
            values = obs["policy"].cpu().numpy()
            actions = []
            clouds = dict(enumerate(env.lidar_clouds())) if step % 10 == 0 else {}
            for scene, state in zip(scenes, values, strict=True):
                if not a.forest:
                    scene.advance(step * 0.01, state[:3], np.linalg.norm(state[3:6]))
            contexts = frontend.contexts(planners, values, clouds)
            prepared = perf_counter()
            results = solver.solve(contexts)
            solved = perf_counter()
            for i, (planner, context, state, result) in enumerate(
                zip(planners, contexts, values, results, strict=True)
            ):
                status, solution = result["status"], result["solution"]
                if status not in ("solved", "solved inaccurate") and not first_failure[i]:
                    first_failure[i] = True
                    problem = planner.mpc.problem(
                        result["state"], context.reference,
                        context.corridors if context.valid else [np.empty((0, 4))] * planner.mpc.n,
                    )
                    np.savez_compressed(
                        a.output / f"first-failed-qp-{i}.npz",
                        H=problem[0],
                        g=problem[1],
                        A=problem[2],
                        lower=problem[3],
                        upper=problem[4],
                        state=problem[5],
                        status=status,
                        step=step,
                    )
                x, j, ok = planner.consume_state(
                    result["state"], np.zeros(3 * planner.mpc.n) if solution is None else solution,
                    status, step * 0.01,
                )
                yaw = (
                    np.arctan2(planner.goal[1] - state[1], planner.goal[0] - state[0])
                    if planner.cfg.params["yaw_ctrl_flag"]
                    else 0.0
                )
                action = flatness(x[6:], j, state[9:].reshape(3, 3), yaw)
                action[0] = np.clip(action[0], 0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
                actions.append(action)
                v = np.r_[x[3:], j]
                lo = np.r_[planner.mpc.lower, planner.mpc.jlow[:3]]
                hi = np.r_[planner.mpc.upper, planner.mpc.jhigh[:3]]
                if not done[i]:
                    violations[i] += bool(
                        np.any((v < lo - 0.001 - 0.001 * np.abs(lo)) | (v > hi + 0.001 + 0.001 * np.abs(hi)))
                    )
            obs, reward, term, trunc, extra = env.step(
                torch.as_tensor(np.asarray(actions), device=env.device, dtype=torch.float32)
            )
            timings.append(
                [(prepared - tick_start) * 1000, (solved - prepared) * 1000, (perf_counter() - solved) * 1000]
            )
            actual = obs["policy"].cpu().numpy()
            collisions = extra["collision"].cpu().numpy()
            successes = extra["success"].cpu().numpy()
            for i in range(a.num_envs):
                if done[i]:
                    continue
                min_clear[i] = min(
                    min_clear[i], env.min_clearance[i].item() if a.forest else scenes[i].clearance(actual[i, :3])
                )
                traces[i].append(np.r_[step * 0.01, actual[i, :9], actions[i], min_clear[i]])
                reached = bool(successes[i])
                collision = bool(collisions[i])
                timeout = step == a.max_steps - 1
                if reached or collision or timeout:
                    done[i] = True
                    result = {
                        "env_id": i,
                        "backend": "newton-mjwarp",
                        "profile": a.profile,
                        "recovery_events": planners[i].recovery_events,
                        "brake_steps": planners[i].brake_steps,
                        "solver": "turbompc-cuda",
                        "success": bool(reached and not collision and violations[i] == 0),
                        "goal_crossed": reached,
                        "collision": collision,
                        "timeout": timeout,
                        "steps": step + 1,
                        "violations": int(violations[i]),
                        "minimum_point_clearance_m": float(min_clear[i]),
                        "minimum_physics_step_clearance_m": env.min_clearance[i].item(),
                        "trigger_speed_mps": scenes[i].trigger_speed,
                        "qp_failures": planners[i].failures,
                        "frontend_failures": planners[i].frontend_failures,
                        "final_position": actual[i, :3].tolist(),
                    }
                    records.append(result)
                    np.savez_compressed(a.output / f"trace-{i}.npz", trace=np.asarray(traces[i]))
                    print(json.dumps(result), flush=True)
                    (a.output / "records.json").write_text(json.dumps(records, indent=2))
            if done.all():
                break
            if step % 100 == 0:
                print("step", step, "p", actual[0, :3], flush=True)
        np.save(a.output / "stage-times-ms.npy", np.array(timings))
        (a.output / "run.json").write_text(
            json.dumps(
                {
                    "wall_s": perf_counter() - start,
                    "num_envs": a.num_envs,
                    "planner_workers": a.planner_workers,
                    "steps": step + 1,
                    "complete": bool(done.all()),
                    "scenario": kind,
                    "sensor": "mid360-multimesh",
                    "profile": a.profile,
                    "goal_x": cfg.goal_x,
                    "stride": a.stride,
                    "trigger_distance_m": a.distance,
                    "solver": "turbompc-cuda",
                    "dt": cfg.sim.dt,
                    "decimation": cfg.decimation,
                    "timing_columns": ["sensor_frontend_qp_assembly_ms", "qp_solve_ms", "control_physics_ms"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
