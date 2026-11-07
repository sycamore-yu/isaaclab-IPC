#!/usr/bin/env python3
"""Measure complete Newton/PPO updates through the production AC-MPC entry point.

The first iteration includes shape compilation and is reported separately.
Subsequent iterations can still compile a new solver/backward branch. Validate
warm status with profile_acmpc_updates.py; two iterations do not establish a p95.
The solver-only forward/backward benchmark is a separate experiment.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/benchmark_acmpc_updates.py",
    "scripts/acmpc.py",
    "src/isaaclab_ipc/rl/ctbr_actor.py",
    "src/isaaclab_ipc/rl/ctbr_observation.py",
    "src/isaaclab_ipc/rl/config.py",
    "src/isaaclab_ipc/tasks/ctbr_env.py",
    "src/isaaclab_ipc/turbompc/ctbr.py",
    "src/isaaclab_ipc/turbompc/quadrotor.py",
    "src/isaaclab_ipc/turbompc/solver.py",
)

# This subprocess wrapper adds CPU-only compiler counters around the existing
# PPO update. It neither synchronizes CUDA nor replaces the production loop.
COMPILE_AUDIT_RUNNER = """
import json
from pathlib import Path
import runpy
import sys
from rsl_rl.algorithms import PPO
from torch._dynamo.utils import counters
entry, output, *arguments = sys.argv[1:]
original = PPO.update
events = []
def update(self, *values, **kwargs):
    before = dict(counters['stats'])
    result = original(self, *values, **kwargs)
    events.append({'before': before, 'after': dict(counters['stats'])})
    return result
PPO.update = update
sys.argv = [entry, *arguments]
try:
    runpy.run_path(entry, run_name='__main__')
finally:
    PPO.update = original
    if Path(output).is_dir():
        (Path(output) / 'compile-counters.json').write_text(json.dumps(events, indent=2))
"""


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def read_iteration_times(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(directory)).Reload()
    tags = {
        "collection_s": "Perf/collection_time",
        "ppo_update_s": "Perf/learning_time",
        "samples_per_s": "Perf/total_fps",
        "solved_fraction": "Control/solved",
    }
    rows = {}
    for name, tag in tags.items():
        for event in events.Scalars(tag):
            rows.setdefault(event.step, {"iteration": event.step})[name] = event.value
    result = [rows[index] for index in sorted(rows)]
    counters = json.loads((directory / "compile-counters.json").read_text())
    if len(counters) != len(result):
        raise RuntimeError("Compiler counters must cover every PPO update")
    previous = None
    for row, compile_event in zip(result, counters, strict=True):
        row["collection_and_update_s"] = row["collection_s"] + row["ppo_update_s"]
        row["compile_graphs_before_update"] = compile_event["before"].get("unique_graphs", 0)
        row["compile_graphs_after_update"] = compile_event["after"].get("unique_graphs", 0)
        row["no_new_graphs_since_previous_update"] = (
            previous is not None and row["compile_graphs_after_update"] == previous
        )
        previous = row["compile_graphs_after_update"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--mini-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--task", default="Isaac-ACMPC-CTBR-Direct")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.iterations < 2 or min(*args.batches, args.rollout_steps, args.mini_batches, args.epochs) < 1:
        parser.error("Require positive batch/rollout/PPO counts and at least two iterations")
    if any(batch * args.rollout_steps % args.mini_batches for batch in args.batches):
        parser.error("The rollout size must divide evenly into PPO mini-batches")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    report = {
        "scope": "Actual Newton collection and complete RSL-RL PPO update via scripts/acmpc.py",
        "task": args.task,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "mini_batches": args.mini_batches,
        "epochs": args.epochs,
        "timing": "TensorBoard Perf/collection_time and Perf/learning_time from the production runner",
        "first_iteration": "Includes lazy compilation; a persistent compiler cache may already be populated",
        "post_first_samples": args.iterations - 1,
        "steady_state_verified": False,
        "post_first_scope": "Per-iteration graph counters identify further specialization; a single warm sample does not establish a latency percentile",
        "source_sha256": source_hashes(),
        "environment": {
            name: environment.get(name)
            for name in ("TURBOMPC_NATIVE_BUILD_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_COMPILE_THREADS", "OMP_NUM_THREADS")
        },
        "runs": [],
    }
    for batch in args.batches:
        run_directory = directory / f"B{batch}"
        entry_arguments = [
            "train",
            "--task", args.task, "--output", str(run_directory),
            "--num-envs", str(batch), "--iterations", str(args.iterations),
            f"agent.num_steps_per_env={args.rollout_steps}",
            f"agent.algorithm.num_mini_batches={args.mini_batches}",
            f"agent.algorithm.num_learning_epochs={args.epochs}",
            *args.override,
        ]
        command = [sys.executable, "-u", "-c", COMPILE_AUDIT_RUNNER,
                   str(ROOT / "scripts/acmpc.py"), str(run_directory), *entry_arguments]
        print(json.dumps({"starting_num_envs": batch, "command": command}), flush=True)
        tick = perf_counter()
        with (directory / f"B{batch}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        row = {
            "num_envs": batch,
            "rollout_samples": batch * args.rollout_steps,
            "ppo_actor_batch_size": batch * args.rollout_steps // args.mini_batches,
            "optimizer_steps_per_update": args.mini_batches * args.epochs,
            "command": command,
            "exit_code": result.returncode,
            "process_wall_s": perf_counter() - tick,
            "source_sha256_after": source_hashes(),
        }
        if result.returncode == 0:
            metrics = json.loads((run_directory / "metrics.json").read_text())
            rows = read_iteration_times(run_directory)
            if len(rows) != args.iterations:
                raise RuntimeError(f"Expected {args.iterations} timing samples, found {len(rows)}")
            row.update(
                first_iteration=rows[0], subsequent_iterations=rows[1:],
                cost_network_max_parameter_change=metrics["cost_network_max_parameter_change"],
                checkpoint_action_match=metrics["checkpoint_action_match"],
                completed_episodes=metrics["completed_episodes"],
                solver=metrics["mpc"],
                physical_model_contract=metrics["physical_model_contract"],
            )
        report["runs"].append(row)
        report["steady_state_verified"] = all(
            run["exit_code"] == 0 and any(
                value["no_new_graphs_since_previous_update"] for value in run["subsequent_iterations"]
            ) for run in report["runs"]
        )
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in row.items() if key not in ("command", "solver", "physical_model_contract", "source_sha256_after")}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
