"""Replay the same observed cloud/state sequence through serial and parallel IPC frontends."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from isaaclab_ipc.frontend import PlanningBatch
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner


def measure(cloud, num_envs, steps, workers, cached):
    planners = [Planner(Config.load(), [8, 0, 1], cache_corridors=cached) for _ in range(num_envs)]
    times = []
    digest = hashlib.sha256()
    with PlanningBatch(workers) as batch:
        for step in range(steps):
            state = np.tile([step * 0.005, 0, 1, 0.5, 0, 0, 0, 0, 0.0], (num_envs, 1))
            clouds = dict.fromkeys(range(num_envs), cloud) if step % 10 == 0 else {}
            begin = perf_counter()
            results = batch.contexts(planners, state, clouds)
            times.append((perf_counter() - begin) * 1000)
            for result in results:
                digest.update(bytes([result.valid]))
                digest.update(result.state.tobytes())
                digest.update(result.reference.tobytes())
                for faces in result.corridors:
                    digest.update(np.asarray(faces).tobytes())
    stats = dict(
        workers=workers,
        cache=cached,
        steps=steps,
        num_envs=num_envs,
        median_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        total_ms=sum(times),
        native_corridor_calls=sum(p.corridor_evaluations for p in planners),
        cache_hits=sum(p.corridor_cache_hits for p in planners),
        output_sha256=digest.hexdigest(),
    )
    del planners
    gc.collect()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cloud = np.load(args.cloud, allow_pickle=False)
    rows = [
        measure(cloud, args.num_envs, args.steps, workers, cache)
        for workers, cache in ((1, False), (1, True), (4, True))
    ]
    assert len({row["output_sha256"] for row in rows}) == 1, "Planning outputs changed"
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
