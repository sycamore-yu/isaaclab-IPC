"""Paired official/compact sensor validation and bounded IPC simulation scaling.

Run each variant in a fresh process. GPU process memory includes native Warp allocations;
Torch allocator statistics alone do not. Query timing is separate from environment stepping.
"""

import argparse
import json
import os
import resource
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import warp as wp

from isaaclab_ipc.mid360 import mid360_config
from isaaclab_ipc.tasks.env import IPCEnv, make_cfg


def process_gpu_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        pid, _, memory = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return float(memory.strip())
    return None


def statistics(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("official", "compact"), default="compact")
    parser.add_argument("--scenario", choices=("forest", "avoid"), default="forest")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-hits", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg(
        num_envs=args.num_envs,
        device=args.device,
        scenario=args.scenario,
        forest_usd=Path("build/forest/forest.usdc").resolve(),
    )
    cfg.scene.mid360 = mid360_config(args.scenario, optimized=args.sensor == "compact")
    cfg.automatic_reset = False
    cfg.episode_length_s = 120.0
    gpu = args.device.startswith("cuda")

    def sync():
        if gpu:
            torch.cuda.synchronize(args.device)

    started = perf_counter()
    with IPCEnv(cfg) as env:
        env.reset(seed=0)
        sensor = env.scene["mid360"]
        sync()
        setup_s = perf_counter() - started
        initial = sensor.data.ray_hits_w.torch.clone()
        local = initial - env.scene.env_origins[:, None, :]
        finite = torch.isfinite(local).all(-1)
        if args.num_envs > 1:
            torch.testing.assert_close(local[0], local[-1], atol=1e-4, rtol=1e-5)
        ids = sensor._mesh_ids_wp.numpy()
        if args.sensor == "compact":
            assert len(np.unique(ids)) == (1 if args.scenario == "forest" else 2)
        memory = [process_gpu_mib()] if gpu else []
        timings = []
        query_mask = wp.full(env.num_envs, True, dtype=wp.bool, device=env.device)
        # SensorBase.force_recompute does not mark fresh environments outdated.
        # Explicitly query every ray, without advancing the sensor's simulated clock.
        for _ in range(2):
            sensor._update_buffers_impl(query_mask)
        sync()
        for _ in range(args.queries):
            sync()
            t0 = perf_counter()
            sensor._update_buffers_impl(query_mask)
            sync()
            timings.append((perf_counter() - t0) * 1000)
        action = torch.tensor([9.81, 0, 0, 0], device=env.device).expand(env.num_envs, -1)
        window_before = env.mid360_window_start.clone()
        step_times = []
        for tick in range(args.steps):
            sync()
            t0 = perf_counter()
            env.step(action)
            if (tick + 1) % 10 == 0:
                env.lidar_clouds()
            sync()
            step_times.append((perf_counter() - t0) * 1000)
        hits = sensor.data.ray_hits_w.torch.clone()
        if args.steps >= 10:
            assert not torch.equal(window_before, env.mid360_window_start)
        position = env.body_state()[0][:, :3]
        ranges = (hits - position[:, None, :]).norm(dim=-1)
        assert ranges[torch.isfinite(ranges)].max().item() <= 40.001
        if args.save_hits:
            np.savez(
                args.output / "hits.npz",
                initial=initial.cpu().numpy(),
                after=hits.cpu().numpy(),
                pose=env.body_state()[0].cpu().numpy(),
                windows=env.mid360_window_start.cpu().numpy(),
                env_origins=env.scene.env_origins.cpu().numpy(),
            )
        if args.num_envs > 1:
            other_hits = hits[1].clone()
            other_window = env.mid360_window_start[1].clone()
            env._reset_idx(torch.tensor([0], dtype=torch.int32, device=env.device))
            torch.testing.assert_close(sensor.data.ray_hits_w.torch[1], other_hits)
            torch.testing.assert_close(env.mid360_window_start[1], other_window)
        # Count only persistent ray-sized arrays, separately from the once-per-sensor table.
        buffers = {}
        for name in ("ray_starts", "ray_directions", "_ray_starts_w", "_ray_directions_w", "_ray_distance_wp"):
            array = getattr(sensor, name, None)
            if array is not None:
                array = array.warp if hasattr(array, "warp") else array
                buffers[name] = int(array.size * wp.types.type_size_in_bytes(array.dtype))
        buffers["hits"] = int(initial.numel() * initial.element_size())
        if gpu:
            memory.append(process_gpu_mib())
        result = dict(
            variant=args.sensor,
            scenario=args.scenario,
            device=args.device,
            num_envs=env.num_envs,
            sensor_class=type(sensor).__name__,
            rays=sensor.num_rays,
            setup_s=setup_s,
            unique_mesh_ids=len(np.unique(ids)),
            initial_hits_per_env=finite.sum(-1).cpu().tolist(),
            persistent_ray_buffer_bytes=buffers,
            query_ms=statistics(timings),
            query_mask="all environments active; warmed real ray queries, not cached sensor access",
            simulation_control_step_ms=statistics(step_times),
            process_gpu_memory_mib_samples=memory,
            host_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            parity_checks="environment transforms, max range, window changes and partial reset passed",
            note="Simulation timing includes 10 Hz cloud transfer, excludes A*/SFC/QP/PPO; shared GPU.",
        )
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
