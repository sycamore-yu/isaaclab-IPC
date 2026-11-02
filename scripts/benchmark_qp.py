"""Batched QP throughput with matching nominal OSQP tolerances; no physics timing."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.qp import BatchedQP

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
m = MPC(Config.load())
rng = np.random.default_rng(42)
fixtures = []
for _ in range(8):
    x = np.r_[0, 0, 1, rng.uniform(-0.5, 0.5, 6)]
    ref = np.zeros((15, 9))
    ref[:, :3] = x[:3] + np.arange(15)[:, None] * rng.uniform(-0.15, 0.15, 3)
    ref[:, 2] = 1
    planes = np.array([[0, 1, 0, -0.8], [0, -1, 0, -0.8], [0, 0, 1, -2.5], [0, 0, -1, 0.5]])
    fixtures.append(m.problem(x, ref, [planes] * 15))
records = []
for batch in [1, 8, 64]:
    problems = [fixtures[i % 8] for i in range(batch)]
    solver = BatchedQP(batch, 45, len(fixtures[0][3]))
    solver.solve(problems)
    m.solve(problems[0])
    cpu = []
    gpu = []
    for _ in range(3):
        start = time.perf_counter()
        cpu_values = [m.solve(problem) for problem in problems]
        cpu.append((time.perf_counter() - start) * 1000)
        torch.cuda.synchronize()
        start = time.perf_counter()
        values, status, pr, du, it = solver.solve(problems)
        torch.cuda.synchronize()
        gpu.append((time.perf_counter() - start) * 1000)
    record = {
        "batch": batch,
        "cpu_batch_median_ms": float(np.median(cpu)),
        "gpu_batch_median_ms": float(np.median(gpu)),
        "gpu_per_qp_ms": float(np.median(gpu) / batch),
        "gpu_statuses": status,
        "gpu_iterations": it,
        "max_first_jerk_difference_cpu": float(abs(values[:, :3] - np.array([c[0] for c in cpu_values])[:, :3]).max()),
        "max_primal_residual": float(pr.max()),
        "max_dual_residual": float(du.max()),
    }
    records.append(record)
    print(json.dumps(record), flush=True)
    (a.output / "benchmark.json").write_text(
        json.dumps({"shared_gpu": True, "eps_abs": 0.001, "eps_rel": 0.001, "records": records}, indent=2)
    )
