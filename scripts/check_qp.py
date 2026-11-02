"""Compare current CPU/CUDA solvers with the original Docker OSQP-Eigen binary."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.qp import BatchedQP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    m = MPC(Config.load())
    rng = np.random.default_rng(42)
    problems = []
    for i in range(8):
        state = np.r_[0, 0, 1, rng.uniform(-0.5, 0.5, 6)]
        ref = np.zeros((15, 9))
        ref[:, :3] = state[:3] + np.arange(15)[:, None] * rng.uniform(-0.15, 0.15, 3)
        ref[:, 2] = 1.0
        planes = np.array([[0, 1, 0, -0.8], [0, -1, 0, -0.8], [0, 0, 1, -2.5], [0, 0, -1, 0.5]])
        problem = m.problem(state, ref, [planes] * 15)
        problems.append(problem)
        h, g, aa, lower, u, _ = problem
        # Preserve IPC's lower-only continuity entries to exercise Data.tpp.
        source_h = h.copy()
        for j in range(1, 15):
            source_h[j * 3 : (j + 1) * 3, (j - 1) * 3 : j * 3] -= 0.4 * np.eye(3)
        with (out / f"{i}.txt").open("w") as f:
            f.write(f"45 {len(lower)}\n")
            for v in [source_h, g, aa, np.maximum(lower, -1e30), np.minimum(u, 1e30)]:
                np.savetxt(f, v.reshape(1, -1), fmt="%.17g")
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{__import__('os').getuid()}:{__import__('os').getgid()}",
        "-v",
        f"{Path('results/oracle').resolve()}:/bin-oracle:ro",
        "-v",
        f"{out}:/out",
        "ipc-hku-mars:noetic",
        "bash",
        "-c",
        "for i in $(seq 0 7); do /bin-oracle/qp_oracle /out/$i.txt /out/$i.out || exit; /bin-oracle/qp_oracle /out/$i.txt /out/$i-strict.out strict || exit; done",
    ]
    subprocess.run(command, check=True)
    expected = np.array([np.loadtxt(out / f"{i}.out")[1:] for i in range(8)])
    strict = np.array([np.loadtxt(out / f"{i}-strict.out")[1:] for i in range(8)])
    cpu = np.array([m.solve(problem)[0] for problem in problems])
    solver = BatchedQP(
        8, 45, max(len(p[3]) for p in problems), device="cuda:0", max_iterations=16384, eps_abs=1e-5, eps_rel=0.0
    )
    gpu, status, pr, du, it = solver.solve(problems)
    result = {
        "cases": 8,
        "cpu_first_jerk_max_difference": float(abs(cpu[:, :3] - expected[:, :3]).max()),
        "gpu_first_jerk_max_difference": float(abs(gpu[:, :3] - expected[:, :3]).max()),
        "gpu_vs_strict_first_jerk_max_difference": float(abs(gpu[:, :3] - strict[:, :3]).max()),
        "gpu_statuses": status,
        "gpu_primal_residuals": pr.tolist(),
        "gpu_dual_residuals": du.tolist(),
        "gpu_iterations": it,
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert all(s == "solved" for s in status), result
    assert abs(gpu[:, :3] - strict[:, :3]).max() < 0.01, result


if __name__ == "__main__":
    main()
