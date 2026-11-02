"""Test a captured failing QP with original OSQP-Eigen and a feasibility LP."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("input", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
q = np.load(a.input)
H, g, A, lower, u = (q[k] for k in ["H", "g", "A", "lower", "upper"])
with (a.output / "input.txt").open("w") as f:
    f.write(f"{len(g)} {len(lower)}\n")
    for v in [H, g, A, np.maximum(lower, -1e30), np.minimum(u, 1e30)]:
        np.savetxt(f, v.reshape(1, -1), fmt="%.17g")
subprocess.run(
    [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{Path('results/oracle').resolve()}:/bin-oracle:ro",
        "-v",
        f"{a.output.resolve()}:/out",
        "ipc-hku-mars:noetic",
        "/bin-oracle/qp_oracle",
        "/out/input.txt",
        "/out/output.txt",
    ],
    check=True,
)
original = np.loadtxt(a.output / "output.txt")
finite_l = np.isfinite(lower)
finite_u = np.isfinite(u)
lp = linprog(
    np.zeros(len(g)),
    A_ub=np.vstack((A[finite_u], -A[finite_l])),
    b_ub=np.r_[u[finite_u], -lower[finite_l]],
    bounds=[(None, None)] * len(g),
    method="highs",
)
# A single uniform bound relaxation gives a physical-units diagnostic, not a replacement controller.
rows = np.vstack((A[finite_u], -A[finite_l]))
bounds = np.r_[u[finite_u], -lower[finite_l]]
relax = linprog(
    np.r_[np.zeros(len(g)), 1.0],
    A_ub=np.column_stack((rows, -np.ones(len(rows)))),
    b_ub=bounds,
    bounds=[(None, None)] * len(g) + [(0, None)],
    method="highs",
)
result = {
    "source_input": str(a.input),
    "step": int(q["step"]),
    "state": q["state"].tolist(),
    "candidate_status": str(q["status"]),
    "original_solver_success": bool(original[0]),
    "lp_feasible": bool(lp.success),
    "lp_status": lp.message,
    "minimum_uniform_constraint_relaxation": float(relax.x[-1]) if relax.success else None,
}
(a.output / "diagnosis.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
