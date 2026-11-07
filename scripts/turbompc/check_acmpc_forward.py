"""Compare frozen 28N MPC solutions with an independent NumPy/SciPy control NLP.

This offline oracle eliminates every state with the already pinned NumPy CTBR
map. It does not call the production dynamics, SQP, ADMM or gradient code.
The terminal control remains an optimization variable, matching the cost nodes.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests/test_turbompc_ctbr.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(path, numpy_step):
    metadata = json.loads(path.with_suffix(".json").read_text())
    model = metadata["model"]
    if metadata["acceleration_constraints"]:
        raise ValueError("This oracle covers the primary CTBR input/velocity/SFC constraints")
    rows = []
    with np.load(path) as frozen:
        for batch in range(len(frozen["initial_state"])):
            initial = frozen["initial_state"][batch]
            diagonal, cross, linear = (frozen[key][batch] for key in ("D", "E", "q"))
            matrix, lower, upper = (frozen[key][batch] for key in ("G", "lower", "upper"))
            native = frozen["trajectory"][batch]
            nodes = len(native)
            active_rows = np.any(matrix != 0, axis=-1)
            has_lower = active_rows & np.isfinite(lower)
            has_upper = active_rows & np.isfinite(upper)

            def rollout(flat_controls, nodes=nodes, initial=initial):
                controls = np.asarray(flat_controls).reshape(nodes, 4)
                trajectory = np.empty((nodes, 14), dtype=np.float64)
                state = initial.copy()
                for k in range(nodes):
                    trajectory[k] = np.r_[state, controls[k]]
                    if k + 1 < nodes:
                        state = numpy_step(trajectory[k], dt=model["dt"],
                                           mass=model["mass"], gravity=model["gravity"])
                return trajectory

            def cost_of(trajectory, diagonal=diagonal, cross=cross, linear=linear):
                return float(
                    .5 * np.einsum("ki,kij,kj->", trajectory, diagonal, trajectory)
                    + np.sum(linear * trajectory)
                    + np.einsum("ki,kij,kj->", trajectory[1:], cross, trajectory[:-1])
                )

            def objective(controls):
                return cost_of(rollout(controls))

            def margins(controls, matrix=matrix, lower=lower, upper=upper,
                        has_lower=has_lower, has_upper=has_upper):
                value = np.einsum("kri,ki->kr", matrix, rollout(controls))
                return np.r_[(value - lower)[has_lower], (upper - value)[has_upper]]

            hover = np.tile([model["mass"] * model["gravity"], 0., 0., 0.], (nodes, 1))
            warm = hover.copy()
            warm[:-1] = frozen["obs_ctbr_warm_start"][batch]
            guesses = [("frozen_warm_start", warm)]
            if not np.array_equal(warm, hover):
                guesses.append(("hover", hover))
            references = []
            native_cost = cost_of(native)
            native_rollout = rollout(native[:, 10:].reshape(-1))
            for label, guess in guesses:
                result = minimize(objective, guess.reshape(-1), method="SLSQP", jac="3-point",
                    constraints={"type": "ineq", "fun": margins},
                    options={"maxiter": 500, "ftol": 1e-11, "finite_diff_rel_step": 1e-5})
                reference = rollout(result.x)
                violation = max(0.0, -float(margins(result.x).min()))
                control_error = float(np.abs(reference[:, 10:] - native[:, 10:]).max())
                state_error = float(np.abs(reference[:, :10] - native[:, :10]).max())
                objective_error = abs(float(result.fun) - native_cost)
                # Native stationarity is 1e-4 and ADMM absolute tolerance is 1e-5.
                # These checks compare numerical local solutions, not global optimality.
                passed = bool(result.success and violation < 1e-7 and control_error < 2e-3
                              and state_error < 2e-4 and objective_error < 2e-4)
                references.append(dict(initialization=label, success=bool(result.success),
                    message=str(result.message), iterations=int(result.nit),
                    objective=float(result.fun), objective_abs_difference=objective_error,
                    control_max_abs_difference=control_error, state_max_abs_difference=state_error,
                    reference_constraint_violation=violation, passed=passed))
            dynamics_error = float(np.abs(native_rollout[:, :10] - native[:, :10]).max())
            rows.append(dict(batch_index=batch, native_objective=native_cost,
                native_numpy_reintegration_error=dynamics_error,
                native_reintegrated_constraint_violation=max(0.0, -float(margins(native[:, 10:].reshape(-1)).min())),
                comparisons=references, passed=dynamics_error < 1e-5 and all(r["passed"] for r in references)))
    return dict(input=str(path), input_sha256=digest(path), metadata_sha256=digest(path.with_suffix(".json")),
        cost_nodes=model["cost_nodes"], case=metadata["case"], model=model,
        rows=rows, passed=all(row["passed"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    numpy_step = runpy.run_path(str(ORACLE))["numpy_step"]
    cases = [compare(path, numpy_step) for path in args.inputs]
    report = dict(scope="Independent eliminated-control NLP; local solution and feasibility comparison",
        method="SciPy SLSQP with central finite-difference objective gradients; frozen warm start and distinct hover guess",
        numpy=np.__version__, scipy=scipy.__version__, oracle_sha256=digest(ORACLE),
        script_sha256=digest(Path(__file__)), cases=cases, passed=all(case["passed"] for case in cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
