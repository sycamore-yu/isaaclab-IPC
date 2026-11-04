"""Bounded M1/M2 audit of the fixed official-linear B1/N15 acceptance case.

Observe instance-local SQP assembly/ADMM calls, then replay identical captured
QP arrays through both raw solvers. No upstream files or algorithms are edited.
This is an eager diagnostic, not a performance or gradient acceptance run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = "a771284"
CORE_DIRECTORIES = ("src/isaaclab_ipc/turbompc", "native/turbompc")
SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".cuh", ".h"}
QP_FIELDS = ("D", "E", "q", "A0", "A_minus", "A_plus", "c0", "c_dyn", "G", "lower", "upper", "slack_weight")
STATE_FIELDS = ("x", "z", "y", "lambda0", "lambda_dyn", "xi", "rho")
RESIDUAL_NAMES = ("primal_residual", "dual_residual", "primal_residual_limit", "dual_residual_limit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    paths = list((ROOT / "src/isaaclab_ipc/turbompc").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("check_acceptance.py")]
    paths += [p for p in (ROOT / "native/turbompc").rglob("*") if p.suffix in {".cpp", ".cu", ".cuh", ".h"}]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def anchor_sources():
    """Check core bytes against the frozen anchor while allowing later docs commits."""
    git = ["git", "-C", str(ROOT)]
    anchor = subprocess.check_output([*git, "rev-parse", ANCHOR], text=True).strip()
    names = subprocess.check_output(
        [*git, "ls-tree", "-r", "--name-only", anchor, "--", *CORE_DIRECTORIES], text=True
    ).splitlines()
    expected = {
        name: hashlib.sha256(subprocess.check_output([*git, "show", f"{anchor}:{name}"])).hexdigest()
        for name in names
        if Path(name).suffix in SOURCE_SUFFIXES
    }
    actual = {
        str(p.relative_to(ROOT)): digest(p)
        for directory in CORE_DIRECTORIES
        for p in (ROOT / directory).rglob("*")
        if p.is_file() and p.suffix in SOURCE_SUFFIXES
    }
    return {
        "anchor": anchor,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": bool(expected) and expected == actual,
    }


def provenance(source, repository):
    """Read source/build records without loading Torch, JAX, or CUDA libraries."""
    import check_acceptance as fixture

    spec = importlib.util.find_spec("turbompc")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Installed upstream turbompc package not found")
    installed = Path(next(iter(spec.submodule_search_locations)))
    return {
        "frozen_core": anchor_sources(),
        "verified_upstream_sources": fixture.verify_original(source, repository, installed),
        "build_records": fixture.verified_build_records(),
    }


def preflight(source):
    original = source / "turbompc/solvers/turbompc_solver.py"
    native = ROOT / "src/isaaclab_ipc/turbompc/solver.py"
    names = {}
    for label, path in (("A", original), ("B", native)):
        tree = ast.parse(path.read_text())
        names[label] = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    checks = {
        "A_build_qp": "_build_qp" in names["A"],
        "A_solve": "solve" in names["A"],
        "B_qp": "_qp" in names["B"],
        "B_solve": "solve" in names["B"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "actual_upstream_hook": "_build_qp",
        "gpu_initialized": False,
    }


def cpu(value, *, batched=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.array(value, copy=True)
    except Exception as error:
        raise RuntimeError(
            "Capture encountered a JAX tracer; eager SQP observation is unavailable at this boundary"
        ) from error
    if batched:
        if array.ndim == 0 or array.shape[0] != 1:
            raise ValueError(f"This audit requires B1, received {array.shape}")
        array = array[0].copy()
    return array


def qp_a(qp):
    return {
        "D": cpu(qp.cost.D),
        "E": cpu(qp.cost.E),
        "q": cpu(qp.cost.q),
        "A0": cpu(qp.eq.A0),
        "A_minus": cpu(qp.eq.A_minus),
        "A_plus": cpu(qp.eq.A_plus),
        "c0": cpu(qp.eq.c0),
        "c_dyn": cpu(qp.eq.c),
        "G": cpu(qp.ineq.G),
        "lower": cpu(qp.ineq.l),
        "upper": cpu(qp.ineq.u),
        "slack_weight": cpu(qp.ineq.slack_penalization_weight),
    }


def qp_b(qp, slack_weight):
    result = {key: cpu(getattr(qp, key), batched=True) for key in QP_FIELDS if key != "slack_weight"}
    result["slack_weight"] = np.asarray(slack_weight, dtype=np.float64)
    return result


def state_a(state):
    mapping = dict(x="x_blocks", z="z_g", y="y_g", lambda0="y_f_0", lambda_dyn="y_f_dyn", xi="xi_g", rho="rho_bar")
    return {key: cpu(getattr(state, attr)) for key, attr in mapping.items()}


def state_b(state):
    return {key: cpu(getattr(state, key), batched=True) for key in STATE_FIELDS}


def result_a(result):
    return {
        **state_a(result.state),
        "status": int(cpu(result.stats.status)),
        "iterations": int(cpu(result.stats.num_iter)),
        "residuals": np.asarray([float(cpu(getattr(result.stats, key))) for key in RESIDUAL_NAMES]),
    }


def result_b(result):
    return {
        **state_b(result),
        "status": int(cpu(result.status, batched=True)),
        "iterations": int(cpu(result.iterations, batched=True)),
        "residuals": cpu(result.residuals, batched=True),
    }


def zero_state(qp, rho):
    return {
        "x": np.zeros_like(qp["q"]),
        "z": np.clip(np.zeros_like(qp["lower"]), qp["lower"], qp["upper"]),
        "y": np.zeros_like(qp["lower"]),
        "lambda0": np.zeros_like(qp["c0"]),
        "lambda_dyn": np.zeros_like(qp["c_dyn"]),
        "xi": np.zeros_like(qp["lower"]),
        "rho": np.asarray(rho, dtype=np.float64),
    }


@contextmanager
def temporary_method(instance, name, replacement):
    missing = object()
    previous = instance.__dict__.get(name, missing)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if previous is missing:
            delattr(instance, name)
        else:
            setattr(instance, name, previous)


def classify(trace):
    for index, assembly in enumerate(trace["assemblies"]):
        used = [i for i, solve in enumerate(trace["solves"]) if solve["assembly_index"] == index]
        assembly["used_by_qp_solves"] = used
        assembly["role"] = (
            "initial_assembly" if index == 0 else "assembly_for_next_iteration" if used else "final_evaluation_only"
        )
    return trace


def capture_a(runtime, case, fixture):
    import jax
    import jax.numpy as jnp

    solver = runtime.backend.solver
    if solver.program.rescale_optimization_variables or solver.program.use_slack_variables:
        raise RuntimeError("Audit requires physical coordinates and no slack")
    trace, references = {"assemblies": [], "solves": []}, []
    original_build, original_admm = solver._build_qp, solver._admm_solver_fwd.solve

    def build(states, controls, slacks, params, admm_state, **kwargs):
        qp = original_build(states, controls, slacks, params, admm_state, **kwargs)
        trace["assemblies"].append(
            {
                "after_qp_solve_count": len(trace["solves"]),
                "linearization_point": np.concatenate((cpu(states), cpu(controls)), axis=-1),
                "qp": qp_a(qp),
                "scale_variables_argument": bool(kwargs["scale_variables"]),
                "physical_coordinates": True,
                "hessian": str(kwargs["hessian"]),
            }
        )
        references.append(qp)
        return qp

    def solve_qp(qp_data, admm_state0=None, rho_bar=1.0, alpha=1.0):
        data = qp_a(qp_data)
        warm = zero_state(data, rho_bar) if admm_state0 is None else state_a(admm_state0)
        result = original_admm(qp_data, admm_state0, rho_bar, alpha)
        trace["solves"].append(
            {
                "assembly_index": len(trace["assemblies"]) - 1,
                "qp": data,
                "warm_start": warm,
                "result": result_a(result),
                "rho_argument": float(rho_bar),
                "alpha_argument": float(alpha),
            }
        )
        return result

    params, _ = runtime.problem.params_from_context(jnp.asarray(case["state"][0]))
    weights = runtime.problem.cost_weights(jnp.asarray(case["costs"][0]))
    seed = solver.initial_guess(params)._replace(
        states=jnp.asarray(case["seed"][0, :, : fixture.NX]),
        controls=jnp.asarray(case["seed"][0, :, fixture.NX :]),
        admm_state=None,
    )
    with temporary_method(solver, "_build_qp", build), temporary_method(solver._admm_solver_fwd, "solve", solve_qp):
        with jax.disable_jit():
            solution = solver.solve(seed, params, weights)
    trace["final"] = {
        "trajectory": np.concatenate((cpu(solution.states), cpu(solution.controls)), axis=-1),
        "status": int(cpu(solution.status)),
        "sqp_iterations": int(cpu(solution.num_iter)),
        "convergence_error": float(cpu(solution.convergence_error)),
        "public_admm_iterations_history": cpu(solution.admm_iters),
        "public_admm_iterations_total": int(np.sum(cpu(solution.admm_iters))),
    }
    return classify(trace), references[0]


def capture_b(runtime):
    import torch

    solver = runtime.solver
    if solver.config.compile_kernels:
        raise RuntimeError("Native capture requires eager _qp")
    trace = {"assemblies": [], "solves": []}
    original_build, original_admm = solver._qp, solver.admm.solve

    def build(inputs, point):
        qp = original_build(inputs, point)
        trace["assemblies"].append(
            {
                "after_qp_solve_count": len(trace["solves"]),
                "linearization_point": cpu(point, batched=True),
                "qp": qp_b(qp, solver.admm.config.slack_weight),
                "physical_coordinates": True,
            }
        )
        return qp

    def solve_qp(qp, warm_start=None):
        data = qp_b(qp, solver.admm.config.slack_weight)
        warm = zero_state(data, solver.admm.config.rho) if warm_start is None else state_b(warm_start)
        result = original_admm(qp, warm_start)
        trace["solves"].append(
            {
                "assembly_index": len(trace["assemblies"]) - 1,
                "qp": data,
                "warm_start": warm,
                "result": result_b(result),
            }
        )
        return result

    with temporary_method(solver, "_qp", build), temporary_method(solver.admm, "solve", solve_qp):
        with torch.no_grad():
            _, solution, _ = runtime.forward()
    trace["final"] = {
        "trajectory": cpu(solution.trajectory, batched=True),
        "status": int(cpu(solution.status, batched=True)),
        "sqp_iterations": int(cpu(solution.iterations, batched=True)),
        "convergence_error": float(np.max(cpu(solution.errors))),
        "public_admm_iterations_last": int(cpu(solution.admm.iterations, batched=True)),
    }
    return classify(trace)


def comparison(reference, value, atol, rtol=0.0):
    reference, value = np.asarray(reference), np.asarray(value)
    if reference.shape != value.shape:
        return {"passed": False, "A_shape": list(reference.shape), "B_shape": list(value.shape)}
    finite = bool(np.isfinite(reference).all() and np.isfinite(value).all())
    difference = np.abs(reference - value)
    return {
        "passed": bool(finite and np.all(difference <= atol + rtol * np.abs(reference))),
        "max_abs": float(difference.max(initial=0.0)),
        "shape": list(reference.shape),
        "A_dtype": str(reference.dtype),
        "B_dtype": str(value.dtype),
        "finite": finite,
    }


def fields_comparison(a, b, keys, atol, rtol=0.0):
    values = {key: comparison(a[key], b[key], atol, rtol) for key in keys}
    return {"fields": values, "passed": all(value["passed"] for value in values.values())}


def independent_residuals(qp, state, admm):
    """Dense NumPy assembly, independent of both solver residual implementations."""
    t, n = qp["q"].shape
    n0, nx, m = qp["A0"].shape[0], qp["A_minus"].shape[1], qp["G"].shape[1]
    p, c, g = np.zeros((t * n, t * n)), np.zeros((n0 + (t - 1) * nx, t * n)), np.zeros((t * m, t * n))
    c[:n0, :n] = qp["A0"]
    for k in range(t):
        p[k * n : (k + 1) * n, k * n : (k + 1) * n] = qp["D"][k]
        g[k * m : (k + 1) * m, k * n : (k + 1) * n] = qp["G"][k]
        if k:
            p[k * n : (k + 1) * n, (k - 1) * n : k * n] = qp["E"][k - 1]
            p[(k - 1) * n : k * n, k * n : (k + 1) * n] = qp["E"][k - 1].T
            row = n0 + (k - 1) * nx
            c[row : row + nx, (k - 1) * n : k * n] = qp["A_minus"][k - 1]
            c[row : row + nx, k * n : (k + 1) * n] = qp["A_plus"][k - 1]
    x, q, z = state["x"].ravel(), qp["q"].ravel(), state["z"].ravel()
    rhs = np.concatenate((qp["c0"], qp["c_dyn"].ravel()))
    dual = np.concatenate((state["lambda0"], state["lambda_dyn"].ravel()))

    def norm(value):
        return float(np.max(np.abs(value), initial=0.0))

    cx, gx, px = c @ x, g @ x, p @ x
    aty = c.T @ dual + g.T @ state["y"].ravel()
    equality = norm(cx - rhs)
    inequality = float(np.maximum(np.maximum(qp["lower"].ravel() - gx, gx - qp["upper"].ravel()), 0).max(initial=0))
    primal, stationarity = max(equality, norm(gx - z)), norm(px + q + aty)
    limits = (
        admm["eps_abs"] + admm["eps_rel"] * max(norm(cx), norm(gx), norm(rhs), norm(z)),
        admm["eps_abs"] + admm["eps_rel"] * max(norm(px), norm(aty), norm(q)),
    )
    return {
        "equality_max_abs": equality,
        "bound_max_violation": inequality,
        "residual_vector": np.array([primal, stationarity, *limits]),
        "finite": bool(
            np.isfinite(x).all() and np.isfinite([equality, inequality, primal, stationarity, *limits]).all()
        ),
    }


def raw_comparison(qp, a, b, fixture, *, expected_status=0):
    atol, rtol = fixture.ADMM["eps_abs"], fixture.ADMM["eps_rel"]
    residuals = {label: independent_residuals(qp, value, fixture.ADMM) for label, value in (("A", a), ("B", b))}
    reports = {
        label: comparison(value["residuals"], residuals[label]["residual_vector"], atol, rtol)
        for label, value in (("A", a), ("B", b))
    }
    states = fields_comparison(a, b, STATE_FIELDS, fixture.THRESHOLDS["u0_max_abs"], rtol)
    parity = comparison(a["residuals"], b["residuals"], atol, rtol)
    gates = {
        "expected_status": a["status"] == b["status"] == expected_status,
        "finite_independent_residuals": all(v["finite"] for v in residuals.values()),
        "reported_residuals_verified": all(v["passed"] for v in reports.values()),
        "residual_parity": parity["passed"],
    }
    if expected_status == 0:
        gates.update(
            state_parity=states["passed"],
            iterations_equal=a["iterations"] == b["iterations"],
            physical_feasibility=all(
                max(v["equality_max_abs"], v["bound_max_violation"]) <= fixture.THRESHOLDS["physical_violation"]
                for v in residuals.values()
            ),
        )
    return {
        "A_status": a["status"],
        "B_status": b["status"],
        "A_iterations": a["iterations"],
        "B_iterations": b["iterations"],
        "state_parity": states,
        "residual_parity": parity,
        "independent_residuals": residuals,
        "reported_residual_checks": reports,
        "gates": gates,
        "passed": all(gates.values()),
        "infeasible_note": None
        if expected_status == 0
        else "No optimizer exists: raw iterates/iteration counts are reported, not treated as optimal solutions.",
    }


def compare_traces(a, b, fixture):
    atol, rtol = fixture.ADMM["eps_abs"], fixture.ADMM["eps_rel"]
    assemblies, solves = [], []
    for index, (aa, bb) in enumerate(zip(a["assemblies"], b["assemblies"], strict=False)):
        values = fields_comparison(aa["qp"], bb["qp"], QP_FIELDS, atol, rtol)
        point = comparison(aa["linearization_point"], bb["linearization_point"], fixture.THRESHOLDS["u0_max_abs"])
        assemblies.append(
            {
                "index": index,
                "A_role": aa["role"],
                "B_role": bb["role"],
                "data": values,
                "linearization_point": point,
                "passed": values["passed"] and point["passed"] and aa["role"] == bb["role"],
            }
        )
    for index, (aa, bb) in enumerate(zip(a["solves"], b["solves"], strict=False)):
        data = fields_comparison(aa["qp"], bb["qp"], QP_FIELDS, atol, rtol)
        warm = fields_comparison(
            aa["warm_start"], bb["warm_start"], STATE_FIELDS, fixture.THRESHOLDS["u0_max_abs"], rtol
        )
        output = raw_comparison(aa["qp"], aa["result"], bb["result"], fixture)
        # Verify that each recorded solve actually consumed its tagged assembly.
        consumed = {
            label: fields_comparison(trace["assemblies"][value["assembly_index"]]["qp"], value["qp"], QP_FIELDS, 0.0)
            for label, trace, value in (("A", a, aa), ("B", b, bb))
        }
        solves.append(
            {
                "index": index,
                "data": data,
                "warm_start": warm,
                "raw_result": output,
                "consumed_assembly": consumed,
                "passed": data["passed"]
                and warm["passed"]
                and output["passed"]
                and all(v["passed"] for v in consumed.values()),
            }
        )
    final = comparison(a["final"]["trajectory"], b["final"]["trajectory"], fixture.THRESHOLDS["u0_max_abs"])
    totals = {
        "A_captured_each": [v["result"]["iterations"] for v in a["solves"]],
        "B_captured_each": [v["result"]["iterations"] for v in b["solves"]],
        "A_public_total": a["final"]["public_admm_iterations_total"],
        "B_public_last": b["final"]["public_admm_iterations_last"],
        "semantics": "Upstream public history sums all SQP QPs; Native MPCResult.admm retains only the last QP result.",
    }
    totals.update(A_captured_total=sum(totals["A_captured_each"]), B_captured_total=sum(totals["B_captured_each"]))
    gates = {
        "assembly_counts_equal": len(a["assemblies"]) == len(b["assemblies"]) > 0,
        "solve_counts_equal": len(a["solves"]) == len(b["solves"]) > 0,
        "every_assembly": bool(assemblies) and all(v["passed"] for v in assemblies),
        "every_raw_qp": bool(solves) and all(v["passed"] for v in solves),
        "sqp_solved": a["final"]["status"] == b["final"]["status"] == 0,
        "sqp_iteration_counts": a["final"]["sqp_iterations"] == b["final"]["sqp_iterations"] == len(a["solves"]),
        "final_trajectory": final["passed"],
        "public_iteration_semantics": totals["A_public_total"] == totals["A_captured_total"]
        and bool(totals["B_captured_each"])
        and totals["B_public_last"] == totals["B_captured_each"][-1],
    }
    return {
        "assemblies": assemblies,
        "qp_solves": solves,
        "admm_iterations": totals,
        "final_trajectory": final,
        "gates": gates,
        "passed": all(gates.values()),
    }


def replay(original, baseline, native, fixture, *, infeasible):
    import jax
    import jax.numpy as jnp
    import torch

    from isaaclab_ipc.turbompc.native import QPData

    witness = None
    if infeasible:
        data = qp_a(original)
        matches = np.flatnonzero(np.all(data["G"][0] == data["A0"][0], axis=-1))
        if len(matches) != 1:
            raise RuntimeError("Cannot form the requested simple initial-state/box infeasibility witness")
        row = int(matches[0])
        upper = data["upper"].copy()
        upper[0, row] = data["c0"][0] - 1.0
        if not data["lower"][0, row] < upper[0, row]:
            raise RuntimeError("Infeasible test must retain valid lower <= upper bounds")
        witness = {
            "initial_equality_row": 0,
            "inequality_row_at_stage0": row,
            "forced_value": float(data["c0"][0]),
            "upper_bound": float(upper[0, row]),
            "gap": 1.0,
        }
        original = replace(original, ineq=replace(original.ineq, u=jnp.asarray(upper)))
    data = qp_a(original)

    def tensor(value):
        return torch.as_tensor(value.copy(), device="cuda:0", dtype=torch.float64).unsqueeze(0)

    copied = QPData(**{key: tensor(value) for key, value in data.items() if key != "slack_weight"})
    identical = fields_comparison(data, qp_b(copied, native.solver.admm.config.slack_weight), QP_FIELDS, 0.0)
    with jax.disable_jit():
        a = result_a(
            baseline.backend.solver._admm_solver_fwd.solve(
                original,
                admm_state0=None,
                rho_bar=fixture.ADMM["rho"],
                alpha=fixture.ADMM["relaxation_parameter"],
            )
        )
    b = result_b(native.solver.admm.solve(copied, warm_start=None))
    result = raw_comparison(data, a, b, fixture, expected_status=2 if infeasible else 0)
    result.update(
        exact_same_qp_arrays=identical,
        logical_infeasibility_witness=witness,
        passed=result["passed"] and identical["passed"],
    )
    return result, {"qp": data, "A": a, "B": b}


def flatten_arrays(value, prefix, output):
    if isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            flatten_arrays(child, f"{prefix}_{key}", output)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            flatten_arrays(child, f"{prefix}_{index}", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--preflight", action="store_true", help="CPU AST/source/build checks only; no Torch/JAX imports"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("TURBOMPC_BASELINE_SOURCE", ROOT / "build/turbompc/upstream")),
    )
    parser.add_argument("--upstream-repo", type=Path, default=ROOT / "build/reference")
    parser.add_argument("--infeasible", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.source)
        records = provenance(args.source, args.upstream_repo)
        result["frozen_anchor"] = records["frozen_core"]["anchor"]
        result["frozen_core_matches"] = records["frozen_core"]["passed"]
        result["upstream_files_verified"] = len(records["verified_upstream_sources"])
        result["build_ids"] = {key: value["build_id"] for key, value in records["build_records"].items()}
        result["passed"] = result["passed"] and result["frozen_core_matches"]
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.output is None:
        parser.error("--output is required for the GPU audit")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Use a fresh output name")
    import check_acceptance as fixture

    report = {
        "scope": "M1/M2 fixed B1/N15 official-linear QP audit; no gradient/performance claim",
        "time_utc": datetime.now(UTC).isoformat(),
        "harness_sha256": digest(__file__),
        "case": {"batch": 1, "horizon": fixture.N, "nx": fixture.NX, "nu": fixture.NU, "dtype": "float64"},
        "thresholds": fixture.THRESHOLDS,
        "solver_contract": fixture.SOLVER,
        "assembly_comparison": {
            "atol": fixture.ADMM["eps_abs"],
            "rtol": fixture.ADMM["eps_rel"],
            "coordinates": "physical [state,control] at each stage; no row sign/scaling normalization",
        },
        "passed": False,
    }
    artifacts, baseline, native = {}, None, None
    before = source_snapshot()
    try:
        if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
            raise RuntimeError("Use scripts/turbompc/env.sh cu128")
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        report["worktree_commit"] = revision
        report["provenance"] = provenance(args.source, args.upstream_repo)
        if not report["provenance"]["frozen_core"]["passed"]:
            raise RuntimeError("Core source differs from frozen anchor a771284")
        sys.path.insert(0, str(ROOT / "src"))
        import jax
        import torch

        jax.config.update("jax_enable_x64", True)
        if jax.__version__ != "0.6.2" or torch.version.cuda != "12.8":
            raise RuntimeError("Expected original JAX 0.6.2 and Torch CUDA 12.8")
        case = fixture.case_bank(1)
        report["case_sha256"] = fixture.array_hash(case)
        flatten_arrays(case, "input", artifacts)
        baseline = fixture.Baseline(case, args.source)
        native = fixture.Native(case, baseline.constants, compile_kernels=False)
        a, original_qp = capture_a(baseline, case, fixture)
        b = capture_b(native)
        report["A_trace"], report["B_trace"] = a, b
        flatten_arrays(a, "A", artifacts)
        flatten_arrays(b, "B", artifacts)
        report["sqp_qp_parity"] = compare_traces(a, b, fixture)
        report["raw_feasible"], raw = replay(original_qp, baseline, native, fixture, infeasible=False)
        flatten_arrays(raw, "raw_feasible", artifacts)
        if args.infeasible:
            report["raw_infeasible"], raw = replay(original_qp, baseline, native, fixture, infeasible=True)
            flatten_arrays(raw, "raw_infeasible", artifacts)
        report["versions"] = {"jax": jax.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda}
        report["loaded_libraries"] = fixture.loaded_libraries()
        report["loaded_library_sha256"] = {path: digest(path) for path in report["loaded_libraries"]}
        expected = {
            str(Path(path).resolve()): sha
            for record in report["provenance"]["build_records"].values()
            for path, sha in record["artifact_sha256"].items()
        }
        libraries = {
            path: sha
            for path, sha in report["loaded_library_sha256"].items()
            if "_ffi.so" in path or "_turbompc_native.so" in path
        }
        report["loaded_build_hashes_match"] = bool(libraries) and all(
            expected.get(str(Path(path).resolve())) == sha for path, sha in libraries.items()
        )
        report["gates"] = {
            "sqp_qp_parity": report["sqp_qp_parity"]["passed"],
            "identical_raw_feasible_qp": report["raw_feasible"]["passed"],
            "infeasible_requested_check": not args.infeasible or report["raw_infeasible"]["passed"],
            "loaded_build_hashes_match": report["loaded_build_hashes_match"],
        }
        report["passed"] = all(report["gates"].values())
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        for runtime in (native, baseline):
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as error:
                    report.setdefault("close_errors", []).append(f"{type(error).__name__}: {error}")
                    report["passed"] = False
        report["sources_before"], report["sources_after"] = before, source_snapshot()
        report["sources_unchanged"] = report["sources_before"] == report["sources_after"]
        report["passed"] = report["passed"] and report["sources_unchanged"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **artifacts)
        with args.output.open("x") as handle:
            json.dump(fixture.json_safe(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {"output": str(args.output), "passed": report["passed"], "error": report.get("error")}, indent=2
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
