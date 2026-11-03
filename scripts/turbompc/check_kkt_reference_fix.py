"""Check the isolated JAX IR=2 candidate numerically; this is not A or M4.

Select its source with command-local PYTHONPATH AFTER scripts/turbompc/env.sh.
The installed baseline is hashed before and after, never modified. Outputs must
be new files. --preflight-only checks inputs/provenance without importing JAX.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROLE = "IR=2 JAX candidate accuracy diagnostic; not unchanged A and not M4"
LIBRARY = "libcudss_sparse_kkt_ffi.so"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/turbompc/kkt-reference-fix/cu128/manifest.json")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/data/turbompc_kkt_accuracy_reproducer.npz")
    parser.add_argument("--native-reference", type=Path, default=ROOT / "results/turbompc-port/sweep-linear-cu128-v1/torch-b1.npz")
    parser.add_argument("--reference", type=Path, default=ROOT / "build/reference", help="Read-only pinned upstream Git repository")
    parser.add_argument("--output", type=Path, help="New JSON report; raw arrays use the same stem with .npz")
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs and provenance on CPU, then exit")
    args = parser.parse_args()
    if not args.preflight_only and args.output is None:
        parser.error("--output is required for the GPU numerical diagnostic")
    if args.output is not None and args.output.suffix != ".json":
        parser.error("--output must end in .json")
    return args


def preflight(args):
    if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
        raise RuntimeError("Run through scripts/turbompc/env.sh cu128")
    manifest = json.loads(args.manifest.read_text())
    source = args.manifest.resolve().parent / "source"
    python_path = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if not python_path or python_path[0] != source:
        raise RuntimeError("Set PYTHONPATH to the candidate source AFTER env.sh, for this process only")
    spec = importlib.util.find_spec("turbompc")
    if spec is None or Path(spec.origin).resolve() != source / "turbompc/__init__.py":
        raise RuntimeError("Python would not import the explicitly selected candidate package")
    expected = source / "turbompc/solvers/_ffi" / LIBRARY
    baseline = Path(importlib.metadata.distribution("turbompc").locate_file("turbompc/solvers/_ffi")) / LIBRARY
    baseline = baseline.resolve()
    before = sha256(baseline)
    if manifest["profile"] != "cu128" or manifest["kkt_ir_steps"] != 2:
        raise RuntimeError("Manifest does not describe the cu128 IR=2 candidate")
    if expected != Path(manifest["library"]).resolve() or expected == baseline:
        raise RuntimeError("Candidate and baseline library locations are inconsistent")
    if sha256(expected) != manifest["library_sha256"]:
        raise RuntimeError("Candidate library SHA256 differs from its build manifest")
    if baseline != Path(manifest["installed_baseline_kkt"]).resolve():
        raise RuntimeError("The installed baseline location differs from the build manifest")
    if before != manifest["installed_baseline_kkt_sha256_before"] or before != manifest["installed_baseline_kkt_sha256_after"]:
        raise RuntimeError("The installed baseline hash differs from the preserved original")
    if any(importlib.metadata.version(package) != "0.6.2" for package in ("jax", "jaxlib")):
        raise RuntimeError("This diagnostic requires the pinned JAX/jaxlib 0.6.2 environment")

    # These project helpers import only NumPy/stdlib until prepare_jax is called.
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_turbompc import prepare_jax
    from turbompc_benchmark_common import UPSTREAM_COMMIT, make_case, physical_residuals

    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Candidate commit differs from the original benchmark commit")
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError("The read-only benchmark reference is not the pinned upstream commit")
    case = make_case(1, 15, False)
    with np.load(args.native_reference, allow_pickle=False) as saved:
        native = {key: saved[key].copy() for key in ("trajectory", "cost_gradient", "state_gradient", "status")}
        for key, value in case.items():
            if f"input_{key}" not in saved or not np.array_equal(saved[f"input_{key}"], value):
                raise RuntimeError(f"Saved native input_{key} differs from the unchanged old B1 case")
    for key, shape in (("trajectory", (1, 16, 12)), ("cost_gradient", (1, 12)), ("state_gradient", (1, 9))):
        if native[key].shape != shape or native[key].dtype != np.float64 or not np.isfinite(native[key]).all():
            raise RuntimeError(f"Invalid saved native {key}")
    if not np.all(native["status"] == 0):
        raise RuntimeError("The saved native comparison did not solve successfully")
    with np.load(args.fixture, allow_pickle=False) as saved:
        fixture = {key: saved[key].copy() for key in ("row", "col", "values", "rhs")}
    row, col, values, rhs = (fixture[key] for key in ("row", "col", "values", "rhs"))
    if row.shape != (41,) or rhs.shape != (40,) or values.dtype != np.float64 or rhs.dtype != np.float64:
        raise RuntimeError("Expected the saved 40x40 float64 KKT fixture")
    if row[0] != 0 or row[-1] != len(values) or len(col) != len(values) or np.any(np.diff(row) < 0):
        raise RuntimeError("Invalid fixture CSR structure")
    if np.any(col < 0) or np.any(col >= 40) or not np.isfinite(values).all() or not np.isfinite(rhs).all():
        raise RuntimeError("Invalid fixture indices or numerical values")
    dense = np.zeros((40, 40), dtype=np.float64)
    np.add.at(dense, (np.repeat(np.arange(40), np.diff(row)), col), values)
    fixture["dense"] = dense
    fixture["numpy_solution"] = np.linalg.solve(dense, rhs)
    provenance = {
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "upstream_commit": UPSTREAM_COMMIT, "upstream_reference": str(args.reference.resolve()),
        "candidate_source": str(source),
        "expected_candidate_library": str(expected), "expected_candidate_library_sha256": sha256(expected),
        "installed_baseline_library": str(baseline), "installed_baseline_sha256_before": before,
        "native_reference": str(args.native_reference.resolve()), "native_reference_sha256": sha256(args.native_reference),
        "fixture": str(args.fixture.resolve()), "fixture_sha256": sha256(args.fixture),
        "script_sha256": sha256(Path(__file__)),
        "benchmark_helper_sha256": sha256(ROOT / "scripts/benchmark_turbompc.py"),
        "case_helper_sha256": sha256(ROOT / "scripts/turbompc_benchmark_common.py"),
        "same_saved_inputs": True,
    }
    return SimpleNamespace(
        source=source, expected=expected, baseline=baseline, before=before,
        case=case, native=native, fixture=fixture, provenance=provenance,
        prepare_jax=prepare_jax, physical_residuals=physical_residuals,
    )


def run_diagnostic(args, inputs, report, arrays):
    import jax
    import jax.numpy as jnp
    from turbompc.solvers.backward import backward_kkt_cudss_ffi as backend

    loaded = Path(backend._LIB._name).resolve()
    if loaded != inputs.expected or not Path(backend.__file__).resolve().is_relative_to(inputs.source):
        raise RuntimeError(f"Wrong KKT candidate actually loaded: {loaded}")
    if sha256(loaded) != report["provenance"]["expected_candidate_library_sha256"]:
        raise RuntimeError("Actually loaded KKT candidate hash differs from the manifest")
    report["provenance"].update(actual_loaded_library=str(loaded), actual_loaded_library_sha256=sha256(loaded))
    jax.config.update("jax_enable_x64", True)
    report["versions"] = {"jax": jax.__version__, "jaxlib": importlib.metadata.version("jaxlib"), "numpy": np.__version__}
    fixture = inputs.fixture
    pattern = SimpleNamespace(rowPtr=fixture["row"], colIdx=fixture["col"], N_kkt=40, nnz=len(fixture["values"]))
    workspace = backend.CudssSparseKktWorkspace(pattern, np.float64)
    try:
        result = backend._call_ffi(jnp.asarray(fixture["values"]), jnp.asarray(fixture["rhs"]), workspace)
        result.block_until_ready()
        solution = np.asarray(result)
    finally:
        workspace.close()
    residual = float(np.max(np.abs(fixture["dense"] @ solution - fixture["rhs"])))
    error = float(np.max(np.abs(solution - fixture["numpy_solution"])))
    report["fixture"] = {"dimension": 40, "max_abs_residual": residual, "max_abs_solution_error": error}
    arrays.update({f"fixture_{key}": value for key, value in fixture.items() if key != "dense"})
    arrays["fixture_candidate_solution"] = solution
    report["checks"].update(fixture_residual=residual <= 1e-9, fixture_solution=error <= 1e-9)

    helper_args = SimpleNamespace(backend="jax", nonlinear=False, reference=args.reference, backward_backend="direct_cudss_ffi")
    runtime = inputs.prepare_jax(inputs.case, helper_args)
    try:
        result = runtime.full()
        runtime.ready(result)
        candidate = runtime.export(result)
        direction = np.random.default_rng(45).normal(size=inputs.case["costs"].shape)
        direction /= np.linalg.norm(direction)
        h = 1e-4
        tensor_direction = runtime.tensor(direction)
        plus = runtime.fd(tensor_direction, h)
        minus = runtime.fd(tensor_direction, -h)
        numerical = (plus - minus) / (2 * h)
    finally:
        runtime.close()
    native = inputs.native
    analytic = float(np.sum(candidate["cost_gradient"] * direction))
    native_analytic = float(np.sum(native["cost_gradient"] * direction))
    fd_relative = abs(analytic - numerical) / max(abs(numerical), 1e-12)
    gradient_relative = float(np.linalg.norm(candidate["cost_gradient"] - native["cost_gradient"]) / max(np.linalg.norm(native["cost_gradient"]), 1e-15))
    trajectory_error = float(np.max(np.abs(candidate["trajectory"] - native["trajectory"])))
    control_error = float(np.max(np.abs(candidate["trajectory"][..., 9:] - native["trajectory"][..., 9:])))
    residuals = inputs.physical_residuals(candidate["trajectory"], inputs.case, False)
    report["jerk_case"] = {
        "batch": 1, "horizon": 15, "nx": 9, "nu": 3, "dtype": "float64", "input_seed": 719,
        "status": candidate["status"].tolist(), "physical_residuals": residuals,
        "trajectory_max_abs_vs_native": trajectory_error, "control_max_abs_vs_native": control_error,
        "cost_gradient_relative_vs_native": gradient_relative,
        "cost_gradient_max_abs_vs_native": float(np.max(np.abs(candidate["cost_gradient"] - native["cost_gradient"]))),
        "state_gradient_max_abs_vs_native": float(np.max(np.abs(candidate["state_gradient"] - native["state_gradient"]))),
        "directional_finite_difference": {
            "direction_seed": 45, "h": h, "loss_plus": plus, "loss_minus": minus,
            "candidate_analytic": analytic, "saved_native_analytic": native_analytic, "numerical": numerical,
            "candidate_relative_error": fd_relative,
            "saved_native_relative_error": abs(native_analytic - numerical) / max(abs(numerical), 1e-12),
        },
    }
    report["checks"].update(
        candidate_solved=bool(np.all(candidate["status"] == 0)),
        physical_constraints=max(residuals.values()) <= 1e-5,
        controls_vs_saved_native=control_error <= 1e-4,
        full_cost_gradient_vs_saved_native=gradient_relative <= 1e-3,
        directional_finite_difference=fd_relative <= 1e-3,
        finite_outputs=all(bool(np.isfinite(value).all()) for value in candidate.values()),
    )
    arrays.update({f"candidate_{key}": value for key, value in candidate.items()})
    arrays.update({f"native_{key}": value for key, value in native.items()})
    arrays.update({f"input_{key}": value for key, value in inputs.case.items()})
    arrays["fd_direction"] = direction


def main():
    args = options()
    if args.output is not None and (args.output.exists() or args.output.with_suffix(".npz").exists()):
        raise FileExistsError("Choose a new output stem; existing results are preserved")
    inputs = preflight(args)
    if args.preflight_only:
        after = sha256(inputs.baseline)
        if after != inputs.before:
            raise RuntimeError("Installed baseline changed during CPU preflight")
        print(json.dumps({"role": ROLE, "cpu_preflight_only": True, "gpu_started": False, "provenance": inputs.provenance, "installed_baseline_sha256_after": after}, indent=2))
        return 0
    report = {"role": ROLE, "m4_result": False, "timings_collected": False, "provenance": inputs.provenance, "checks": {}}
    arrays = {}
    exit_code = 0
    try:
        run_diagnostic(args, inputs, report, arrays)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        exit_code = 1
    finally:
        after = sha256(inputs.baseline)
        report["provenance"]["installed_baseline_sha256_after"] = after
        report["checks"]["installed_baseline_unchanged"] = after == inputs.before
        report["candidate_accuracy_checks_passed"] = "error" not in report and all(report["checks"].values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if arrays:
            with args.output.with_suffix(".npz").open("xb") as output:
                np.savez(output, **arrays)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
        print(json.dumps(report, indent=2))
    return exit_code or int(not report["candidate_accuracy_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
