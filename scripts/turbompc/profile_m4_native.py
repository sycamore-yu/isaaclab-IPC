"""Bounded B16 diagnosis: unchanged current/old native cores and CPU/CUDA trace.

This is neither an A/B acceptance run nor a replacement baseline. It consumes
the failed final M4 inputs, preserves every diagnostic pair, and never edits
the current or historical solver sources.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from functools import wraps
from pathlib import Path

import check_acceptance as acceptance
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(acceptance.json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def old_package(directory, historical, revision):
    revision = subprocess.check_output(["git", "rev-parse", revision], text=True).strip()
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    sources = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/isaaclab_ipc/turbompc"], text=True
    ).splitlines()
    for suffix in sources:
        if Path(suffix).suffix != ".py" or Path(suffix).parent != Path("src/isaaclab_ipc/turbompc"):
            continue
        name = Path(suffix).name
        contents = subprocess.check_output(["git", "show", revision + ":" + suffix])
        expected = [value for path, value in historical.items() if path.endswith("/" + suffix)]
        assert expected == [hashlib.sha256(contents).hexdigest()], (name, expected)
        target = directory / name
        with target.open("xb") as handle:
            handle.write(contents)
        hashes[str(target)] = sha(target)
    spec = importlib.util.spec_from_file_location(
        "turbompc_profile_" + revision[:12], directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, {"revision": revision, "source_sha256": hashes}


def old_runtime(current, module):
    # Native.forward only uses its explicit MPCInputs/solver attributes, so the
    # same official benchmark adapter can call a separately imported old core.
    runtime = copy.copy(current)
    runtime.MPCInputs = module.MPCInputs
    dynamics = module.LinearDynamics(
        current.solver.dynamics.matrix, nx=acceptance.NX, offset=current.solver.dynamics.offset
    )
    config = asdict(current.solver.config)
    assert config["require_qp_convergence"] is False
    old_fields = {field.name for field in fields(module.SQPConfig)}
    if "require_qp_convergence" not in old_fields:
        config.pop("require_qp_convergence")
    assert config.keys() == old_fields
    runtime.solver = module.TorchMPC(
        dynamics, module.SQPConfig(**config), module.ADMMConfig(**asdict(current.solver.admm.config))
    )
    if hasattr(current, "_assemble"):
        import torch

        # A copied compiled bound method would still close over current's
        # MPCInputs type. Bind the class function to this historical runtime.
        runtime._assemble = type(runtime)._assemble.__get__(runtime, type(runtime))
        if runtime.solver.config.compile_kernels:
            runtime._assemble = torch.compile(runtime._assemble, fullgraph=True, dynamic=False)
    return runtime


def old_adapter(directory, historical, revision):
    """Load the exact historical adapter while requiring one unchanged core."""
    revision = subprocess.check_output(["git", "rev-parse", revision], text=True).strip()
    relative = "scripts/turbompc/check_acceptance.py"
    contents = subprocess.check_output(["git", "show", revision + ":" + relative])
    digest = hashlib.sha256(contents).hexdigest()
    assert [value for path, value in historical.items() if path.endswith("/" + relative)] == [digest]
    current_path = ROOT / relative

    def outside_native(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ClassDef) and node.name == "Native")]
        return ast.dump(tree, include_attributes=False)

    assert outside_native(contents) == outside_native(current_path.read_bytes()), "Non-Native harness AST changed"
    sources = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/isaaclab_ipc/turbompc"], text=True
    ).splitlines()
    shared_core = {}
    for source in sources:
        if Path(source).suffix != ".py" or Path(source).parent != Path("src/isaaclab_ipc/turbompc"):
            continue
        pinned = subprocess.check_output(["git", "show", revision + ":" + source])
        shared_core[str(ROOT / source)] = hashlib.sha256(pinned).hexdigest()
    assert shared_core == {str(path): sha(path) for path in (ROOT / "src/isaaclab_ipc/turbompc").glob("*.py")}
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "check_acceptance.py"
    with target.open("xb") as handle:
        handle.write(contents)
    module = acceptance.load_example("turbompc_profile_adapter_" + revision[:12], target)
    assert not hasattr(module.Native, "_assemble") and hasattr(acceptance.Native, "_assemble")
    return module, {
        "revision": revision,
        "source_sha256": {str(target): digest},
        "shared_core_sha256": shared_core,
        "non_native_ast_unchanged": True,
        "old_native_loaded_from_exact_git_source": True,
        "current_native_source": str(current_path),
        "current_native_file_sha256": sha(current_path),
    }


def measured(runtime):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = acceptance.full(runtime)
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


@contextmanager
def instrument(runtime):
    """Ranges only; restore every temporary method after profiling."""
    from torch.profiler import record_function

    import isaaclab_ipc.turbompc.native as native
    import isaaclab_ipc.turbompc.solver as solver_module

    restored = []

    def mark(owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            with record_function("phase/" + label):
                return original(*args, **kwargs)

        restored.append((owner, name, original))
        setattr(owner, name, wrapped)

    class WorkspaceRange:
        def __init__(self, value):
            self.value = value

        def solve(self, *args):
            with record_function("phase/admm_native_binding_including_metadata_checks"):
                return self.value.solve(*args)

        def close(self):
            self.value.close()

    solver = runtime.solver
    workspaces = solver.admm._workspaces.copy()
    try:
        mark(runtime, "forward", "forward_adapter_and_sqp")
        for name, label in (
            ("solve", "sqp_total"),
            ("_qp", "qp_assembly"),
            ("_errors", "nonlinear_errors"),
            ("_line_search", "merit_line_search"),
            ("_dynamics_hessian", "backward_dynamics_hessian"),
            ("_active_sets", "backward_active_sets"),
            ("_parameter_gradients", "backward_parameter_gradients"),
        ):
            mark(solver, name, label)
        mark(native.QPData, "contiguous", "qp_contiguous_inputs")
        mark(solver_module, "_choose_admm", "warm_state_selection")
        mark(solver.admm, "solve", "admm_total")
        mark(solver.admm, "_schur", "admm_schur_assembly")
        for stream, workspace in workspaces.items():
            solver.admm._workspaces[stream] = WorkspaceRange(workspace)
        for adjoint in solver._adjoints.values():
            mark(adjoint, "solve", "implicit_kkt_total")
            mark(adjoint, "_assemble", "implicit_kkt_csr_assembly")
            mark(adjoint.solver, "solve", "implicit_kkt_native_binding")
        yield
    finally:
        solver.admm._workspaces = workspaces
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)


def check_batch_cache(current, constants, input_report, arrays):
    """Exercise four shapes after timing; this is not an acceptance suite."""
    import torch

    limit = torch._dynamo.config.recompile_limit
    assert limit == 8
    result = {
        "role": "Same-process compiled-cache smoke only; not M4 acceptance",
        "recompile_limit_before": limit,
        "batches": [],
    }
    with np.load(input_report.with_suffix(".npz"), allow_pickle=False) as saved:
        cases = {
            batch: {key: saved[f"b{batch}_input_{key}"].copy() for key in ("state", "costs", "seed")}
            for batch in (1, 16, 64, 256)
        }
    for batch, case in cases.items():
        assert all(np.array_equal(value, acceptance.case_bank(batch)[key]) for key, value in case.items())
        arrays.update({f"cache_b{batch}_input_{key}": value for key, value in case.items()})
        entry = {"batch": batch, "snapshots": {}, "passed": False}
        result["batches"].append(entry)
        runtime = None
        try:
            runtime = current if batch == 16 else acceptance.Native(case, constants, compile_kernels=True)
            snapshots = {}
            for label in ("learnable_before", "detached", "learnable_after"):
                if label == "detached":
                    mean, solution, callbacks = runtime.forward(runtime.costs.detach(), runtime.state.detach())
                    value = (mean, solution, None, callbacks)
                else:
                    value = acceptance.full(runtime)
                snapshot = runtime.snapshot(value)
                snapshots[label] = snapshot
                arrays.update({f"cache_b{batch}_{label}_{key}": tensor for key, tensor in snapshot.items()})
                residual = acceptance.physical(snapshot, case, constants)
                entry["snapshots"][label] = {
                    "finite": bool(all(np.isfinite(tensor).all() for tensor in snapshot.values())),
                    "solved": bool(
                        np.all(snapshot["status"] == 0)
                        and np.all(snapshot["valid"])
                        and np.max(snapshot["convergence"]) <= acceptance.SOLVER["tol_convergence"]
                    ),
                    "physical_maximum": residual["maximum"],
                }
            first = snapshots["learnable_before"]
            entry["same_forward_max_abs"] = max(
                float(np.max(np.abs(first[key] - snapshot[key])))
                for label, snapshot in snapshots.items()
                if label != "learnable_before"
                for key in ("states", "controls", "u0")
            )
            entry["exact_status_and_iterations"] = all(
                np.array_equal(first[key], snapshot[key])
                for snapshot in snapshots.values()
                for key in ("status", "valid", "iterations", "admm_status", "admm_iterations")
            )
            entry["repeated_full_comparison"] = acceptance.compare(first, snapshots["learnable_after"], case, constants)
            entry["passed"] = bool(
                all(
                    value["finite"] and value["solved"] and value["physical_maximum"] <= 1e-5
                    for value in entry["snapshots"].values()
                )
                and entry["same_forward_max_abs"] <= 1e-10
                and entry["exact_status_and_iterations"]
                and entry["repeated_full_comparison"]["passed"]
            )
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            if runtime is not None and runtime is not current:
                runtime.close()
        print("cache smoke", batch, "passed", entry["passed"], "error", entry.get("error"), flush=True)
    result["recompile_limit_after"] = torch._dynamo.config.recompile_limit
    result["passed"] = all(entry["passed"] for entry in result["batches"]) and result["recompile_limit_after"] == limit
    return result


def event_summary(event):
    return {
        "key": event.key,
        "count": event.count,
        "cpu_total_ms": event.cpu_time_total / 1000,
        "self_cpu_ms": event.self_cpu_time_total / 1000,
        "device_total_ms": event.device_time_total / 1000,
        "self_device_ms": event.self_device_time_total / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument(
        "--input-report",
        type=Path,
        default=ROOT / "results/turbompc-port/candidate-ir2-official-linear-cu128-final-v1.json",
    )
    parser.add_argument("--baseline-revision", default="a771284")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "results/turbompc-port/candidate-ir2-official-linear-cu128-v1.json",
    )
    parser.add_argument(
        "--compare-only", action="store_true", help="Skip profiler; compare the old and current core only"
    )
    parser.add_argument(
        "--compare-adapter",
        action="store_true",
        help="With --compare-only, load the exact old Native adapter and require the same pinned core",
    )
    parser.add_argument(
        "--check-batch-cache", action="store_true", help="After timing, exercise four shapes and FD metadata"
    )
    args = parser.parse_args()
    if args.compare_adapter and not args.compare_only:
        parser.error("--compare-adapter requires --compare-only")
    assert args.pairs == args.warmup == 10 and args.profile_steps == 5
    args.output.mkdir(parents=True, exist_ok=False)
    failed_path = args.input_report.resolve()
    historical_path = args.baseline_report.resolve()
    failed = json.loads(failed_path.read_text())
    historical = json.loads(historical_path.read_text())
    expected = failed["provenance_before"]["source_sha256"]
    source_before = {path: sha(path) for path in expected}
    source_before[str(Path(__file__).resolve())] = sha(__file__)
    source_changes = {
        path: {"before": expected[path], "current": value}
        for path, value in source_before.items()
        if path in expected and value != expected[path]
    }
    if args.compare_adapter:
        assert set(source_changes) == {str(ROOT / "scripts/turbompc/check_acceptance.py")}
    elif args.compare_only:
        # Allow the deliberately optimized core while keeping the benchmark,
        # official reference and build sources pinned to the failed experiment.
        assert all(Path(path).parent == ROOT / "src/isaaclab_ipc/turbompc" for path in source_changes)
    else:
        assert not source_changes
    artifacts_before = {path: sha(path) for path in failed["provenance_before"]["artifact_sha256"]}
    assert artifacts_before == failed["provenance_before"]["artifact_sha256"]
    with np.load(failed_path.with_suffix(".npz"), allow_pickle=False) as archive:
        case = {key: archive[f"b16_input_{key}"].copy() for key in ("state", "costs", "seed")}
    assert all(np.array_equal(value, acceptance.case_bank(16)[key]) for key, value in case.items())
    constants = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in yaml.safe_load(
            (Path(failed["provenance_before"]["candidate_package"]) / "problems/params/linear.yaml").read_text()
        )["dynamics_state_dot_params"].items()
    }
    snapshot_loader = old_adapter if args.compare_adapter else old_package
    module, old_provenance = snapshot_loader(
        ROOT / "build/turbompc/profile-snapshots" / args.output.name,
        historical["provenance_before"]["source_sha256"],
        args.baseline_revision,
    )
    import torch

    assert "jax" not in sys.modules
    assert torch.version.cuda.startswith("12.8")
    if not args.compare_only:
        assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    current = acceptance.Native(case, constants, compile_kernels=True)
    if args.compare_adapter:
        old = module.Native(case, constants, compile_kernels=True)
        assert type(old.solver) is type(current.solver)
        assert asdict(old.solver.config) == asdict(current.solver.config)
        assert asdict(old.solver.admm.config) == asdict(current.solver.admm.config)
    else:
        old = old_runtime(current, module)
    runtimes = {"old": old, "current": current}
    report = {
        "role": "B16 performance diagnosis only; not M4 acceptance and not a new A baseline",
        "comparison_target": "Native adapter, shared pinned core" if args.compare_adapter else "native core",
        "failed_report": {"path": str(failed_path), "sha256": sha(failed_path)},
        "historical_report": {"path": str(historical_path), "sha256": sha(historical_path)},
        "input_sha256": acceptance.array_hash(case),
        "old": old_provenance,
        "current_source_sha256_before": source_before,
        "intentional_core_changes_from_failed_run": {} if args.compare_adapter else source_changes,
        "intentional_adapter_changes_from_failed_run": source_changes if args.compare_adapter else {},
        "artifact_sha256_before": artifacts_before,
        "current_core_sha256": {
            path: value for path, value in source_before.items() if "/src/isaaclab_ipc/turbompc/" in path
        },
        "current_sqp": asdict(current.solver.config),
        "admm": asdict(current.solver.admm.config),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "warmups": {},
        "pairs": [],
        "profile_steps": 0 if args.compare_only else args.profile_steps,
    }
    monitor = acceptance.GPULoad()
    monitor.start()
    arrays = {"input_" + key: value for key, value in case.items()}
    try:
        for label, runtime in runtimes.items():
            report["warmups"][label] = []
            for _ in range(args.warmup):
                _, elapsed = measured(runtime)
                report["warmups"][label].append(elapsed)
            print(label, "10 full warmups complete", flush=True)
        for index in range(args.pairs):
            order = ("old", "current") if index % 2 == 0 else ("current", "old")
            pair = {"index": index, "order": list(order), "start_time_ns": time.time_ns()}
            snapshots = {}
            for label in order:
                result, elapsed = measured(runtimes[label])
                pair[label + "_ms"] = elapsed
                snapshots[label] = runtimes[label].snapshot(result)
            comparison = acceptance.compare(snapshots["old"], snapshots["current"], case, constants)
            pair.update(
                end_time_ns=time.time_ns(),
                comparison_passed=comparison["passed"],
                control_max_abs=comparison["u0_max_abs"],
                cost_gradient_relative=comparison["cost_gradient"]["maximum_relative"],
                state_gradient_relative=comparison["state_gradient"]["maximum_relative"],
                trajectory_max_abs=comparison["full_trajectory_max_abs"],
                exact_status_and_iterations={
                    key: bool(np.array_equal(snapshots["old"][key], snapshots["current"][key]))
                    for key in ("status", "valid", "iterations", "admm_status", "admm_iterations")
                },
            )
            report["pairs"].append(pair)
            for label, snapshot in snapshots.items():
                arrays.update({f"pair_{index:02d}_{label}_{key}": value for key, value in snapshot.items()})
                arrays.update({label + "_" + key: value for key, value in snapshot.items()})
        report["paired_timing"] = {
            label: {
                "p50_ms": float(np.median([pair[label + "_ms"] for pair in report["pairs"]])),
                "p95_ms": float(np.percentile([pair[label + "_ms"] for pair in report["pairs"]], 95)),
            }
            for label in runtimes
        }
        report["current_over_old_timing"] = {
            key: report["paired_timing"]["current"][key] / report["paired_timing"]["old"][key]
            for key in ("p50_ms", "p95_ms")
        }
        print("paired diagnosis", report["paired_timing"], flush=True)
        if not args.compare_only:
            run_profile(current, args, report)
        report["diagnostic_pairs_numerically_equal"] = all(pair["comparison_passed"] for pair in report["pairs"])
        report["diagnostic_status_and_iterations_equal"] = all(
            all(pair["exact_status_and_iterations"].values()) for pair in report["pairs"]
        )
        assert report["diagnostic_pairs_numerically_equal"] and report["diagnostic_status_and_iterations_equal"]
        if args.check_batch_cache:
            report["batch_cache_smoke"] = check_batch_cache(current, constants, failed_path, arrays)
            assert report["batch_cache_smoke"]["passed"], "Same-process batch cache smoke failed"
    except Exception as error:
        import traceback

        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        for runtime in runtimes.values():
            runtime.close()
        monitor.close()
        report["gpu_load_samples"] = monitor.samples
        report["current_source_sha256_after"] = {path: sha(path) for path in source_before}
        report["source_guard_passed"] = source_before == report["current_source_sha256_after"]
        report["old_source_guard_passed"] = all(
            sha(path) == value for path, value in old_provenance["source_sha256"].items()
        )
        report["artifact_sha256_after"] = {path: sha(path) for path in artifacts_before}
        report["artifact_guard_passed"] = artifacts_before == report["artifact_sha256_after"]
        report["native_libraries"] = acceptance.loaded_libraries()
        report["producer_sha256"] = sha(__file__)
        with (args.output / "producer.py").open("xb") as handle:
            handle.write(Path(__file__).read_bytes())
        with (args.output / "arrays.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        report["artifact_sha256"] = {path.name: sha(path) for path in args.output.iterdir() if path.is_file()}
        write_json(args.output / "summary.json", report)
    assert report["source_guard_passed"] and report["old_source_guard_passed"] and report["artifact_guard_passed"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_timing": report["paired_timing"],
                "current_over_old_timing": report["current_over_old_timing"],
                "source_guard_passed": report["source_guard_passed"],
                "status_and_iterations_equal": report["diagnostic_status_and_iterations_equal"],
                "batch_cache_smoke_passed": report.get("batch_cache_smoke", {}).get("passed"),
                "profile_cuda_events": report.get("profile_cuda_events"),
            },
            indent=2,
        ),
        flush=True,
    )


def run_profile(current, args, report):
    import torch

    with instrument(current):
        # Wrappers do not alter numerical work. Warm again before tracing
        # so any instrumentation-sensitive compile guard is paid upfront.
        report["instrumented_warmup_ms"] = [measured(current)[1] for _ in range(3)]
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            elapsed = []
            for _ in range(args.profile_steps):
                with torch.profiler.record_function("phase/full_forward_backward_and_final_sync"):
                    _, duration = measured(current)
                elapsed.append(duration)
                profiler.step()
        report["profile_wall_ms"] = elapsed
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    events = [event_summary(event) for event in profiler.key_averages()]
    write_json(args.output / "operator-summary.json", events)
    report["phase_totals_over_five_steps"] = [row for row in events if row["key"].startswith("phase/")]
    report["cpu_top_self"] = sorted(events, key=lambda row: row["self_cpu_ms"], reverse=True)[:25]
    report["cuda_top_self"] = sorted(events, key=lambda row: row["self_device_ms"], reverse=True)[:20]
    report["synchronization_and_host_scalar_reads"] = [
        row
        for row in events
        if any(token in row["key"] for token in ("Synchronize", "aten::item", "_local_scalar_dense"))
    ]
    report["profile_notes"] = [
        "Inclusive nested ranges must not be added together; self CPU times exclude child events.",
        "Profiler overhead is excluded from old/current paired timing; its durations diagnose attribution only.",
        "ADMM binding includes metadata shape/type/device checks, output allocation and native launch; no rebuilt inner ranges.",
        "C++ metadata validation has no finite-value GPU scan; its isolated cost cannot be inferred from this trace.",
    ]
    report["profile_cuda_events"] = sum(
        event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events()
    )
    assert report["profile_cuda_events"] > 0


if __name__ == "__main__":
    main()
