"""IR2 candidate comparison using the official interface and unchanged M4 checks.

This experiment never authorizes replacing original A and never passes M4.
Select the isolated candidate explicitly with command-local PYTHONPATH after
env.sh. --preflight-only validates provenance without importing JAX or Torch.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import check_acceptance as acceptance
import numpy as np
from record_build import pinned_snapshot

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = "candidate-only-not-authorized-original-A"
KKT_NAME = "libcudss_sparse_kkt_ffi.so"
KKT_SOURCE = "turbompc/solvers/backward/csrc/cudss_sparse_kkt.cu"
CANDIDATE_HASH = "1c194207f9f8930c9072e440b19646d4adfeaa238185c53865da5008d1af30dc"
ORIGINAL_HASH = "3adceabed469d2c2d074cc098acede55ebd9d6f1b4eca146096bfc528d112817"


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "build/turbompc/kkt-reference-fix/cu128/manifest.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "build/reference")
    parser.add_argument("--upstream-repo", type=Path, default=ROOT / "build/reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=list(acceptance.REQUIRED_BATCHES))
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.pairs, args.warmup, *args.batches) < 1 or len(set(args.batches)) != len(args.batches):
        parser.error("positive distinct batches, positive pairs and warmup are required")
    if args.output is None and not args.preflight_only:
        parser.error("--output is required for GPU experiments")
    if args.output is not None:
        if args.output.suffix != ".json":
            parser.error("--output must end in .json")
        args.output = args.output.absolute()
        for path in (args.output, args.output.with_suffix(".npz")):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Preserve existing result: {path}")
    args.manifest = args.manifest.resolve()
    args.reference = args.reference.resolve()
    args.upstream_repo = args.upstream_repo.resolve()
    return args


def preflight(args):
    """Check the exact candidate, untouched installed A, and guarded B on CPU."""
    if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
        raise RuntimeError("Run through bash scripts/turbompc/env.sh cu128")
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "jax", "jaxlib", "turbompc", "nvidia-cudss-cu12")
    }
    if versions["torch"] != "2.11.0+cu128" or any(versions[name] != "0.6.2" for name in ("jax", "jaxlib")):
        raise RuntimeError("Requires the pinned Torch 2.11.0+cu128 and JAX/jaxlib 0.6.2 profile")
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest["upstream_commit"] != acceptance.REVISION
        or manifest["profile"] != "cu128"
        or manifest["kkt_ir_steps"] != 2
        or manifest["library_sha256"] != CANDIDATE_HASH
    ):
        raise RuntimeError("Manifest is not the reviewed, fixed-hash IR2 candidate")
    source = args.manifest.parent / "source"
    package = source / "turbompc"
    candidate = (package / "solvers/_ffi" / KKT_NAME).resolve()
    installed = Path(importlib.metadata.distribution("turbompc").locate_file("turbompc")).resolve()
    original = installed / "solvers/_ffi" / KKT_NAME
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Installed A is outside the selected profile")
    if (
        candidate != Path(manifest["library"]).resolve()
        or original != Path(manifest["installed_baseline_kkt"]).resolve()
        or candidate == original
        or acceptance.sha256(candidate) != CANDIDATE_HASH
        or acceptance.sha256(original) != ORIGINAL_HASH
        or manifest["installed_baseline_kkt_sha256_before"] != ORIGINAL_HASH
        or manifest["installed_baseline_kkt_sha256_after"] != ORIGINAL_HASH
    ):
        raise RuntimeError("Candidate or preserved original KKT path/hash differs")
    search = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    spec = importlib.util.find_spec("turbompc")
    if not search or search[0] != source or spec is None or Path(spec.origin).resolve() != package / "__init__.py":
        raise RuntimeError("Set PYTHONPATH to the candidate source explicitly AFTER env.sh")

    snapshot = pinned_snapshot(source, args.upstream_repo, acceptance.REVISION)
    changed = [name for name, value in snapshot["files"].items() if not value["matches_pin"]]
    if changed != [KKT_SOURCE]:
        raise RuntimeError(f"Candidate differs from pinned sources beyond the KKT patch: {changed}")
    original_text = subprocess.check_output(
        ["git", "-C", str(args.upstream_repo), "show", f"{acceptance.REVISION}:{KKT_SOURCE}"], text=True
    )
    anchor = "        CUDSS_CHECK(cudssConfigCreate(&config));\n"
    addition = (
        "        int refinement_steps = 2;\n"
        "        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS,\n"
        "                                   &refinement_steps, sizeof(refinement_steps)));\n"
    )
    patched = original_text.replace(anchor, anchor + addition, 1)
    if original_text.count(anchor) != 1 or (source / KKT_SOURCE).read_text() != patched:
        raise RuntimeError("Candidate KKT source is not exactly the reviewed three-line IR2 insertion")
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{KKT_SOURCE}",
            tofile=f"b/{KKT_SOURCE}",
        )
    )
    if manifest["patch"] != diff or (args.manifest.parent / "kkt-ir2.patch").read_text() != diff:
        raise RuntimeError("Candidate patch record differs from the actual source diff")
    if acceptance.sha256(source / KKT_SOURCE) != manifest["candidate_kkt_source_sha256"]:
        raise RuntimeError("Candidate source hash differs from its build manifest")

    sources = {str(source / name): value["sha256"] for name, value in snapshot["files"].items()}
    for location in (package, installed):
        sources.update(acceptance.verify_original(args.reference, args.upstream_repo, location))
    for path in (
        Path(__file__).resolve(),
        Path(acceptance.__file__).resolve(),
        ROOT / "scripts/turbompc/record_build.py",
        ROOT / "scripts/turbompc/env.sh",
        args.manifest,
        args.manifest.parent / "kkt-ir2.patch",
        *sorted((ROOT / "src/isaaclab_ipc/turbompc").glob("*.py")),
    ):
        sources[str(path)] = acceptance.sha256(path)

    # The upstream loader is stdlib-only; importing this file does not load CUDA.
    loader = acceptance.load_example("turbompc.solvers._candidate_lookup", package / "solvers/ffi_loader.py")
    selected = {}
    for path in sorted((installed / "solvers/_ffi").glob("*.so")):
        actual = loader.find_ffi_lib(path.name).resolve()
        expected = candidate if path.name == KKT_NAME else path
        if actual != expected:
            raise RuntimeError(f"Unexpected FFI selection for {path.name}: {actual}")
        selected[path.name] = str(actual)
    if not {KKT_NAME, "libadmm_cudss_ffi.so"}.issubset(selected):
        raise RuntimeError("Required candidate KKT and installed ADMM FFI libraries are missing")
    builds = acceptance.verified_build_records()
    binaries = {
        str(Path(path).resolve()): digest
        for record in builds.values()
        for path, digest in record["artifact_sha256"].items()
    }
    binaries[str(candidate)] = CANDIDATE_HASH
    for name, digest in manifest.get("unchanged_eigen_copies", {}).items():
        path = Path(name).resolve()
        original_copy = installed / path.relative_to(package)
        if acceptance.sha256(path) != digest or acceptance.sha256(original_copy) != digest:
            raise RuntimeError("Optional candidate Eigen copy differs from the installed original")
        binaries[str(path)] = digest
    native_directory = (ROOT / "build/turbompc/native/cu128").resolve()
    if Path(os.environ.get("TURBOMPC_NATIVE_BUILD_DIR", "")).resolve() != native_directory:
        raise RuntimeError("Native build override must select this verified worktree's cu128 build")
    if not all(Path(path).resolve().is_relative_to(native_directory) for path in builds["B"]["artifact_sha256"]):
        raise RuntimeError("B build record points outside this verified worktree")
    return {
        "versions": versions,
        "upstream_revision": acceptance.REVISION,
        "candidate_package": str(package),
        "candidate_kkt": str(candidate),
        "installed_original_kkt": str(original),
        "candidate_tracked_files_checked": len(snapshot["files"]),
        "changed_tracked_files": changed,
        "exact_three_line_patch": diff,
        "selected_ffi": selected,
        "source_sha256": sources,
        "artifact_sha256": binaries,
        "build_records": builds,
    }


def loaded_provenance(before):
    paths = [Path(path).resolve() for path in acceptance.loaded_libraries()]
    solvers = [path for path in paths if path.name.endswith("_ffi.so") or "_turbompc_native" in path.name]
    hashes = {str(path): acceptance.sha256(path) for path in solvers}
    ffi = [path for path in solvers if path.name.endswith("_ffi.so")]
    native = [path for path in solvers if "_turbompc_native" in path.name]
    cuda = [
        path
        for path in paths
        if not path.name.endswith("_ffi.so")
        and path.name.startswith(("libcudss", "libcudart", "libcublas", "libcusparse", "libnvrtc", "libnvJitLink"))
    ]
    checks = {
        "required_candidate_and_admm_loaded": {KKT_NAME, "libadmm_cudss_ffi.so"}.issubset(p.name for p in ffi),
        "ffi_paths_match_explicit_selection": all(before["selected_ffi"].get(p.name) == str(p) for p in ffi),
        "one_native_from_guarded_build": len(native) == 1
        and str(native[0]) in before["build_records"]["B"]["artifact_sha256"],
        "loaded_solver_hashes_match_records": bool(hashes)
        and all(before["artifact_sha256"].get(path) == digest for path, digest in hashes.items()),
        "original_kkt_not_loaded": Path(before["installed_original_kkt"]) not in solvers,
        "cuda12_profile_libraries": bool(cuda)
        and all(
            path.is_relative_to(Path(sys.prefix).resolve())
            or path.is_relative_to(Path(os.environ["CUDA_HOME"]).resolve())
            for path in cuda
        )
        and not any("libcudart.so.13" in str(path) for path in paths),
    }
    return {
        "paths": [str(path) for path in paths],
        "solver_sha256": hashes,
        "checks": checks,
        "passed": all(checks.values()),
    }


def finish_gates(args, report):
    """Candidate success is independent of authorization to replace original A."""
    entries = report["batches"]
    gates = {
        "gpu_experiment": not args.preflight_only,
        "all_required_batches": sorted(args.batches) == list(acceptance.REQUIRED_BATCHES)
        and sorted(entry["batch"] for entry in entries) == list(acceptance.REQUIRED_BATCHES),
        "all_batch_gates_passed": bool(entries) and all(entry.get("passed", False) for entry in entries),
        "at_least_100_pairs_per_batch": bool(entries) and all(len(entry.get("pairs", [])) >= 100 for entry in entries),
        "at_least_10_warmups_per_backend": bool(entries)
        and all(len(entry.get("warmup_full_ms", {}).get(label, [])) >= 10 for entry in entries for label in ("A", "B")),
        "sources_and_builds_unchanged": bool(report.get("provenance_before"))
        and report.get("provenance_before") == report.get("provenance_after"),
        "loaded_library_provenance": report.get("loaded_libraries", {}).get("passed", False),
        "original_kkt_preserved": report.get("original_kkt_sha256_before") == ORIGINAL_HASH
        and report.get("original_kkt_sha256_after") == ORIGINAL_HASH,
        "gpu_load_recorded": any(
            sample.get("gpus", {}).get("returncode") == 0 for sample in report["gpu_load_samples"]
        ),
        "no_driver_error": "error" not in report and "verification_error" not in report,
    }
    report["candidate_gates"] = gates
    report["candidate_suite_passed"] = all(gates.values())
    report["qualification"] = QUALIFICATION
    report["m4_gate_passed"] = False


def save_report(output, report, artifacts):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, **artifacts)
    with output.open("x") as handle:
        json.dump(acceptance.json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    args = options()
    report = {
        "qualification": QUALIFICATION,
        "scope": "IR2 candidate official MPCBackend + MPC.mean_only vs native TorchMPC; decision evidence only",
        "roles": {"A": "explicit IR2 candidate; NOT original A", "B": "native TorchMPC"},
        "time_utc": datetime.now(UTC).isoformat(),
        "mode": "cpu-preflight" if args.preflight_only else "gpu-candidate-experiment",
        "python": sys.version,
        "executable": sys.executable,
        "horizon": acceptance.N,
        "nx": acceptance.NX,
        "nu": acceptance.NU,
        "dt": acceptance.DT,
        "dtype": "float64",
        "thresholds": acceptance.THRESHOLDS,
        "solver_contract": acceptance.SOLVER,
        "requested_batches": args.batches,
        "requested_pairs": args.pairs,
        "requested_warmup": args.warmup,
        "compile": args.compile,
        "case_definition": "Unmodified check_acceptance.case_bank and Baseline/Native adapters",
        "loss": "sum over rows of 0.5 * ||u0||^2; no batch averaging",
        "timing_contract": (
            "same run_batch: device-synchronized full forward+backward, alternating AB/BA, no row/outlier removal"
        ),
        "shared_gpu": True,
        "gpu_load_columns": {
            "gpus": (
                "uuid,name,driver_version,memory.used MiB,memory.total MiB,utilization.gpu %,utilization.memory %,"
                "power.draw W,temperature.gpu C,clocks.sm MHz,pstate"
            ),
            "compute_processes": "gpu_uuid,pid,process_name,used_gpu_memory MiB",
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "TURBOMPC_CUDA_PROFILE",
                "TURBOMPC_NATIVE_BUILD_DIR",
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "MAX_JOBS",
                "TORCHINDUCTOR_COMPILE_THREADS",
            )
        },
        "arrays_file": str(args.output.with_suffix(".npz")) if args.output else None,
        "batches": [],
        "candidate_preflight_passed": False,
        "candidate_suite_passed": False,
        "m4_gate_passed": False,
    }
    artifacts, monitor = {}, None
    try:
        report["provenance_before"] = preflight(args)
        original = report["provenance_before"]["installed_original_kkt"]
        report["original_kkt_sha256_before"] = acceptance.sha256(original)
        report["candidate_preflight_passed"] = True
        if not args.preflight_only:
            monitor = acceptance.GPULoad()
            monitor.start()
            start = time.perf_counter()
            sys.path.insert(0, str(ROOT / "src"))
            import jax
            import torch
            import turbompc

            jax.config.update("jax_enable_x64", True)
            if (
                torch.version.cuda != "12.8"
                or not torch.cuda.is_available()
                or any(device.platform not in {"gpu", "cuda"} for device in jax.devices())
            ):
                raise RuntimeError("Both backends must use the cu128 GPU profile")
            if Path(turbompc.__file__).resolve().parent != Path(report["provenance_before"]["candidate_package"]):
                raise RuntimeError("Runtime imported a different TurboMPC package")
            report["gpu"] = {
                "torch": torch.cuda.get_device_name(0),
                "torch_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                "jax": [str(device) for device in jax.devices()],
            }
            report["shared_startup_seconds"] = time.perf_counter() - start
            for batch in args.batches:
                acceptance.run_batch(args, batch, report, artifacts)
            report["loaded_libraries"] = loaded_provenance(report["provenance_before"])
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if monitor is not None:
            monitor.close()
        report["gpu_load_samples"] = monitor.samples if monitor is not None else []
        if "provenance_before" in report:
            try:
                report["original_kkt_sha256_after"] = acceptance.sha256(
                    report["provenance_before"]["installed_original_kkt"]
                )
                report["provenance_after"] = preflight(args)
            except Exception as error:
                report["verification_error"] = f"{type(error).__name__}: {error}"
        finish_gates(args, report)
        report["completed_utc"] = datetime.now(UTC).isoformat()
        if args.output is not None:
            save_report(args.output, report, artifacts)
        print(
            json.dumps(
                {
                    "output": str(args.output) if args.output else None,
                    "qualification": report["qualification"],
                    "candidate_preflight_passed": report["candidate_preflight_passed"],
                    "candidate_suite_passed": report["candidate_suite_passed"],
                    "m4_gate_passed": report["m4_gate_passed"],
                    "source_and_build_guard_passed": report["candidate_gates"]["sources_and_builds_unchanged"],
                    "original_kkt_preserved": report["candidate_gates"]["original_kkt_preserved"],
                    "error": report.get("error", report.get("verification_error")),
                },
                indent=2,
            ),
            flush=True,
        )
    if args.preflight_only:
        return (
            0
            if all(
                report["candidate_gates"][key]
                for key in ("sources_and_builds_unchanged", "original_kkt_preserved", "no_driver_error")
            )
            else 1
        )
    return 0 if report["candidate_suite_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
