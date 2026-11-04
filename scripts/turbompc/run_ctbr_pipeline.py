"""Run the existing CTBR CLI and record execution provenance, not task acceptance."""

import argparse
import json
import os
import runpy
import subprocess
import sys
import sysconfig
import traceback
from pathlib import Path

from record_build import file_record, utc_now

ROOT = Path(__file__).resolve().parents[2]


def extensions():
    return {
        "geometry": ROOT / "src/isaaclab_ipc" / ("_geometry" + sysconfig.get_config_var("EXT_SUFFIX")),
        "native": Path(os.environ.get("TURBOMPC_NATIVE_BUILD_DIR", ROOT / "build/turbompc/native/cu130"))
        / "_turbompc_native.so",
    }


def snapshot(checkpoint=None):
    suffixes = {".py", ".json", ".yaml", ".yml", ".usd", ".usda", ".usdc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
    paths = {
        p
        for directory in ("src", "native", "third_party/pathcover")
        for p in (ROOT / directory).rglob("*")
        if p.is_file() and p.suffix in suffixes
    }
    paths.update(
        ROOT / name
        for name in (
            "scripts/acmpc.py",
            "scripts/turbompc/env.sh",
            "scripts/build_native.py",
            "scripts/build_turbompc.py",
            "pyproject.toml",
            "native/geometry.cpp",
            "build/sensors/mid360_raydirs.npy",
        )
    )
    paths.update((Path(__file__), Path(file_record.__code__.co_filename), *extensions().values()))
    paths.add(extensions()["native"].parent / "build-manifest.json")
    if checkpoint is not None:
        paths.add(checkpoint)
    return {str(p): {**file_record(p), "resolved_path": str(p.resolve())} for p in sorted(paths)}


def runtime_provenance(before):
    mapped = {
        line.split(maxsplit=5)[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6
    }
    paths = sorted(
        p
        for p in mapped
        if p.startswith("/")
        and any(
            token in Path(p).name for token in ("_geometry", "_turbompc_native", "libcudss.so", "libcudart.so", "jax")
        )
    )
    libraries, library_errors = {}, {}
    for path in paths:
        try:
            libraries[path] = file_record(path)
        except (OSError, RuntimeError) as error:
            library_errors[path] = repr(error)
    modules = {
        name: str(getattr(module, "__file__", ""))
        for name, module in list(sys.modules.items())
        if name == "torch" or name.startswith(("isaaclab_ipc", "_turbompc_native", "jax"))
    }
    jax_modules = [name for name in sys.modules if name == "jax" or name.startswith(("jax.", "jaxlib"))]
    project_modules = {
        name: p for name, p in modules.items() if name == "isaaclab_ipc" or name.startswith("isaaclab_ipc.")
    }
    torch = sys.modules.get("torch")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    expected = {str(p.resolve()): before.get(str(p), {}).get("sha256") for p in extensions().values()}
    expected_names = {p.name for p in extensions().values()}
    actual = {str(Path(p).resolve()): item["sha256"] for p, item in libraries.items() if Path(p).name in expected_names}
    extensions_match = bool(actual) and actual == expected
    cudss = [Path(p).resolve() for p in paths if Path(p).name.startswith("libcudss.so")]
    manifest = json.loads((extensions()["native"].parent / "build-manifest.json").read_text())
    built = {str(Path(p).resolve()): item["sha256"] for p, item in manifest["artifacts_after_call"].items()}
    checks = {
        "library_reads_succeeded": not library_errors,
        "two_extensions_match_before": extensions_match,
        "native_build_manifest": manifest.get("status") == "succeeded"
        and manifest.get("source_guard_passed", False)
        and built.get(str(extensions()["native"].resolve())) == expected[str(extensions()["native"].resolve())],
        "private_cudss_loaded": bool(cudss)
        and bool(os.environ.get("CUDSS_DIR"))
        and all(p.is_relative_to(Path(os.environ["CUDSS_DIR"]).resolve()) for p in cudss),
        "cuda13": cuda == "13.0"
        and os.environ.get("TURBOMPC_CUDA_PROFILE") == "cu130"
        and any(Path(p).name.startswith("libcudart.so.13") for p in paths)
        and not any(Path(p).name.startswith("libcudart.so.12") for p in paths),
        "no_jax_observed": not jax_modules and not any("jax" in p for p in paths),
        "project_module_paths": bool(project_modules)
        and all(
            Path(p).resolve().is_relative_to(ROOT / "src")
            or (
                name == "isaaclab_ipc._geometry"
                and extensions_match
                and Path(p).resolve() == extensions()["geometry"].resolve()
            )
            for name, p in project_modules.items()
        ),
    }
    return {
        "mapped_paths": paths,
        "libraries": libraries,
        "library_errors": library_errors,
        "modules": modules,
        "jax_modules": jax_modules,
        "torch": str(getattr(torch, "__version__", None)),
        "torch_cuda": cuda,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("mode", choices=("evaluate", "train"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path)
    target, _ = cli.parse_known_args(command)
    report_path, result_dir = args.report.resolve(), target.output.resolve()
    if sum(part == "--output" or part.startswith("--output=") for part in command) != 1:
        parser.error("Exactly one CLI --output is required")
    if report_path.is_relative_to(result_dir):
        parser.error("--report must be outside the new CLI --output directory")
    before_path = report_path.with_suffix(".before.json")
    harness_path = report_path.with_suffix(".harness.py")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Execution and provenance only; navigation/training acceptance is independent",
        shared_gpu=True,
        started_utc=utc_now(),
        command=[str(ROOT / "scripts/acmpc.py"), *command],
        executable=sys.executable,
        python=sys.version,
        mode=target.mode,
        result_directory=str(result_dir),
        before_report=str(before_path),
        harness=str(harness_path),
        sources_before={},
        exit_code=None,
        error=None,
        execution_provenance_passed=False,
        environment={
            name: os.environ.get(name)
            for name in (
                "TURBOMPC_CUDA_PROFILE",
                "TURBOMPC_NATIVE_BUILD_DIR",
                "CUDSS_DIR",
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
            )
        },
    )
    with report_path.open("x") as final:
        try:
            if result_dir.exists() or result_dir.is_symlink():
                raise FileExistsError(f"CLI output must be new: {result_dir}")
            with harness_path.open("xb") as handle:
                handle.write(Path(__file__).read_bytes())
            report["git_head"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
            checkpoint = target.checkpoint.resolve() if target.checkpoint is not None else None
            report["sources_before"] = snapshot(checkpoint)
            with before_path.open("x") as handle:
                json.dump(report, handle, indent=2)
            previous_argv, previous_path = sys.argv, sys.path[:]
            try:
                sys.argv = report["command"][:]
                sys.path.insert(0, str(ROOT / "src"))
                sys.path.insert(1, str(ROOT / "scripts"))
                runpy.run_path(str(ROOT / "scripts/acmpc.py"), run_name="__main__")
                report["exit_code"] = 0
            finally:
                sys.argv, sys.path[:] = previous_argv, previous_path
        except SystemExit as error:
            report["exit_code"] = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
            if report["exit_code"]:
                report.update(error=repr(error), traceback=traceback.format_exc())
        except BaseException as error:
            report.update(
                exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                error=repr(error),
                traceback=traceback.format_exc(),
            )
        finally:
            for name, collect in (
                ("sources_after", lambda: snapshot(target.checkpoint.resolve() if target.checkpoint else None)),
                ("runtime", lambda: runtime_provenance(report["sources_before"])),
                ("result_paths", lambda: sorted(str(p) for p in result_dir.rglob("*") if p.is_file())),
            ):
                try:
                    report[name] = collect()
                except BaseException as error:
                    report.setdefault("collection_errors", {})[name] = repr(error)
            report["sources_unchanged"] = bool(report["sources_before"]) and (
                report["sources_before"] == report.get("sources_after")
            )
            report["execution_completed"] = report["exit_code"] == 0 and report["error"] is None
            report["execution_provenance_passed"] = (
                report["execution_completed"]
                and report["sources_unchanged"]
                and report.get("runtime", {}).get("passed", False)
                and not report.get("collection_errors")
            )
            report["completed_utc"] = utc_now()
            json.dump(report, final, indent=2, allow_nan=False)
            final.write("\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "exit_code": report["exit_code"],
                "execution_provenance_passed": report["execution_provenance_passed"],
            }
        ),
        flush=True,
    )
    return 0 if report["execution_provenance_passed"] else (report["exit_code"] or 1)


if __name__ == "__main__":
    raise SystemExit(main())
