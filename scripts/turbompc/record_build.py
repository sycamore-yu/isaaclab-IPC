"""Record observed inputs/artifacts of existing incremental TurboMPC builds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import stat
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

CODE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh", ".inl", ".ipp", ".tpp"}
ENVIRONMENT_KEYS = (
    "VIRTUAL_ENV",
    "TURBOMPC_CUDA_PROFILE",
    "CUDA_HOME",
    "CUDACXX",
    "CUDSS_DIR",
    "CXX",
    "CC",
    "PYTORCH_NVCC",
    "NVCC_CCBIN",
    "TORCH_CUDA_ARCH_LIST",
    "MAX_JOBS",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_ARGS",
    "TURBOMPC_CUDA",
    "TURBOMPC_EIGEN",
)


def utc_now():
    return datetime.now(UTC).isoformat()


def identity(info):
    # atime is excluded: compiler reads must not trip the guard.
    return {
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def file_record(path, *, git_format=None, git_mode=None):
    """Hash one stable read; Git symlink blobs contain the link target bytes."""
    path = Path(path)
    link_before = path.lstat()
    is_link = stat.S_ISLNK(link_before.st_mode)
    if git_mode is not None:
        expected_link = git_mode == "120000"
        if is_link != expected_link or (not is_link and not stat.S_ISREG(link_before.st_mode)):
            raise RuntimeError(f"Git file type differs: {path}")
        if not is_link and bool(link_before.st_mode & 0o111) != (git_mode == "100755"):
            raise RuntimeError(f"Git executable mode differs: {path}")
    symlink_blob = git_mode == "120000"
    before = link_before if symlink_blob else path.stat()
    digest = hashlib.sha256()
    blob = hashlib.new(git_format) if git_format else None
    if blob is not None:
        blob.update(f"blob {before.st_size}\0".encode())
    if symlink_blob:
        chunk = os.fsencode(os.readlink(path))
        digest.update(chunk)
        blob.update(chunk)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                if blob is not None:
                    blob.update(chunk)
    after = path.lstat() if symlink_blob else path.stat()
    if identity(before) != identity(after) or identity(link_before) != identity(path.lstat()):
        raise RuntimeError(f"File changed while hashing: {path}")
    result = {**identity(after), "sha256": digest.hexdigest()}
    if is_link:
        result.update(symlink_target=os.readlink(path), symlink_identity=identity(link_before))
    if blob is not None:
        result["git_blob"] = blob.hexdigest()
    return result


def native_snapshot(root, drivers):
    root = Path(root).resolve()
    files = {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
    }
    files.update({f"build-driver:{Path(path).resolve()}": Path(path) for path in drivers})
    records = {}
    for name, path in files.items():
        try:
            records[name] = file_record(path)
        except Exception as error:
            records[name] = {"error": f"{type(error).__name__}: {error}"}
    return {
        "root": str(root),
        "files": records,
        "valid": bool(files) and all("error" not in value for value in records.values()),
    }


def pinned_snapshot(root, repository, revision):
    root, repository = Path(root).resolve(), Path(repository).resolve()
    command = ["git", "-C", str(repository)]
    resolved = subprocess.check_output([*command, "rev-parse", "--verify", f"{revision}^{{commit}}"], text=True).strip()
    if resolved != revision:
        raise RuntimeError(f"Expected full pinned commit {revision}, got {resolved}")
    git_format = subprocess.check_output([*command, "rev-parse", "--show-object-format"], text=True).strip()
    tree = subprocess.check_output([*command, "ls-tree", "-r", "-z", revision])
    records, gitlinks = {}, {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.decode("ascii").split()
        relative = os.fsdecode(name)
        if kind != "blob":
            gitlinks[relative] = {"mode": mode, "type": kind, "object": expected}
            continue
        try:
            value = file_record(root / relative, git_format=git_format, git_mode=mode)
            value.update(expected_git_blob=expected, expected_git_mode=mode, matches_pin=value["git_blob"] == expected)
        except Exception as error:
            value = {
                "expected_git_blob": expected,
                "expected_git_mode": mode,
                "matches_pin": False,
                "error": f"{type(error).__name__}: {error}",
            }
        records[relative] = value
    return {
        "root": str(root),
        "repository": str(repository),
        "revision": revision,
        "git_object_format": git_format,
        "files": records,
        "gitlinks": gitlinks,
        "untracked_products": "ignored; only pinned tracked blobs are verified",
        "valid": bool(records) and all(value["matches_pin"] for value in records.values()),
    }


def command_version(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as error:
        return {"command": command, "error": f"{type(error).__name__}: {error}"}


def new_manifest(kind):
    started = datetime.now(UTC)
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower() in {
            "torch",
            "jax",
            "jaxlib",
            "turbompc",
            "scikit-build-core",
            "pybind11",
            "ninja",
        } or name.lower().startswith(("nvidia-", "cuda-")):
            packages[name] = distribution.version
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = os.environ.get("PYTORCH_NVCC") if kind == "native" else os.environ.get("CUDACXX")
    return {
        "schema_version": 1,
        "kind": kind,
        "status": "started",
        "build_id": f"{started:%Y%m%dT%H%M%S.%fZ}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        "started_utc": started.isoformat(),
        "incremental": True,
        "provenance_scope": "Observed inputs and artifacts of this incremental call; cached outputs may be reused. This is not a hermetic build attestation.",
        "python": sys.version,
        "executable": sys.executable,
        "invocation": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "packages_before": packages,
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "compiler_versions": {
            "cxx": command_version([*shlex.split(os.environ.get("CXX", "c++")), "--version"]),
            "nvcc": command_version([*shlex.split(nvcc or str(cuda / "bin/nvcc")), "--version"]),
            "cmake": command_version(["cmake", "--version"]),
            "ninja": command_version(["ninja", "--version"]),
        },
        "errors": [],
    }


def build_files_snapshot(root, *, recursive=False):
    root = Path(root)
    records = {}
    for name in ("build.ninja", ".ninja_log", "CMakeCache.txt"):
        paths = root.rglob(name) if recursive and root.exists() else [root / name]
        for path in sorted(paths):
            if not path.is_file():
                continue
            try:
                value = file_record(path)
                lines = path.read_text(errors="replace").splitlines()
                if name == ".ninja_log":
                    value.update(line_count=len(lines), last_entries=lines[-20:])
                elif name == "build.ninja":
                    value["compiler_and_flags"] = [
                        line
                        for line in lines
                        if line.startswith(
                            (
                                "cxx =",
                                "nvcc =",
                                "cflags =",
                                "post_cflags =",
                                "cuda_cflags =",
                                "cuda_post_cflags =",
                                "ldflags =",
                                "  command =",
                            )
                        )
                    ]
                records[str(path.resolve())] = value
            except Exception as error:
                records[str(path)] = {"error": f"{type(error).__name__}: {error}"}
    return records


def finalize(directory, report, after, artifact_paths, build_files_after):
    report["sources_after"] = after
    before_files = report.get("sources_before", {}).get("files", {})
    after_files = after.get("files", {})
    report["changed_inputs"] = sorted(
        key for key in before_files.keys() | after_files.keys() if before_files.get(key) != after_files.get(key)
    )
    report["source_guard_passed"] = bool(
        report.get("sources_before", {}).get("valid") and after.get("valid") and not report["changed_inputs"]
    )
    if not report["source_guard_passed"]:
        report["errors"].append("Source verification or before/after identity guard failed; do not accept this build.")
    report["artifacts_after_call"] = {}
    for path in sorted(set(Path(path).resolve() for path in artifact_paths)):
        try:
            report["artifacts_after_call"][str(path)] = file_record(path)
        except Exception as error:
            report["errors"].append(f"Artifact {path}: {type(error).__name__}: {error}")
    if not report["artifacts_after_call"]:
        report["errors"].append("No solver shared libraries were recorded after the call.")
    report["build_files_after"] = build_files_after
    report["finished_utc"] = utc_now()
    report["status"] = "failed" if report["errors"] else "succeeded"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history = directory / f"build-manifest-{report['build_id']}.json"
    report["history_file"] = str(history.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with history.open("x") as handle:
        handle.write(payload)
    temporary = directory / f".build-manifest-{report['build_id']}.tmp"
    with temporary.open("x") as handle:
        handle.write(payload)
    os.replace(temporary, directory / "build-manifest.json")
    print(f"Build provenance ({report['status']}): {history}", flush=True)


def baseline(args):
    source = args.source.resolve()
    report = new_manifest("baseline")
    report.update(
        upstream_revision=args.revision,
        source_directory=str(source),
        private_source_marker=(source / ".source-revision").read_text().strip()
        if (source / ".source-revision").is_file()
        else None,
    )
    report["build_driver_hashes"] = {
        str(path): file_record(path)
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("build_baseline.sh").resolve(),
        )
    }
    build_tree = source / "build/skbuild"
    report["build_files_before"] = build_files_snapshot(build_tree, recursive=True)
    commands = [
        [
            args.uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(Path(sys.prefix) / "bin/python"),
            "--no-deps",
            "--no-build-isolation",
            str(source),
        ],
        [args.uv, "--no-config", "pip", "check", "--python", str(Path(sys.prefix) / "bin/python")],
    ]
    report["commands"] = [{"argv": command, "returncode": None} for command in commands]
    after, artifacts, returncode = {"valid": False, "files": {}}, [], 0
    try:
        report["sources_before"] = pinned_snapshot(source, args.upstream, args.revision)
        if not report["sources_before"]["valid"]:
            raise RuntimeError("Private source differs from pinned Git blobs; build was not started.")
        for command in report["commands"]:
            command["started_utc"] = utc_now()
            result = subprocess.run(command["argv"], check=False)
            command.update(returncode=result.returncode, finished_utc=utc_now())
            if result.returncode:
                returncode = result.returncode
                raise RuntimeError(f"Build command exited {result.returncode}: {command['argv']}")
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        returncode = returncode or 1
    finally:
        try:
            after = pinned_snapshot(source, args.upstream, args.revision)
        except Exception as error:
            report["errors"].append(f"Source-after check: {type(error).__name__}: {error}")
        try:
            distribution = importlib.metadata.distribution("turbompc")
            installed = Path(distribution.locate_file("turbompc")).resolve()
            if not installed.is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(f"Installed package is outside this venv: {installed}")
            artifacts = list(installed.rglob("*.so"))
            required = {"libadmm_cudss_ffi.so", "libcudss_sparse_kkt_ffi.so"}
            if not required.issubset({path.name for path in artifacts}):
                raise RuntimeError("Installed default forward/backward FFI libraries are missing")
            report.update(installed_package=str(installed), installed_version=distribution.version)
        except Exception as error:
            report["errors"].append(f"Installed artifacts: {type(error).__name__}: {error}")
        finalize(args.manifest_dir, report, after, artifacts, build_files_snapshot(build_tree, recursive=True))
    return returncode or (0 if report["status"] == "succeeded" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    command = subparsers.add_parser("baseline", help="Record the existing baseline install/check calls")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--upstream", type=Path, required=True)
    command.add_argument("--revision", required=True)
    command.add_argument("--manifest-dir", type=Path, required=True)
    command.add_argument("--uv", required=True)
    args = parser.parse_args()
    return baseline(args)


if __name__ == "__main__":
    raise SystemExit(main())
