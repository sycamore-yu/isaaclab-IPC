"""Build a review-only JAX KKT IR=2 candidate without installing or running it."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

REVISION = "7050a4f367d08504ecb6f5820d9161e704df5816"
KKT_SOURCE = Path("turbompc/solvers/backward/csrc/cudss_sparse_kkt.cu")
LIBRARY = "libcudss_sparse_kkt_ffi.so"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_ffi(source, name):
    # Load only this stdlib-only helper. Importing turbompc.solvers normally
    # imports solver backends too, which is unnecessary for this CPU check.
    path = source / "turbompc/solvers/ffi_loader.py"
    spec = importlib.util.spec_from_file_location("turbompc.solvers._candidate_lookup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_ffi_lib(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Read-only repository containing the pinned commit")
    args = parser.parse_args()
    if os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu128":
        parser.error("Run through: bash scripts/turbompc/env.sh cu128 python ...")
    if importlib.metadata.version("jaxlib") != "0.6.2":
        parser.error("This candidate requires the installed JAX/jaxlib 0.6.2 baseline environment")

    root = Path(__file__).resolve().parents[2]
    candidate = root / "build/turbompc/kkt-reference-fix/cu128"
    source, build = candidate / "source", candidate / "build"
    candidate.mkdir(parents=True, exist_ok=True)
    package = importlib.metadata.distribution("turbompc")
    installed_root = Path(package.locate_file("turbompc"))
    installed_kkt = installed_root / "solvers/_ffi" / LIBRARY
    baseline_hash = sha256(installed_kkt)
    marker = candidate / "source-revision.txt"
    if not source.exists():
        archive = subprocess.check_output(["git", "-C", str(args.upstream), "archive", REVISION])
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(source, filter="data")
        marker.write_text(REVISION + "\n")
    elif not marker.exists() or marker.read_text().strip() != REVISION:
        raise RuntimeError(f"Unrecognized existing source at {source}; retained without changes")

    original = subprocess.check_output(
        ["git", "-C", str(args.upstream), "show", f"{REVISION}:{KKT_SOURCE.as_posix()}"], text=True
    )
    anchor = "        CUDSS_CHECK(cudssConfigCreate(&config));\n"
    refinement = (
        "        int refinement_steps = 2;\n"
        "        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS,\n"
        "                                   &refinement_steps, sizeof(refinement_steps)));\n"
    )
    if original.count(anchor) != 1:
        raise RuntimeError("Pinned KKT initialization no longer matches the reviewed patch")
    patched = original.replace(anchor, anchor + refinement, 1)
    target = source / KKT_SOURCE
    if target.read_text() not in (original, patched):
        raise RuntimeError(f"Unexpected edits in {target}; retained without changes")
    target.write_text(patched)
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{KKT_SOURCE.as_posix()}",
            tofile=f"b/{KKT_SOURCE.as_posix()}",
        )
    )
    (candidate / "kkt-ir2.patch").write_text(patch)

    cuda = Path(os.environ["CUDA_HOME"])
    cudss = Path(os.environ["CUDSS_DIR"])
    site = Path(importlib.metadata.distribution("torch").locate_file(""))
    libraries = sorted(str(path) for path in (site / "nvidia").glob("*/lib") if path.is_dir())
    libraries.append(str(cuda / "lib64"))
    rpath = ";".join(libraries)
    commands = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DTURBOMPC_CUDA=ON",
            "-DTURBOMPC_EIGEN=OFF",
            "-DTURBOMPC_CUDSS=ON",
            "-DTURBOMPC_POST_BUILD_CHECK=OFF",
            "-DCMAKE_CUDA_ARCHITECTURES=89",
            f"-DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc'}",
            f"-DCUDAToolkit_ROOT={cuda}",
            f"-DCUDSS_DIR={cudss}",
            f"-DCUDSS_LIB={cudss / 'lib/libcudss.so.1'}",
            f"-DCMAKE_BUILD_RPATH={rpath}",
            f"-DCMAKE_INSTALL_RPATH={rpath}",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags",
        ],
        ["cmake", "--build", str(build), "--target", "cudss_sparse_kkt_ffi", "--parallel", "2"],
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = candidate / f"build-{stamp}.log"
    build_environment = dict(os.environ, TURBOMPC_CUDA="ON", TURBOMPC_EIGEN="OFF")
    try:
        with log_path.open("x") as log:
            for command in commands:
                print("Building review-only KKT candidate:", command[0:2], flush=True)
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, env=build_environment, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Candidate build failed; log retained at {log_path}") from error
    finally:
        if sha256(installed_kkt) != baseline_hash:
            raise RuntimeError("Installed baseline KKT changed during the candidate build")

    overlay = source / "turbompc/solvers/_ffi" / LIBRARY
    overlay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / LIBRARY, overlay)
    # Preserve optional CPU backend availability when selecting this source
    # tree. These are unchanged copies of already-installed baseline binaries.
    eigen_copies = {}
    for binary in (installed_root / "solvers/linear_systems_solvers").glob("_eigen_block_tridi*.so"):
        destination = source / "turbompc" / binary.relative_to(installed_root)
        shutil.copy2(binary, destination)
        eigen_copies[str(destination)] = sha256(destination)
    selected = locate_ffi(source, LIBRARY)
    forward = locate_ffi(source, "libadmm_cudss_ffi.so")
    if selected != overlay or forward != installed_root / "solvers/_ffi/libadmm_cudss_ffi.so":
        raise RuntimeError("Candidate/installed FFI discovery did not match the intended isolation")
    manifest = {
        "role": "review-only candidate; not the unchanged A baseline and not an M4 result",
        "upstream_commit": REVISION,
        "profile": "cu128",
        "kkt_ir_steps": 2,
        "gpu_validation_performed": False,
        "original_kkt_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "candidate_kkt_source_sha256": sha256(target),
        "patch": patch,
        "library": str(overlay),
        "library_sha256": sha256(overlay),
        "installed_baseline_kkt": str(installed_kkt),
        "installed_baseline_kkt_sha256_before": baseline_hash,
        "installed_baseline_kkt_sha256_after": sha256(installed_kkt),
        "selected_forward_library": str(forward),
        "unchanged_eigen_copies": eigen_copies,
        "commands": commands,
        "log": str(log_path),
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    (candidate / f"manifest-{stamp}.json").write_text(encoded)
    (candidate / "manifest.json").write_text(encoded)
    print("Built candidate (no GPU validation):", overlay)
    print("Review manifest:", candidate / "manifest.json")


if __name__ == "__main__":
    main()
