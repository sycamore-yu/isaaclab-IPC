"""Build the pinned TurboMPC kernels as a standalone PyTorch CUDA extension."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "turbompc"))
import record_build


def main():
    root = Path(__file__).resolve().parents[1]
    native = root / "native" / "turbompc"
    import torch

    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("A CUDA-enabled PyTorch build is required")
    major, minor = cuda_version.split(".")[:2]
    profile = f"cu{major}{minor}"
    cuda_roots = [Path(p) / "nvidia" / f"cu{major}" for p in sys.path if p]
    compiler_roots = [p for p in cuda_roots if (p / "bin" / "nvcc").is_file()]
    cuda = Path(os.environ["CUDA_HOME"]) if "CUDA_HOME" in os.environ else next(iter(compiler_roots), None)
    if cuda is None:
        raise RuntimeError("CUDA_HOME must point to the toolkit matching this environment's PyTorch")
    os.environ["CUDA_HOME"] = str(cuda)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    libraries = [p / "lib" for p in cuda_roots if (p / "lib").is_dir()]
    libraries += [
        Path(p) / "nvidia" / component / "lib"
        for p in sys.path
        if p
        for component in ("cublas", "cusparse", "cuda_runtime")
        if (Path(p) / "nvidia" / component / "lib").is_dir()
    ]
    if (cuda / "lib64").is_dir():
        libraries.insert(0, cuda / "lib64")

    def library(name):
        for directory in libraries:
            matches = sorted(directory.glob(f"{name}.so*"), key=lambda p: len(p.name))
            if matches:
                return matches[0]
        raise RuntimeError(f"Missing {name} in matching CUDA {major} library directories")

    cudss = library("libcudss")
    cusparse = library("libcusparse")
    runtime = library("libcudart")
    # Wheels omit the development symlink used by torch's '-lcudart' linker flag.
    runtime_link = cuda / "lib" / "libcudart.so"
    if not runtime_link.exists() and runtime.parent == runtime_link.parent:
        runtime_link.symlink_to(runtime.name)

    from torch.utils.cpp_extension import load

    default_build = root / "build" / "turbompc" / "native" / profile
    build = Path(os.environ.get("TURBOMPC_NATIVE_BUILD_DIR", default_build)).resolve()
    build.mkdir(parents=True, exist_ok=True)
    sources = [
        native / "binding.cpp",
        native / "upstream" / "admm" / "admm_cudss.cu",
        native / "upstream" / "backward" / "cudss_sparse_kkt.cu",
        native / "upstream" / "common" / "block_tridi.cu",
    ]
    include = [
        native,
        native / "upstream" / "admm",
        native / "upstream" / "backward",
        native / "upstream" / "common",
        cudss.parent.parent / "include",
    ]
    load_options = {
        "name": "_turbompc_native",
        "sources": [str(p) for p in sources],
        "extra_include_paths": [str(p) for p in include],
        "extra_cflags": ["-O3", "-std=c++17"],
        "extra_cuda_cflags": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        "extra_ldflags": [str(cudss), str(cusparse), *[f"-Wl,-rpath,{p}" for p in libraries]],
        "build_directory": str(build),
        "verbose": True,
    }
    report = record_build.new_manifest("native")
    report.update(
        profile=profile,
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        load_options=load_options,
        build_directory=str(build),
    )
    drivers = (Path(__file__).resolve(), Path(record_build.__file__).resolve())
    report["sources_before"] = record_build.native_snapshot(native, drivers)
    report["build_files_before"] = record_build.build_files_snapshot(build)
    artifacts, error = [], None
    try:
        if not report["sources_before"]["valid"]:
            raise RuntimeError("Cannot hash stable native source inputs; build was not started")
        module = load(**load_options)
        binary = Path(module.__file__).resolve()
        if not binary.is_relative_to(build):
            raise RuntimeError(f"Loaded extension is outside the requested build directory: {binary}")
        artifacts = [binary]
        report["loaded_extension"] = str(binary)
    except Exception as failure:
        error = failure
        report["errors"].append(f"{type(failure).__name__}: {failure}")
        artifacts = list(build.glob("_turbompc_native*.so"))
    finally:
        record_build.finalize(
            build,
            report,
            record_build.native_snapshot(native, drivers),
            artifacts,
            record_build.build_files_snapshot(build),
        )
    if report["status"] != "succeeded":
        raise RuntimeError("Native build provenance failed: " + "; ".join(report["errors"])) from error
    print(f"Built native TurboMPC extension for PyTorch CUDA {cuda_version} ({profile}) in {build}")


if __name__ == "__main__":
    main()
