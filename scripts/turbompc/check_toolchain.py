"""Compile and run a small float64 Torch/CUDA/cuDSS probe in the selected profile.

This checks installation, device pointers and the current CUDA stream. It does
not implement MPC or measure solver performance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve the existing check result: {args.output}")
    profile = os.environ["TURBOMPC_CUDA_PROFILE"]
    cuda_major = {"cu128": 12, "cu130": 13}[profile]
    cuda_home = Path(os.environ["CUDA_HOME"])
    cudss_dir = Path(os.environ["CUDSS_DIR"])
    package = f"nvidia-cudss-cu{cuda_major}"
    report = {
        "scope": "toolchain smoke check; not MPC acceptance or a performance benchmark",
        "profile": profile,
        "time_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "cuda_home": str(cuda_home),
        "cudss_dir": str(cudss_dir),
        "status": "failed",
    }
    try:
        import torch
        from torch.utils.cpp_extension import load

        if int(torch.version.cuda.split(".")[0]) != cuda_major:
            raise RuntimeError(f"Torch CUDA {torch.version.cuda} does not match {profile}")
        if not torch.cuda.is_available():
            raise RuntimeError("Torch cannot access a CUDA device")
        report.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudss_package": {package: importlib.metadata.version(package)},
            "nvcc": subprocess.check_output([str(cuda_home / "bin/nvcc"), "--version"], text=True).strip(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "driver": subprocess.check_output([
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ], text=True).strip(),
            "cudss_jit_disable": os.environ.get("CUDSS_JIT_DISABLE", "unset"),
        })
        site = Path(sysconfig.get_path("purelib"))
        wheel_libraries = sorted(path for path in (site / "nvidia").glob("*/lib") if path.is_dir())
        cudss_library = cudss_dir / "lib/libcudss.so.1"
        if not cudss_library.is_file():
            raise FileNotFoundError(cudss_library)
        source = Path(__file__).resolve().parent
        os.environ["MAX_JOBS"] = str(max(1, min(int(os.environ.get("MAX_JOBS", "2")), 2)))
        # DT_RPATH also resolves cuDSS's transitive NVRTC dependency. Only this
        # profile's wheel libraries are searched; no global LD_LIBRARY_PATH is needed.
        extension = load(
            name=f"turbompc_toolchain_smoke_{profile}",
            sources=[str(source / "smoke_bindings.cpp"), str(source / "smoke_cuda.cu")],
            extra_include_paths=[str(cudss_dir / "include")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            extra_ldflags=[str(cudss_library), "-Wl,--disable-new-dtags", *[
                f"-Wl,-rpath,{path}" for path in wheel_libraries
            ]],
            verbose=True,
        )
        versions = extension.compiled_versions()
        report["cuda_compile_version"] = versions[0]
        report["cuda_runtime_version"] = versions[1]
        report["cuda_driver_api_version"] = versions[2]
        report["cudss_runtime_version"] = ".".join(map(str, versions[3:]))
        if versions[0] // 1000 != cuda_major or versions[1] // 1000 != cuda_major:
            raise RuntimeError("Compiled or loaded CUDA runtime has the wrong major version")

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            input_tensor = torch.arange(4096, device="cuda", dtype=torch.float64)
            output, kernel_stream, input_pointer = extension.affine(input_tensor)
            # A=[[4,1,0],[1,3,1],[0,1,2]], x=[1,2,3], b=[6,10,8].
            offsets = torch.tensor([0, 1, 3, 5], device="cuda", dtype=torch.int32)
            columns = torch.tensor([0, 0, 1, 1, 2], device="cuda", dtype=torch.int32)
            values = torch.tensor([4, 1, 3, 1, 2], device="cuda", dtype=torch.float64)
            rhs = torch.tensor([6, 10, 8], device="cuda", dtype=torch.float64)
            solution, solver_stream = extension.sparse_solve(offsets, columns, values, rhs)
        stream.synchronize()
        expected = 2 * torch.arange(4096, dtype=torch.float64) + 1
        affine_error = (output.cpu() - expected).abs().max().item()
        solution_cpu = solution.cpu()
        matrix = torch.tensor([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=torch.float64)
        rhs_cpu = torch.tensor([6, 10, 8], dtype=torch.float64)
        residual = (matrix @ solution_cpu - rhs_cpu).abs().max().item()
        solution_error = (solution_cpu - torch.tensor([1, 2, 3], dtype=torch.float64)).abs().max().item()
        stream_matches = kernel_stream == solver_stream == stream.cuda_stream
        nondefault = stream.cuda_stream != torch.cuda.default_stream().cuda_stream
        pointer_matches = input_pointer == input_tensor.data_ptr()
        report["checks"] = {
            "affine_max_abs_error": affine_error,
            "cudss_max_abs_residual": residual,
            "cudss_max_abs_solution_error": solution_error,
            "cudss_solution": solution_cpu.tolist(),
            "current_stream_matches": stream_matches,
            "nondefault_stream": nondefault,
            "torch_input_pointer_matches": pointer_matches,
            "stream_pointer": stream.cuda_stream,
        }
        library_names = ("libcuda", "libcudart", "libcublas", "libcusparse", "libcudss", "libnvrtc", "libnvJitLink")
        report["loaded_cuda_libraries"] = sorted({
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if len(line.split()) >= 6 and Path(line.split()[-1]).name.startswith(library_names)
        })
        report["extension"] = extension.__file__
        if not (affine_error == 0.0 and residual <= 1e-12 and solution_error <= 1e-12
                and stream_matches and nondefault and pointer_matches):
            raise RuntimeError("CUDA/cuDSS numerical, pointer or stream check failed")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
