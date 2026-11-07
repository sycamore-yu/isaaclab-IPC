"""Exact CPU/CUDA graph comparisons for the native precomputed-neighbour path."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/pathcover_native/jps"


def compile_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def legal_moves_cpu_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    directory = tmp_path_factory.mktemp("dmp_legal_moves_cpu")
    binary = directory / "test"
    compile_command([
        compiler, "-O1", "-std=c++17", "-I/usr/include/eigen3", f"-I{VENDOR / 'include'}",
        str(ROOT / "tests/dmp_legal_moves_test.cpp"),
        str(VENDOR / "src/distance_map_planner/graph_search.cpp"),
        str(VENDOR / "src/distance_map_planner/distance_map_planner.cpp"), "-o", str(binary),
    ])
    return binary


@pytest.fixture(scope="module")
def legal_moves_cuda_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    cuda = Path(os.environ.get("CUDA_HOME", ROOT / "build/turbompc/toolchains/cuda-13.0.2"))
    nvcc = cuda / "bin/nvcc"
    if compiler is None or not nvcc.is_file():
        pytest.skip("native compiler or CUDA toolkit is unavailable")
    directory = tmp_path_factory.mktemp("dmp_legal_moves_cuda")
    obj = directory / "distance_map.o"
    binary = directory / "test"
    compile_command([
        str(nvcc), "-O3", "-std=c++17", f"-arch=sm_{os.environ.get('PATHCOVER_CUDA_ARCH', '89')}",
        "--expt-relaxed-constexpr", "-c", str(ROOT / "native/pathcover_distance_map.cu"), "-o", str(obj),
    ])
    library = cuda / ("lib" if (cuda / "lib/libcudart.so").exists() else "lib64")
    compile_command([
        compiler, "-O1", "-std=c++17", "-DPATHCOVER_TEST_CUDA", "-I/usr/include/eigen3",
        f"-I{VENDOR / 'include'}", f"-I{ROOT / 'native'}", f"-I{cuda / 'include'}",
        str(ROOT / "tests/dmp_legal_moves_test.cpp"),
        str(VENDOR / "src/distance_map_planner/graph_search.cpp"),
        str(VENDOR / "src/distance_map_planner/distance_map_planner.cpp"), str(obj),
        f"-L{library}", f"-Wl,-rpath,{library}", "-lcudart", "-o", str(binary),
    ])
    return binary


def run_case(binary, case):
    result = subprocess.run([str(binary), case], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("CUDA device or driver is unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", ["cpu_equivalence", "lifecycle"])
def test_precomputed_legal_moves_cpu(legal_moves_cpu_binary, case):
    run_case(legal_moves_cpu_binary, case)


@pytest.mark.parametrize("case", ["gpu_masks", "gpu_equivalence"])
def test_precomputed_legal_moves_cuda(legal_moves_cuda_binary, case):
    run_case(legal_moves_cuda_binary, case)
