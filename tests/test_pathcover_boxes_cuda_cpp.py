"""GPU box boundaries/order against both existing CPU compression oracles."""

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _oracle_source():
    """Reuse the existing CPU geometry fixtures, replacing only the tested call."""
    tree = ast.parse(Path(__file__).with_name("test_pathcover_boxes_cpp.py").read_text())
    source = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE" for target in node.targets)
    )
    source = source.replace(
        '#include "pathcover_boxes.h"',
        '#include "pathcover_boxes_cuda.h"\n#include <future>',
    )
    source = source.replace(
        "using Key=std::array<int,3>;",
        "using Key=std::array<int,3>;\n"
        "std::vector<Box> compare_gpu(const VoxelBitGrid& grid);\n"
        "void workspace_growth_and_instances();",
    )
    source = source.replace("pathcover_boxes::compress(grid)", "compare_gpu(grid)")
    source = source.replace(
        'else if(name=="adjacent_row_order") adjacent_row_order();',
        'else if(name=="adjacent_row_order") adjacent_row_order();\n'
        '        else if(name=="workspace_growth_and_instances") workspace_growth_and_instances();',
    )
    return source + r"""

std::vector<Box> compare_gpu(const VoxelBitGrid& grid) {
    static pathcover_cuda::BoxCompressor gpu;
    const auto expected=pathcover_boxes::compress(grid);
    auto actual=gpu.compress(grid);
    require(same(actual,expected),"GPU coordinates or ordered boxes differ from V21 CPU");
    return actual;
}

void workspace_growth_and_instances() {
    const auto cells=fill(Cell(-70,-3,-2),Cell(131,4,3));
    auto first=makeGrid(cells);
    const auto before=compare_gpu(first);
    const auto larger=occupancy_inflation::snapWindow(Cell(-350,-8,-5),Cell(350,8,5));
    require(first.growToInclude(larger,1u<<20),"grow grid window");
    require(same(before,compare_gpu(first)),"growing padding changed the box array");

    const std::vector<Cell> other_cells{Cell(64,-7,-4),Cell(65,-7,-4),Cell(-65,9,5)};
    auto second=makeGrid(other_cells);
    const auto expected_other=pathcover_boxes::compress(second);
    pathcover_cuda::BoxCompressor other_gpu;
    auto a=std::async(std::launch::async,[&]{return compare_gpu(first);});
    auto b=std::async(std::launch::async,[&]{return other_gpu.compress(second);});
    const auto first_result=a.get();
    const auto other_result=b.get();
    require(same(first_result,before),"concurrent instance changed first output");
    require(same(other_result,expected_other),"concurrent instance changed second output");
    exactUnion(cells,first_result);
    exactUnion(other_cells,other_result);

    second.clearBits();
    require(other_gpu.compress(second).empty(),"clear retained a previous GPU result");
    first.release();
    require(compare_gpu(first).empty(),"release retained GPU boxes");
    auto small=makeGrid({Cell(-1,0,0)});
    exactUnion({Cell(-1,0,0)},compare_gpu(small));
    require(same(other_result,expected_other),"subsequent calls mutated an owned snapshot");
}
"""


@pytest.fixture(scope="module")
def cuda_boxes_binary(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    compiler = shutil.which("g++")
    root = Path(__file__).resolve().parents[1]
    cuda = Path(os.environ.get("CUDA_HOME", root / "build/turbompc/toolchains/cuda-13.0.2"))
    if compiler is None or not (cuda / "bin/nvcc").is_file():
        pytest.skip("C++ compiler or configured CUDA toolkit is unavailable")
    vendor = root / "third_party/pathcover_native/occupancy_inflation"
    directory = tmp_path_factory.mktemp("pathcover_boxes_cuda")
    source = directory / "boxes.cpp"
    source.write_text(_oracle_source())
    executable = directory / "boxes"
    host_object, grid_object, cuda_object = (directory / name for name in ("boxes.o", "grid.o", "cuda.o"))
    includes = ["-I/usr/include/eigen3", f"-I{root / 'native'}", f"-I{vendor / 'include'}"]
    host_flags = ["-O3", "-DNDEBUG", "-std=c++17", "-fopenmp", "-pthread", *includes]
    cuda_lib = cuda / "lib" if (cuda / "lib/libcudart.so").is_file() else cuda / "lib64"
    commands = [
        [compiler, *host_flags, "-c", str(source), "-o", str(host_object)],
        [compiler, *host_flags, "-c", str(vendor / "src/voxel_bit_grid.cpp"), "-o", str(grid_object)],
        [
            str(cuda / "bin/nvcc"), "-O3", "-DNDEBUG", "-std=c++17", "-arch=sm_89",
            "--expt-relaxed-constexpr", "-Xcompiler=-fopenmp,-pthread", *includes,
            f"-I{cuda / 'include/cccl'}", "-c", str(root / "native/pathcover_boxes_cuda.cu"),
            "-o", str(cuda_object),
        ],
        [
            compiler, "-fopenmp", "-pthread", str(host_object), str(grid_object), str(cuda_object),
            f"-L{cuda_lib}", f"-Wl,-rpath,{cuda_lib}", "-lcudart", "-o", str(executable),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "empty_and_reset", "solid_negative_words", "holes", "checkerboard",
    "footprints_and_gaps", "partial_word", "adjacent_row_order", "workspace_growth_and_instances",
])
def test_cuda_box_compression_matches_ordered_cpu(cuda_boxes_binary, case):
    result = subprocess.run([str(cuda_boxes_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
