"""CUDA fine-volume projection against integer overlap and the CPU production map."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r'''
#include "pathcover_grid_cuda.h"
#include "pathcover_grid.h"
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using Cell=Eigen::Vector3i;
using occupancy_inflation::VoxelBitGrid;
using Ratio=pathcover_grid::ResolutionRatio;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }

VoxelBitGrid makeGrid(const std::vector<Cell>& cells) {
    VoxelBitGrid grid;
    if(cells.empty()) return grid;
    Cell lower=cells.front(),upper=lower;
    for(const auto& cell:cells) {lower=lower.cwiseMin(cell);upper=upper.cwiseMax(cell);}
    const auto window=occupancy_inflation::snapWindow(lower,upper);
    require(grid.reset(window,window.wordCount()),"grid allocation");
    for(const auto& cell:cells) {
        const Cell local=cell-window.base;
        grid.setLocal(local.x(),local.y(),local.z());
    }
    return grid;
}

std::vector<signed char> overlapOracle(const std::vector<Cell>& fine,Ratio ratio,Cell origin,Cell dimensions) {
    std::vector<signed char> expected(dimensions.prod(),0);
    for(int z=0;z<dimensions.z();++z) for(int y=0;y<dimensions.y();++y) for(int x=0;x<dimensions.x();++x) {
        const Cell coarse=origin+Cell(x,y,z);
        for(const auto& voxel:fine) {
            bool overlap=true;
            for(int axis=0;axis<3;++axis) {
                overlap=overlap && std::int64_t(voxel[axis])*ratio.first < std::int64_t(coarse[axis]+1)*ratio.second
                    && std::int64_t(voxel[axis]+1)*ratio.first > std::int64_t(coarse[axis])*ratio.second;
            }
            if(overlap) {expected[x+dimensions.x()*(y+dimensions.y()*z)]=100;break;}
        }
    }
    return expected;
}

void compare(pathcover_cuda::GridProjector& gpu,const std::vector<Cell>& occupied,Ratio ratio,Cell origin,Cell dimensions) {
    const auto fine=makeGrid(occupied);
    const auto actual=gpu.project(fine,ratio,origin,dimensions);
    require(actual==overlapOracle(occupied,ratio,origin,dimensions),"CUDA differs from integer positive-volume intersection");
    std::vector<signed char> cpu(actual.size(),0);
    std::vector<int> indexes;
    pathcover_grid::coarsenOccupiedVolumes(fine,ratio,origin,dimensions,cpu,indexes);
    require(actual==cpu,"CUDA differs from production fine-volume projection");
    // Merge only cloud occupancy. Workspace cells that were already occupied
    // must survive future cloud resets and must not enter the cloud index list.
    std::vector<signed char> merged(actual.size(),0), expected=merged;
    merged.front()=expected.front()=100;
    indexes.clear();
    pathcover_grid::coarsenOccupiedVolumes(fine,ratio,origin,dimensions,expected,indexes);
    std::vector<int> actual_indexes;
    for(std::size_t i=0;i<actual.size();++i) if(actual[i] && merged[i]==0) {
        merged[i]=100; actual_indexes.push_back(static_cast<int>(i));
    }
    require(merged==expected,"workspace boundary merge changed");
    for(int i:actual_indexes) merged[i]=0;
    require(merged.front()==100,"cloud reset removed workspace boundary");
}

void boundaries() {
    pathcover_cuda::GridProjector gpu;
    compare(gpu,{},Ratio(2,5),Cell(-3,-2,-1),Cell(9,7,5));
    compare(gpu,{Cell(-1,0,0),Cell(0,1,0),Cell(1,0,-1)},Ratio(1,1),Cell(-3,-3,-3),Cell(8,8,8));
    compare(gpu,{Cell(-66,-3,-2),Cell(-65,-3,-2),Cell(-1,0,0),Cell(0,0,0),Cell(63,2,1),Cell(64,2,1)},
            Ratio(2,5),Cell(-28,-3,-3),Cell(56,7,7));
    compare(gpu,{Cell(-9,262,29),Cell(-8,263,30)},Ratio(2,5),Cell(-30,-220,0),Cell(60,440,14));
}

void varied_ratios() {
    std::mt19937 random(241);
    std::vector<Cell> occupied;
    for(int i=0;i<80;++i) occupied.emplace_back(int(random()%140)-70,int(random()%10)-5,int(random()%8)-4);
    pathcover_cuda::GridProjector gpu;
    for(const Ratio ratio:{Ratio(1,3),Ratio(2,5),Ratio(3,4)}) {
        compare(gpu,occupied,ratio,Cell(-55,-5,-4),Cell(110,12,10));
    }
}

void padding_and_reuse() {
    pathcover_cuda::GridProjector gpu;
    const occupancy_inflation::VoxelWindow window{Cell(-73,-2,-1),Cell(65,3,2)};
    VoxelBitGrid grid;
    require(grid.reset(window,window.wordCount()),"partial grid allocation");
    std::vector<Cell> occupied;
    for(int z=0;z<2;++z) for(int y=0;y<3;++y) for(int x=0;x<65;++x) {
        grid.setLocal(x,y,z);occupied.push_back(window.base+Cell(x,y,z));
    }
    for(int z=0;z<2;++z) for(int y=0;y<3;++y) grid.data()[window.rowOffset(y,z)+1]=~std::uint64_t{0};
    const Ratio ratio(2,5); const Cell origin(-35,-4,-3),dimensions(40,9,7);
    const auto actual=gpu.project(grid,ratio,origin,dimensions);
    require(actual==overlapOracle(occupied,ratio,origin,dimensions),"padding leaked into coarse occupancy");
    grid.clearBits();
    const auto empty=gpu.project(grid,ratio,origin,dimensions);
    require(std::all_of(empty.begin(),empty.end(),[](signed char c){return c==0;}),"clear retained previous projection");
    grid.release();
    require(gpu.project(grid,ratio,origin,dimensions)==empty,"release retained previous projection");
    compare(gpu,{Cell(-150,-7,-3),Cell(145,8,5)},Ratio(1,2),Cell(-80,-6,-4),Cell(160,14,10));
    compare(gpu,{Cell(0,0,0)},Ratio(1,1),Cell(-1,-1,-1),Cell(3,3,3));
    require(actual==overlapOracle(occupied,ratio,origin,dimensions),"later calls mutated an owned result");
}

void independent_instances() {
    auto first=std::async(std::launch::async,[]{
        pathcover_cuda::GridProjector gpu;
        compare(gpu,{Cell(-64,0,0),Cell(65,1,2)},Ratio(2,5),Cell(-28,-2,-1),Cell(60,6,5));
    });
    auto second=std::async(std::launch::async,[]{
        pathcover_cuda::GridProjector gpu;
        compare(gpu,{Cell(-3,-2,-1),Cell(4,2,1)},Ratio(3,4),Cell(-4,-4,-4),Cell(10,10,10));
    });
    first.get();second.get();
}

int main(int argc,char** argv) {
    try {
        const std::string name=argc>1?argv[1]:"";
        if(name=="boundaries") boundaries();
        else if(name=="varied_ratios") varied_ratios();
        else if(name=="padding_and_reuse") padding_and_reuse();
        else if(name=="independent_instances") independent_instances();
        else throw std::runtime_error("unknown case");
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
'''


@pytest.fixture(scope="module")
def cuda_projection_binary(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    compiler = shutil.which("g++")
    root = Path(__file__).resolve().parents[1]
    cuda = Path(os.environ.get("CUDA_HOME", root / "build/turbompc/toolchains/cuda-13.0.2"))
    if compiler is None or not (cuda / "bin/nvcc").is_file():
        pytest.skip("C++ compiler or configured CUDA toolkit is unavailable")
    vendor = root / "third_party/pathcover_native/occupancy_inflation"
    directory = tmp_path_factory.mktemp("pathcover_grid_cuda")
    source = directory / "projection.cpp"
    source.write_text(SOURCE)
    executable = directory / "projection"
    host_object, grid_object, cuda_object = (directory / name for name in ("projection.o", "grid.o", "cuda.o"))
    includes = ["-I/usr/include/eigen3", f"-I{root / 'native'}", f"-I{vendor / 'include'}"]
    host_flags = ["-O3", "-DNDEBUG", "-std=c++17", "-fopenmp", "-pthread", *includes]
    cuda_lib = cuda / "lib" if (cuda / "lib/libcudart.so").is_file() else cuda / "lib64"
    commands = [
        [compiler, *host_flags, "-c", str(source), "-o", str(host_object)],
        [compiler, *host_flags, "-c", str(vendor / "src/voxel_bit_grid.cpp"), "-o", str(grid_object)],
        [str(cuda / "bin/nvcc"), "-O3", "-DNDEBUG", "-std=c++17", "-arch=sm_89",
         "--expt-relaxed-constexpr", "-Xcompiler=-fopenmp,-pthread", *includes,
         "-c", str(root / "native/pathcover_grid_cuda.cu"), "-o", str(cuda_object)],
        [compiler, "-fopenmp", "-pthread", str(host_object), str(grid_object), str(cuda_object),
         f"-L{cuda_lib}", f"-Wl,-rpath,{cuda_lib}", "-lcudart", "-o", str(executable)],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", ["boundaries", "varied_ratios", "padding_and_reuse", "independent_instances"])
def test_cuda_projection_matches_volume_contract(cuda_projection_binary, case):
    result = subprocess.run([str(cuda_projection_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
