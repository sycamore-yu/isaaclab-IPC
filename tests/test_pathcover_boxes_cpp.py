"""Exact voxel union and reset contracts for integer box compression."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include "pathcover_boxes.h"
#include <array>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

using pathcover_boxes::Box;
using pathcover_boxes::Cell;
using occupancy_inflation::VoxelBitGrid;
using Key=std::array<int,3>;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}
Key key(const Cell& cell) { return {cell.x(),cell.y(),cell.z()}; }
std::set<Key> canonical(const std::vector<Cell>& cells) {
    std::set<Key> result;
    for(const auto& cell:cells) result.insert(key(cell));
    return result;
}
void exactUnion(const std::vector<Cell>& cells,const std::vector<Box>& boxes) {
    std::set<Key> covered;
    std::int64_t volume=0;
    for(const auto& box:boxes) {
        require((box.upper.array()>box.lower.array()).all(),"nonpositive box extent");
        const Cell size=box.upper-box.lower;
        volume+=std::int64_t(size.x())*size.y()*size.z();
        for(int z=box.lower.z();z<box.upper.z();++z)
            for(int y=box.lower.y();y<box.upper.y();++y)
                for(int x=box.lower.x();x<box.upper.x();++x)
                    require(covered.insert({x,y,z}).second,"boxes overlap on a voxel");
    }
    require(covered==canonical(cells),"box union differs from occupied voxels");
    require(volume==std::int64_t(covered.size()),"box volume counts duplicates");
}
bool same(const std::vector<Box>& a,const std::vector<Box>& b) {
    if(a.size()!=b.size()) return false;
    for(std::size_t i=0;i<a.size();++i)
        if((a[i].lower-b[i].lower).norm()!=0 || (a[i].upper-b[i].upper).norm()!=0) return false;
    return true;
}
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
std::vector<Box> checkBoth(const std::vector<Cell>& cells) {
    const auto from_cells=pathcover_boxes::compress(cells);
    const auto grid=makeGrid(cells);
    const auto from_grid=pathcover_boxes::compress(grid);
    exactUnion(cells,from_cells);
    exactUnion(cells,from_grid);
    require(same(from_cells,from_grid),"list and packed-grid compression differ");
    return from_grid;
}
std::vector<Cell> fill(const Cell& lower,const Cell& upper) {
    std::vector<Cell> cells;
    for(int z=lower.z();z<upper.z();++z) for(int y=lower.y();y<upper.y();++y)
        for(int x=lower.x();x<upper.x();++x) cells.emplace_back(x,y,z);
    return cells;
}
void empty_and_reset() {
    require(checkBoth({}).empty(),"empty input produced a box");
    auto grid=makeGrid({Cell(-64,-1,0),Cell(63,1,2)});
    require(pathcover_boxes::compress(grid).size()==2,"separate cells merged");
    grid.clearBits();
    require(pathcover_boxes::compress(grid).empty(),"clear retained boxes");
    const occupancy_inflation::VoxelWindow window{Cell(-128,-8,-8),Cell(256,16,16)};
    require(grid.reset(window,window.wordCount()),"reset failed");
    require(pathcover_boxes::compress(grid).empty(),"reset retained boxes");
    grid.setLocal(128,8,8);
    exactUnion({Cell(0,0,0)},pathcover_boxes::compress(grid));
    grid.release();
    require(pathcover_boxes::compress(grid).empty(),"release retained boxes");
}
void solid_negative_words() {
    const Cell lower(-70,-3,-4),upper(130,4,2);
    auto cells=fill(lower,upper);
    const std::vector<Cell> duplicates(cells.begin(),cells.begin()+100);
    cells.insert(cells.end(),duplicates.begin(),duplicates.end());
    std::mt19937 generator(19);
    std::shuffle(cells.begin(),cells.end(),generator);
    const auto boxes=checkBoth(cells);
    require(boxes.size()==1,"solid cuboid was not fully merged");
    require((boxes[0].lower-lower).norm()==0 && (boxes[0].upper-upper).norm()==0,"negative or word-boundary extent changed");
    std::reverse(cells.begin(),cells.end());
    require(same(boxes,pathcover_boxes::compress(cells)),"input order affected boxes");
}
void holes() {
    auto cells=fill(Cell(-4,-3,-2),Cell(5,4,3));
    cells.erase(std::remove_if(cells.begin(),cells.end(),[](const Cell& cell){
        return (cell.x()==0 && cell.y()==0) || (cell.x()==2 && cell.y()==1 && cell.z()==0);
    }),cells.end());
    const auto boxes=checkBoth(cells);
    require(boxes.size()>1,"hole was covered by one bounding box");
}
void checkerboard() {
    std::vector<Cell> cells;
    for(int z=-3;z<3;++z) for(int y=-3;y<4;++y) for(int x=-4;x<4;++x)
        if((x+y+z)%2==0) cells.emplace_back(x,y,z);
    const auto boxes=checkBoth(cells);
    require(boxes.size()==cells.size(),"checkerboard filled an empty neighbour");
}
void footprints_and_gaps() {
    std::vector<Cell> cells;
    for(int z:{0,1,3}) {
        auto a=fill(Cell(-2,0,z),Cell(1,2,z+1));cells.insert(cells.end(),a.begin(),a.end());
        auto b=fill(Cell(-2,3,z),Cell(2,4,z+1));cells.insert(cells.end(),b.begin(),b.end());
    }
    const auto boxes=checkBoth(cells);
    require(boxes.size()==4,"footprint equality or consecutive z contract changed");
    auto narrow=fill(Cell(-2,0,4),Cell(0,2,5));cells.insert(cells.end(),narrow.begin(),narrow.end());
    require(checkBoth(cells).size()==5,"different x interval was merged");
}
void partial_word() {
    const occupancy_inflation::VoxelWindow window{Cell(-73,-2,-1),Cell(65,3,2)};
    VoxelBitGrid grid;
    require(grid.reset(window,window.wordCount()),"partial-word grid allocation");
    std::vector<Cell> cells;
    for(int z=0;z<2;++z) for(int y=0;y<3;++y) for(int x=0;x<65;++x) {
        grid.setLocal(x,y,z);cells.push_back(window.base+Cell(x,y,z));
    }
    // Padding outside the logical window is not an occupied voxel.
    for(int z=0;z<2;++z) for(int y=0;y<3;++y)
        grid.data()[window.rowOffset(y,z)+1]=~uint64_t{0};
    const auto boxes=pathcover_boxes::compress(grid);
    exactUnion(cells,boxes);
    require(boxes.size()==1,"last word broke a continuous run");
    require((boxes[0].upper-(window.base+window.dims)).norm()==0,"padding leaked into volume");
}
void adjacent_row_order() {
    std::vector<Cell> cells;
    const std::vector<std::vector<std::pair<int,int>>> rows{
        {{-70,-65},{-2,2},{7,8},{62,66}},
        {{-70,-64},{-2,2},{5,6},{62,66}},
        {{-2,1},{7,8},{63,66}},
        {},
        {{-70,-65},{-2,2},{7,8},{62,66}},
    };
    for(int z:{-1,0,2}) for(std::size_t row=0;row<rows.size();++row)
        for(const auto& interval:rows[row]) for(int x=interval.first;x<interval.second;++x)
            cells.emplace_back(x,static_cast<int>(row)-2,z);
    const auto boxes=checkBoth(cells);
    bool merged=false;
    for(const auto& box:boxes) merged=merged || (box.lower==Cell(-2,-2,-1) && box.upper==Cell(2,0,1));
    require(merged,"identical adjacent rows/planes did not merge");
    // Different ends at the same x start, missing intervals, empty rows and
    // missing z planes must not carry a stale active-row interval.
    std::mt19937 random(21);
    for(int trial=0;trial<48;++trial){
        cells.clear();
        for(int z=-2;z<3;++z)for(int y=-4;y<5;++y)for(int x=-12;x<13;++x)
            if(random()%10 < static_cast<unsigned>(1+trial%9))cells.emplace_back(x,y,z);
        checkBoth(cells);
    }
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        const std::string name=argv[1];
        if(name=="empty_and_reset") empty_and_reset();
        else if(name=="solid_negative_words") solid_negative_words();
        else if(name=="holes") holes();
        else if(name=="checkerboard") checkerboard();
        else if(name=="footprints_and_gaps") footprints_and_gaps();
        else if(name=="partial_word") partial_word();
        else if(name=="adjacent_row_order") adjacent_row_order();
        else return 3;
        std::cout<<name<<" passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def boxes_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native/occupancy_inflation"
    directory = tmp_path_factory.mktemp("pathcover_boxes")
    source = directory / "boxes.cpp"
    executable = directory / "boxes"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O2", "-std=c++17", "-I/usr/include/eigen3", f"-I{root / 'native'}",
        f"-I{vendor / 'include'}", str(source), str(vendor / "src/voxel_bit_grid.cpp"),
        "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "empty_and_reset", "solid_negative_words", "holes", "checkerboard",
    "footprints_and_gaps", "partial_word", "adjacent_row_order",
])
def test_exact_voxel_box_compression(boxes_binary, case):
    result = subprocess.run([str(boxes_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
