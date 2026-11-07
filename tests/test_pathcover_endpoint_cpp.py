"""Fine geometry and bounded endpoint links for the observed Dense failure."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include "pathcover_endpoint.h"
#include "pathcover_grid.h"
#include "corridor/PathCover.hpp"
#include <iostream>

using V = Eigen::Vector3d;
using I = Eigen::Vector3i;
using occupancy_inflation::VoxelBitGrid;
using namespace pathcover_endpoint;

VoxelBitGrid makeGrid(const std::vector<I>& indices) {
    VoxelBitGrid grid;
    if (indices.empty()) return grid;
    I first = indices.front(), last = first;
    for (const auto& index : indices) { first = first.cwiseMin(index); last = last.cwiseMax(index); }
    const auto window = occupancy_inflation::snapWindow(first, last);
    if (!grid.reset(window, window.wordCount())) throw std::runtime_error("grid allocation");
    for (const auto& index : indices) {
        const I local = index-window.base;
        grid.setLocal(local.x(),local.y(),local.z());
    }
    return grid;
}

void require(bool condition, int code) {
    if (!condition) throw std::runtime_error("endpoint regression " + std::to_string(code));
}

int main() {
    const std::vector<I> source{{0,-379,12},{0,-378,12},{0,-379,13},{0,-378,13},
                                {0,-380,14},{0,-379,14},{0,-378,14}};
    auto fine_grid = makeGrid(source);
    const V lower(-7.3,-54.8,.2), upper(7.3,54.8,3.3), origin(-7.5,-55.,0.);
    const I dimensions(60,440,14), origin_index(-30,-220,0);
    std::vector<signed char> cells(dimensions.prod(),0);
    const V search_lower(-7.25,-54.75,.25), search_upper(7.25,54.75,3.25);
    for (int z=0;z<dimensions.z();++z) for(int y=0;y<dimensions.y();++y) for(int x=0;x<dimensions.x();++x) {
        const V center=origin+(V(x,y,z).array()+.5).matrix()*.25;
        if ((center.array()<search_lower.array()).any() || (center.array()>search_upper.array()).any())
            cells[x+dimensions.x()*(y+dimensions.y()*z)]=100;
    }
    std::vector<int> occupied;
    pathcover_grid::coarsenOccupiedVolumes(fine_grid,{2,5},origin_index,dimensions,cells,occupied);
    const auto original_cells=cells;
    FineGeometry fine(fine_grid,.1,lower,upper);
    CoarseGrid coarse{cells,dimensions,origin,.25};
    const std::vector<V> seed{
        V(.2945846915245056,-38.26182556152344,1.3028671741485596),
        V(.2936957584915512,-38.11167442111221,1.330392438489267),
        V(.2399769759157811,-37.98323914461027,1.3914357090867218)};
    require(!coarse.free(coarse.index(seed[2])) && fine.pointFree(seed[2]),1);
    for (int i=0;i<3;++i) require(fine.segmentFree(seed[i],seed[(i+1)%3]),2);
    const auto normal=connect(seed[0],fine,coarse,1.0);
    require(normal.endpoint_in_coarse_free_cell && normal.anchors.size()==1 && normal.visited_cells==0,3);
    const auto connected=connect(seed[2],fine,coarse,1.0);
    require(!connected.endpoint_in_coarse_free_cell && !connected.anchors.empty(),4);
    require(connected.visited_cells<=9261,5);
    for(const auto& anchor:connected.anchors) {
        require(coarse.free(anchor.coarse_index),6);
        require((anchor.points.front()-seed[2]).norm()==0.0,7);
        for(std::size_t i=1;i<anchor.points.size();++i)
            require(fine.segmentFree(anchor.points[i-1],anchor.points[i]),8);
    }
    require(cells==original_cells,9);
    require(connect(seed[2],fine,coarse,.001).anchors.empty(),10);
    require(connect(V(.05,-37.95,1.45),fine,coarse,1.0).anchors.empty(),11);

    // Preserve the three-point initial hull, but use authoritative fine volumes.
    Eigen::Matrix<double,Eigen::Dynamic,3> A(6,3), output;
    A.topRows(3)=-Eigen::Matrix3d::Identity();A.bottomRows(3)=Eigen::Matrix3d::Identity();
    Eigen::VectorXd b(6), offsets;b.head(3)=-lower;b.tail(3)=upper;
    std::vector<int> remaining;
    bool coarse_rejected=false;
    try { PathCover::RISP<double,3>({V(.125,-37.875,1.375)},seed[0],A,b,output,offsets,remaining,.01,.125,&seed); }
    catch(const PathCover::SeedRegionCollision&) { coarse_rejected=true; }
    require(coarse_rejected,12);
    const auto centers=fine.occupiedCenters();
    PathCover::setRandomSeed(0);
    PathCover::RISP<double,3>(centers,seed[0],A,b,output,offsets,remaining,.01,.05,&seed);
    for(const auto& point:seed) require((output*point-offsets).maxCoeff()<1e-10,13);
    for(const auto& center:centers)
        require((output*center-offsets-.05*output.cwiseAbs().rowwise().sum()).maxCoeff()>0,14);

    // Exact physical ceiling 3.30 m is distinct from the coarse 3.25 m boundary.
    const V boundary(1.0,-38.0,3.28);
    require(fine.pointFree(boundary) && !coarse.free(coarse.index(boundary)),15);
    const auto boundary_links=connect(boundary,fine,coarse,1.0);
    require(!boundary_links.anchors.empty(),16);
    for(const auto& anchor:boundary_links.anchors) for(const auto& point:anchor.points)
        require(point.z()<=3.3,17);
    require(!fine.pointFree(V(1.,-38.,3.31)),18);

    // Slab tests include exact face/edge/corner contacts, in both directions.
    for(const V offset : {V(0,0,0),V(-2,-3,-4)}) {
        const V box=offset+V(.5,.5,.5);
        require(FineGeometry::segmentIntersectsBox(offset+V(-1,.5,.5),offset+V(2,.5,.5),box,.5),19);
        require(FineGeometry::segmentIntersectsBox(offset+V(-1,0,0),offset+V(2,0,0),box,.5),20);
        require(FineGeometry::segmentIntersectsBox(offset+V(-1,-1,-1),offset,box,.5),21);
        require(!FineGeometry::segmentIntersectsBox(offset+V(-1,-.01,0),offset+V(2,-.01,0),box,.5),22);
    }

    // A bounded local route must turn around an obstacle; direct LOS is blocked.
    std::vector<I> wall;
    for(int y=-2;y<=3;++y)for(int z=6;z<=15;++z)wall.emplace_back(4,y,z);
    auto bent_grid=makeGrid(wall);
    FineGeometry bent_fine(bent_grid,.1,V(-2,-2,0),V(2,2,2));
    const I bent_dimensions(16,16,8), anchor_index(11,8,4);
    std::vector<signed char> bent_cells(bent_dimensions.prod(),100);
    CoarseGrid bent_coarse{bent_cells,bent_dimensions,V(-2,-2,0),.25};
    bent_cells[bent_coarse.linear(anchor_index)]=0;
    const V endpoint(.05,.05,1.05);
    require(!bent_fine.segmentFree(endpoint,bent_coarse.center(anchor_index)),23);
    const auto bent=connect(endpoint,bent_fine,bent_coarse,1.0);
    require(bent.anchors.size()==1 && bent.anchors[0].points.size()>2,24);
    for(std::size_t i=1;i<bent.anchors[0].points.size();++i)
        require(bent_fine.segmentFree(bent.anchors[0].points[i-1],bent.anchors[0].points[i]),25);

    // Do not stop at the first free coarse component. The fine route crosses
    // occupied -> free A -> occupied -> free B, inside the same fixed window.
    auto bridge_grid=makeGrid({I(0,0,0),I(5,0,0)});
    FineGeometry bridge_fine(bridge_grid,.1,V(0,0,0),V(1,.25,.25));
    std::vector<signed char> bridge_cells(4,0);
    std::vector<int> bridge_indices;
    pathcover_grid::coarsenOccupiedVolumes(bridge_grid,{2,5},I::Zero(),I(4,1,1),bridge_cells,bridge_indices);
    CoarseGrid bridge_coarse{bridge_cells,I(4,1,1),V::Zero(),.25};
    const auto bridge=connect(V(.15,.18,.18),bridge_fine,bridge_coarse,1.0);
    require(bridge.anchors.size()==2,26);
    require(bridge.anchors[0].coarse_index.x()==1 && bridge.anchors[1].coarse_index.x()==3,27);
    std::cout << "fine geometry, bounded links, ceiling and bent connection passed; Dense anchors "
              << connected.anchors.size() << ", visited " << connected.visited_cells << "\n";
}
"""


def test_cpp_fine_geometry_and_endpoint_connections(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native"
    source = tmp_path / "pathcover_endpoint.cpp"
    executable = tmp_path / "pathcover_endpoint"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-O2", "-std=c++17", "-I/usr/include/eigen3", f"-I{root / 'native'}",
        f"-I{vendor}", f"-I{vendor / 'occupancy_inflation/include'}", str(source),
        str(vendor / "occupancy_inflation/src/voxel_bit_grid.cpp"), "-lqhullcpp", "-lqhull_r",
        "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
