"""Raw-box/body-ellipsoid adapters preserve search occupancy and SFC separation."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include "pathcover_motion_seed.h"
#include "pathcover_grid.h"
#include "corridor/PathCover.hpp"
#include <iostream>

using V = Eigen::Vector3d;
using I = Eigen::Vector3i;
using occupancy_inflation::VoxelBitGrid;
using A = Eigen::Matrix<double,Eigen::Dynamic,3>;

void require(bool condition, int code) {
    if (!condition) throw std::runtime_error("box-ellipsoid adapter regression " + std::to_string(code));
}

VoxelBitGrid grid(const std::vector<I>& cells) {
    VoxelBitGrid result;
    if (cells.empty()) return result;
    I lo=cells.front(),hi=lo;
    for (const auto& cell:cells) {lo=lo.cwiseMin(cell);hi=hi.cwiseMax(cell);}
    const auto window=occupancy_inflation::snapWindow(lo,hi);
    require(result.reset(window,window.wordCount()),1);
    for (const auto& cell:cells) {
        const I local=cell-window.base;
        result.setLocal(local.x(),local.y(),local.z());
    }
    return result;
}

int main() {
    // Exact V28 failure point and its unique compatible raw source cell.
    // This is the one-box geometric fixture, not a reconstruction of the
    // unavailable complete historical NanoVoxMap snapshot.
    const I source(-11,260,27);
    auto raw=grid({source});
    std::vector<I> dilation;
    for(int z=-2;z<=2;++z)for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x)
        dilation.push_back(source+I(x,y,z));
    auto inflated=grid(dilation);
    const V radii=V::Constant(.2),lower(-2.8,24.2,.2),upper(1.8,29.8,3.8);
    const V p0(-.8190380334854126,26.266569137573242,2.998993158340454);
    const pathcover_clearance::Box box{source.cast<double>()*.1,(source+I::Ones()).cast<double>()*.1};
    pathcover_endpoint::FineGeometry legacy(inflated,.1,lower,upper);
    pathcover_endpoint::FineGeometry exact(inflated,.1,lower,upper,&raw,radii);
    require(!legacy.pointFree(p0) && exact.pointFree(p0),2);
    require(pathcover_clearance::pointBoxClosest(p0,box).squared_distance>.2*.2,3);
    const V contact(-.8,26.05,2.75);
    require(!exact.pointFree(contact),4);
    const std::vector<uint64_t> raw_words(raw.data(),raw.data()+raw.wordCount());
    const std::vector<uint64_t> inflated_words(inflated.data(),inflated.data()+inflated.wordCount());

    const V origin(-3,24,0);
    const I dimensions(20,24,16),origin_index(-12,96,0);
    std::vector<signed char> cells(dimensions.prod(),0);
    std::vector<int> indices;
    pathcover_grid::coarsenOccupiedVolumes(inflated,{2,5},origin_index,dimensions,cells,indices);
    const auto original_cells=cells;
    pathcover_endpoint::CoarseGrid coarse{cells,dimensions,origin,.25};
    require(!coarse.free(coarse.index(p0)),5);
    const auto connected=pathcover_endpoint::connect(p0,exact,coarse,1.0);
    require(!connected.endpoint_in_coarse_free_cell && !connected.anchors.empty(),6);
    for (const auto& anchor:connected.anchors) {
        require(coarse.free(anchor.coarse_index),7);
        for(std::size_t i=1;i<anchor.points.size();++i)
            require(pathcover_clearance::segmentFree(anchor.points[i-1],anchor.points[i],box,radii),8);
    }
    require(cells==original_cells && std::equal(raw_words.begin(),raw_words.end(),raw.data())
        && std::equal(inflated_words.begin(),inflated_words.end(),inflated.data()),9);
    // Non-lattice endpoint links must test their full segments: free endpoints
    // alone do not certify a connector or a line-of-sight simplification.
    const V a(-1.4,26.05,2.75),b(-.7,26.05,2.75);
    require(exact.pointFree(a) && exact.pointFree(b) && !exact.segmentFree(a,b),10);
    // Six-axis neighboring fine centers are a narrower case: same-lattice raw
    // boxes and an axis-aligned ellipsoid make two free centers sufficient.
    for(int z=-3;z<=3;++z)for(int y=-3;y<=3;++y)for(int x=-3;x<=3;++x) {
        const I index=source+I(x,y,z);
        const V first=exact.center(index);
        if(!exact.centerFree(index))continue;
        for(int axis=0;axis<3;++axis) {
            const I next=index+I::Unit(axis);
            if(exact.centerFree(next))
                require(pathcover_clearance::segmentFree(first,exact.center(next),box,radii),19);
        }
    }

    const std::vector<V> seed{p0,
        V(-.73049842728415,26.356514930237807,2.965623288247394),
        V(-.6391069224359335,26.464028352218023,2.9189673365366264)};
    const pathcover_motion_seed::Triangle preferred{seed[0],seed[1],seed[2]};
    const pathcover_motion_seed::TipSegment tips{seed[2],seed[2]};
    const auto unchanged=pathcover_motion_seed::select(preferred,tips,exact);
    require(!unchanged.adjusted,11);
    for(int i=0;i<3;++i)require((unchanged.points[i].array()==preferred[i].array()).all(),12);

    A bounds(6,3),faces,raw_faces;
    bounds.topRows(3)=-Eigen::Matrix3d::Identity();bounds.bottomRows(3)=Eigen::Matrix3d::Identity();
    Eigen::VectorXd limits(6),offsets,raw_offsets;
    limits.head(3)=-lower;limits.tail(3)=upper;
    const std::vector<V> centers{(box.lower+box.upper)*.5};
    const std::vector<V> halves{(box.upper-box.lower)*.5};
    std::vector<int> remaining;
    PathCover::setRandomSeed(0);
    PathCover::RISP<double,3>(centers,p0,bounds,limits,faces,offsets,remaining,.01,halves,
        &seed,&raw_faces,&raw_offsets,radii);
    for(const auto& p:seed)require((faces*p-offsets).maxCoeff()<1e-10,13);
    bool certificate=false;
    for(int i=0;i<raw_faces.rows();++i) {
        const V n=raw_faces.row(i);
        if(-pathcover_clearance::boxSupportUpper(-n,box,radii)>raw_offsets[i])certificate=true;
    }
    require(certificate,14);
    bool rejected=false;
    const std::vector<V> collision(3,contact);
    try { PathCover::RISP<double,3>(centers,contact,bounds,limits,faces,offsets,remaining,
        .01,halves,&collision,nullptr,nullptr,radii); }
    catch(const PathCover::SeedRegionCollision&) {rejected=true;}
    require(rejected,15);

    const pathcover_motion_seed::Box obstacle{V(0,0,0),V(1,1,1)};
    const pathcover_motion_seed::Triangle blocked{V(-1,-.5,.5),V(-.8,-.5,.5),V(2,.5,.5)};
    const pathcover_motion_seed::TipSegment range{V(2,-.5,.5),V(2,2,.5)};
    const auto selected=pathcover_motion_seed::select(blocked,range,{obstacle},V::Constant(-3),V::Constant(3),radii);
    require(selected.adjusted && selected.alpha>=0 && selected.alpha<=1,16);
    const auto clearance=pathcover_clearance::triangleClosest(selected.points,{obstacle.lower,obstacle.upper},radii);
    require(clearance.free && clearance.seed_support-clearance.obstacle_support>1e-10,17);

    // The all-zero body is the existing closed raw-AABB geometry, including
    // the original SAT/LP and default RISP arithmetic.
    pathcover_endpoint::FineGeometry zero(raw,.1,lower,upper,&raw,V::Zero());
    pathcover_endpoint::FineGeometry old_zero(raw,.1,lower,upper);
    for(const auto& p:std::array<V,3>{p0,contact,(box.lower+box.upper)*.5})
        require(zero.pointFree(p)==old_zero.pointFree(p),18);

    // Integer endpoints and an outward-rounded broad phase must not skip the
    // exact closed-contact test. Center +/- half shrinks the first raw face;
    // even integer multiplication alone shrinks the positive-radius example.
    auto raw_zero=grid({I(1,0,0)});
    pathcover_endpoint::FineGeometry zero_contact(raw_zero,.1,V::Constant(-2),V::Constant(2),
        &raw_zero,V::Zero());
    require(.1 < (1.5*.1)-.05,20);
    require(!zero_contact.pointFree(V(.1,.05,.05)),21);
    require(!zero_contact.segmentFree(V(.1,-.1,.05),V(.1,.2,.05)),22);
    require(zero_contact.pointFree(V(.1-1e-8,.05,.05)),23);
    auto raw_positive=grid({I(9,0,0)});
    std::vector<I> positive_dilation;
    for(int z=-2;z<=2;++z)for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x)
        positive_dilation.push_back(I(9+x,y,z));
    auto inflated_positive=grid(positive_dilation);
    pathcover_endpoint::FineGeometry positive_contact(inflated_positive,.1,V::Constant(-2),V::Constant(2),
        &raw_positive,radii);
    require(7*.1 > .7,24);
    require(!positive_contact.pointFree(V(.7,.05,.05)),25);
    require(!positive_contact.segmentFree(V(.7,-.1,.05),V(.7,.2,.05)),26);
    require(positive_contact.pointFree(V(.7-1e-8,.05,.05)),27);
    bool seed_contact_rejected=false;
    const V touching(.7,.05,.05);
    try { pathcover_motion_seed::select({touching,touching,touching},
        {touching,touching},positive_contact); }
    catch(const pathcover_motion_seed::NoFeasibleSeed&) {seed_contact_rejected=true;}
    require(seed_contact_rejected,28);
    std::cout << "box-ellipsoid adapters passed; anchors " << connected.anchors.size() << "\n";
}
"""


def test_cpp_box_ellipsoid_geometry_adapters(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native"
    source = tmp_path / "box_ellipsoid_adapters.cpp"
    executable = tmp_path / "box_ellipsoid_adapters"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-O2", "-std=c++17", "-I/usr/include/eigen3", f"-I{root / 'native'}",
        f"-I{vendor}", f"-I{vendor / 'trajectory'}", f"-I{vendor / 'occupancy_inflation/include'}",
        str(source), str(vendor / "occupancy_inflation/src/voxel_bit_grid.cpp"),
        "-lqhullcpp", "-lqhull_r", "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
