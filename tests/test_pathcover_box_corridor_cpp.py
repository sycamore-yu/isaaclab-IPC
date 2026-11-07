"""RISP box support and equivalent facet reduction, without planning or MINCO."""

import shutil
import subprocess
from pathlib import Path

import pytest


SOURCE = r"""
#include "corridor/PathCover.hpp"
#include <iostream>

using V = Eigen::Vector3d;
using AType = Eigen::Matrix<double,Eigen::Dynamic,3>;

bool equivalent(const AType& raw_A, const Eigen::VectorXd& raw_b,
                const AType& A, const Eigen::VectorXd& b) {
    // Retained rows give raw subset reduced; vertices certify the reverse.
    for (int j=0;j<A.rows();++j) {
        bool found=false;
        for (int k=0;k<raw_A.rows();++k)
            if ((A.row(j).array()==raw_A.row(k).array()).all() && b[j]==raw_b[k]) found=true;
        if (!found) return false;
    }
    Eigen::MatrixX4d h(A.rows(),4); h.leftCols<3>()=A; h.col(3)=-b;
    V interior;
    if (!geo_utils::findInterior(h,interior)) return false;
    Eigen::Matrix3Xd vertices;
    geo_utils::enumerateVs(h,interior,vertices,1e-9);
    if (vertices.cols()<4 || !vertices.allFinite()) return false;
    Eigen::MatrixXd residual=(raw_A*vertices).colwise()-raw_b;
    residual.array().colwise()/=raw_A.rowwise().norm().array();
    return residual.maxCoeff()<1e-9;
}

bool certificates(const AType& raw_A, const Eigen::VectorXd& raw_b,
                  const std::vector<V>& centers, const std::vector<V>& halves) {
    for (std::size_t i=0;i<centers.size();++i) {
        const Eigen::VectorXd gap=raw_A*centers[i]-raw_b-raw_A.cwiseAbs()*halves[i];
        if (gap.maxCoeff()<=0.0) return false;
    }
    return true;
}

int main() {
    AType bounds(6,3), A, raw_A;
    bounds.topRows<3>()=V::Ones().asDiagonal(); bounds.bottomRows<3>()=-Eigen::Matrix3d::Identity();
    Eigen::VectorXd limits=Eigen::VectorXd::Constant(6,3.0), b, raw_b;
    std::vector<int> decay;
    const std::vector<V> centers{{2,0,0},{-1,1,0},{0,-2,.2},{.2,.2,2}};
    const std::vector<V> halves{{.7,.2,.4},{.2,.6,.5},{.3,.7,.4},{.5,.3,.8}};
    const std::vector<V> motion{{0,0,0},{.15,.05,.05},{.3,.1,.1}};
    PathCover::setRandomSeed(11);
    PathCover::RISP<double,3>(centers,motion[0],bounds,limits,A,b,decay,.01,halves,&motion,&raw_A,&raw_b);
    if (!certificates(raw_A,raw_b,centers,halves) || !equivalent(raw_A,raw_b,A,b)) return 1;
    for (const auto& point:motion) if ((A*point-b).maxCoeff()>1e-10) return 2;
    auto collision=motion; collision[2]=centers[0];
    bool rejected=false;
    try {
        PathCover::RISP<double,3>(centers,motion[0],bounds,limits,A,b,decay,.01,halves,&collision);
    } catch (const PathCover::SeedRegionCollision&) { rejected=true; }
    if (!rejected) return 3;

    // Uniform and per-box APIs execute identical support arithmetic.
    AType uniform_A; Eigen::VectorXd uniform_b;
    PathCover::setRandomSeed(23);
    PathCover::RISP<double,3>(centers,motion[0],bounds,limits,uniform_A,uniform_b,decay,.01,.1);
    PathCover::setRandomSeed(23);
    PathCover::RISP<double,3>(centers,motion[0],bounds,limits,A,b,decay,.01,std::vector<V>(4,V::Constant(.1)));
    if (A.rows()!=uniform_A.rows() || !(A.array()==uniform_A.array()).all()
        || !(b.array()==uniform_b.array()).all()) return 4;

    // A workspace-boundary seed is legal even though it is not a dual origin.
    const V boundary(3,0,3);
    PathCover::RISP<double,3>({},boundary,bounds,limits,A,b,decay,.01,.1,nullptr,&raw_A,&raw_b);
    if (A.rows()!=6 || (A*boundary-b).maxCoeff()>1e-12 || !equivalent(raw_A,raw_b,A,b)) return 5;

    // A redundant x<=1 face separates this whole box. After removing it,
    // y>=0 and x+y<=1 jointly exclude the box; neither alone is a certificate.
    raw_A.resize(7,3); raw_b.resize(7);
    raw_A << -1,0,0, 0,-1,0, 1,1,0, 0,0,1, 0,0,-1, 1,0,0, 0,1,0;
    raw_b << 0,0,1,1,1,1,1;
    A=raw_A; b=raw_b;
    PathCover::noredund<double,3>(A,b,V(.2,.2,0));
    const std::vector<V> far_box{{1.2,0,0}}, far_half{{.1,1,.1}};
    if (A.rows()!=5 || !certificates(raw_A,raw_b,far_box,far_half)
        || certificates(A,b,far_box,far_half) || !equivalent(raw_A,raw_b,A,b)) return 6;

    std::vector<AType> As, raw_As;
    std::vector<Eigen::VectorXd> bs, raw_bs;
    std::vector<V> seeds;
    std::vector<std::vector<int>> histories;
    PathCover::setRandomSeed(11);
    PathCover::pathCover<double,3>(centers,{motion[0],V(.5,0,0)},bounds,limits,
        As,bs,seeds,histories,3,.01,halves,motion,&raw_As,&raw_bs);
    if (As.empty() || As.size()!=raw_As.size() || bs.size()!=raw_bs.size()) return 7;
    for (std::size_t i=0;i<As.size();++i)
        if (!certificates(raw_As[i],raw_bs[i],centers,halves)
            || !equivalent(raw_As[i],raw_bs[i],As[i],bs[i])) return 8;
    std::cout << "anisotropic boxes, boundary seed, raw certificates and facet equivalence passed\n";
}
"""


def test_cpp_box_support_and_equivalent_redundancy_removal(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "box_corridor.cpp"
    executable = tmp_path / "box_corridor"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-O1", "-std=c++17", "-I/usr/include/eigen3",
        f"-I{root / 'third_party/pathcover_native'}", str(source),
        "-lqhullcpp", "-lqhull_r", "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
