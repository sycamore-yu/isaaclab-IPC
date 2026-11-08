"""Local RISP must preserve the global geometry and covered path contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include "PathCover.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

using V = Eigen::Vector3d;
using A = Eigen::Matrix<double,Eigen::Dynamic,3>;
using B = Eigen::VectorXd;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}
struct Problem {
    std::vector<V> centers{V(4,3,1),V(40,0,1),V(-40,0,1),V(0,30,1),V(0,-30,1)};
    std::vector<V> widths{V::Constant(.3),V::Constant(.5),V::Constant(.5),V::Constant(.5),V::Constant(.5)};
    std::vector<V> path{V(0,0,1),V(2,0,1),V(4,0,1),V(6,0,1),V(8,0,1),V(10,0,1)};
    std::vector<V> region{V(0,0,1),V(.2,-.1,1),V(.4,0,1.2)};
    V radii=V::Constant(.25);
    A bounds_A;
    B bounds_b;
    Problem() {
        bounds_A.setZero(6,3); bounds_b.resize(6);
        for(int i=0;i<3;++i){bounds_A(i,i)=-1;bounds_A(i+3,i)=1;}
        bounds_b<<50,40,-.25,50,40,8;
    }
};
struct Result {
    std::vector<A> a,raw_a;
    std::vector<B> b,raw_b;
    std::vector<V> seeds;
    std::vector<std::vector<int>> remaining;
};
Result run(const Problem& p,double extent,bool omit_argument=false) {
    Result result;
    PathCover::setRandomSeed(0);sdlp::permutation_rng.seed(0);
    if(omit_argument)
        PathCover::pathCover<double,3>(p.centers,p.path,p.bounds_A,p.bounds_b,
            result.a,result.b,result.seeds,result.remaining,8,.01,p.widths,p.region,
            &result.raw_a,&result.raw_b,p.radii);
    else
        PathCover::pathCover<double,3>(p.centers,p.path,p.bounds_A,p.bounds_b,
            result.a,result.b,result.seeds,result.remaining,8,.01,p.widths,p.region,
            &result.raw_a,&result.raw_b,p.radii,extent);
    return result;
}
Eigen::MatrixX4d halfspaces(const A& a,const B& b) {
    Eigen::MatrixX4d h(a.rows(),4);h.leftCols<3>()=a;h.col(3)=-b;return h;
}
double violation(const A& a,const B& b,const V& point) {
    return ((a*point-b).array()/a.rowwise().norm().array()).maxCoeff();
}
std::uint64_t fingerprint(const Result& result) {
    std::uint64_t hash=UINT64_C(1469598103934665603);
    auto word=[&](std::uint64_t value){
        for(int i=0;i<8;++i){hash^=(value>>(8*i))&255;hash*=UINT64_C(1099511628211);}
    };
    auto number=[&](double value){std::uint64_t raw;std::memcpy(&raw,&value,sizeof(raw));word(raw);};
    auto matrices=[&](const std::vector<A>& a,const std::vector<B>& b){
        word(a.size());
        for(std::size_t i=0;i<a.size();++i){
            word(a[i].rows());
            for(int row=0;row<a[i].rows();++row){
                for(int j=0;j<3;++j)number(a[i](row,j));number(-b[i][row]);
            }
        }
    };
    matrices(result.a,result.b);matrices(result.raw_a,result.raw_b);
    word(result.seeds.size());for(const auto& point:result.seeds)for(int j=0;j<3;++j)number(point[j]);
    word(result.remaining.size());
    for(const auto& row:result.remaining){word(row.size());for(int count:row)word(count);}
    return hash;
}

void zero_mode() {
    const Problem problem;
    const auto explicit_zero=run(problem,0.);
    const auto default_zero=run(problem,0.,true);
    // Captured from the pre-local cached-support implementation, with this
    // deterministic fixture. The hash includes raw/reduced rows, seed and decay.
    constexpr std::uint64_t previous=UINT64_C(0xde39196c4c92fa7d);
    require(fingerprint(explicit_zero)==previous,"zero mode changed old RISP output");
    require(fingerprint(default_zero)==previous,"default mode changed old RISP output");
}

void hull_certificates_and_path() {
    const Problem problem;
    const auto result=run(problem,8.);
    require(result.a.size()>=2,"fixture did not cross a local boundary");
    require((result.seeds.back()-problem.path.back()).norm()<1e-9,"covered path lost the endpoint");
    require(result.seeds.size()==result.a.size()+1,"corridor/seed schedule changed");
    for(std::size_t k=0;k<result.a.size();++k){
        const auto h=halfspaces(result.a[k],result.b[k]);
        V inner;
        require(geo_utils::findInterior(h,inner),"corridor lacks an interior");
        Eigen::Matrix3Xd vertices;
        geo_utils::enumerateVs(h,inner,vertices,1e-9);
        V lower=result.seeds[k],upper=lower;
        if(k==0)for(const auto& point:problem.region){lower=lower.cwiseMin(point);upper=upper.cwiseMax(point);}
        lower.array()-=8.;upper.array()+=8.;
        for(int i=0;i<vertices.cols();++i){
            require(violation(result.raw_a[k],result.raw_b[k],vertices.col(i))<1e-8,"reduction enlarged raw SFC");
            require(violation(problem.bounds_A,problem.bounds_b,vertices.col(i))<1e-8,"workspace was replaced");
            require((lower-vertices.col(i)).maxCoeff()<1e-8 && (vertices.col(i)-upper).maxCoeff()<1e-8,
                    "corridor escaped its local boundary");
        }
        if(k==0)for(const auto& point:problem.region)
            require(violation(result.a[k],result.b[k],point)<1e-8,"initial seed hull was clipped");
        require(result.seeds[k+1].x()>=result.seeds[k].x(),"corridor order reversed");
        require(violation(result.a[k],result.b[k],result.seeds[k])<1e-8,"current seed is outside corridor");
        require(violation(result.a[k],result.b[k],result.seeds[k+1])<1e-8,"covered endpoint is outside corridor");
        for(const auto& point:problem.path)
            if(point.x()>=result.seeds[k].x() && point.x()<=result.seeds[k+1].x())
                require(violation(result.a[k],result.b[k],point)<1e-8,"covered input path left corridor");
        if(k){
            const auto previous=halfspaces(result.a[k-1],result.b[k-1]);
            require(geo_utils::overlapRadius(previous,h,inner)>1e-7,"adjacent corridors lost overlap");
        }
        require(result.remaining[k].front()<int(problem.centers.size()),"no remote boxes were culled");
        // Every original obstacle needs a raw separating plane, even if it
        // was omitted before RISP. Do not test only the retained subset.
        for(std::size_t i=0;i<problem.centers.size();++i){
            const pathcover_clearance::Box box{problem.centers[i]-problem.widths[i],problem.centers[i]+problem.widths[i]};
            bool separated=false;
            for(int row=0;row<result.raw_a[k].rows();++row){
                const V normal=-result.raw_a[k].row(row).transpose();
                const double minimum=-pathcover_clearance::PreparedBoxSupport(normal,problem.radii).upper(box);
                separated=separated || minimum>result.raw_b[k][row];
            }
            require(separated,"omitted original obstacle has no raw SFC certificate");
        }
    }
}

void contact_and_straddling() {
    Problem problem;
    problem.path={V(0,0,1),V(1,0,1)};problem.region={V(0,0,1),V(0,0,1),V(0,0,1)};
    const double contact=8.25,one=std::nextafter(contact,INFINITY),two=std::nextafter(one,INFINITY);
    struct Case {double center,half;bool keep;};
    // Exact contact, one-ULP uncertainty, and a provably separated two-ULP
    // case on both sides; the last box defeats center-distance-only cropping.
    for(const auto& item:std::array<Case,7>{{{contact,0.,true},{one,0.,true},{two,0.,false},
        {-contact,0.,true},{-one,0.,true},{-two,0.,false},{20.,13.,true}}}){
        problem.centers={V(item.center,0,1)};problem.widths={V(item.half,.1,.1)};
        const auto result=run(problem,8.);
        const bool retained=result.remaining.front().front()==1;
        require(retained==item.keep,"axis culling changed closed-contact semantics");
        if(!retained){
            // All quantities in these fixtures are dyadic; long double gives
            // an independent exact axis-intersection oracle at this magnitude.
            const long double lower=static_cast<long double>(item.center-item.half)-.25L;
            const long double upper=static_cast<long double>(item.center+item.half)+.25L;
            require(upper < -8.L || lower > 8.L,"an intersecting obstacle was omitted");
        }
    }
}

int main(int argc,char** argv) {
    try {
        if(argc!=2)return 2;
        const std::string name=argv[1];
        if(name=="zero_mode")zero_mode();
        else if(name=="hull_certificates_and_path")hull_certificates_and_path();
        else if(name=="contact_and_straddling")contact_and_straddling();
        else return 3;
        std::cout<<name<<" passed\n";
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
"""


@pytest.fixture(scope="module")
def local_corridor_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("pathcover_local_corridor")
    source = directory / "local_corridor.cpp"
    executable = directory / "local_corridor"
    source.write_text(SOURCE)
    result = subprocess.run([
        compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-fopenmp", "-pthread",
        "-I/usr/include/eigen3", f"-I{root / 'native'}",
        f"-I{root / 'third_party/pathcover_native/corridor'}",
        f"-I{root / 'third_party/pathcover_native/trajectory'}",
        str(source), "-lqhullcpp", "-lqhull_r", "-o", str(executable),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", [
    "zero_mode", "hull_certificates_and_path", "contact_and_straddling",
])
def test_local_corridor_geometry_contract(local_corridor_binary, case):
    result = subprocess.run([str(local_corridor_binary), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
