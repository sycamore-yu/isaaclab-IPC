"""Continuous S3 position objective, envelope derivatives and legacy contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "gcopter_solver.hpp"

using Solver = trajectory_server::GcopterSolver<3>;
using Coeff = Eigen::Matrix<double,6,3>;
using pathcover_continuous_position::maximum;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void near(double a, double b, double tolerance, const char* message) {
    require(std::isfinite(a) && std::isfinite(b) &&
            std::abs(a-b) <= tolerance*std::max({1.,std::abs(a),std::abs(b)}), message);
}
struct Evaluation {
    double cost = 0;
    Eigen::VectorXd time = Eigen::VectorXd::Zero(1);
    Eigen::MatrixX3d coeff = Eigen::MatrixX3d::Zero(6,3);
    Solver::ConstraintMetrics metrics;
};
Evaluation penalty(const Coeff& c, double duration, const Eigen::MatrixX4d& planes,
                   bool continuous, bool omit_argument=false) {
    Evaluation result;
    flatness::FlatnessMap map;
    map.reset(.75,9.81,0,0,0,.001);
    Eigen::VectorXd times=Eigen::VectorXd::Constant(1,duration), bounds(5), weights(5);
    Eigen::VectorXi indices=Eigen::VectorXi::Zero(1);
    bounds << 100,100,M_PI,0,1000;
    weights << 1e5,0,0,0,0;
    const Solver::PolyhedraH polygons{planes};
    if (omit_argument)
        Solver::attachPenaltyFunctional<true>(times,c,indices,polygons,.01,64,
            bounds,weights,map,result.metrics,1e-6,polygons,result.cost,result.time,result.coeff);
    else
        Solver::attachPenaltyFunctional<true>(times,c,indices,polygons,.01,64,
            bounds,weights,map,result.metrics,1e-6,polygons,result.cost,result.time,result.coeff,continuous);
    return result;
}
Eigen::MatrixX4d face(double offset=0) {
    Eigen::MatrixX4d h(1,4); h << 0,1,0,offset; return h;
}
void polynomials() {
    Coeff c=Coeff::Zero();
    c(0,1)=-.3; c(0,2)=1;
    Eigen::MatrixX4d h(3,4);
    h << 0,1,0,0, 0,1,0,-1, 0,1,0,0;
    auto m=maximum(c,1.,h);
    require(m.finite && m.value==-.3 && m.face==0 && m.alpha==0, "negative constant/tie changed");
    c(0,1)=.01; c(1,1)=.02;
    m=maximum(c,1.,h);
    near(m.value,.03,1e-14,"endpoint maximum"); require(m.alpha==1.,"endpoint time");
    c(0,1)=.02; c(1,1)=-.03;
    m=maximum(c,1.,h);
    near(m.value,.02,1e-14,"initial maximum"); require(m.alpha==0.,"initial time");
    c.setZero(); c(0,2)=1;
    // .01 - (t-.5)^4: a repeated stationary root.
    c.col(1) << -.0525,.5,-1.5,2.,-1.,0.;
    m=maximum(c,1.,h);
    near(m.value,.01,1e-12,"repeated stationary maximum");
    c(3,0)=std::numeric_limits<double>::quiet_NaN();
    require(!maximum(c,1.,h).finite,"nonfinite coefficient hidden");
    c.setZero(); c(0,2)=1; h(2,3)=std::numeric_limits<double>::infinity();
    require(!maximum(c,1.,h).finite,"later nonfinite plane hidden");
    h=face(); c(4,1)=1e308;
    require(!maximum(c,1.,h).finite,"derivative overflow hidden");
    const std::array<double,6> overflow{1e308,-1e308,1e308,-1e308,0.,0.};
    require(!std::isfinite(pathcover_continuous_position::bernsteinEvaluationUpperBound(overflow)),
            "uncertain Bernstein bound did not retain roots");
}
void objective_and_derivatives() {
    // The exact positive maximum is strictly between adjacent 65-node samples.
    const double tstar=25.5/64., delta=5e-5;
    Coeff c=Coeff::Zero(); c(0,2)=1;
    c(0,1)=delta-tstar*tstar; c(1,1)=2*tstar; c(2,1)=-1;
    const auto h=face();
    const auto integral=penalty(c,1.,h,false);
    const auto exact=penalty(c,1.,h,true);
    require(integral.cost==0 && integral.metrics.sfc<0, "sampling fixture does not isolate missed peak");
    near(exact.cost,1e5*(delta-.5e-6),1e-10,"continuous penalty missed the between-node maximum");
    require(exact.metrics.sfc==integral.metrics.sfc,"quadrature diagnostic changed semantics");
    require(exact.metrics.finite && exact.coeff.allFinite(),"finite continuous gradient rejected");
    for (bool endpoint : {false,true}) {
        Coeff current=c;
        double duration=1.3;
        if (endpoint) { current.setZero(); current(0,2)=1; current(0,1)=.01; current(1,1)=.02; }
        const auto center=penalty(current,duration,h,true);
        Coeff direction;
        for (int i=0;i<6;++i) for(int d=0;d<3;++d) direction(i,d)=std::sin(3*i+d+.31);
        direction.normalize();
        const double step=1e-7;
        const double fd_c=(penalty(current+step*direction,duration,h,true).cost-
                           penalty(current-step*direction,duration,h,true).cost)/(2*step);
        near(fd_c,center.coeff.cwiseProduct(direction).sum(),2e-6,"coefficient envelope derivative");
        const double fd_t=(penalty(current,duration+step,h,true).cost-
                           penalty(current,duration-step,h,true).cost)/(2*step);
        near(fd_t,center.time(0),2e-6,"explicit-time envelope derivative");
    }
}
void legacy_and_templates() {
    Coeff c=Coeff::Zero(); c(0,1)=.02; c(0,2)=1;
    const auto h=face();
    const auto original_default=penalty(c,.3,h,false,true);
    const auto explicit_integral=penalty(c,.3,h,false);
    require(std::memcmp(&original_default.cost,&explicit_integral.cost,sizeof(double))==0 &&
            std::memcmp(original_default.time.data(),explicit_integral.time.data(),sizeof(double))==0 &&
            std::memcmp(original_default.coeff.data(),explicit_integral.coeff.data(),18*sizeof(double))==0,
            "default integral changed");
    near(original_default.cost,1e5*(.02-.5e-6)*.3,1e-12,"integral duration factor changed");
    Solver solver;
    require(!solver.continuousMaxPositionPenalty,"continuous objective was enabled implicitly");
    solver.setContinuousMaxPositionPenalty(true);
    require(solver.continuousMaxPositionPenalty && !solver.getInitialForce().enabled,"mode coupled to free force");
    solver.setContinuousMaxPositionPenalty(false);
    require(!solver.continuousMaxPositionPenalty,"mode cannot be disabled");
    [[maybe_unused]] auto instantiate_s4=&trajectory_server::GcopterSolver<4>::optimize;
    trajectory_server::GcopterSolver<4> s4;
    s4.setContinuousMaxPositionPenalty(false);
    bool rejected=false;
    try { s4.setContinuousMaxPositionPenalty(true); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"unsupported S4 mode silently removed the position objective");
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"one contract name required");
        const std::string mode=argv[1];
        if(mode=="polynomials") polynomials();
        else if(mode=="derivatives") objective_and_derivatives();
        else if(mode=="legacy") legacy_and_templates();
        else throw std::runtime_error("unknown contract");
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
"""


@pytest.fixture(scope="module")
def continuous_contract(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/pathcover_native/trajectory"
    folder = tmp_path_factory.mktemp("continuous-position")
    source, binary = folder / "contract.cpp", folder / "contract"
    source.write_text(SOURCE)
    header = (vendor / "gcopter_solver.hpp").read_text()
    pivot = header.index("#elif PATHCOVER_DIM == 3")
    (folder / "gcopter_solver.hpp").write_text(
        header[:pivot] + header[pivot:].replace("    private:", "    public:")
    )
    completed = subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
         "-I/usr/include/eigen3", f"-I{folder}", f"-I{vendor}", str(source), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return binary


@pytest.mark.parametrize("mode", ["polynomials", "derivatives", "legacy"])
def test_continuous_position_contract(continuous_contract, mode):
    completed = subprocess.run([str(continuous_contract), mode], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_planar_default_templates_still_compile(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    vendor = Path(__file__).resolve().parents[1] / "third_party/pathcover_native/trajectory"
    source = tmp_path / "planar.cpp"
    source.write_text(
        '#include <cstdint>\n#include "gcopter_solver.hpp"\n'
        'int main() { [[maybe_unused]] auto a=&trajectory_server::GcopterSolver<3>::optimize;'
        ' [[maybe_unused]] auto b=&trajectory_server::GcopterSolver<4>::optimize; }\n'
    )
    completed = subprocess.run(
        [compiler, "-O2", "-std=c++17", "-DPATHCOVER_DIM=2",
         "-I/usr/include/eigen3", f"-I{vendor}", str(source), "-o", str(tmp_path / "planar")],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
