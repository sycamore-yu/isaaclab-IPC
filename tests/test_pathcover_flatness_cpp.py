"""Finite zero-drag flatness and nonfinite penalty rejection contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <math.h>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "gcopter_solver.hpp"

using Solver = trajectory_server::GcopterSolver<3>;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class A, class B>
void close(const A &a, const B &b, const char *message) {
    require(a.allFinite() && b.allFinite(), message);
    for (int i=0; i<a.size(); ++i)
        require(std::abs(a[i]-b[i]) <= 2e-13*std::max({1.,std::abs(a[i]),std::abs(b[i])}), message);
}

void finite_flatness() {
    const Eigen::Vector3d velocity(2.,-1.,.3), jerk(7.,-4.,2.);
    const Eigen::Vector3d gp(.2,-.4,.3), gv(-.1,.3,.7), go(.3,-.5,.2);
    const Eigen::Vector4d gq(.2,-.3,.4,.5);
    for (const double gravity : {9.81,0.}) {
        for (const Eigen::Vector3d acceleration :
             {Eigen::Vector3d(1.,-.7,.6),Eigen::Vector3d(-.0,0.,1.)}) {
            flatness::FlatnessMap generic, fast;
            generic.reset(.75,gravity,0.,0.,0.,.001);
            fast.reset(.75,gravity,0.,0.,0.,.001);
            require(fast.supportsZeroDragZeroYaw(),"valid zero-drag specialization unavailable");
            double old_thrust,new_thrust,psi,dpsi;
            Eigen::Vector4d old_quat,new_quat;
            Eigen::Vector3d old_omega,new_omega,op,ov,oa,oj,np,nv,na,nj;
            generic.forward(velocity,acceleration,jerk,0.,0.,old_thrust,old_quat,old_omega);
            fast.forwardZeroDragZeroYaw(acceleration,jerk,new_thrust,new_quat,new_omega);
            require(old_thrust==new_thrust,"finite thrust changed");
            close(old_quat,new_quat,"finite quaternion changed");
            close(old_omega,new_omega,"finite body rate changed");
            generic.backward(gp,gv,.7,gq,go,op,ov,oa,oj,psi,dpsi);
            fast.backwardZeroDragZeroYaw(gp,gv,.7,gq,go,np,nv,na,nj);
            close(op,np,"position gradient changed"); close(ov,nv,"velocity gradient changed");
            close(oa,na,"acceleration gradient changed"); close(oj,nj,"jerk gradient changed");
        }
    }
    for (int coefficient=0;coefficient<3;++coefficient) {
        flatness::FlatnessMap map;
        map.reset(.75,9.81,coefficient==0?.1:0.,coefficient==1?.1:0.,coefficient==2?.1:0.,.001);
        require(!map.supportsZeroDragZeroYaw(),"nonzero drag selected zero-drag kernel");
    }
    flatness::FlatnessMap invalid;
    invalid.reset(0.,9.81,0.,0.,0.,.001);
    require(!invalid.supportsZeroDragZeroYaw(),"zero mass selected specialization");
    invalid.reset(.75,9.81,0.,0.,0.,0.);
    require(!invalid.supportsZeroDragZeroYaw(),"undefined velocity smoothing selected specialization");
}

struct Evaluation {
    double cost, time_gradient;
    bool gradient_finite, quality_finite, values_finite;
    bool rejected() const {
        return !std::isfinite(cost) || !quality_finite || !values_finite;
    }
};
template<bool Specialized>
Evaluation penalty(const Eigen::MatrixX3d &coefficients, double duration=.3,
                   double gravity=9.81, double rate_weight=1e4) {
    flatness::FlatnessMap map;
    map.reset(.75,gravity,0.,0.,0.,.001);
    Eigen::VectorXd times=Eigen::VectorXd::Constant(1,duration);
    Eigen::VectorXi indices=Eigen::VectorXi::Zero(1);
    Solver::PolyhedraH planes(1,Eigen::MatrixX4d(6,4));
    planes[0] << 1,0,0,-100,-1,0,0,-100,0,1,0,-100,
                 0,-1,0,-100,0,0,1,-100,0,0,-1,-100;
    Eigen::VectorXd bounds(5),weights(5);
    bounds << 3.,2.99,.7854,2.094972067039106,10.47486033519553;
    weights << 1e5,100.,rate_weight,1e4,1e4;
    Eigen::VectorXd gradT=Eigen::VectorXd::Zero(1);
    Eigen::MatrixX3d gradC=Eigen::MatrixX3d::Zero(6,3);
    Solver::ConstraintMetrics metrics;
    double cost=0.;
    Solver::attachPenaltyFunctional<Specialized>(times,coefficients,indices,planes,.01,64,
        bounds,weights,map,metrics,1e-6,planes,cost,gradT,gradC);
    return {cost,gradT[0],gradT.allFinite() && gradC.allFinite(),metrics.finite,metrics.values().allFinite()};
}

void caller_rejection() {
    Eigen::MatrixX3d base=Eigen::MatrixX3d::Zero(6,3); base(0,2)=1.;
    require(!penalty<false>(base).rejected() && !penalty<true>(base).rejected(),"finite hover rejected");
    const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
    for (int row : {1,3}) for (double value : {nan,inf,-inf,1e200}) {
        auto coefficients=base;coefficients(row,0)=value;
        const auto old=penalty<false>(coefficients),fast=penalty<true>(coefficients);
        require(old.rejected() && fast.rejected(),"nonfinite velocity/jerk accepted");
        require(old.gradient_finite==fast.gradient_finite && old.quality_finite==fast.quality_finite,
                "nonfinite caller category changed");
    }
    require(penalty<false>(base,.3,0.).rejected() && penalty<true>(base,.3,0.).rejected(),
            "zero force was replaced by a valid attitude");
    auto inverted=base;inverted(2,2)=(-9.81-1.)*.5;
    require(penalty<false>(inverted).rejected() && penalty<true>(inverted).rejected(),
            "undefined inverted attitude accepted");
    auto active=base;active(3,0)=100.;
    for (double weight : {nan,inf,1e308}) {
        const auto old=penalty<false>(active,.3,9.81,weight),fast=penalty<true>(active,.3,9.81,weight);
        require(old.rejected() && fast.rejected(),"nonfinite penalty/adjoint accepted");
    }
}

void snap_time_gradient() {
    Eigen::MatrixX3d coefficients(6,3);
    coefficients << .2,-.4,1., 3.5,.2,-.1, .5,-.4,.3,
                    30.,-20.,10., 5.,-2.,1., 1.,.5,-.5;
    const auto central=penalty<true>(coefficients);
    require(!central.rejected() && central.gradient_finite,"finite snap fixture is invalid");
    const double step=1e-6;
    const double fd=(penalty<true>(coefficients,.3+step).cost-
                     penalty<true>(coefficients,.3-step).cost)/(2*step);
    require(std::abs(fd-central.time_gradient) <= 2e-5*std::max(1.,std::abs(fd)),
            "time gradient lost the active snap contribution");
}

void original_sfc_metrics_with_per_face_margins() {
    Solver::PolyhedraH raw(1,Eigen::MatrixX4d(6,4));
    raw[0] << 1,0,0,-2,-1,0,0,-2,0,1,0,-3,
              0,-1,0,-3,0,0,1,-4,0,0,-1,-4;
    auto inner=raw;
    inner[0](0,3)+=.15; inner[0](1,3)+=.02;
    Eigen::MatrixX3d coefficients=Eigen::MatrixX3d::Zero(6,3);
    coefficients.row(0) << 1.9,0.,1.;
    Eigen::VectorXd times=Eigen::VectorXd::Constant(1,.3),bounds(5),weights(5);
    Eigen::VectorXi indices=Eigen::VectorXi::Zero(1);
    bounds << 3.,2.99,.7854,2.094972067039106,22.0725;
    weights << 1e5,100.,1e4,1e4,1e4;
    flatness::FlatnessMap map; map.reset(.75,9.81,0.,0.,0.,.001);
    Solver::ConstraintMetrics metrics;
    Eigen::VectorXd gradT=Eigen::VectorXd::Zero(1);
    Eigen::MatrixX3d gradC=Eigen::MatrixX3d::Zero(6,3);
    double cost=0.;
    Solver::attachPenaltyFunctional<true>(times,coefficients,indices,inner,.01,64,
        bounds,weights,map,metrics,1e-6,raw,cost,gradT,gradC);
    require(std::abs(metrics.sfc+.1)<1e-12,"SFC diagnostic used the inward penalty domain");
    require(std::abs(cost-1e5*(.05-0.5e-6)*.3)<1e-8,"soft face offset was lost from the objective");
}

int main() {
    finite_flatness();
    caller_rejection();
    snap_time_gradient();
    original_sfc_metrics_with_per_face_margins();
}
"""


def _compile_contract(directory, vendor):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    source = directory / "contract.cpp"
    binary = directory / "contract"
    source.write_text(SOURCE)
    header = (vendor / "gcopter_solver.hpp").read_text()
    pivot = header.index("#elif PATHCOVER_DIM == 3")
    # Expose the actual private quadrature helper to this test. Its arithmetic
    # remains unchanged; dependent headers still come from the production tree.
    (directory / "gcopter_solver.hpp").write_text(
        header[:pivot] + header[pivot:].replace("    private:", "    public:")
    )
    subprocess.run(
        [
            compiler, "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
            "-I/usr/include/eigen3", f"-I{directory}", f"-I{vendor}",
            str(source), "-o", str(binary),
        ],
        check=True, capture_output=True, text=True,
    )
    return binary


def test_zero_drag_flatness_preserves_finite_math_and_rejects_undefined_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = _compile_contract(tmp_path, root / "third_party/pathcover_native/trajectory")
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

