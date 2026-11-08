"""Continuous S3 position objective, envelope derivatives and legacy contracts."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <math.h> // QuickHull uses the unqualified floating-point abs overload.
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
void positive_penalty_maximum() {
    const auto compare=[](const Coeff& c,double duration,const Eigen::MatrixX4d& h) {
        const auto signed_max=maximum(c,duration,h), positive=maximum<true>(c,duration,h);
        require(signed_max.finite==positive.finite,"penalty pruning changed finite classification");
        if(!signed_max.finite) {
            require(std::isnan(positive.value),"penalty pruning hid invalid arithmetic");
        } else if(signed_max.value>=0.) {
            require(std::memcmp(&signed_max.value,&positive.value,sizeof(double))==0 &&
                    std::memcmp(&signed_max.alpha,&positive.alpha,sizeof(double))==0 &&
                    signed_max.face==positive.face,"zero/positive maximum or first witness changed");
        } else {
            require(std::isfinite(positive.value) && positive.value<=0.,"negative face created a penalty");
        }
    };
    Coeff c=Coeff::Zero(); c(0,2)=1.;
    Eigen::MatrixX4d h(3,4); h << 0,1,0,0, 0,1,0,-1, 0,1,0,0;
    for(double value:{-.3,-1e-13,-std::numeric_limits<double>::denorm_min(),-0.,0.,
            std::numeric_limits<double>::denorm_min(),1e-13,1e-6,
            std::nextafter(1e-6,0.),std::nextafter(1e-6,1.),.1}) {
        c(0,1)=value; compare(c,1.,h);
    }
    c(0,1)=-.3;
    const auto inactive=maximum<true>(c,1.,h);
    require(inactive.face==-1 && inactive.value==0.,"strictly negative faces were not omitted");
    const auto zero=penalty(c,1.,h,true);
    require(zero.metrics.finite && zero.cost==0. && zero.time.isZero(0.) &&
            zero.coeff.isZero(0.),"inactive sentinel reached a face or added a gradient");
    for(double shift:{-.02,0.,1e-15,5e-5,.1}) {
        c.col(1) << -.0625+shift,.5,-1.5,2.,-1.,0.; compare(c,1.,h);
        c.col(1) << shift-1.,2.,-1.,0.,0.,0.; compare(c,1.,h);
        c.col(1) << shift,.02,0.,0.,0.,0.; compare(c,.7,h);
    }
    // A safely negative first face must not hide overflow on a later face.
    c.setZero(); c(0,2)=1.; c(5,1)=std::numeric_limits<double>::max()/3.;
    Eigen::MatrixX4d later(2,4); later << 0,0,1,-10, 0,1,0,-10;
    compare(c,1.,later);
    c.setZero(); c(0,2)=1.; compare(c,1e80,h);
    c(3,0)=std::numeric_limits<double>::quiet_NaN(); compare(c,1.,h);
    c(3,0)=std::numeric_limits<double>::infinity(); compare(c,1.,h);
    c.setZero(); c(0,2)=1.; c(0,1)=-10.;
    later(1,3)=std::numeric_limits<double>::infinity(); compare(c,1.,later);
    // An overflowing Bernstein guard retains the original evaluation path.
    c.setZero(); c(0,1)=1e308; c(1,1)=-1e308; compare(c,1.,h);
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
    require(std::isinf(exact.metrics.sfc) && exact.metrics.sfc<0 &&
            exact.metrics.positionPieces==1,"continuous SFC diagnostic was not deferred");
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
Eigen::MatrixX4d box(double lower_x, double upper_x) {
    Eigen::MatrixX4d h(6,4);
    h << -1,0,0,lower_x, 1,0,0,-upper_x,
         0,-1,0,-.8, 0,1,0,-.9,
         0,0,-1,.1, 0,0,1,-2.1;
    return h;
}
void setup_diagnostic_problem(Solver& solver) {
    sdlp::permutation_rng.seed(0);
    Eigen::Matrix3d initial=Eigen::Matrix3d::Zero(), terminal=Eigen::Matrix3d::Zero();
    initial.col(0) << 0,0,1;
    initial.col(1) << .3,.1,0;
    terminal.col(0) << 2.5,.2,1.1;
    const Solver::PolyhedraH corridors{box(-1,1.4),box(.5,3.2)};
    Eigen::VectorXd bounds(5), weights(5), physics(6);
    bounds << 3,2.99,M_PI/2,2.094972067039106,22.0725;
    weights << 1e5,100,1e4,1e4,1e4;
    physics << .75,9.81,0,0,0,.001;
    require(solver.setup(100,initial,terminal,corridors,.1,2.,1,.01,64,
                         bounds,weights,physics),"diagnostic fixture setup failed");
    solver.setPositionPenaltySmoothing(1e-6);
    solver.setContinuousMaxPositionPenalty(true);
    solver.enableInitialForce(Eigen::Vector3d::UnitZ(),0,22.0725,.75*9.81);
}
Evaluation dynamic_penalty(const Coeff& c,double duration,bool drag=false,bool continuous=true) {
    Evaluation result;
    flatness::FlatnessMap map;
    map.reset(.75,9.81,drag?.1:0.,drag?.12:0.,drag?.03:0.,.001);
    Eigen::VectorXd times=Eigen::VectorXd::Constant(1,duration), bounds(5), weights(5);
    Eigen::VectorXi indices=Eigen::VectorXi::Zero(1);
    // These synthetic limits activate every dynamic penalty. Position weight
    // is zero to isolate the quadrature whose evaluation order changed.
    bounds << 1.,.02,.08,7.,7.5;
    weights << 0.,100.,1e4,1e4,1e4;
    const Solver::PolyhedraH polygons{box(-1.,3.)};
    if (drag)
        Solver::attachPenaltyFunctional<false>(times,c,indices,polygons,.01,64,
            bounds,weights,map,result.metrics,1e-6,polygons,result.cost,result.time,result.coeff,continuous);
    else
        Solver::attachPenaltyFunctional<true>(times,c,indices,polygons,.01,64,
            bounds,weights,map,result.metrics,1e-6,polygons,result.cost,result.time,result.coeff,continuous);
    return result;
}
void dynamic_derivatives() {
    Coeff c;
    c.col(0) << 0,1.6,.3,.2,-.07,.02;
    c.col(1) << .2,-.8,.6,-.15,.04,-.005;
    c.col(2) << 1,.3,.1,.04,-.02,.002;
    constexpr double duration=.7, step=1e-6;
    for (bool drag : {false,true}) {
        const auto center=dynamic_penalty(c,duration,drag);
        require(center.metrics.finite && center.coeff.allFinite() && center.time.allFinite(),
                "finite dynamic penalty rejected");
        require(center.metrics.velocitySquared>1. && center.metrics.rateSquared>.02*.02 &&
                center.metrics.tilt>.08 && center.metrics.maxThrust>7.5,
                "dynamic fixture left a penalty inactive");
        for (int trial=0;trial<4;++trial) {
            Coeff direction;
            for (int row=0;row<6;++row) for(int axis=0;axis<3;++axis)
                direction(row,axis)=std::sin(.31+1.3*row+2.1*axis+.7*trial);
            direction.normalize();
            const double numeric=(dynamic_penalty(c+step*direction,duration,drag).cost-
                                  dynamic_penalty(c-step*direction,duration,drag).cost)/(2*step);
            near(numeric,center.coeff.cwiseProduct(direction).sum(),2e-6,
                 "dynamic coefficient derivative");
        }
        const double numeric=(dynamic_penalty(c,duration+step,drag).cost-
                              dynamic_penalty(c,duration-step,drag).cost)/(2*step);
        near(numeric,center.time(0),2e-6,"dynamic explicit-time derivative");
    }
}
void full_objective_derivatives() {
    Solver solver;
    setup_diagnostic_problem(solver);
    Eigen::VectorXd x(solver.temporalDim+solver.spatialDim+1);
    Eigen::Map<Eigen::VectorXd> tau(x.data(),solver.temporalDim);
    Eigen::Map<Eigen::VectorXd> xi(x.data()+solver.temporalDim,solver.spatialDim);
    solver.applyInitialForce(solver.initialForce.etaGuess);
    Solver::setInitial(solver.shortPath,solver.allocSpeed,solver.pieceIdx,solver.points,solver.times);
    require(solver.initializeTimeScale()>=0,"full derivative fixture initialization failed");
    Solver::backwardT(solver.times,tau);
    Solver::backwardP(solver.points,solver.vPolyIdx,solver.vPolytopes,xi);
    x(x.size()-1)=solver.initialForce.etaGuess;
    Eigen::VectorXd gradient(x.size());
    const double cost=Solver::costFunctional(&solver,x,gradient);
    require(std::isfinite(cost) && gradient.allFinite(),"full derivative fixture is nonfinite");
    constexpr double step=1e-6;
    for (int trial=0;trial<4;++trial) {
        Eigen::VectorXd direction(x.size()),unused(x.size());
        for(int i=0;i<direction.size();++i) direction(i)=std::sin(.8+.6*i+1.2*trial);
        direction.normalize();
        const double numeric=(Solver::costFunctional(&solver,x+step*direction,unused)-
                              Solver::costFunctional(&solver,x-step*direction,unused))/(2*step);
        near(numeric,gradient.dot(direction),2e-6,"full MINCO/free-force objective derivative");
    }
}
void derivative_coefficient_overflow() {
    Coeff c=Coeff::Zero();
    c(0,2)=1.; c(5,0)=std::numeric_limits<double>::max()/20.;
    require(!std::isfinite(60.*c(5,0)),"overflow fixture does not overflow derivative premultiplication");
    // On this tiny interval the original time-scaled derivatives remain finite.
    // Integral mode keeps that original expression; position weight is zero.
    const auto original=dynamic_penalty(c,1e-80,false,false);
    const auto selected=dynamic_penalty(c,1e-80,false,true);
    require(original.metrics.finite && selected.metrics.finite &&
            std::isfinite(original.cost) && std::isfinite(selected.cost) &&
            original.coeff.allFinite() && selected.coeff.allFinite() &&
            original.time.allFinite() && selected.time.allFinite(),
            "new derivative premultiplication hid a finite original evaluation");
    require(original.cost==selected.cost &&
            (original.coeff.array()==selected.coeff.array()).all() &&
            (original.time.array()==selected.time.array()).all() &&
            (original.metrics.values().tail<5>().array()==selected.metrics.values().tail<5>().array()).all(),
            "derivative-overflow branch did not retain the original expression");
}
double quadrature_sfc(const Trajectory<5>& trajectory, const Solver& solver,
                      const Solver::PolyhedraH& corridors) {
    // Independently evaluate the returned/trial polynomial through Piece::getPos.
    // Do not call the deferred-metric helper or its monomial-basis implementation.
    double result=-INFINITY;
    for (int i=0;i<trajectory.getPieceNum();++i) {
        const auto& planes=corridors[solver.getPiecePolytopeIndices()(i)];
        const double step=trajectory[i].getDuration()/solver.integralRes;
        for (int j=0;j<=solver.integralRes;++j) {
            const Eigen::Vector3d position=trajectory[i].getPos(j*step);
            for (int k=0;k<planes.rows();++k)
                result=std::max(result,planes.row(k).head<3>().dot(position)+planes(k,3));
        }
    }
    return result;
}
void finalized_quadrature_diagnostics(bool iteration_limit) {
    Solver solver;
    setup_diagnostic_problem(solver);
    Trajectory<5> returned;
    const double cost=solver.optimize(returned,1e-5,iteration_limit ? 1 : 1000);
    Trajectory<5> inspected;
    if (iteration_limit) {
        require(solver.getStatus()==lbfgs::LBFGSERR_MAXIMUMITERATION &&
                std::isinf(cost) && returned.getPieceNum()==0,
                "iteration-limit failure contract changed");
        // On failure optimize() must retain diagnostics for the last evaluated
        // MINCO trial, even though it does not return that trajectory.
        solver.minco.getTrajectory(inspected);
    } else {
        require(solver.getStatus()>=0 && std::isfinite(cost) && returned.getPieceNum()>0,
                "diagnostic fixture did not optimize successfully");
        inspected=returned;
    }
    const auto& metrics=solver.getConstraintMetrics();
    require(inspected.getPieceNum()>1 && metrics.positionPieces==inspected.getPieceNum(),
            "diagnostic fixture lost its multiple visited pieces");
    require(metrics.finite && metrics.values().allFinite(),"final diagnostics were left pending");
    for (int i=0;i<inspected.getPieceNum();++i)
        near(inspected[i].getDuration(),solver.getTrialDurations()(i),1e-14,
             "inspected polynomial does not match the retained trial durations");
    const double original=quadrature_sfc(inspected,solver,solver.hPolytopes);
    const double shifted=quadrature_sfc(inspected,solver,solver.penaltyPolytopes);
    near(metrics.sfc,original,1e-12,"final SFC diagnostic differs from independent quadrature");
    require(shifted-original>.01,"fixture does not distinguish original and inward-offset planes");
    // The unchanged integral evaluator is an independent oracle on the same
    // polynomial. Horner changes the rounding order of dynamic diagnostics;
    // the final SFC diagnostic still uses the original quadrature expression.
    flatness::FlatnessMap map;
    map.reset(.75,9.81,0,0,0,.001);
    Solver::ConstraintMetrics integral_metrics;
    Eigen::VectorXd grad_time=Eigen::VectorXd::Zero(solver.getTrialDurations().size());
    Eigen::MatrixX3d grad_coeff=Eigen::MatrixX3d::Zero(solver.minco.getCoeffs().rows(),3);
    double ignored_cost=0;
    Solver::attachPenaltyFunctional<true>(solver.getTrialDurations(),solver.minco.getCoeffs(),
        solver.getPiecePolytopeIndices(),solver.penaltyPolytopes,solver.smoothEps,solver.integralRes,
        solver.magnitudeBd,solver.penaltyWt,map,integral_metrics,solver.positionSmoothEps,
        solver.hPolytopes,ignored_cost,grad_time,grad_coeff,false);
    require(metrics.sfc==integral_metrics.sfc && metrics.finite==integral_metrics.finite,
            "final SFC/finite diagnostic differs from original quadrature");
    const auto actual=metrics.values(), expected=integral_metrics.values();
    for(int i=1;i<actual.size();++i)
        near(actual(i),expected(i),1e-12,"final dynamic QUALITY differs from original quadrature");
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"one contract name required");
        const std::string mode=argv[1];
        if(mode=="polynomials") polynomials();
        else if(mode=="positive_penalty") positive_penalty_maximum();
        else if(mode=="derivatives") objective_and_derivatives();
        else if(mode=="legacy") legacy_and_templates();
        else if(mode=="success_diagnostics") finalized_quadrature_diagnostics(false);
        else if(mode=="failure_diagnostics") finalized_quadrature_diagnostics(true);
        else if(mode=="dynamic_derivatives") dynamic_derivatives();
        else if(mode=="full_derivatives") full_objective_derivatives();
        else if(mode=="derivative_overflow") derivative_coefficient_overflow();
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


@pytest.mark.parametrize(
    "mode", ["polynomials", "positive_penalty", "derivatives", "legacy", "success_diagnostics", "failure_diagnostics",
             "dynamic_derivatives", "full_derivatives", "derivative_overflow"]
)
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
