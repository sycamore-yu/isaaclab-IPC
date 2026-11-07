"""Regression checks for observed GCOPTER geometry and line-search failures."""

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <cstdint>
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp"
#include "gcopter_solver.hpp"
#include "corridor/PathCover.hpp"
#include <iostream>
#include <limits>

double objective(void*, const Eigen::VectorXd& x, Eigen::VectorXd& gradient) {
    if (x[0] <= 0.0) {
        gradient[0] = std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::infinity();
    }
    gradient[0] = 2.0 * (x[0] - 0.05);
    return (x[0] - 0.05) * (x[0] - 0.05);
}

int main() {
    // Dual coordinates below one used to reach integer abs in QuickHull's
    // tetrahedron test when the wrapper's incidental math.h include was absent.
    Eigen::MatrixX4d h(6, 4);
    h << 1, 0, 0, -2, -1, 0, 0, -2,
         0, 1, 0, -3, 0, -1, 0, -3,
         0, 0, 1, -4, 0, 0, -1, -4;
    Eigen::Matrix3Xd vertices;
    if (!geo_utils::enumerateVs(h, vertices) || vertices.cols() != 8 || !vertices.allFinite()) return 1;
    const Eigen::MatrixXd residual = (h.leftCols<3>() * vertices).colwise() + h.col(3);
    if (residual.maxCoeff() > 1e-10) return 2;

    // Real high-speed v5: both measured position and the previous trajectory
    // entered an occupied voxel although its centre was outside the point SFC.
    const Eigen::Vector3d centre(0.75, -8.550000190734863, 0.44999998807907104);
    const Eigen::Vector3d seed(0.7052647471427917, -8.879378318786621, 0.6411795616149902);
    const Eigen::Vector3d measured(0.6379122734069824, -8.437137603759766, 0.4850492775440216);
    const Eigen::Vector3d previous_reference(0.6374621221279937, -8.437586106507663, 0.48315698254176714);
    Eigen::Matrix<double,Eigen::Dynamic,3> bounds(6,3), faces;
    bounds.topRows(3)=Eigen::Matrix3d::Identity();bounds.bottomRows(3)=-Eigen::Matrix3d::Identity();
    Eigen::VectorXd offsets, limits=Eigen::VectorXd::Constant(6,60.0);
    std::vector<int> remaining;
    PathCover::RISP<double,3>({centre},seed,bounds,limits,faces,offsets,remaining,0.01,0.15);
    const Eigen::VectorXd separation=faces*centre-offsets-0.15*faces.cwiseAbs().rowwise().sum();
    if (separation.maxCoeff() <= 0.0) return 4;
    if ((faces*measured-offsets).maxCoeff() <= 0.0) return 5;
    if ((faces*previous_reference-offsets).maxCoeff() <= 0.0) return 6;

    // Dense V9: a point-seed cut excluded the unavoidable first two Euler
    // positions. The protected convex hull must retain them and exclude the box.
    const Eigen::Vector3d box(0.875, -46.625, 2.125);
    const std::vector<Eigen::Vector3d> motion{
        {1.051566243171692, -46.819114685058594, 2.1741671562194824},
        {1.1382163717527647, -46.67283078298952, 2.148625385405014},
        {1.2164971029061002, -46.53000705842147, 2.1042063020933632}};
    PathCover::RISP<double,3>({box},motion[0],bounds,limits,faces,offsets,remaining,0.01,0.125);
    if (((faces*motion[2]-offsets).array() / faces.rowwise().norm().array()).maxCoeff() <= 0.04) return 7;
    PathCover::RISP<double,3>({box},motion[0],bounds,limits,faces,offsets,remaining,0.01,0.125,&motion);
    for (const auto& point : motion)
        if ((faces*point-offsets).maxCoeff() > 1e-10) return 8;
    if ((faces*box-offsets-0.125*faces.cwiseAbs().rowwise().sum()).maxCoeff() <= 0.0) return 9;
    auto intersecting = motion;
    intersecting[1] = box;
    bool rejected = false;
    try {
        PathCover::RISP<double,3>({box},motion[0],bounds,limits,faces,offsets,remaining,0.01,0.125,&intersecting);
    } catch (const PathCover::SeedRegionCollision&) { rejected = true; }
    if (!rejected) return 10;

    // The normalized first descent step is outside x > 0. The same line search
    // must reject that trial, shrink it, and converge to the interior optimum.
    Eigen::VectorXd x(1); x << 0.2;
    double value;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.g_epsilon = 1e-10;
    parameters.max_iterations = 100;
    const int status = lbfgs::lbfgs_optimize(x, value, objective, nullptr, nullptr, nullptr, parameters);
    if (status < 0 || std::abs(x[0] - 0.05) > 1e-8 || !std::isfinite(value)) return 3;
    // Real Forest V10: the one-piece free-space optimum is already within
    // one objective ULP, so the gradient criterion must stop before line search.
    trajectory_server::GcopterSolver<3> solver;
    Eigen::Matrix3d initial_pva, terminal_pva = Eigen::Matrix3d::Zero();
    initial_pva << 6.363571643829346, 2.7366533279418945, -2.3191123008728027,
                   0.26794102787971497, -0.23469772934913635, -0.3318706750869751,
                   2.16351318359375, -0.7082198262214661, -2.2158825397491455;
    terminal_pva.col(0) << 8.0, 0.0, 1.0;
    Eigen::MatrixX4d wide = h; wide.col(3).setConstant(-100.0);
    Eigen::VectorXd dynamic_limits(5), penalty_weights(5), physics(6);
    dynamic_limits << 6.0, 3.0, 1.2, 0.0, 22.0725;
    penalty_weights << 1e4, 1e3, 1e4, 1e4, 1e4;
    physics << 0.75, 9.81, 0.0, 0.0, 0.0, 0.001;
    if (!solver.setup(100.0, initial_pva, terminal_pva, {wide}, 0.0, 2.0, 1, 0.01,
                      64, dynamic_limits, penalty_weights, physics)) return 11;
    solver.setPositionPenaltySmoothing(1e-6);
    Trajectory<5> trajectory;
    const double minco_cost = solver.optimize(trajectory, 1e-4, 1000);
    if (solver.getStatus() < 0 || trajectory.getPieceNum() != 1 || !std::isfinite(minco_cost)) return 12;
    // Independent 80-digit quintic jerk-energy + 100*T stationary point.
    if (std::abs(trajectory[0].getDuration() - 1.4983553795894553) > 1e-7
        || std::abs(minco_cost - 185.8492727009114) > 1e-8) return 13;

    std::cout << "bounded vertices and nonfinite-trial backtracking passed\n";
}
"""


def test_cpp_geometry_and_nonfinite_trial_regressions(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "pathcover_numerics.cpp"
    executable = tmp_path / "pathcover_numerics"
    source.write_text(SOURCE)
    subprocess.run([
        compiler, "-O1", "-std=c++17", "-I/usr/include/eigen3",
        f"-I{root / 'third_party/pathcover_native/trajectory'}",
        f"-I{root / 'third_party/pathcover_native'}", str(source),
        "-lqhullcpp", "-lqhull_r", "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
