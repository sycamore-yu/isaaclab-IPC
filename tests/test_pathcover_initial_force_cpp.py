"""Native CTBR initial-force derivatives, template isolation and JSON metadata."""

import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import PathCoverConfig

SOURCE = r"""
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "gcopter_solver.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void adjoint() {
    minco::MINCO_S3NU<> minco;
    Eigen::Matrix3d initial = Eigen::Matrix3d::Zero(), terminal = initial;
    initial.col(0) << 0.1, -0.2, 1.0;
    initial.col(1) << 0.5, 0.2, -0.1;
    terminal.col(0) << 2.2, 0.8, 1.5;
    const Eigen::Vector3d axis = Eigen::Vector3d(0.2, -0.1, 1.0).normalized();
    const double mass = 0.75, gravity = 9.81, upper = 22.0725;
    const double eta = std::atanh(2.0 * 6.0 / upper - 1.0);
    Eigen::Matrix3Xd points(3, 2);
    points << .7, 1.4, .1, .4, 1.2, 1.1;
    Eigen::Vector3d times(0.8, 1.1, 0.9);
    minco.setConditions(initial, terminal, 3);
    minco::MINCO_S3NU<>::CoeffMat linear(18, 3);
    for (int i = 0; i < linear.rows(); ++i)
        for (int j = 0; j < 3; ++j) linear(i, j) = .01 * (i + 1) * (j - 1);
    auto evaluate = [&](double value) {
        const double force = upper * .5 * (1.0 + std::tanh(value));
        minco.setHeadAcceleration(axis * (force / mass) - Eigen::Vector3d(0, 0, gravity));
        minco.setParameters(points, times);
        double energy;
        minco.getEnergy(energy);
        return energy + minco.getCoeffs().cwiseProduct(linear).sum();
    };
    evaluate(eta);
    minco::MINCO_S3NU<>::CoeffMat partial;
    Eigen::VectorXd partial_time;
    minco.getEnergyPartialGradByCoeffs(partial);
    minco.getEnergyPartialGradByTimes(partial_time);
    partial += linear;
    Eigen::Matrix3Xd gp, gp_legacy;
    Eigen::VectorXd gt, gt_legacy;
    Eigen::Vector3d ga;
    minco.propogateGrad(partial, partial_time, gp_legacy, gt_legacy);
    minco.propogateGrad(partial, partial_time, gp, gt, &ga);
    require((gp.array() == gp_legacy.array()).all() && (gt.array() == gt_legacy.array()).all(),
            "optional head adjoint changed the existing point/time gradient");
    const double tanh_eta = std::tanh(eta);
    const double analytic = ga.dot(axis) / mass * .5 * upper * (1.0 - tanh_eta * tanh_eta);
    const double step = 1e-6;
    const double numerical = (evaluate(eta + step) - evaluate(eta - step)) / (2 * step);
    require(std::isfinite(analytic) && std::isfinite(numerical), "nonfinite boundary derivative");
    require(std::abs(analytic - numerical) / std::max(1.0, std::abs(analytic)) < 1e-6,
            "head-force adjoint disagrees with independent central differences");
    std::cout << "force_eta_gradient " << analytic << " " << numerical << '\n';
}

template<class Minco, int BoundaryColumns>
void legacy_minco() {
    Minco minco;
    Eigen::Matrix<double, 3, BoundaryColumns> initial, terminal;
    initial.setZero(); terminal.setZero(); terminal(0, 0) = 1;
    minco.setConditions(initial, terminal, 1);
    Eigen::Matrix3Xd points(3, 0), gp;
    Eigen::VectorXd times = Eigen::VectorXd::Constant(1, 1.2), partial_time, gt;
    minco.setParameters(points, times);
    typename Minco::CoeffMat partial;
    minco.getEnergyPartialGradByCoeffs(partial);
    minco.getEnergyPartialGradByTimes(partial_time);
    minco.propogateGrad(partial, partial_time, gp, gt);
    require(partial.allFinite() && gt.allFinite(), "legacy MINCO gradient changed");
}

void templates() {
    legacy_minco<minco::MINCO_S2NU<>, 2>();
    legacy_minco<minco::MINCO_S4NU<>, 4>();
    // Instantiating optimize also instantiates its actual objective callback.
    [[maybe_unused]] auto default_s4 = &trajectory_server::GcopterSolver<4>::optimize;
    trajectory_server::GcopterSolver<4> unsupported;
    bool rejected = false;
    try { unsupported.enableInitialForce(Eigen::Vector3d::UnitZ(), 0, 22, 7); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "free force silently enabled for S4");
}

void solver_boundary() {
    using Solver = trajectory_server::GcopterSolver<3>;
    Eigen::Matrix3d initial = Eigen::Matrix3d::Zero(), terminal = initial;
    initial.col(0) << 0, 0, 1;
    initial.col(1) << .5, .1, 0;
    initial.col(2) << -.2, .3, -.5;
    terminal.col(0) << 2, 0, 1;
    Eigen::MatrixX4d corridor(6, 4);
    corridor << -1,0,0,-5, 1,0,0,-5, 0,-1,0,-5, 0,1,0,-5, 0,0,-1,0, 0,0,1,-5;
    Eigen::VectorXd bounds(5), weights(5), physics(6);
    bounds << 3, 2.99, M_PI_2, 2.0, 22.0725;
    weights << 1e5, 100, 1e4, 1e4, 1e4;
    physics << .75, 9.81, 0, 0, 0, 1e-3;
    Solver solver;
    auto setup = [&]() {
        sdlp::permutation_rng.seed(0);
        require(solver.setup(100, initial, terminal, {corridor}, 0, 2, 1, .01, 64,
                             bounds, weights, physics), "simple corridor setup failed");
        solver.setPositionPenaltySmoothing(1e-6);
    };
    setup();
    const Eigen::Vector3d axis = Eigen::Vector3d(.1, .2, 1).normalized();
    const double guess = .75 * axis.dot(initial.col(2) + Eigen::Vector3d(0,0,9.81));
    solver.enableInitialForce(axis, 0, 22.0725, guess);
    require(!solver.getInitialForce().guessWasProjected, "an interior force guess was projected");
    require(solver.getInitialForce().guessRaw == guess, "raw force guess was changed");
    Trajectory<5> trajectory;
    const double objective = solver.optimize(trajectory, 1e-4, 1000);
    require(solver.getStatus() >= 0 && std::isfinite(objective) && trajectory.getPieceNum() == 1,
            "free initial force did not return a finite reference");
    require((trajectory.getPos(0) - initial.col(0)).norm() < 1e-12 &&
            (trajectory.getVel(0) - initial.col(1)).norm() < 1e-12, "p0/v0 changed");
    const auto force = solver.getInitialForce();
    const Eigen::Vector3d acceleration = trajectory.getAcc(0);
    require(force.force >= 0 && force.force <= 22.0725, "optimized force left the physical interval");
    require((acceleration + Eigen::Vector3d(0,0,9.81) - axis * (force.force/.75)).norm() < 1e-12,
            "returned acceleration does not use measured force direction");
    require((acceleration - initial.col(2)).norm() > 1e-4, "reference still fixes measured acceleration");
    setup();
    require(!solver.getInitialForce().enabled, "setup retained the previous free-force mode");
    const double fixed_objective = solver.optimize(trajectory, 1e-4, 1000);
    require(std::isfinite(fixed_objective) && solver.getStatus() >= 0, "fixed PVA mode failed");
    require((trajectory.getAcc(0) - initial.col(2)).norm() < 1e-12, "standalone changed fixed a0");
    solver.enableInitialForce(axis, 0, 22.0725, -10);
    require(solver.getInitialForce().guess > 0 && solver.getInitialForce().guess < 22.0725 &&
            std::isfinite(solver.getInitialForce().etaGuess), "lower guess projection is not finite");
    solver.enableInitialForce(axis, 0, 22.0725, 30);
    require(solver.getInitialForce().guess > 0 && solver.getInitialForce().guess < 22.0725 &&
            std::isfinite(solver.getInitialForce().etaGuess), "upper guess projection is not finite");
    bool rejected = false;
    try { solver.enableInitialForce(2 * axis, 0, 22.0725, guess); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "nonunit initial axis was silently normalized");
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected one test mode");
        const std::string mode(argv[1]);
        if (mode == "adjoint") adjoint();
        else if (mode == "templates") templates();
        else if (mode == "solver") solver_boundary();
        else throw std::runtime_error("unknown test mode");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
"""


@pytest.fixture(scope="module")
def force_contract(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path_factory.mktemp("pathcover-initial-force")
    source, binary = folder / "contract.cpp", folder / "contract"
    source.write_text(SOURCE)
    result = subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
         "-I/usr/include/eigen3", f"-I{root / 'third_party/pathcover_native/trajectory'}",
         str(source), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("mode", ["adjoint", "templates", "solver"])
def test_initial_force_native_contract(force_contract, mode):
    result = subprocess.run([str(force_contract), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def boundary_metadata(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    pybind = pytest.importorskip("pybind11")
    root = Path(__file__).resolve().parents[1]
    native = (root / "native/pathcover_native.cpp").read_text()
    options = native[native.index("struct Options {"):native.index("struct MappingStats {")]
    boundary = native[native.index("struct InitialBoundary {"):native.index("struct PlanResult {")]
    source = r"""
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
namespace py = pybind11;
using Vec3 = Eigen::Vector3d;
""" + options + boundary + r"""
PYBIND11_MODULE(_initial_boundary_metadata, module) {
    module.def("read", [](const py::dict& config, const Vec3& acceleration,
                          const std::optional<Vec3>& axis) {
        const Options options(config);
        const auto value = initialBoundaryForInput(acceleration, axis, options);
        return py::make_tuple(initialBoundaryDict(value), value.permitsHold(options.mass * options.gravity));
    });
}
"""
    folder = tmp_path_factory.mktemp("initial-boundary-metadata")
    cpp = folder / "contract.cpp"
    binary = folder / ("_initial_boundary_metadata" + sysconfig.get_config_var("EXT_SUFFIX"))
    cpp.write_text(source)
    result = subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", "-I/usr/include/eigen3",
         f"-I{pybind.get_include()}", f"-I{sysconfig.get_path('include')}", str(cpp), "-o", str(binary)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    spec = importlib.util.spec_from_file_location("_initial_boundary_metadata", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_metadata_is_json_and_hold_uses_the_10d_state(boundary_metadata):
    options = PathCoverConfig(thrust_min=0, nominal_thrust_min=2, thrust_max=22.0725).native_options()
    value, hold = boundary_metadata.read(options, [0.2, -0.3, -1], [0, 0, 1])
    assert hold  # Measured a is not a CTBR state component.
    assert value["mode"] == "ctbr_collective"
    assert value["reference_acceleration"] is None and value["force_n"] is None
    assert abs(value["force_guess_raw_n"] - value["force_guess_n"]) <= np.spacing(value["force_guess_raw_n"])
    assert not value["force_guess_was_projected"]
    assert json.loads(json.dumps(value))["thrust_axis"] == [0, 0, 1]
    assert json.loads(json.dumps(value))["measured_acceleration"] == [.2, -.3, -1]
    tilted = np.array([.1, 0, 1])
    tilted /= np.linalg.norm(tilted)
    _, hold = boundary_metadata.read(options, [0, 0, 0], tilted)
    assert not hold
    insufficient_force = dict(options, thrust_max=5.0)
    _, hold = boundary_metadata.read(insufficient_force, [0, 0, 0], [0, 0, 1])
    assert not hold
    fixed, hold = boundary_metadata.read(options, [0, 0, 0], None)
    assert hold and fixed["mode"] == "fixed_pva" and fixed["thrust_axis"] is None
    _, hold = boundary_metadata.read(options, [.2, 0, 0], None)
    assert not hold


def test_boundary_projection_and_invalid_axis_remain_explicit(boundary_metadata):
    options = PathCoverConfig(thrust_min=0, nominal_thrust_min=2, thrust_max=22.0725).native_options()
    value, _ = boundary_metadata.read(options, [0, 0, -20], [0, 0, 1])
    assert value["force_guess_raw_n"] < 0 < value["force_guess_n"]
    assert value["force_guess_was_projected"]
    assert value["force_lower_n"] == 0
    assert value["measured_acceleration"] == (0, 0, -20)
    # This non-round interior force has a one-ULP fraction round trip. The
    # projection flag describes clamp activity, not arithmetic roundoff.
    unaligned = dict(options, thrust_min=0.13, nominal_thrust_min=2.0, thrust_max=21.913)
    value, _ = boundary_metadata.read(unaligned, [.123, .234, -2.345], [0, 0, 1])
    assert value["force_lower_n"] < value["force_guess_raw_n"] < value["force_upper_n"]
    assert value["force_guess_raw_n"] != value["force_guess_n"]
    assert not value["force_guess_was_projected"]
    for axis in ([0, 0, 0], [0, 0, 2], [np.nan, 0, 1]):
        with pytest.raises(ValueError, match="finite unit vector"):
            boundary_metadata.read(options, [0, 0, 0], axis)
