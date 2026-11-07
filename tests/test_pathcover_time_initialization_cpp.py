"""Real N1 and multipiece regressions for derivative-based time initialization."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <math.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include "gcopter_solver.hpp"

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        const bool invalid_rho = argc == 2 && std::string(argv[1]) == "invalid-rho";
        Eigen::Matrix3d initial, terminal;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) std::cin >> initial(r, c);
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r) std::cin >> terminal(r, c);
        int count;
        std::cin >> count;
        std::vector<Eigen::MatrixX4d> corridors(count);
        for (auto& corridor : corridors) {
            int faces;
            std::cin >> faces;
            corridor.resize(faces, 4);
            for (int r = 0; r < faces; ++r)
                for (int c = 0; c < 4; ++c) std::cin >> corridor(r, c);
        }
        double time_weight, margin, tolerance, smoothing, oracle_time, oracle_cost;
        int pieces, quadrature, iterations, has_oracle;
        unsigned seed;
        std::cin >> time_weight >> margin >> pieces >> quadrature >> tolerance >> iterations;
        Eigen::VectorXd bounds(5), weights(5), physics(6);
        for (int i = 0; i < 5; ++i) std::cin >> bounds[i];
        for (int i = 0; i < 5; ++i) std::cin >> weights[i];
        for (int i = 0; i < 6; ++i) std::cin >> physics[i];
        std::cin >> smoothing >> seed >> has_oracle >> oracle_time >> oracle_cost;
        require(bool(std::cin), "invalid frozen problem");
        require(count > 0 && pieces > 0, "fixture requires a fixed positive piece count");
        if (invalid_rho) time_weight = 0.0;

        sdlp::permutation_rng.seed(seed);
        trajectory_server::GcopterSolver<3> solver;
        require(solver.setup(time_weight, initial, terminal, corridors, margin,
                    2.0, pieces, .01, quadrature, bounds, weights, physics),
                "frozen corridor setup failed");
        solver.setPositionPenaltySmoothing(smoothing);
        Trajectory<5> trajectory;
        const double objective = solver.optimize(trajectory, tolerance, iterations);
        const auto& initialization = solver.getTimeInitialization();
        std::cout << std::setprecision(17)
                  << "status=" << solver.getStatus()
                  << " initialization_status=" << initialization.status
                  << " initialization_evaluations=" << initialization.evaluations
                  << " initialization_scale=" << initialization.scale
                  << " objective=" << objective << '\n';
        if (invalid_rho) {
            require(initialization.status == lbfgs::LBFGSERR_INVALID_FUNCVAL,
                    "zero time weight did not reject scalar initialization");
            require(initialization.evaluations == 0,
                    "invalid time weight reached MINCO scalar evaluation");
            require(solver.getStatus() == initialization.status &&
                        !std::isfinite(objective) && trajectory.getPieceNum() == 0,
                    "initialization failure was not propagated as an empty failed solve");
            return 0;
        }
        require(initialization.status >= 0,
                "energy/time initialization failed");
        require(std::isfinite(initialization.scale) && initialization.scale > 0.0,
                "initializer did not return a finite positive time scale");
        require(initialization.evaluations > 0 && initialization.evaluations <= 64,
                "initializer exceeded its total scalar evaluation budget");
        require(solver.getStatus() >= 0 && std::isfinite(objective),
                "main solve failed after scalar initialization");
        require(trajectory.getPieceNum() == count * pieces,
                "the fixed piece count changed");
        double duration = 0.0;
        for (int i = 0; i < trajectory.getPieceNum(); ++i) {
            const double piece_duration = trajectory[i].getDuration();
            require(std::isfinite(piece_duration) && piece_duration > 0.0,
                    "trajectory contains an invalid duration");
            require(trajectory[i].getCoeffMat().allFinite(),
                    "trajectory contains nonfinite coefficients");
            duration += piece_duration;
        }
        const auto& quality = solver.getConstraintMetrics();
        require(quality.finite && quality.values().allFinite(),
                "quadrature quality diagnostics became nonfinite");
        if (has_oracle) {
            require(count == 1 && pieces == 1, "the independent oracle is for N1 only");
            std::cout << "duration=" << duration << " oracle_time=" << oracle_time
                      << " time_error=" << duration - oracle_time << '\n';
            // The 100-digit quintic oracle uses the exact binary64 PVA.
            // The root's dimensionless 1e-8 derivative tolerance corresponds
            // to about 6.2 ns here. 10 ns allows double rounding and still
            // rejects the old initializer's 93 ns error. No physical bound changes.
            require(std::abs(duration - oracle_time) < 1e-8,
                    "duration differs from independent high-precision optimum");
            require(std::abs(objective - oracle_cost) < 1e-8,
                    "inactive-penalty objective differs from the independent oracle");
            require(quality.sfc < 0.0,
                    "N1 trajectory lost its original-corridor clearance");
        }

        Eigen::Matrix3d start, end;
        start.col(0) = trajectory.getPos(0.0);
        start.col(1) = trajectory.getVel(0.0);
        start.col(2) = trajectory.getAcc(0.0);
        end.col(0) = trajectory.getPos(duration);
        end.col(1) = trajectory.getVel(duration);
        end.col(2) = trajectory.getAcc(duration);
        require((start - initial).cwiseAbs().maxCoeff() < 1e-9,
                "measured initial PVA was changed");
        require((end - terminal).cwiseAbs().maxCoeff() < 1e-9,
                "terminal PVA was changed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
"""


def _input_text(fixture):
    problem = fixture["problem"]
    config = problem["config"]
    rows = []
    for key in ("initial_pva", "terminal_pva"):
        rows.append([problem[key][r][c] for c in range(3) for r in range(3)])
    rows.append([len(problem["corridors"])])
    for corridor in problem["corridors"]:
        rows.append([len(corridor)])
        rows.extend(corridor)
    rows.append([
        config["time_weight"], config["safety_margin"], config["pieces_per_polytope"],
        config["integral_resolution"], config["relative_tolerance"], config["max_iterations"],
    ])
    rows.append([
        config["nominal_velocity"],
        config["body_rate_max"] - config["reference_body_rate_margin"],
        config["tilt_max"], config["thrust_min"], config["thrust_max"],
    ])
    rows.append(config["penalty_weights"])
    rows.append([config["mass"], config["gravity"], 0.0, 0.0, 0.0, 1e-3])
    rows.append([config["position_penalty_smoothing_m"], config["random_seed"]])
    text = "\n".join(" ".join(format(value, ".17g") for value in row) for row in rows)
    oracle = fixture.get("oracle")
    if oracle is None:
        return text + "\n0 0 0\n"
    return (
        text + "\n1 " + oracle["optimum_time_s_decimal"] + " "
        + oracle["optimum_objective_decimal"] + "\n"
    )


def _compile(directory, vendor):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    source = directory / "contract.cpp"
    binary = directory / "contract"
    source.write_text(SOURCE)
    result = subprocess.run(
        [
            compiler, "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
            "-I/usr/include/eigen3", f"-I{directory}", f"-I{vendor}",
            str(source), "-o", str(binary),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.fixture(scope="module")
def native_time_solver(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    return _compile(
        tmp_path_factory.mktemp("native-time-initialization"),
        root / "third_party/pathcover_native/trajectory",
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["pathcover_newton_n1_v25.json", "pathcover_dense_seq64_v24.json"],
    ids=["newton-n1-quintic-oracle", "dense-seq64-multipiece"],
)
def test_real_time_initialization_regressions(native_time_solver, fixture_name):
    fixture = json.loads((Path(__file__).with_name("data") / fixture_name).read_text())
    result = subprocess.run(
        [str(native_time_solver)], input=_input_text(fixture), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_zero_time_weight_is_an_explicit_initialization_failure(native_time_solver):
    fixture = json.loads(
        (Path(__file__).with_name("data") / "pathcover_newton_n1_v25.json").read_text()
    )
    result = subprocess.run(
        [str(native_time_solver), "invalid-rho"],
        input=_input_text(fixture), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

