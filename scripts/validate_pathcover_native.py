"""Reproduce frozen-input native PathCover correctness and stage timing evidence."""

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverSegment,
    PathCoverState,
    PathCoverTrajectory,
    validate_pathcover_plan,
)

PIN = "8723a05d1993bf0199a453b47dc6ee167d493c23"
ORACLE_SOURCE = r"""
#include <math.h> // Upstream QuickHull calls unqualified abs(T); retain its floating overload.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "gcopter_solver.hpp"
int main(int argc, char** argv) {
#ifdef PATHCOVER_PORTED
    sdlp::permutation_rng.seed(argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 5489u);
#endif
    Eigen::Matrix3d initial, terminal = Eigen::Matrix3d::Zero();
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) std::cin >> initial(r,c);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) std::cin >> terminal(r,c);
    int n; std::cin >> n;
    std::vector<Eigen::MatrixX4d> corridor;
    for (int i = 0; i < n; ++i) {
        int rows; std::cin >> rows;
        Eigen::MatrixX4d poly(rows,4);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < 4; ++c) std::cin >> poly(r,c);
        corridor.push_back(poly);
    }
    double time_weight, margin, tolerance;
    int pieces, quadrature, iterations;
    std::cin >> time_weight >> margin >> pieces >> quadrature >> tolerance >> iterations;
    Eigen::VectorXd limits(5), weights(5), physics(6);
    for (int i = 0; i < 5; ++i) std::cin >> limits(i);
    for (int i = 0; i < 5; ++i) std::cin >> weights(i);
    for (int i = 0; i < 6; ++i) std::cin >> physics(i);
    trajectory_server::GcopterSolver<3> solver;
    if (!solver.setup(time_weight, initial, terminal, corridor, margin, 2.0,
                      pieces, 0.01, quadrature, limits, weights, physics)) return 2;
#ifdef PATHCOVER_PORTED
    double position_smoothing; std::cin >> position_smoothing;
    solver.setPositionPenaltySmoothing(position_smoothing);
#endif
    Trajectory<5> trajectory;
    const double objective = solver.optimize(trajectory, tolerance, iterations);
    if (!std::isfinite(objective)) {
#ifdef PATHCOVER_PORTED
        std::cerr << "trial_durations_s " << solver.getTrialDurations().transpose() << "\n";
#endif
        return 3;
    }
    std::cout << std::setprecision(17) << trajectory.getPieceNum() << " " << objective << "\n";
    for (int i = 0; i < trajectory.getPieceNum(); ++i) {
        std::cout << trajectory[i].getDuration() << "\n";
        const auto coefficients = trajectory[i].getCoeffMat();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 6; ++c) std::cout << coefficients(r,c) << " ";
        std::cout << "\n";
    }
}
"""


def fixture():
    y, z = np.meshgrid(np.arange(-0.6, 0.61, 0.1), np.arange(0.4, 2.61, 0.1), indexing="ij")
    cloud = np.stack((np.full_like(y, 1.4), y, z), axis=-1).reshape(-1, 3)
    state = PathCoverState(np.array([-1.0, 0.0, 1.5]), np.zeros(3), np.zeros(3), time=7.0)
    config = PathCoverConfig(
        bounds_lower=(-2.0, -3.0, 0.0), bounds_upper=(6.0, 3.0, 3.0),
        resolution=0.2, robot_radius=0.2, max_velocity=2.0, max_iterations=256,
        random_seed=5489,
    )
    return cloud, state, np.array([5.0, 0.0, 1.5]), config


def official_gcopter(upstream, plan, state, config):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected PathCover {PIN}, found {revision}")
    root = Path(__file__).resolve().parents[1]
    directory = root / "build/pathcover_native"
    directory.mkdir(parents=True, exist_ok=True)
    source, binary = directory / "upstream_gcopter.cpp", directory / "upstream_gcopter"
    if not source.exists() or source.read_text() != ORACLE_SOURCE:
        source.write_text(ORACLE_SOURCE)
    if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        subprocess.run(
            ["g++", "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
             "-I/usr/include/eigen3", f"-I{upstream / 'src/trajectory_server/include'}",
             str(source), "-o", str(binary)], check=True,
        )
    values = [*state.position, *state.velocity, *state.acceleration, *plan.local_goal, *np.zeros(6), len(plan.corridors)]
    for corridor in plan.corridors:
        values.extend([len(corridor), *corridor.ravel()])
    options = config.native_options()
    values.extend([
        config.time_weight, config.safety_margin, config.pieces_per_polytope, config.integral_resolution,
        config.relative_tolerance, config.max_iterations,
        options["nominal_velocity"], config.body_rate_max, config.tilt_max, config.thrust_min, options["thrust_max"],
        *config.penalty_weights, config.mass, config.gravity, 0.0, 0.0, 0.0, 0.001,
    ])
    serialized = " ".join(str(value) for value in values)
    (directory / "upstream_gcopter_input.txt").write_text(serialized + "\n")
    completed = subprocess.run([str(binary)], input=serialized, text=True, capture_output=True)
    if completed.returncode:
        return {"revision": revision, "success": False, "returncode": completed.returncode,
                "diagnostic": "\n".join(value.strip() for value in (completed.stdout, completed.stderr) if value.strip()),
                "scope": "unmodified official GCOPTER on identical PVA/corridors/dynamic parameters; upstream position smoothing retained"}
    output = np.fromstring(completed.stdout, sep=" ")
    pieces, objective = int(output[0]), float(output[1])
    data = output[2:].reshape(pieces, 19)
    coefficients = data[:, 1:].reshape(pieces, 3, 6)
    try:
        trajectory = PathCoverTrajectory(coefficients, data[:, 0], state.yaw, config.body_yaw_rate_max,
                                        config.yaw_sample_dt, config.yaw_low_speed, config.gravity)
    except (ValueError, PathCoverPlanningError) as error:
        return {"revision": revision, "success": True, "trajectory_valid": False,
                "official_objective": objective, "diagnostic": str(error),
                "scope": "official optimization returned a polynomial outside the heading construction domain"}
    boundaries = np.concatenate(([0.0], np.cumsum(data[:, 0])))
    segments = tuple(PathCoverSegment(original.corridor_index, float(boundaries[i]),
                                     float(boundaries[i + 1]), float(data[i, 0]))
                     for i, original in enumerate(plan.segments))
    official_plan = replace(plan, trajectory=trajectory, segments=segments,
                            valid_until=plan.start_time + trajectory.duration,
                            metrics={"optimizer_status": 1, "objective": objective})
    numerics = validate_pathcover_plan(official_plan, config)
    numerics.pop("optimizer_status")  # The unmodified wrapper exposes success, not its exact L-BFGS status.
    return {
        "revision": revision, "success": True, "official_objective": objective,
        "objective_abs_difference": abs(objective - plan.metrics["objective"]),
        "duration_max_abs_difference_s": float(np.max(np.abs(data[:, 0] - plan.trajectory.durations))),
        "coefficient_max_abs_difference": float(np.max(np.abs(coefficients - plan.trajectory.coefficients))),
        "numerics": numerics,
        "official_position_penalty_smoothing_m": 0.01,
        "ported_position_penalty_smoothing_m": config.position_penalty_smoothing_m,
        "scope": "unmodified official GCOPTER on the identical PVA, ordered corridors and dynamic parameters",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="MARSIM frontend-input.npz containing real observed LiDAR returns")
    parser.add_argument("--benchmark", choices=("high_speed", "dense"), default="high_speed")
    parser.add_argument("--search-resolution", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--upstream", type=Path, help="Read-only pinned PathCover checkout for a GCOPTER oracle")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/pathcover-native/validation.json"))
    args = parser.parse_args()
    cloud, state, goal, config = fixture()
    if args.input:
        from isaaclab_ipc.assets.quadrotor import MASS
        from isaaclab_ipc.benchmark_config import load_case
        from isaaclab_ipc.ros_ctbr_policy import benchmark_frontend_config
        from isaaclab_ipc.turbompc.ctbr import CTBRConfig

        frozen = np.load(args.input)
        raw = frozen["state"]
        w, x, y, z = raw[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        state = PathCoverState(raw[:3], raw[7:10], frozen["acceleration"], float(yaw), float(frozen["now"]))
        cloud, goal = frozen["cloud"], frozen["goal"]
        config = benchmark_frontend_config(load_case(args.benchmark), CTBRConfig(mass=MASS))
    if args.search_resolution is not None:
        config = replace(config, search_resolution=args.search_resolution)
    if args.random_seed is not None:
        config = replace(config, random_seed=args.random_seed)
    report = {"upstream_revision": PIN, "config": asdict(config), "points": len(cloud), "backends": {}}
    plans, frontends = {}, {}
    for backend in ("cpu", "cuda"):
        selected = replace(config, use_cuda=backend == "cuda", cuda_min_grid_voxels=0)
        frontend = NativePathCoverFrontend(selected)
        plan = frontend.plan(cloud, state, goal)
        plans[backend], frontends[backend] = plan, frontend
        # Replan from a frozen map/state: map integration is recorded separately in the first solve.
        timings = [frontend.plan(None, state, goal).metrics for _ in range(args.repeats)]
        latency = {key: {"median": float(np.median([row[key] for row in timings])),
                         "p95": float(np.percentile([row[key] for row in timings], 95))}
                   for key in ("search_ms", "jps_ms", "distance_map_ms", "dmp_ms", "corridor_ms", "trajectory_ms", "total_ms")}
        report["backends"][backend] = {
            "first_plan": plan.metrics, "frozen_replan_ms": latency, "repeats": args.repeats,
            "corridors": len(plan.corridors), "pieces": len(plan.segments),
            "numerics": validate_pathcover_plan(plan, selected),
        }
    np.testing.assert_array_equal(frontends["cpu"].inflated_cloud(), frontends["cuda"].inflated_cloud())
    np.testing.assert_array_equal(frontends["cpu"]._native.potential_map(), frontends["cuda"]._native.potential_map())
    np.testing.assert_allclose(plans["cpu"].path, plans["cuda"].path, atol=1e-12)
    coefficient_error = float(np.max(np.abs(
        plans["cpu"].trajectory.coefficients - plans["cuda"].trajectory.coefficients,
    )))
    report["cpu_cuda"] = {"inflated_map_exact": True, "potential_map_exact": True, "path_equal": True,
                          "coefficient_max_abs_difference": coefficient_error}
    if args.upstream:
        report["official_gcopter"] = official_gcopter(args.upstream, plans["cpu"], state, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
