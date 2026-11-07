"""Replay a captured GCOPTER problem against the port and a pinned official checkout."""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from validate_pathcover_native import ORACLE_SOURCE, PIN

from isaaclab_ipc.pathcover_native import (
    PathCoverConfig,
    PathCoverPlan,
    PathCoverSegment,
    PathCoverTrajectory,
    validate_pathcover_plan,
)


def serialized_problem(problem, config):
    values = [*np.asarray(problem["initial_pva"]).ravel(order="F"),
              *np.asarray(problem["terminal_pva"]).ravel(order="F"), len(problem["corridors"])]
    for polytope in problem["corridors"]:
        values.extend([len(polytope), *np.asarray(polytope).ravel()])
    values.extend([
        config.time_weight, config.safety_margin, config.pieces_per_polytope, config.integral_resolution,
        config.relative_tolerance, config.max_iterations,
        config.max_velocity, config.body_rate_max, config.tilt_max, config.thrust_min,
        config.native_options()["thrust_max"], *config.penalty_weights,
        config.mass, config.gravity, 0.0, 0.0, 0.0, 0.001, config.position_penalty_smoothing_m,
    ])
    return " ".join(str(value) for value in values)


def build_solver(root, include, name, ported):
    folder = root / "build/pathcover_native"
    folder.mkdir(parents=True, exist_ok=True)
    source, binary = folder / f"replay_{name}.cpp", folder / f"replay_{name}"
    if not source.exists() or source.read_text() != ORACLE_SOURCE:
        source.write_text(ORACLE_SOURCE)
    latest = max(source.stat().st_mtime_ns, max(path.stat().st_mtime_ns for path in include.rglob("*.hpp")))
    if not binary.exists() or binary.stat().st_mtime_ns < latest:
        subprocess.run(["g++", "-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3",
                        *(["-DPATHCOVER_PORTED"] if ported else []), "-I/usr/include/eigen3",
                        f"-I{include}", str(source), "-o", str(binary)], check=True)
    return binary


def replay(binary, text, problem, config, ported):
    completed = subprocess.run([str(binary), str(config.random_seed)], input=text,
                               text=True, capture_output=True)
    result = {"success": completed.returncode == 0, "returncode": completed.returncode,
              "initialization_seed": config.random_seed if ported else 5489}
    if completed.returncode:
        result["diagnostic"] = "\n".join(value.strip() for value in (completed.stdout, completed.stderr) if value.strip())
        return result
    data = np.fromstring(completed.stdout, sep=" ")
    pieces, objective = int(data[0]), float(data[1])
    data = data[2:].reshape(pieces, 19)
    try:
        trajectory = PathCoverTrajectory(data[:, 1:].reshape(pieces, 3, 6), data[:, 0],
                                        initial_yaw=problem.get("initial_yaw", 0.0),
                                        body_yaw_rate_max=config.body_yaw_rate_max,
                                        yaw_sample_dt=config.yaw_sample_dt, yaw_low_speed=config.yaw_low_speed,
                                        gravity=config.gravity)
    except ValueError as error:
        result.update(trajectory_valid=False, diagnostic=str(error), objective=objective,
                      durations_s=data[:, 0].tolist(), coefficients=data[:, 1:].reshape(pieces, 3, 6).tolist())
        return result
    boundaries = np.concatenate(([0.0], np.cumsum(data[:, 0])))
    segments = tuple(PathCoverSegment(i // config.pieces_per_polytope, float(boundaries[i]),
                                     float(boundaries[i + 1]), float(data[i, 0])) for i in range(pieces))
    plan = PathCoverPlan(trajectory, tuple(problem["corridors"]), segments, 0.0, trajectory.duration,
                         np.stack((problem["initial_pva"][:, 0], problem["terminal_pva"][:, 0])),
                         problem["terminal_pva"][:, 0], {"optimizer_status": 1, "objective": objective})
    numerics = validate_pathcover_plan(plan, config)
    numerics.pop("optimizer_status")
    result.update(objective=objective, durations_s=data[:, 0].tolist(),
                  coefficients=trajectory.coefficients.tolist(), numerics=numerics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="torch.save of failed_problem(), or ROS failed-frontend-input.pt")
    parser.add_argument("--upstream", type=Path, help="Pinned unmodified PathCover checkout")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    captured = torch.load(args.input, map_location="cpu", weights_only=False)
    problem = captured.get("gcopter_problem", captured)
    if problem is None or "initial_pva" not in problem:
        raise ValueError("Capture does not contain NativePathCoverFrontend.failed_problem()")
    if "state" in captured:
        state = np.asarray(captured["state"]).reshape(-1)
        w, x, y, z = state[3:7]
        problem["initial_yaw"] = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    elif "current_state" in captured:
        state = captured["current_state"]
        problem["initial_yaw"] = float(state["yaw"] if isinstance(state, dict) else state.yaw)
    config_values = dict(problem["config"])
    # Offline migration of frozen V1-V7 evidence. The production API has only
    # body_yaw_rate_max; new replays use the V8 heading construction explicitly.
    if "yaw_rate_max" in config_values:
        config_values["body_yaw_rate_max"] = config_values.pop("yaw_rate_max")
    # Earlier frozen problems used the shared upstream 0.01 position width.
    config_values.setdefault("position_penalty_smoothing_m", 0.01)
    config = PathCoverConfig(**config_values)
    if args.max_iterations is not None:
        config = replace(config, max_iterations=args.max_iterations)
    root = Path(__file__).resolve().parents[1]
    text = serialized_problem(problem, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".input.txt").write_text(text + "\n")
    force = np.asarray(problem["initial_pva"])[:, 2] + np.array([0.0, 0.0, config.gravity])
    norm = float(np.linalg.norm(force))
    result = {"input": str(args.input), "captured_status": int(problem["optimizer_status"]),
              "yaw_profile": "body-r constrained heading, V8 semantics",
              "captured_trial_durations": [float(value) if np.isfinite(value) else None
                                           for value in problem.get("trial_durations", [])],
              "config": config.native_options(), "initial": {
                  "speed_m_s": float(np.linalg.norm(problem["initial_pva"][:, 1])),
                  "thrust_n": config.mass * norm,
                  "tilt_rad": float(np.arccos(np.clip(force[2] / norm, -1, 1))) if norm else None,
              }}
    ported = build_solver(root, root / "third_party/pathcover_native/trajectory", "ported", True)
    result["ported"] = replay(ported, text, problem, config, True)
    if args.upstream:
        revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != PIN:
            raise ValueError(f"Expected PathCover {PIN}, found {revision}")
        official = build_solver(root, args.upstream / "src/trajectory_server/include", "official", False)
        result["official"] = replay(official, text, problem, config, False)
        result["official"]["revision"] = revision
        result["official"]["boundary_projection"] = "upstream boundary-state clamping retained"
        result["official"]["position_penalty_smoothing_m"] = 0.01
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
