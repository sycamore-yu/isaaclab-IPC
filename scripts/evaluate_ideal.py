"""ROS-free ideal-tracker diagnostic; this is not a physics evaluation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from isaaclab_ipc.frontend import PlanningBatch
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner, flatness
from isaaclab_ipc.scenes import AvoidScene
from isaaclab_ipc.turbompc.ipc import FixedIPCSolver


def run(stride, distance, output):
    cfg = Config.load("avoid", stride, profile="ideal")
    planner = Planner(cfg, [10, 0, 1])
    scene = AvoidScene(distance)
    speed = 0.0
    x = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0.0])
    rot = np.eye(3)
    thrust = 9.81
    clearance = np.inf
    trace = []
    infos = []
    violations = 0
    with PlanningBatch() as batch, FixedIPCSolver(cfg.params) as solver:
        for step in range(3500):
            t = step * 0.01
            for cloud in scene.advance(t, x[:3], speed):
                planner.update_cloud(cloud, x[:3])
            state = np.r_[x[:3], x[3:6], rot[:, 2] * thrust - [0, 0, 9.81]]
            frontend_started = perf_counter()
            context = batch.contexts([planner], [state], {})[0]
            frontend_ms = (perf_counter() - frontend_started) * 1000
            solve_started = perf_counter()
            result = solver.solve([context])[0]
            solve_ms = (perf_counter() - solve_started) * 1000
            command, jerk, ok = planner.consume_state(
                result["state"], np.zeros(45) if result["solution"] is None else result["solution"],
                result["status"], t,
            )
            info = dict(frontend_ms=frontend_ms, solve_ms=solve_ms)
            infos.append(info)
            action = flatness(command[6:], jerk, rot)
            # IPC's internal thrust estimate is unclipped even when MAVROS output clips.
            thrust = action[0]
            old = x[:3].copy()
            x = command
            total = command[6:] + [0, 0, 9.81]
            z = total / np.linalg.norm(total)
            y = np.cross(z, [1, 0, 0])
            y /= np.linalg.norm(y)
            rot = np.column_stack((np.cross(y, z), y, z))
            speed = np.linalg.norm(x[:3] - old) / 0.01
            clearance = min(clearance, scene.clearance(x[:3]))
            values = np.r_[x[3:], jerk]
            lo = np.r_[planner.mpc.lower, planner.mpc.jlow[:3]]
            hi = np.r_[planner.mpc.upper, planner.mpc.jhigh[:3]]
            violations += bool(
                np.any((values < lo - 0.001 - 0.001 * np.abs(lo)) | (values > hi + 0.001 + 0.001 * np.abs(hi)))
            )
            trace.append(np.r_[t, x, jerk, clearance, speed])
            if np.linalg.norm(x[:3] - [10, 0, 1]) <= 0.5:
                break
    output.mkdir(parents=True, exist_ok=False)
    reached = np.linalg.norm(x[:3] - [10, 0, 1]) <= 0.5
    result = {
        "backend": "ideal-tracker",
        "profile": cfg.profile,
        "goal_crossed": bool(reached),
        "minimum_clearance_m": float(clearance),
        "violations": violations,
        "success": bool(reached and clearance > 0 and not violations),
        "steps": len(trace),
        "qp_failures": planner.failures,
        "frontend_failures": planner.frontend_failures,
        "trigger_speed_mps": scene.trigger_speed,
        "peak_fd_speed_mps": max(row[-1] for row in trace),
        "mean_frontend_ms": float(np.mean([i["frontend_ms"] for i in infos])),
        "mean_solve_ms": float(np.mean([i["solve_ms"] for i in infos])),
    }
    np.savez_compressed(output / "trace.npz", trace=np.array(trace))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.stride, a.distance, a.output)
