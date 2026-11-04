"""Bounded CTBR hover/step tests using actual Newton motors at 100 Hz."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASES = (("hover", 2.0, (0.0, 0.0, 1.0)), ("position_step", 4.0, (0.5, 0.2, 1.1)))
THRESHOLDS = {
    "prediction_constraint": 1e-5,
    "quaternion_norm": 1e-5,
    "hover_position": 0.03,
    "hover_speed": 0.05,
    "step_terminal_position": 0.10,
    "step_terminal_speed": 0.15,
    "actual_state_bound_tolerance": 0.02,
}


def source_snapshot():
    paths = list((ROOT / "src/isaaclab_ipc/turbompc").glob("*.py"))
    paths += [
        ROOT / "src/isaaclab_ipc" / name
        for name in (
            "tasks/env.py",
            "tasks/events.py",
            "tasks/ctbr_env.py",
            "assets/quadrotor.py",
            "assets/rate_control.py",
            "assets/newton_multirotor.py",
            "assets/data/quadrotor.usda",
        )
    ]
    paths.append(Path(__file__))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def np_step(z, cfg):
    """Independent author equations for checking accepted prediction dynamics."""
    p, q, v, force, omega = z[..., :3], z[..., 3:7], z[..., 7:10], z[..., 10:11], z[..., 11:14]
    w, x, y, zz = np.moveaxis(q, -1, 0)
    body_z = np.stack((2 * (x * zz + w * y), 2 * (y * zz - w * x), w * w - x * x - y * y + zz * zz), -1)
    acc = body_z * force / cfg.mass + np.array([0.0, 0.0, -cfg.gravity])
    angle = np.linalg.norm(omega, axis=-1, keepdims=True) * cfg.dt / 2
    increment = omega * (cfg.dt / 2) * np.sinc(angle / np.pi)
    scalar = q[..., :1] * np.cos(angle) - (q[..., 1:] * increment).sum(-1, keepdims=True)
    vector = q[..., :1] * increment + np.cos(angle) * q[..., 1:] + np.cross(q[..., 1:], increment)
    q_next = np.concatenate((scalar, vector), -1)
    q_next /= np.linalg.norm(q_next, axis=-1, keepdims=True) + 1e-8
    return np.concatenate((p + cfg.dt * v, q_next, v + cfg.dt * acc), -1), acc


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    before = source_snapshot()
    report = {
        "scope": "Newton motor/rigid-body CTBR validation; no ideal state replacement",
        "cases": [],
        "thresholds": THRESHOLDS,
        "sources_before": before,
        "passed": False,
    }
    artifacts, env, solver = {}, None, None
    try:
        if torch.version.cuda != "13.0" or os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu130":
            raise RuntimeError("Use the private cu130 wrapper with the main Isaac Lab Python")
        import warp as wp

        from isaaclab_ipc.assets.quadrotor import MASS
        from isaaclab_ipc.tasks.ctbr_env import checked_articulation_mass
        from isaaclab_ipc.tasks.env import IPCEnv, make_cfg
        from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
        from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
        from isaaclab_ipc.turbompc.ctbr_polish import POLISH_ERROR_NAMES, POLISH_FAILURE_REASONS, CTBRPolisher
        from isaaclab_ipc.turbompc.quadrotor import body_rates_from_isaac, controls_to_actions, state_from_isaac

        torch.manual_seed(2026)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = CTBRConfig(mass=MASS)
        sqp = SQPConfig(
            max_iter=60,
            tolerance=1e-7,
            linesearch="merit",
            compile_kernels=not args.no_compile,
            require_qp_convergence=True,
        )
        admm = ADMMConfig(max_iter=32768, eps_abs=1e-8, eps_rel=0.0)
        cfg = make_cfg(num_envs=1, scenario="empty")
        # Empty-field tests have no scanner. Only its two event terms are disabled.
        cfg.events.reset_scan = None
        cfg.events.advance_scan = None
        cfg.automatic_reset = False
        cfg.goal_x = 100.0  # Keep navigation termination outside these fixed tracking experiments.
        env = IPCEnv(cfg)
        report["actual_body_names"] = list(env.robot.body_names)
        report["actual_link_mass_kg"] = env.robot.data.body_mass.torch.cpu().tolist()
        masses = checked_articulation_mass(env.robot, config.mass)
        if tuple(cfg.sim.gravity) != (0.0, 0.0, -config.gravity):
            raise RuntimeError("Newton and CTBR gravity disagree")
        builder = CTBRQPBuilder(config).to("cuda:0")
        polisher = CTBRPolisher()
        solver = TorchMPC(builder.dynamics(), sqp, admm, constraints=builder.path_constraints(), polisher=polisher)
        report.update(
            model=asdict(config),
            sqp=asdict(sqp),
            admm=asdict(admm),
            polish={
                "max_iter": polisher.max_iter,
                "tolerance": polisher.tolerance,
                "max_linear_refinements": polisher.max_linear_refinements,
            },
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            actual_articulation_mass_kg=masses.cpu().tolist(),
            control_dt=env.step_dt,
            physics_dt=cfg.sim.dt,
            physical_environment="empty field with ground",
            environment_goal_x_for_termination_only=cfg.goal_x,
        )
        start = perf_counter()
        for case_index, (name, seconds, target_tuple) in enumerate(CASES):
            env.reset(seed=2026 + case_index)
            target = np.array(target_tuple)
            count = round(seconds / env.step_dt)
            trace, maximum_prediction_error = [], 0.0
            # Keep the live list registered so a mid-case exception preserves every completed step.
            artifacts[f"{name}_actual_trace"] = trace
            case = {"name": name, "target": target.tolist(), "requested_steps": count, "passed": False}
            report["cases"].append(case)
            reference = torch.zeros((1, config.horizon, 10), device="cuda:0", dtype=torch.float64)
            reference[..., :3] = torch.tensor(target, device="cuda:0")
            reference[..., 3] = 1.0
            costs = torch.zeros((1, 7), device="cuda:0", dtype=torch.float64)
            for index in range(count):
                with torch.inference_mode():
                    pose, twist = env.body_state()
                    state = state_from_isaac(pose, twist, env.scene.env_origins)
                    obs = dict(
                        ctbr_state=state,
                        ctbr_reference=reference,
                        planes=state.new_zeros((1, config.horizon, 0, 4)),
                        plane_mask=torch.zeros((1, config.horizon, 0), device=state.device, dtype=torch.bool),
                    )
                    data, guess = builder(obs, costs)
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    result = solver.solve(data, guess)
                    torch.cuda.synchronize()
                    solve_ms = (perf_counter() - tick) * 1000
                    z = result.trajectory[0].cpu().numpy()
                    native_status, qp_status = int(result.status[0]), int(result.admm.status[0])
                    predicted, acceleration = np_step(z[:-1], config)
                    violation = max(
                        float(np.abs(z[0, :10] - state[0].cpu().numpy()).max()),
                        float(np.abs(predicted - z[1:, :10]).max()),
                        float(np.maximum(np.asarray(config.velocity_lower) - z[1:, 7:10], 0).max()),
                        float(np.maximum(z[1:, 7:10] - np.asarray(config.velocity_upper), 0).max()),
                        float(np.maximum(builder.control_lower.cpu().numpy() - z[:-1, 10:], 0).max()),
                        float(np.maximum(z[:-1, 10:] - builder.control_upper.cpu().numpy(), 0).max()),
                        float(np.maximum(np.asarray(config.acceleration_lower) - acceleration, 0).max()),
                        float(np.maximum(acceleration - np.asarray(config.acceleration_upper), 0).max()),
                    )
                    maximum_prediction_error = max(maximum_prediction_error, violation)
                    artifacts[f"{name}_prediction_{index}"] = z
                    if result.polish is not None:
                        for field in fields(result.polish):
                            value = getattr(result.polish, field.name)
                            if isinstance(value, torch.Tensor):
                                artifacts[f"{name}_polish_{field.name}_{index}"] = value.cpu().numpy()
                    if native_status or qp_status or violation > THRESHOLDS["prediction_constraint"]:
                        case["failed_solve"] = dict(
                            step=index,
                            status=native_status,
                            raw_qp_status=qp_status,
                            errors=result.errors[0].cpu().tolist(),
                            normal_cone=float(result.normal_cone_error[0]),
                            independent_violation=violation,
                        )
                        if result.polish is not None:
                            polish = result.polish
                            case["failed_solve"]["polish"] = dict(
                                status=int(polish.status[0]),
                                iterations=int(polish.iterations[0]),
                                failure_reason=POLISH_FAILURE_REASONS[int(polish.failure_reason[0])],
                                errors=dict(zip(POLISH_ERROR_NAMES, polish.errors[0].cpu().tolist(), strict=True)),
                                linear_residual_history=polish.linear_residual_history[0].cpu().tolist(),
                                linear_refinement_steps=polish.linear_refinement_steps[0].cpu().tolist(),
                                prefix_rows_removed=int(polish.prefix_rows_removed[0]),
                            )
                        for field in fields(result.admm):
                            value = getattr(result.admm, field.name)
                            if isinstance(value, torch.Tensor):
                                artifacts[f"{name}_failed_raw_{field.name}_{index}"] = value.cpu().numpy()
                        break
                    action = controls_to_actions(result.controls[:, 0], config.mass).float()
                    env.step(action)
                    pose, twist = env.body_state()
                    actual = state_from_isaac(pose, twist, env.scene.env_origins)[0].cpu().numpy()
                    rates = body_rates_from_isaac(pose, twist)[0].cpu().numpy()
                    measured_acc = env.acceleration()[0].cpu().numpy()
                    motor_force = float(wp.to_torch(env.robot.applied_thrust_wp).sum())
                    trace.append(
                        np.r_[
                            actual,
                            rates,
                            measured_acc,
                            action[0].cpu().numpy(),
                            motor_force,
                            solve_ms,
                            violation,
                            int(result.iterations[0]),
                        ]
                    )
                    if env.hit.any():
                        case["collision_step"] = index
                        break
                if (index + 1) % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "step": index + 1,
                                "position": actual[:3].tolist(),
                                "position_error": float(np.linalg.norm(actual[:3] - target)),
                            }
                        ),
                        flush=True,
                    )
            values = np.asarray(trace).reshape(-1, 24)
            artifacts[f"{name}_actual_trace"] = values
            if len(values):
                position_error = np.linalg.norm(values[:, :3] - target, axis=-1)
                speed = np.linalg.norm(values[:, 7:10], axis=-1)
                q_error = np.abs(np.linalg.norm(values[:, 3:7], axis=-1) - 1)
                actual_state = np.concatenate((values[:, 7:10], values[:, 13:16]), -1)
                lo = np.array((*config.velocity_lower, *config.acceleration_lower))
                hi = np.array((*config.velocity_upper, *config.acceleration_upper))
                actual_bound_violation = float(np.maximum(np.maximum(lo - actual_state, actual_state - hi), 0).max())
                case.update(
                    completed_steps=len(values),
                    terminal_position_error=float(position_error[-1]),
                    terminal_speed=float(speed[-1]),
                    maximum_position_error=float(position_error.max()),
                    maximum_speed=float(speed.max()),
                    quaternion_norm_error=float(q_error.max()),
                    maximum_prediction_violation=maximum_prediction_error,
                    actual_state_bound_violation=actual_bound_violation,
                    minimum_physics_clearance=float(env.min_clearance[0]),
                    solve_p50_ms=float(np.median(values[:, 21])),
                    solve_p95_ms=float(np.quantile(values[:, 21], 0.95)),
                )
                tracking = (
                    (position_error.max() <= THRESHOLDS["hover_position"] and speed.max() <= THRESHOLDS["hover_speed"])
                    if name == "hover"
                    else (
                        position_error[-1] <= THRESHOLDS["step_terminal_position"]
                        and speed[-1] <= THRESHOLDS["step_terminal_speed"]
                    )
                )
                case["passed"] = bool(
                    len(values) == count
                    and not env.hit.any()
                    and tracking
                    and q_error.max() <= THRESHOLDS["quaternion_norm"]
                    and actual_bound_violation <= THRESHOLDS["actual_state_bound_tolerance"]
                    and maximum_prediction_error <= THRESHOLDS["prediction_constraint"]
                )
            print(json.dumps(safe_json(case)), flush=True)
        report["wall_s"] = perf_counter() - start
        report["passed"] = all(case["passed"] for case in report["cases"]) and len(report["cases"]) == len(CASES)
    except BaseException as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        report["passed"] = False
    finally:
        for resource_name, resource in (("solver", solver), ("environment", env)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    report.setdefault("close_errors", {})[resource_name] = repr(exc)
                    report["passed"] = False
        for key in artifacts:
            if key.endswith("_actual_trace"):
                artifacts[key] = np.asarray(artifacts[key]).reshape(-1, 24)
        after = source_snapshot()
        report["sources_after"] = after
        report["sources_unchanged"] = before == after
        report["passed"] = report["passed"] and before == after
        report["trace_columns"] = [
            "p_xyz",
            "q_wxyz",
            "v_world_xyz",
            "omega_body_xyz",
            "acc_world_xyz",
            "action_specific_thrust_body_rates",
            "motor_total_N",
            "solve_ms",
            "prediction_violation",
            "sqp_iterations",
        ]
        np.savez_compressed(args.output / "traces.npz", **artifacts)
        (args.output / "report.json").write_text(json.dumps(safe_json(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"output": str(args.output), "passed": report["passed"], "error": report.get("error")}), flush=True
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
