"""Fixed CTBR numerical and independent model-loop acceptance, without Newton.

All four predeclared cases and all 112 cost perturbations count. The NumPy
canonical discrete map and finite differences are independent of Torch AD.
The author solver is not a same-problem SFC optimizer; no A/B latency or
nonlinear optimizer parity claim is made here. Newton has a separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr_polish import POLISH_ERROR_NAMES, POLISH_FAILURE_REASONS, CTBRPolisher

ROOT = Path(__file__).resolve().parents[2]
AUTHOR_COMMIT = "14aa349bb1f509c529bf3662c7feb72db68f1cb7"
AUTHOR_SHA = "090d99a39f2cfccf98ae872486f01fbe6a0976c2a3bf499550b7b34379759246"
HORIZON, DT, MASS, GRAVITY, LOOP_STEPS = 15, 0.1, 0.75, 9.81, 60
HOVER = np.array([MASS * GRAVITY, 0.0, 0.0, 0.0])
CONTROL_LOWER = np.array([MASS * 0.08 * GRAVITY / 0.7, -1.5, -1.5, -0.1])
CONTROL_UPPER = np.array([MASS * 0.9 * GRAVITY / 0.7, 1.5, 1.5, 0.1])
ACC_LOWER, ACC_UPPER = np.array([-20.0, -20.0, -10.0]), np.array([20.0, 20.0, 20.0])
FD_STEPS = (1e-3, 5e-4)
MODEL_FD_STEPS = (1e-5, 5e-6)
SQP = SQPConfig(max_iter=60, tolerance=1e-7, linesearch="merit", compile_kernels=False, require_qp_convergence=True)
ADMM = ADMMConfig(max_iter=32768, eps_abs=1e-8, eps_rel=0.0)
THRESHOLDS = {
    "data_max_abs": 1e-11,
    "numpy_model_max_abs": 1e-12,
    "author_model_max_abs": 1e-7,
    "model_jacobian_max_abs": 1e-7,
    "model_jacobian_relative": 1e-6,
    "physical_violation": 1e-5,
    "independent_stationarity": 1e-5,
    "quaternion_norm_error": 1e-5,
    "gradient_relative": 1e-3,
    "gradient_nonzero_floor": 1e-12,
    "active_distance": 1e-6,
    "inactive_distance_margin": 1e-4,
    "hover_position": 0.01,
    "hover_velocity": 0.01,
    "hover_attitude_rad": 0.01,
    "step_final_position": 0.1,
    "step_final_velocity": 0.1,
    "attitude_tail_rms_position": 0.15,
    "attitude_tail_rms_velocity": 0.15,
    "attitude_tail_rms_angle_rad": 0.1,
    "sfc_final_position": 0.05,
    "sfc_final_velocity": 0.1,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    return value.detach().cpu().numpy().copy()


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(child) for child in value]
    return value


def snapshot():
    files = sorted((ROOT / "src/isaaclab_ipc/turbompc").glob("*.py"))
    files.append(Path(__file__))
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def maximum(value):
    return float(np.max(np.abs(value), initial=0.0))


def comparison(actual, reference, tolerance):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        return {"passed": False, "actual_shape": actual.shape, "reference_shape": reference.shape}
    finite = np.isfinite(actual) & np.isfinite(reference)
    infinities_match = np.array_equal(np.isposinf(actual), np.isposinf(reference)) and np.array_equal(
        np.isneginf(actual), np.isneginf(reference)
    )
    difference = maximum(actual[finite] - reference[finite])
    good = infinities_match and not np.isnan(actual).any() and not np.isnan(reference).any()
    return {
        "max_abs": difference,
        "infinities_match": infinities_match,
        "passed": bool(good and difference <= tolerance),
    }


def gradient_comparison(actual, reference):
    denominator = float(np.linalg.norm(reference))
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    relative = float(np.linalg.norm(actual - reference) / max(denominator, THRESHOLDS["gradient_nonzero_floor"]))
    return {
        "reference_l2": denominator,
        "relative_l2": relative,
        "coordinate_abs_error": np.abs(actual - reference),
        "passed": finite and denominator > THRESHOLDS["gradient_nonzero_floor"] and relative <= 1e-3,
    }


def multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    qx = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
    qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return multiply(multiply(qz, qy), qx)


def acceleration(z):
    """Raw Hamilton q [0,0,F] q*, including off-unit SQP intermediate states."""
    qw, qx, qy, qz = np.moveaxis(z[..., 3:7], -1, 0)
    vertical = np.stack(
        (2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), qw * qw - qx * qx - qy * qy + qz * qz), axis=-1
    )
    return vertical * (z[..., 10:11] / MASS) - np.array([0.0, 0.0, GRAVITY])


def step(z):
    """Canonical p Euler, right quaternion exponential, velocity from old q."""
    omega = z[..., 11:14]
    angle = np.linalg.norm(omega, axis=-1, keepdims=True) * DT / 2
    exponential = np.concatenate((np.cos(angle), omega * (DT / 2) * np.sinc(angle / np.pi)), axis=-1)
    q = multiply(z[..., 3:7], exponential)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    return np.concatenate((z[..., :3] + DT * z[..., 7:10], q, z[..., 7:10] + DT * acceleration(z)), axis=-1)


def numerical_jacobian(function, z, epsilon):
    columns = []
    for coordinate in range(z.shape[-1]):
        offset = np.zeros_like(z)
        width = epsilon * np.maximum(1.0, np.abs(z[..., coordinate]))
        offset[..., coordinate] = width
        columns.append((function(z + offset) - function(z - offset)) / (2 * width[..., None]))
    return np.stack(columns, axis=-1)


def angle_error(q, target):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, axis=-1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs(np.sum(q * target, axis=-1)), 0.0, 1.0))


def author_model(path):
    if digest(path) != AUTHOR_SHA:
        raise RuntimeError("The canonical author forward file does not match its pinned SHA256")
    repo = path.parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise RuntimeError("Unexpected author reference commit")
    spec = importlib.util.spec_from_file_location("ctbr_author_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DroneDynamics(device="cpu")
    # Unmodified code, explicitly harmonized physical constants for a value comparison.
    model.mass = MASS
    model.g = torch.tensor([0.0, 0.0, -GRAVITY], dtype=torch.float64)
    return model, {
        "path": str(path),
        "commit": commit,
        "sha256": AUTHOR_SHA,
        "entrypoint": "DroneDynamics.forward",
        "physical_overrides": {"mass": MASS, "g": [0, 0, -GRAVITY], "dt": DT},
        "differences": [
            "Native exponential uses its smooth sinc limit instead of the author's norm+1e-8 angle.",
            "The unconditional quaternion norm+1e-8 projection and raw Hamilton acceleration are retained.",
            "The author's Euler get_jacobian and unit-assuming traced forward are not derivative oracles.",
        ],
    }


def fixed_cases():
    state = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    length = LOOP_STEPS + HORIZON + 1
    hover = np.tile(state, (length, 1))
    position = hover.copy()
    position[:, :3] = (0.35, -0.2, 1.1)
    attitude = np.empty_like(hover)
    attitude[0] = state
    for index in range(length - 1):
        time = index * DT
        rates = np.array(
            [
                0.12 * np.cos(2 * np.pi * time / 3),
                0.08 * np.cos(2 * np.pi * time / 4),
                0.04 * np.cos(2 * np.pi * time / 5),
            ]
        )
        q = attitude[index, 3:7]
        vertical = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
        control = np.r_[MASS * GRAVITY / vertical, rates]
        attitude[index + 1] = step(np.r_[attitude[index], control])
    wall_reference = hover.copy()
    wall_reference[:, 0] = 0.8
    broad = np.array(
        [[1, 0, 0, -4], [-1, 0, 0, -4], [0, 1, 0, -4], [0, -1, 0, -4], [0, 0, 1, -3], [0, 0, -1, 0.2]], dtype=float
    )
    wall = broad.copy()
    wall[0, 3] = -0.2
    bank = []
    for name, reference in zip(
        ("hover", "position_step", "attitude_trajectory", "terminal_sfc"),
        (hover, position, attitude, wall_reference),
        strict=True,
    ):
        initial = state.copy()
        costs = np.array([0.2, -0.1, 0.15, -0.05, 0.03, -0.02, 0.01])
        if name == "hover":
            costs[:] = 0
        if name == "attitude_trajectory":
            initial[3:7] = quaternion(0.08, -0.06, 0.02)
            initial[7:] = (0.03, -0.02, 0.01)
        if name == "terminal_sfc":
            initial[7] = 0.1
        planes = np.tile(broad, (HORIZON, 1, 1))
        if name == "hover":
            planes[:] = 0
        if name == "terminal_sfc":
            planes[-1] = wall
        mask = np.ones(planes.shape[:-1], dtype=bool)
        if name == "hover":
            mask[:] = False
        bank.append(
            {
                "name": name,
                "state": initial,
                "reference": reference,
                "planes": planes,
                "mask": mask,
                "costs": costs,
                "loop_planes": np.tile(wall if name == "terminal_sfc" else broad, (HORIZON, 1, 1)),
            }
        )
    return bank


def observation(bank, *, device="cpu", states=None, time=0, closed_loop=False):
    data = {
        "ctbr_state": np.stack([case["state"] for case in bank]) if states is None else states,
        "ctbr_reference": np.stack([case["reference"][time + 1 : time + HORIZON + 1] for case in bank]),
        "planes": np.stack([case["loop_planes"] if closed_loop else case["planes"] for case in bank]),
        "plane_mask": np.stack([case["mask"] for case in bank]),
    }
    return {key: torch.as_tensor(value, device=device) for key, value in data.items()}


def independent_cost(obs, costs):
    reference = array(obs["ctbr_reference"])
    initial = array(obs["ctbr_state"])
    # Fix quaternion hemisphere once while preparing the observed reference.
    previous = initial[:, 3:7]
    for index in range(HORIZON):
        sign = np.where(np.sum(previous * reference[:, index, 3:7], axis=-1) < 0, -1.0, 1.0)
        reference[:, index, 3:7] *= sign[:, None]
        previous = reference[:, index, 3:7]
    scale = np.exp(np.log(2) * np.tanh(costs[:, :4]))
    state_weights = np.concatenate(
        (
            10 * np.repeat(scale[:, 0:1], 3, axis=-1),
            4 * np.repeat(scale[:, 2:3], 4, axis=-1),
            2 * np.repeat(scale[:, 1:2], 3, axis=-1),
        ),
        axis=-1,
    )
    state_weights = np.tile(state_weights[:, None], (1, HORIZON, 1))
    state_weights[:, -1] *= 5
    control_weights = scale[:, 3:4] * np.array([0.2, 0.1, 0.1, 0.2])
    diagonal = np.zeros((len(costs), HORIZON + 1, 14))
    diagonal[:, 1:, :10] = state_weights
    diagonal[:, :-1, 10:] = control_weights[:, None]
    diagonal[:, -1, 10:] = 1
    q = np.zeros_like(diagonal)
    q[:, 1:, :10] = -state_weights * reference
    bias = 0.2 * np.tanh(costs[:, 4:])
    q[:, 1:, :3] -= 10 * bias[:, None]
    q[:, -1, :3] -= 40 * bias
    q[:, :-1, 10:] = -control_weights[:, None] * HOVER
    q[:, -1, 10:] = -HOVER
    return diagonal[..., None] * np.eye(14), q


def cpu_checks(builder, bank, author, artifacts):
    from isaaclab_ipc.turbompc.quadrotor import body_rates_from_isaac, controls_to_actions, state_from_isaac

    probes = np.tile(np.r_[bank[0]["state"], HOVER], (6, 1))
    probes[1, 11:] = (1e-10, -2e-10, 3e-10)
    probes[2, 3:7], probes[2, 11:], probes[2, 10] = quaternion(0.2, -0.15, 0.3), (0.7, -0.4, 0.03), HOVER[0] * 1.1
    probes[3] = probes[2]
    probes[3, 3:7] *= -1
    probes[4] = probes[2]
    probes[4, 3:7] *= 1.02
    probes[5, 3:7], probes[5, 11:] = quaternion(yaw=np.pi / 2), (0.3, 0, 0)
    z = torch.from_numpy(probes)
    values, jacobian = builder.dynamics().linearize(z)
    oracle = step(probes)
    original = array(author.forward(z[:, :10], z[:, 10:], DT))
    jacobians = {}
    for epsilon in MODEL_FD_STEPS:
        finite_difference = numerical_jacobian(step, probes, epsilon)
        jacobians[str(epsilon)] = {
            **comparison(array(jacobian), finite_difference, THRESHOLDS["model_jacobian_max_abs"]),
            "relative_l2": float(
                np.linalg.norm(array(jacobian) - finite_difference) / np.linalg.norm(finite_difference)
            ),
        }
        jacobians[str(epsilon)]["passed"] &= (
            jacobians[str(epsilon)]["relative_l2"] <= THRESHOLDS["model_jacobian_relative"]
        )
        artifacts[f"model_jacobian_fd_{epsilon}"] = finite_difference
    obs = observation(bank)
    costs = np.stack([case["costs"] for case in bank])
    inputs, guess = builder(obs, torch.from_numpy(costs))
    expected_d, expected_q = independent_cost(obs, costs)
    objective = {
        "D": comparison(array(inputs.D), expected_d, THRESHOLDS["data_max_abs"]),
        "q": comparison(array(inputs.q), expected_q, THRESHOLDS["data_max_abs"]),
        "E": comparison(array(inputs.E), np.zeros_like(array(inputs.E)), THRESHOLDS["data_max_abs"]),
    }
    expected_g = np.zeros_like(array(inputs.G))
    expected_lower = np.full_like(array(inputs.lower), -np.inf)
    expected_upper = np.full_like(array(inputs.upper), np.inf)
    expected_g[:, 1:, :3, 7:10] = np.eye(3)
    expected_lower[:, 1:, :3], expected_upper[:, 1:, :3] = -10, 10
    expected_g[:, :-1, 3:7, 10:] = np.eye(4)
    expected_lower[:, :-1, 3:7], expected_upper[:, :-1, 3:7] = CONTROL_LOWER, CONTROL_UPPER
    planes, mask = array(obs["planes"]), array(obs["plane_mask"])
    expected_g[:, 1:, 7:, :3] = np.where(mask[..., None], planes[..., :3], 0)
    expected_upper[:, 1:, 7:] = np.where(mask, -planes[..., 3], np.inf)
    linear = {
        name: comparison(array(getattr(inputs, name)), expected, THRESHOLDS["data_max_abs"])
        for name, expected in (("G", expected_g), ("lower", expected_lower), ("upper", expected_upper))
    }
    pose = torch.tensor([[2.0, 3.0, 4.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=torch.float64)
    twist = torch.tensor([[0.3, -0.2, 0.1, 0.1, 0.2, 0.3]], dtype=torch.float64)
    origin = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    adapters = {
        "state": comparison(
            array(state_from_isaac(pose, twist, origin)),
            np.array([[1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5), 0.3, -0.2, 0.1]]),
            1e-12,
        ),
        "body_rates": comparison(array(body_rates_from_isaac(pose, twist)), np.array([[0.2, -0.1, 0.3]]), 1e-12),
        "force_to_specific_thrust": comparison(
            array(controls_to_actions(torch.from_numpy(HOVER), MASS)), np.array([GRAVITY, 0, 0, 0]), 1e-12
        ),
    }
    path = builder.path_constraints()
    phi = comparison(array(path(guess)), acceleration(array(guess)), THRESHOLDS["numpy_model_max_abs"])
    stage_mask = array(path.stage_mask)
    mask_good = stage_mask.shape == (HORIZON + 1, 3) and stage_mask[:-1].all() and not stage_mask[-1].any()
    paths = {
        "phi": phi,
        "stage_mask": stage_mask,
        "stage_mask_passed": bool(mask_good),
        "lower": comparison(array(path.lower), ACC_LOWER, 1e-12),
        "upper": comparison(array(path.upper), ACC_UPPER, 1e-12),
    }
    output = {
        "numpy_forward": comparison(array(values), oracle, THRESHOLDS["numpy_model_max_abs"]),
        "author_forward": comparison(array(values), original, THRESHOLDS["author_model_max_abs"]),
        "jacobians": jacobians,
        "objective": objective,
        "linear_constraints": linear,
        "coordinate_and_unit_adapters": adapters,
        "path_constraints": paths,
        "probes": [
            "hover_zero_rate",
            "near_zero_rate",
            "attitude_rate",
            "negative_quaternion",
            "off_unit_quaternion",
            "body_world_axis",
        ],
    }
    output["passed"] = bool(
        output["numpy_forward"]["passed"]
        and output["author_forward"]["passed"]
        and all(item["passed"] for item in jacobians.values())
        and all(item["passed"] for item in objective.values())
        and all(item["passed"] for item in linear.values())
        and all(item["passed"] for item in adapters.values())
        and phi["passed"]
        and mask_good
        and paths["lower"]["passed"]
        and paths["upper"]["passed"]
    )
    artifacts.update(
        model_probes=probes,
        model_native=array(values),
        model_numpy=oracle,
        model_author=original,
        model_jacobian_native=array(jacobian),
    )
    return output


def constraints_and_active(trajectory, obs):
    batch = len(trajectory)
    velocity, control = trajectory[:, 1:, 7:10], trajectory[:, :-1, 10:]
    acc = acceleration(trajectory[:, :-1])
    planes, mask = array(obs["planes"]), array(obs["plane_mask"])
    faces = np.einsum("btfj,btj->btf", planes[..., :3], trajectory[:, 1:, :3]) + planes[..., 3]
    lower_slack = np.concatenate(
        (
            (velocity + 10).reshape(batch, -1),
            (control - CONTROL_LOWER).reshape(batch, -1),
            (acc - ACC_LOWER).reshape(batch, -1),
        ),
        axis=-1,
    )
    upper_slack = np.concatenate(
        (
            (10 - velocity).reshape(batch, -1),
            (CONTROL_UPPER - control).reshape(batch, -1),
            (ACC_UPPER - acc).reshape(batch, -1),
        ),
        axis=-1,
    )
    face_distance = np.where(mask, -faces / np.maximum(np.linalg.norm(planes[..., :3], axis=-1), 1e-12), np.inf)
    lower_slack = np.concatenate((lower_slack, np.full((batch, faces[0].size), np.inf)), axis=-1)
    upper_slack = np.concatenate((upper_slack, face_distance.reshape(batch, -1)), axis=-1)
    distance = np.minimum(lower_slack, upper_slack)
    active_lower = lower_slack <= THRESHOLDS["active_distance"]
    active_upper = upper_slack <= THRESHOLDS["active_distance"]
    inactive = np.where(active_lower | active_upper, np.inf, distance)
    return {
        "velocity": np.max(np.maximum(np.abs(velocity) - 10, 0), axis=(1, 2)),
        "control": np.max(np.maximum(np.maximum(CONTROL_LOWER - control, control - CONTROL_UPPER), 0), axis=(1, 2)),
        "acceleration": np.max(np.maximum(np.maximum(ACC_LOWER - acc, acc - ACC_UPPER), 0), axis=(1, 2)),
        "sfc": np.max(np.where(mask, np.maximum(faces, 0), 0), axis=(1, 2)),
        "active_lower": active_lower,
        "active_upper": active_upper,
        "minimum_inactive_distance": np.min(inactive, axis=-1),
        "sfc_active_count": np.sum((face_distance <= THRESHOLDS["active_distance"]) & mask, axis=(1, 2)),
    }


def assess(inputs, result, obs):
    z = array(result.trajectory)
    data = {name: array(getattr(inputs, name)) for name in inputs.__dataclass_fields__}
    duals = result.polish if result.polish is not None else result.admm
    y = array(duals.y)
    m = data["G"].shape[-2]
    physical = constraints_and_active(z, obs)
    initial = np.max(np.abs(z[:, 0, :10] - array(obs["ctbr_state"])), axis=-1)
    dynamics = np.max(np.abs(step(z[:, :-1]) - z[:, 1:, :10]), axis=(1, 2))
    stationarity = np.einsum("btij,btj->bti", data["D"], z) + data["q"]
    stationarity[:, 1:] += np.einsum("btij,btj->bti", data["E"], z[:, :-1])
    stationarity[:, :-1] += np.einsum("btji,btj->bti", data["E"], z[:, 1:])
    stationarity += np.einsum("btji,btj->bti", data["G"], y[..., :m])
    jd = numerical_jacobian(step, z[:, :-1], MODEL_FD_STEPS[-1])
    ja = numerical_jacobian(acceleration, z[:, :-1], MODEL_FD_STEPS[-1])
    stationarity[:, 0, :10] += array(duals.lambda0)
    stationarity[:, :-1] += np.einsum("btji,btj->bti", jd, array(duals.lambda_dyn))
    stationarity[:, 1:, :10] -= array(duals.lambda_dyn)
    stationarity[:, :-1] += np.einsum("btji,btj->bti", ja, y[:, :-1, m:])
    linear_value = np.einsum("btij,btj->bti", data["G"], z)
    linear_normal = linear_value - np.clip(linear_value + y[..., :m], data["lower"], data["upper"])
    phi = acceleration(z[:, :-1])
    path_normal = phi - np.clip(phi + y[:, :-1, m:], ACC_LOWER, ACC_UPPER)
    records = []
    for row in range(len(z)):
        measures = {key: float(physical[key][row]) for key in ("velocity", "control", "acceleration", "sfc")}
        measures.update(
            initial_equality=float(initial[row]),
            dynamics_equality=float(dynamics[row]),
            quaternion_norm_error=maximum(np.linalg.norm(z[row, :, 3:7], axis=-1) - 1),
            independent_stationarity=maximum(stationarity[row]),
            independent_normal_cone=max(maximum(linear_normal[row]), maximum(path_normal[row])),
            terminal_control_error=maximum(z[row, -1, 10:] - HOVER),
        )
        reported_normal = getattr(result, "normal_cone_error", None)
        finite = bool(
            np.isfinite(z[row]).all() and np.isfinite(y[row]).all() and np.isfinite(list(measures.values())).all()
        )
        good = finite and int(result.status[row]) == 0 and int(result.admm.status[row]) == 0
        good = good and max(measures.values()) <= THRESHOLDS["physical_violation"]
        good = good and reported_normal is not None and float(reported_normal[row]) <= SQP.tolerance
        good = (
            good
            and bool(torch.isfinite(result.errors[row]).all())
            and maximum(array(result.errors[row])) <= SQP.tolerance
        )
        good = good and bool(torch.isfinite(result.admm.residuals[row]).all())
        records.append(
            {
                "status": int(result.status[row]),
                "raw_qp_status": int(result.admm.status[row]),
                "sqp_iterations": int(result.iterations[row]),
                "last_admm_iterations": int(result.admm.iterations[row]),
                "raw_qp_residuals": array(result.admm.residuals[row]),
                "reported_errors": array(result.errors[row]),
                "normal_cone_error": None if reported_normal is None else float(reported_normal[row]),
                "residuals": measures,
                "active_lower_indices": np.flatnonzero(physical["active_lower"][row]),
                "active_upper_indices": np.flatnonzero(physical["active_upper"][row]),
                "minimum_inactive_distance": float(physical["minimum_inactive_distance"][row]),
                "sfc_active_count": int(physical["sfc_active_count"][row]),
                "u0": z[row, 0, 10:].copy(),
                "finite": finite,
                "passed": bool(good),
            }
        )
        records[-1]["dual_source"] = "polished" if result.polish is not None else "raw_admm"
        if result.polish is not None:
            polish = result.polish
            certificate = {
                "status": int(polish.status[row]),
                "iterations": int(polish.iterations[row]),
                "errors": dict(zip(POLISH_ERROR_NAMES, array(polish.errors[row]), strict=True)),
                "failure_reason": POLISH_FAILURE_REASONS[int(polish.failure_reason[row])],
                "prefix_rows_removed": int(polish.prefix_rows_removed[row]),
                "linear_refinement_steps": array(polish.linear_refinement_steps[row]),
            }
            records[-1]["polish"] = certificate
            records[-1]["passed"] &= certificate["status"] == 0
    return records


def save_solver_details(artifacts, prefix, result):
    """Preserve solver points and multipliers for later KKT and physical audits."""
    for name in ("x", "lambda0", "lambda_dyn", "y", "status", "residuals"):
        artifacts[f"{prefix}_raw_{name}"] = array(getattr(result.admm, name))
    if result.polish is not None:
        for name in (
            "lambda0",
            "lambda_dyn",
            "y",
            "active_mask",
            "status",
            "iterations",
            "errors",
            "linear_residuals",
            "linear_residual_history",
            "linear_refinement_steps",
            "failure_reason",
            "prefix_rows_removed",
        ):
            artifacts[f"{prefix}_polish_{name}"] = array(getattr(result.polish, name))


def same_active(a, b):
    return np.array_equal(a["active_lower_indices"], b["active_lower_indices"]) and np.array_equal(
        a["active_upper_indices"], b["active_upper_indices"]
    )


def loss_weights():
    index = np.arange(HORIZON * 4).reshape(HORIZON, 4)
    return (np.sin(0.17 * index + 0.4) + 0.3 * np.cos(0.29 * index)) / HORIZON / np.array([MASS * GRAVITY, 1, 1, 1])


def open_loop(builder, solver, bank, artifacts, output, *, base_only=False):
    obs = observation(bank, device="cuda:0")
    base_costs = np.stack([case["costs"] for case in bank])
    costs = torch.tensor(base_costs, device="cuda:0", dtype=torch.float64, requires_grad=True)
    inputs, guess = builder(obs, costs)
    result = solver.solve(inputs, guess)
    weights = loss_weights()
    records = assess(inputs, result, obs)
    artifacts["base_trajectory"] = array(result.trajectory)
    save_solver_details(artifacts, "base", result)
    artifacts["base_costs"] = base_costs
    artifacts["loss_weights"] = weights
    try:
        loss = (result.controls[:, :-1] * costs.new_tensor(weights)).sum()
        gradients = array(torch.autograd.grad(loss, costs)[0])
        gradient_error = None
    except Exception as error:
        gradients = np.full_like(base_costs, np.nan)
        gradient_error = f"{type(error).__name__}: {error}"
    artifacts["cost_gradient"] = gradients
    reports = [
        {
            "name": case["name"],
            "base": record,
            "autograd": gradients[row],
            "autograd_error": gradient_error,
            "steps": {},
            "expected_sfc_active": case["name"] == "terminal_sfc",
        }
        for row, (case, record) in enumerate(zip(bank, records, strict=True))
    ]
    output.update(
        cases=reports, scheduled_sample_solves=4 if base_only else 116, completed_sample_solves=4, passed=False
    )
    if base_only:
        output["base_diagnostic_passed"] = bool(
            all(record["passed"] for record in records) and np.isfinite(gradients).all()
        )
        return output
    for epsilon in FD_STEPS:
        fd = np.empty_like(base_costs)
        checks = [[] for _ in bank]
        for coordinate in range(7):
            losses = []
            perturbation_records = []
            for sign in (1, -1):
                perturbed = base_costs.copy()
                perturbed[:, coordinate] += sign * epsilon
                with torch.no_grad():
                    data, fixed_guess = builder(obs, costs.new_tensor(perturbed))
                    candidate = solver.solve(data, fixed_guess)
                trajectory = array(candidate.trajectory)
                artifact_key = f"fd_{epsilon}_{coordinate}_{sign}"
                artifacts[artifact_key] = trajectory
                save_solver_details(artifacts, artifact_key, candidate)
                losses.append(np.sum(trajectory[:, :-1, 10:] * weights, axis=(1, 2)))
                perturbation_records.append(assess(data, candidate, obs))
                output["completed_sample_solves"] += len(bank)
            fd[:, coordinate] = (losses[0] - losses[1]) / (2 * epsilon)
            for row in range(len(bank)):
                plus, minus = perturbation_records[0][row], perturbation_records[1][row]
                stable = same_active(records[row], plus) and same_active(records[row], minus)
                stable &= min(plus["minimum_inactive_distance"], minus["minimum_inactive_distance"]) >= 1e-4
                checks[row].append(
                    {
                        "coordinate": coordinate,
                        "plus": plus,
                        "minus": minus,
                        "active_set_unchanged": bool(stable),
                        "passed": bool(plus["passed"] and minus["passed"] and stable),
                    }
                )
            print(f"cost FD h={epsilon:g}, coordinate={coordinate}: completed all 4 cases", flush=True)
        artifacts[f"fd_gradient_{epsilon}"] = fd
        for row, report in enumerate(reports):
            comparison_result = gradient_comparison(gradients[row], fd[row])
            report["steps"][str(epsilon)] = {
                "gradient": fd[row],
                "checks": checks[row],
                "autograd_comparison": comparison_result,
                "passed": comparison_result["passed"] and all(x["passed"] for x in checks[row]),
            }
    for report in reports:
        report["step_consistency"] = gradient_comparison(
            report["steps"][str(FD_STEPS[1])]["gradient"], report["steps"][str(FD_STEPS[0])]["gradient"]
        )
        active_expected = (report["base"]["sfc_active_count"] > 0) == report["expected_sfc_active"]
        report["expected_active_set_passed"] = active_expected
        report["passed"] = bool(
            report["base"]["passed"]
            and active_expected
            and report["base"]["minimum_inactive_distance"] >= 1e-4
            and all(item["passed"] for item in report["steps"].values())
            and report["step_consistency"]["passed"]
        )
    output["passed"] = all(report["passed"] for report in reports)
    return output


def model_loops(builder, solver, bank, artifacts, output):
    states = np.stack([case["state"] for case in bank])
    costs = torch.tensor(np.stack([case["costs"] for case in bank]), device="cuda:0", dtype=torch.float64)
    history, controls, records = [states.copy()], [], []
    output.update(scheduled_sample_solves=240, completed_sample_solves=0, passed=False, step_records=records)
    for index in range(LOOP_STEPS):
        print(f"model loop step {index + 1}/{LOOP_STEPS}: starting all four solves", flush=True)
        obs = observation(bank, device="cuda:0", states=states, time=index, closed_loop=True)
        with torch.no_grad():
            inputs, guess = builder(obs, costs)
            result = solver.solve(inputs, guess)
        artifacts[f"model_loop_prediction_{index:03d}"] = array(result.trajectory)
        save_solver_details(artifacts, f"model_loop_{index:03d}", result)
        checks = assess(inputs, result, obs)
        records.append(checks)
        output["completed_sample_solves"] += len(bank)
        print(
            f"model loop step {index + 1}: status={[c['status'] for c in checks]}, "
            f"raw={[c['raw_qp_status'] for c in checks]}, SQP={[c['sqp_iterations'] for c in checks]}, "
            f"last ADMM={[c['last_admm_iterations'] for c in checks]}",
            flush=True,
        )
        if not all(check["passed"] for check in checks):
            # Keep the failed step; an invalid command is not executed by the independent plant.
            break
        command = array(result.controls[:, 0])
        plant_z = np.concatenate((states, command), axis=-1)
        next_states = step(plant_z)
        planes, mask = array(obs["planes"])[:, 0], array(obs["plane_mask"])[:, 0]
        faces = np.einsum("bfj,bj->bf", planes[..., :3], next_states[:, :3]) + planes[..., 3]
        plant_acc = acceleration(plant_z)
        for row, check in enumerate(checks):
            actual = {
                "velocity": maximum(np.maximum(np.abs(next_states[row, 7:]) - 10, 0)),
                "control": maximum(
                    np.maximum(np.maximum(CONTROL_LOWER - command[row], command[row] - CONTROL_UPPER), 0)
                ),
                "acceleration": maximum(
                    np.maximum(np.maximum(ACC_LOWER - plant_acc[row], plant_acc[row] - ACC_UPPER), 0)
                ),
                "sfc": maximum(np.where(mask[row], np.maximum(faces[row], 0), 0)),
                "quaternion_norm_error": float(abs(np.linalg.norm(next_states[row, 3:7]) - 1)),
                "prediction_first_step": maximum(next_states[row] - array(result.states[row, 1])),
            }
            check["plant_residuals"] = actual
            check["passed"] &= bool(
                np.isfinite(list(actual.values())).all() and max(actual.values()) <= THRESHOLDS["physical_violation"]
            )
        states = next_states
        history.append(states.copy())
        controls.append(command)
        artifacts["model_loop_states"] = np.stack(history, axis=1)
        artifacts["model_loop_controls"] = np.stack(controls, axis=1)
        if not all(check["passed"] for check in checks):
            break
        if (index + 1) % 10 == 0:
            print(f"model loops: {index + 1}/{LOOP_STEPS} steps executed for every case", flush=True)
    history = np.stack(history, axis=1)
    artifacts["model_loop_states"] = history
    artifacts["model_loop_controls"] = np.stack(controls, axis=1) if controls else np.empty((len(bank), 0, 4))
    complete = len(records) == LOOP_STEPS and len(controls) == LOOP_STEPS
    reports = []
    for row, case in enumerate(bank):
        tracked = history[row, 1:]
        target = case["reference"][1 : len(tracked) + 1].copy()
        if case["name"] == "terminal_sfc":
            target[:, 0] = 0.2
        if len(tracked):
            position_error = np.linalg.norm(tracked[:, :3] - target[:, :3], axis=-1)
            velocity_error = np.linalg.norm(tracked[:, 7:] - target[:, 7:], axis=-1)
            orientation_error = angle_error(tracked[:, 3:7], target[:, 3:7])
            metrics = {
                "maximum_position_error": float(position_error.max()),
                "maximum_velocity_error": float(velocity_error.max()),
                "maximum_attitude_error_rad": float(orientation_error.max()),
                "final_position_error": float(position_error[-1]),
                "final_velocity_error": float(velocity_error[-1]),
                "tail_rms_position": float(np.sqrt(np.mean(position_error[-10:] ** 2))),
                "tail_rms_velocity": float(np.sqrt(np.mean(velocity_error[-10:] ** 2))),
                "tail_rms_attitude_rad": float(np.sqrt(np.mean(orientation_error[-10:] ** 2))),
                "quaternion_norm_error": maximum(np.linalg.norm(tracked[:, 3:7], axis=-1) - 1),
            }
            if case["name"] == "hover":
                tracking_passed = (
                    metrics["maximum_position_error"] <= 0.01
                    and metrics["maximum_velocity_error"] <= 0.01
                    and metrics["maximum_attitude_error_rad"] <= 0.01
                )
            elif case["name"] == "position_step":
                tracking_passed = metrics["final_position_error"] <= 0.1 and metrics["final_velocity_error"] <= 0.1
            elif case["name"] == "attitude_trajectory":
                tracking_passed = (
                    metrics["tail_rms_position"] <= 0.15
                    and metrics["tail_rms_velocity"] <= 0.15
                    and metrics["tail_rms_attitude_rad"] <= 0.1
                )
            else:
                tracking_passed = metrics["final_position_error"] <= 0.05 and metrics["final_velocity_error"] <= 0.1
        else:
            metrics, tracking_passed = {}, False
        per_step = [record[row] for record in records]
        passed = complete and tracking_passed and all(record["passed"] for record in per_step)
        reports.append(
            {
                "name": case["name"],
                "executed_steps": len(controls),
                "metrics": metrics,
                "checks": per_step,
                "tracking_passed": bool(tracking_passed),
                "passed": bool(passed),
            }
        )
    output.update(
        {
            "plant": "independent NumPy canonical discrete CTBR; no Newton, motors or collision geometry",
            "reference_sfc": "static x<=0.2 wall at every future node; target is its constrained endpoint",
            "scheduled_sample_solves": 240,
            "completed_sample_solves": len(records) * len(bank),
            "cases": reports,
            "passed": all(item["passed"] for item in reports),
        }
    )
    del output["step_records"]
    return output


def provenance():
    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(line.split()) >= 6 and any(name in line for name in ("_turbompc_native.so", "libcudss.so"))
        }
    )
    libraries = {path: digest(path) for path in paths}
    manifest_path = Path(os.environ["TURBOMPC_NATIVE_BUILD_DIR"]) / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {str(Path(path).resolve()): value["sha256"] for path, value in manifest["artifacts_after_call"].items()}
    native = {path: sha for path, sha in libraries.items() if "_turbompc_native.so" in path}
    passed = bool(native) and all(expected.get(str(Path(path).resolve())) == sha for path, sha in native.items())
    return {
        "libraries_sha256": libraries,
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "build_id": manifest["build_id"],
        "passed": passed and manifest.get("status") == "succeeded" and manifest.get("source_guard_passed", False),
    }


def gpu_snapshot():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"{type(error).__name__}: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--author-source",
        type=Path,
        default=Path(
            os.environ.get(
                "TURBOMPC_CTBR_AUTHOR_SOURCE",
                str(
                    ROOT.parent
                    / "turbompc-torch-native/build/ca-acmpc-reference/differentialMPCPerformance/dynamics/drone_ts.py"
                ),
            )
        ),
    )
    parser.add_argument("--preflight", action="store_true", help="CPU model/data checks only; never an acceptance pass")
    parser.add_argument(
        "--base-only", action="store_true", help="Four nominal solves and VJP only; formal gates remain false"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile tensor kernels; keep the same cases, math and acceptance gates"
    )
    parser.add_argument("--no-polish", action="store_true", help="Explicitly reproduce the earlier unrefined solver")
    args = parser.parse_args()
    sqp = replace(SQP, compile_kernels=args.compile)
    polisher = None if args.no_polish else CTBRPolisher()
    if args.output.exists() or any(args.output.with_suffix(suffix).exists() for suffix in (".npz", ".harness.py")):
        raise FileExistsError("Acceptance outputs are immutable; choose a new output stem")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    harness_path = args.output.with_suffix(".harness.py")
    with harness_path.open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    before = snapshot()
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "kind": "ctbr_m5_model_acceptance",
        "preflight_only": args.preflight,
        "base_diagnostic_only": args.base_only,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources_before": before,
        "executed_harness": str(harness_path),
        "executed_harness_sha256": digest(harness_path),
        "thresholds": THRESHOLDS,
        "fd_steps": FD_STEPS,
        "sqp": asdict(sqp),
        "admm": asdict(ADMM),
        "polish": None
        if polisher is None
        else {
            "max_iter": polisher.max_iter,
            "tolerance": polisher.tolerance,
            "max_linear_refinements": polisher.max_linear_refinements,
            "gradient_scope": "cost parameters D/E/q",
        },
        "horizon": HORIZON,
        "dt": DT,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "passed": False,
        "ctbr_numerical_gate_passed": False,
        "model_closed_loop_gate_passed": False,
        "newton_closed_loop_gate_passed": None,
        "error": None,
        "scope": "Fixed numerical and model-only gates, not M4 performance or Newton navigation success.",
    }
    artifacts, solver = {}, None
    try:
        from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder

        builder = CTBRQPBuilder(CTBRConfig())
        report["model_config"] = asdict(CTBRConfig())
        bank = fixed_cases()
        report["cases"] = bank
        author, report["author"] = author_model(args.author_source)
        report["cpu"] = cpu_checks(builder, bank, author, artifacts)
        if args.preflight:
            report["passed"] = report["cpu"]["passed"]
        else:
            if torch.version.cuda != "13.0" or os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu130":
                raise RuntimeError("Run the cu130 wrapper with the main project's Torch 2.11 cu130 Python")
            report["gpu_before"] = gpu_snapshot()
            builder = builder.to(device="cuda:0")
            solver = TorchMPC(builder.dynamics(), sqp, ADMM, constraints=builder.path_constraints(), polisher=polisher)
            report["open_loop"] = {}
            open_loop(builder, solver, bank, artifacts, report["open_loop"], base_only=args.base_only)
            if not args.base_only:
                report["model_loops"] = {}
                model_loops(builder, solver, bank, artifacts, report["model_loops"])
            report["provenance"] = provenance()
            report["gpu_after"] = gpu_snapshot()
            if args.base_only:
                report["passed"] = bool(
                    report["cpu"]["passed"]
                    and report["open_loop"]["base_diagnostic_passed"]
                    and report["provenance"]["passed"]
                )
            else:
                report["ctbr_numerical_gate_passed"] = report["cpu"]["passed"] and report["open_loop"]["passed"]
                report["model_closed_loop_gate_passed"] = report["model_loops"]["passed"]
                report["passed"] = bool(
                    report["ctbr_numerical_gate_passed"]
                    and report["model_closed_loop_gate_passed"]
                    and report["provenance"]["passed"]
                )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if solver is not None:
            try:
                solver.close()
            except Exception as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
                report["passed"] = False
        report["sources_after"] = snapshot()
        report["sources_unchanged"] = before == report["sources_after"]
        report["passed"] = bool(report["passed"] and report["sources_unchanged"])
        report["ctbr_numerical_gate_passed"] = bool(
            report["ctbr_numerical_gate_passed"] and report["sources_unchanged"]
        )
        report["model_closed_loop_gate_passed"] = bool(
            report["model_closed_loop_gate_passed"] and report["sources_unchanged"]
        )
        report["finished_utc"] = datetime.now(UTC).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **artifacts)
        report["arrays_sha256"] = digest(args.output.with_suffix(".npz"))
        with args.output.open("x") as handle:
            json.dump(safe_json(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "passed": report["passed"],
                    "ctbr_numerical_gate_passed": report["ctbr_numerical_gate_passed"],
                    "model_closed_loop_gate_passed": report["model_closed_loop_gate_passed"],
                    "error": report["error"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
