"""Bounded IPC SFC audit against independent strict OSQP and cost differences.

This checks the existing nine-state jerk problem, not the later CTBR model or
navigation success. Four fixed cases include inactive and duplicate active
planes. Every case and both finite-difference step sizes contribute to the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import osqp
import torch
from scipy import sparse

from isaaclab_ipc.corridor_storage import expand_corridors, pack_corridors
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ipc import IPCQPBuilder

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {
    "data_atol": 1e-10,
    "data_rtol": 1e-12,
    "control_max_abs": 1e-4,
    "physical_violation": 1e-5,
    "gradient_relative": 1e-3,
    "gradient_norm_floor": 1e-12,
    "inactive_distance_margin": 1e-4,
}
FD_STEPS = (1e-4, 5e-5)
OSQP_SETTINGS = {
    "eps_abs": 1e-10,
    "eps_rel": 1e-10,
    "max_iter": 20000,
    "polishing": True,
    "warm_starting": False,
    "verbose": False,
}
# Match TurboMPCActor's absolute stopping policy. Keep the inner threshold
# below SQP's 1e-6 target regardless of the O(1e3) IPC position weights.
ADMM = ADMMConfig(max_iter=16384, eps_abs=1e-7, eps_rel=0.0)
SQP = SQPConfig(max_iter=15, tolerance=1e-6, linesearch="none", compile_kernels=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    files = list((ROOT / "src/isaaclab_ipc/turbompc").glob("*.py"))
    files += [ROOT / "src/isaaclab_ipc" / name for name in ("mpc.py", "corridor_storage.py", "config/avoid.yaml")]
    files += [Path(__file__), ROOT / "tests/test_turbompc_sfc.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def as_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def compare(a, b, atol, rtol=0.0):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"passed": False, "reference_shape": list(a.shape), "actual_shape": list(b.shape)}
    finite = np.isfinite(a) & np.isfinite(b)
    infinities_match = np.array_equal(np.isposinf(a), np.isposinf(b)) and np.array_equal(np.isneginf(a), np.isneginf(b))
    difference = np.abs(a[finite] - b[finite])
    passed = infinities_match and not np.isnan(a).any() and not np.isnan(b).any()
    passed = passed and np.all(difference <= atol + rtol * np.abs(a[finite]))
    return {
        "passed": bool(passed),
        "max_abs": float(difference.max(initial=0)),
        "shape": list(a.shape),
        "infinite_sides_match": bool(infinities_match),
    }


def relative_gradient(actual, reference):
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    norm = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(actual - reference) / max(norm, THRESHOLDS["gradient_norm_floor"]))
    return {
        "relative_l2": error,
        "reference_l2": norm,
        "max_abs": float(np.max(np.abs(actual - reference))),
        "finite": finite,
        "passed": finite and norm > THRESHOLDS["gradient_norm_floor"] and error <= THRESHOLDS["gradient_relative"],
    }


def cases(mpc):
    state = np.array([0.0, 0.0, 1.0, 0.12, -0.04, 0.02, 0.01, -0.02, 0.005])
    time = np.arange(1, mpc.n + 1) * mpc.dt
    velocity = np.array([0.4, -0.12, 0.06])
    reference = np.zeros((mpc.n, 9))
    reference[:, :3] = state[:3] + time[:, None] * velocity
    reference[:, 3:6] = velocity
    box = np.array(
        [
            [1.0, 0.0, 0.0, -2.0],
            [-1.0, 0.0, 0.0, -2.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, -1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, -2.5],
            [0.0, 0.0, -1.0, 0.3],
        ]
    )
    terminal = box.copy()
    terminal[0, 3] = -0.35
    repeated = np.vstack((terminal, 2 * terminal[0]))
    empty = np.empty((0, 4))
    schedules = {
        "no_sfc": [empty] * mpc.n,
        "interior_sfc": [empty] + [box] * (mpc.n - 1),
        "terminal_active_sfc": [empty] + [box] * (mpc.n - 2) + [terminal],
        "redundant_active_sfc": [empty] + [box] * (mpc.n - 2) + [repeated],
    }
    costs = np.array([0.2, -0.1, 0.15, -0.05, 0.07, -0.04, 0.03])
    return [
        {
            "name": name,
            "state": state.copy(),
            "reference": reference.copy(),
            "corridors": schedule,
            "costs": costs.copy(),
            "expect_active": "active" in name,
        }
        for name, schedule in schedules.items()
    ]


def observation(bank, capacity, *, compact=True):
    obs = {
        "mpc_state": torch.from_numpy(np.stack([case["state"] for case in bank])),
        "reference": torch.from_numpy(np.stack([case["reference"] for case in bank])),
    }
    packed = [pack_corridors(case["corridors"], capacity) for case in bank]
    obs.update(
        {
            key: torch.from_numpy(np.stack([value[index] for value in packed]))
            for index, key in enumerate(("planes", "plane_mask", "corridor_index"))
        }
    )
    # False masks must discard data, not merely depend on zero-filled padding.
    obs["planes"][~obs["plane_mask"]] = torch.tensor([17.0, -19.0, 23.0, 29.0], dtype=torch.float64)
    if not compact:
        planes, mask = expand_corridors(obs)
        obs.update(planes=planes.clone(), plane_mask=mask.clone())
        del obs["corridor_index"]
    return obs


def reference_problem(mpc, case, costs):
    """Independent condensed coefficients from the existing IPC CPU implementation."""
    h0, _, a, lower, upper, state = mpc.problem(case["state"], case["reference"], case["corridors"])
    multipliers = np.exp(np.log(2) * np.tanh(costs[:4]))
    q = mpc.q * np.tile(np.repeat(multipliers[:3], 3), mpc.n)
    jerk = h0 - mpc.C.T @ (mpc.q[:, None] * mpc.C)
    h = mpc.C.T @ (q[:, None] * mpc.C) + multipliers[3] * jerk
    g = mpc.C.T @ (q * (mpc.M @ state - case["reference"].ravel()))
    g -= mpc.C[mpc.pi].T @ (mpc.q[mpc.pi] * np.tile(0.2 * np.tanh(costs[4:]), mpc.n))
    return {"H": h, "g": g, "A": a, "lower": lower, "upper": upper, "state": state}


def condensed_residuals(problem, x, y):
    value = problem["A"] @ x
    violation = np.maximum(np.maximum(problem["lower"] - value, value - problem["upper"]), 0)
    stationarity = problem["H"] @ x + problem["g"] + problem["A"].T @ y
    return {
        "physical_violation": float(np.max(violation, initial=0)),
        "stationarity_max_abs": float(np.max(np.abs(stationarity), initial=0)),
    }


def active_set(problem, x, y):
    value = problem["A"] @ x
    row_norm = np.linalg.norm(problem["A"], axis=1)
    distance = np.minimum(value - problem["lower"], problem["upper"] - value) / np.maximum(row_norm, 1e-12)
    active = (distance <= 1e-6) & (row_norm > 0)
    inactive = distance[~active & (row_norm > 0) & np.isfinite(distance)]
    return {
        "mask": active,
        "indices": np.flatnonzero(active),
        "count": int(active.sum()),
        "minimum_inactive_distance": float(inactive.min(initial=np.inf)),
        "active_duals": y[active],
        "active_row_rank": int(np.linalg.matrix_rank(problem["A"][active])) if active.any() else 0,
    }


def solve_osqp(problem):
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(np.triu(problem["H"])),
        q=problem["g"],
        A=sparse.csc_matrix(problem["A"]),
        l=problem["lower"],
        u=problem["upper"],
        **OSQP_SETTINGS,
    )
    result = solver.solve(raise_error=False)
    x, y = np.asarray(result.x), np.asarray(result.y)
    residual = condensed_residuals(problem, x, y)
    good = result.info.status == "solved" and np.isfinite(x).all() and np.isfinite(y).all()
    good = good and max(residual.values()) <= THRESHOLDS["physical_violation"]
    return {
        "x": x,
        "y": y,
        "status": result.info.status,
        "status_val": int(result.info.status_val),
        "iterations": int(result.info.iter),
        "reported_primal": float(result.info.prim_res),
        "reported_dual": float(result.info.dual_res),
        "independent_residuals": residual,
        "active_set": active_set(problem, x, y),
        "passed": bool(good),
    }


def finite_differences(mpc, case, base, loss_weights):
    gradients, steps = {}, {}
    for epsilon in FD_STEPS:
        gradient = np.empty(7)
        checks = []
        for column in range(7):
            offset = np.eye(7)[column] * epsilon
            plus = solve_osqp(reference_problem(mpc, case, case["costs"] + offset))
            minus = solve_osqp(reference_problem(mpc, case, case["costs"] - offset))
            gradient[column] = loss_weights @ (plus["x"] - minus["x"]) / (2 * epsilon)
            stable = np.array_equal(plus["active_set"]["mask"], base["active_set"]["mask"])
            stable = stable and np.array_equal(minus["active_set"]["mask"], base["active_set"]["mask"])
            checks.append(
                {
                    "cost_column": column,
                    "plus_status": plus["status"],
                    "minus_status": minus["status"],
                    "plus_residuals": plus["independent_residuals"],
                    "minus_residuals": minus["independent_residuals"],
                    "active_set_unchanged": bool(stable),
                    "passed": plus["passed"] and minus["passed"] and stable,
                }
            )
        gradients[str(epsilon)] = gradient
        steps[str(epsilon)] = {
            "gradient": gradient,
            "checks": checks,
            "passed": all(value["passed"] for value in checks),
        }
    consistency = relative_gradient(gradients[str(FD_STEPS[1])], gradients[str(FD_STEPS[0])])
    return {
        "steps": steps,
        "step_consistency": consistency,
        "passed": all(value["passed"] for value in steps.values()) and consistency["passed"],
    }


def ordered_rows(value):
    """Map stage [velocity, acceleration, jerk, faces] to IPC's exact row order."""
    batch, stages = value.shape[:2]
    tail = value.shape[3:]
    return np.concatenate(
        (
            value[:, :-1, 6:9].reshape(batch, 3 * (stages - 1), *tail),
            value[:, 1:, 3:6].reshape(batch, 3 * (stages - 1), *tail),
            value[:, 1:, :3].reshape(batch, 3 * (stages - 1), *tail),
            value[:, 1:, 9:].reshape(batch, -1, *tail),
        ),
        axis=1,
    )


def condensed_inputs(inputs, mpc, obs):
    mapping = np.zeros((mpc.n + 1, 12, 3 * mpc.n))
    mapping[1:, :9] = mpc.C.reshape(mpc.n, 9, 3 * mpc.n)
    for k in range(mpc.n):
        mapping[k, 9:, 3 * k : 3 * k + 3] = np.eye(3)
    data = {name: as_numpy(getattr(inputs, name)) for name in inputs.__dataclass_fields__}
    offset = np.zeros_like(data["q"])
    offset[:, 0, :9] = data["initial_state"]
    offset[:, 1:, :9] = (data["initial_state"] @ mpc.M.T).reshape(-1, mpc.n, 9)
    h = np.einsum("tni,btnm,tmj->bij", mapping, data["D"], mapping)
    h += np.einsum("tni,btnm,tmj->bij", mapping[1:], data["E"], mapping[:-1])
    h += np.einsum("tni,btmn,tmj->bij", mapping[:-1], data["E"], mapping[1:])
    linear = (data["D"] @ offset[..., None]).squeeze(-1) + data["q"]
    linear[:, 1:] += (data["E"] @ offset[:, :-1, :, None]).squeeze(-1)
    linear[:, :-1] += (data["E"].swapaxes(-1, -2) @ offset[:, 1:, :, None]).squeeze(-1)
    g = np.einsum("tni,btn->bi", mapping, linear)
    a = ordered_rows(np.einsum("btmn,tnj->btmj", data["G"], mapping))
    shift = (data["G"] @ offset[..., None]).squeeze(-1)
    lower, upper = ordered_rows(data["lower"] - shift), ordered_rows(data["upper"] - shift)
    mask = as_numpy(expand_corridors(obs)[1])
    results, selections = [], []
    for row in range(len(mask)):
        selection = np.r_[np.ones(9 * mpc.n, bool), mask[row].ravel()]
        selections.append(selection)
        results.append(
            {
                "H": h[row],
                "g": g[row],
                "A": a[row, selection],
                "lower": lower[row, selection],
                "upper": upper[row, selection],
                "state": data["initial_state"][row],
            }
        )
    return results, selections


def compare_problem(expected, actual):
    fields = {
        name: compare(expected[name], actual[name], THRESHOLDS["data_atol"], THRESHOLDS["data_rtol"])
        for name in expected
    }
    return {"fields": fields, "passed": all(value["passed"] for value in fields.values())}


def physical_residuals(inputs, trajectory, multipliers, a, b):
    data = {name: as_numpy(getattr(inputs, name)) for name in inputs.__dataclass_fields__}
    initial = trajectory[:, 0, :9] - data["initial_state"]
    dynamics = trajectory[:, :-1, :9] @ a.T + trajectory[:, :-1, 9:] @ b.T - trajectory[:, 1:, :9]
    gx = (data["G"] @ trajectory[..., None]).squeeze(-1)
    violation = np.maximum(np.maximum(data["lower"] - gx, gx - data["upper"]), 0)
    stationarity = (data["D"] @ trajectory[..., None]).squeeze(-1) + data["q"]
    stationarity[:, 1:] += (data["E"] @ trajectory[:, :-1, :, None]).squeeze(-1)
    stationarity[:, :-1] += (data["E"].swapaxes(-1, -2) @ trajectory[:, 1:, :, None]).squeeze(-1)
    stationarity += (data["G"].swapaxes(-1, -2) @ multipliers["y"][..., None]).squeeze(-1)
    stationarity[:, 0, :9] += multipliers["lambda0"]
    stationarity[:, :-1] += multipliers["lambda_dyn"] @ np.concatenate((a, b), axis=-1)
    stationarity[:, 1:, :9] -= multipliers["lambda_dyn"]
    return [
        {
            "initial_equality": float(np.max(np.abs(initial[i]))),
            "dynamics_equality": float(np.max(np.abs(dynamics[i]))),
            "physical_violation": float(np.max(violation[i], initial=0)),
            "stationarity_max_abs": float(np.max(np.abs(stationarity[i]))),
        }
        for i in range(len(trajectory))
    ]


def loaded_provenance():
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
    actual_native = {path: sha for path, sha in libraries.items() if "_turbompc_native.so" in path}
    matches = bool(actual_native) and all(
        expected.get(str(Path(path).resolve())) == sha for path, sha in actual_native.items()
    )
    return {
        "loaded_libraries_sha256": libraries,
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "build_id": manifest["build_id"],
        "incremental": manifest["incremental"],
        "passed": matches and manifest.get("status") == "succeeded" and manifest.get("source_guard_passed", False),
    }


def audit_gpu(builder, obs, bank, mpc, references, case_reports, loss_weights, artifacts):
    if torch.version.cuda != "13.0" or os.environ.get("TURBOMPC_CUDA_PROFILE") != "cu130":
        raise RuntimeError("Use the cu130 wrapper and the project's Torch 2.11 cu130 Python")
    builder = builder.to(device="cuda:0")
    obs = {key: value.to("cuda:0") for key, value in obs.items()}
    costs = torch.tensor(
        np.stack([case["costs"] for case in bank]), device="cuda:0", dtype=torch.float64, requires_grad=True
    )
    inputs, guess = builder(obs, costs)
    native_data, selections = condensed_inputs(inputs, mpc, obs)
    solver = TorchMPC(builder.dynamics(), SQP, ADMM)
    try:
        result = solver.solve(inputs, guess)
        controls = result.controls[:, :-1].flatten(1)
        loss = (controls * controls.new_tensor(loss_weights)).sum()
        trajectory, u = as_numpy(result.trajectory), as_numpy(controls)
        # Preserve forward evidence even if an active-set adjoint raises.
        for i, (case, report) in enumerate(zip(bank, case_reports, strict=True)):
            report["native_forward"] = {
                "status": int(result.status[i]),
                "raw_qp_status": int(result.admm.status[i]),
                "raw_qp_residuals": as_numpy(result.admm.residuals[i]),
                "reported_errors": as_numpy(result.errors[i]),
                "u0": u[i, :3],
            }
            artifacts[f"{case['name']}_native_trajectory"] = trajectory[i]
        gradient = as_numpy(torch.autograd.grad(loss, costs)[0])
        multipliers = {name: as_numpy(getattr(result.admm, name)) for name in ("y", "lambda0", "lambda_dyn")}
        residuals = physical_residuals(inputs, trajectory, multipliers, as_numpy(builder.a), as_numpy(builder.b))
        ordered_duals = ordered_rows(multipliers["y"])
        status, qp_status = as_numpy(result.status), as_numpy(result.admm.status)
        for i, (case, reference, report) in enumerate(zip(bank, references, case_reports, strict=True)):
            name = case["name"]
            actual = native_data[i]
            oracle = report["osqp"]
            gradients = {
                str(step): relative_gradient(gradient[i], report["finite_differences"]["steps"][str(step)]["gradient"])
                for step in FD_STEPS
            }
            solution = compare(oracle["x"], u[i], THRESHOLDS["control_max_abs"])
            u0 = compare(oracle["x"][:3], u[i, :3], THRESHOLDS["control_max_abs"])
            expected_states = (mpc.M @ reference["state"] + mpc.C @ oracle["x"]).reshape(mpc.n, 9)
            states = compare(expected_states, trajectory[i, 1:, :9], THRESHOLDS["control_max_abs"])
            condensed = condensed_residuals(reference, u[i], ordered_duals[i, selections[i]])
            assembly = compare_problem(reference, actual)
            physical_good = max(residuals[i].values()) <= THRESHOLDS["physical_violation"]
            physical_good = physical_good and max(condensed.values()) <= THRESHOLDS["physical_violation"]
            gates = {
                "data_parity": assembly["passed"],
                "sqp_and_qp_solved": status[i] == qp_status[i] == 0,
                "u0": u0["passed"],
                "all_controls": solution["passed"],
                "all_predicted_states": states["passed"],
                "physical_residuals": physical_good,
                "two_step_cost_gradient": all(value["passed"] for value in gradients.values()),
                "terminal_dummy_zero": np.max(np.abs(trajectory[i, -1, 9:])) <= THRESHOLDS["control_max_abs"],
            }
            report["native"] = {
                "status": status[i],
                "raw_qp_status": qp_status[i],
                "raw_qp_residuals": as_numpy(result.admm.residuals[i]),
                "sqp_iterations": int(result.iterations[i]),
                "last_qp_iterations": int(result.admm.iterations[i]),
                "reported_errors": as_numpy(result.errors[i]),
                "independent_multiple_shooting_residuals": residuals[i],
                "independent_condensed_residuals": condensed,
                "data_parity": assembly,
                "u0": u[i, :3],
                "u0_error": u0,
                "all_controls_error": solution,
                "all_states_error": states,
                "cost_gradient": gradient[i],
                "gradient_comparisons": gradients,
                "gates": gates,
                "passed": bool(all(gates.values())),
            }
            for key, value in actual.items():
                artifacts[f"{name}_native_condensed_{key}"] = value
            artifacts[f"{name}_native_trajectory"] = trajectory[i]
            artifacts[f"{name}_native_gradient"] = gradient[i]
        expanded = dict(obs)
        expanded["planes"] = torch.nn.functional.pad(obs["planes"], (0, 0, 0, 32), value=float("nan"))
        expanded["plane_mask"] = torch.nn.functional.pad(obs["plane_mask"], (0, 32))
        with torch.no_grad():
            padded_inputs, padded_guess = builder(expanded, costs.detach())
            padded = solver.solve(padded_inputs, padded_guess)
        padding_error = compare(u, as_numpy(padded.controls[:, :-1].flatten(1)), THRESHOLDS["control_max_abs"])
        padding = {
            "extra_masked_faces": 32,
            "status": as_numpy(padded.status),
            "controls": padding_error,
            "passed": bool((padded.status == 0).all()) and padding_error["passed"],
        }
        return {
            "padding_capacity_check": padding,
            "provenance": loaded_provenance(),
            "gpu_name": torch.cuda.get_device_name(0),
            "native_cost_gradients": gradient,
            "passed": all(report["native"]["passed"] for report in case_reports) and padding["passed"],
        }
    finally:
        solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="CPU matrix, OSQP and seven-cost FD checks only")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--m4-summary", type=Path, default=ROOT / "results/turbompc-port/m4-ir2-accepted-v1.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Choose a new report name; audit evidence is never overwritten")
    before = source_snapshot()
    report = {
        "time_utc": datetime.now(UTC).isoformat(),
        "scope": "IPC SFC numerical/gradient audit; no closed-loop or CTBR claim",
        "thresholds": THRESHOLDS,
        "fd_steps": FD_STEPS,
        "osqp_settings": OSQP_SETTINGS,
        "native_admm": asdict(ADMM),
        "native_sqp": asdict(SQP),
        "native_configuration_origin": "TurboMPCActor numerical settings; eager execution for this independent audit",
        "prior_diagnostics": {
            "sfc-parity-cu130-v1": "4000/1e-10 reached raw QP MAX_ITER on the single active plane; numerical and gradient gates passed",
            "sfc-parity-cu130-v2": "4096/1e-8 relative stopping let QP finish above the absolute SQP target; SQP failed and cost gradients were zero",
        },
        "absolute_stopping_rationale": "ADMM absolute 1e-7 is one tenth of SQP 1e-6; the 16384 budget is only a cap, with ordinary early termination",
        "admm_residual_order": ["primal", "dual", "primal_limit", "dual_limit"],
        "preflight": args.preflight,
        "passed": False,
        "sfc_numerical_gate_passed": None,
        "cases": [],
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "osqp": osqp.__version__,
        },
    }
    artifacts = {}
    try:
        m4 = json.loads(args.m4_summary.read_text())
        report["m4_prerequisite"] = {
            "path": str(args.m4_summary),
            "sha256": digest(args.m4_summary),
            "m4_gate_passed": m4.get("m4_gate_passed"),
            "original_A0_passed": m4.get("original_baseline_passed"),
        }
        if m4.get("m4_gate_passed") is not True:
            raise RuntimeError("The explicitly corrected M4 configuration must pass before this SFC audit")
        report["worktree_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        mpc, builder = MPC(Config.load("avoid")), IPCQPBuilder(Config.load("avoid"))
        bank = cases(mpc)
        obs = observation(bank, args.capacity)
        costs = torch.tensor(np.stack([case["costs"] for case in bank]), dtype=torch.float64)
        inputs, _ = builder(obs, costs)
        structured, _ = condensed_inputs(inputs, mpc, obs)
        references = []
        loss_weights = (np.sin(0.17 * np.arange(45) + 0.4) + 0.3 * np.cos(0.29 * np.arange(45))) / 45
        artifacts["loss_weights"] = loss_weights
        report["case_selection"] = "Four declared cases; no rejection, replacement or gradient-based sample filtering"
        for case, actual in zip(bank, structured, strict=True):
            name = case["name"]
            expected = reference_problem(mpc, case, case["costs"])
            references.append(expected)
            oracle = solve_osqp(expected)
            finite = finite_differences(mpc, case, oracle, loss_weights)
            assembly = compare_problem(expected, actual)
            active = oracle["active_set"]
            active_expected = (active["count"] > 0) if case["expect_active"] else active["count"] == 0
            stable = active["minimum_inactive_distance"] > THRESHOLDS["inactive_distance_margin"]
            gates = {
                "data_parity": assembly["passed"],
                "strict_osqp": oracle["passed"],
                "declared_active_regime": active_expected,
                "inactive_margin": stable,
                "seven_cost_two_step_fd": finite["passed"],
            }
            entry = {
                "name": name,
                "costs": case["costs"],
                "state": case["state"],
                "reference": case["reference"],
                "face_counts_by_time": [len(faces) for faces in case["corridors"]],
                "data_parity": assembly,
                "osqp": oracle,
                "finite_differences": finite,
                "cpu_gates": gates,
                "cpu_passed": bool(all(gates.values())),
            }
            report["cases"].append(entry)
            for key, value in expected.items():
                artifacts[f"{name}_reference_{key}"] = value
                artifacts[f"{name}_builder_condensed_{key}"] = actual[key]
            artifacts[f"{name}_osqp_solution"], artifacts[f"{name}_osqp_dual"] = oracle["x"], oracle["y"]
            for step in FD_STEPS:
                artifacts[f"{name}_fd_{step}"] = finite["steps"][str(step)]["gradient"]
        report["cpu_passed"] = all(case["cpu_passed"] for case in report["cases"])
        if not report["cpu_passed"]:
            raise RuntimeError("CPU data/OSQP/FD prerequisites failed; see every retained case")
        if not args.preflight:
            report["gpu"] = audit_gpu(builder, obs, bank, mpc, references, report["cases"], loss_weights, artifacts)
            report["sfc_numerical_gate_passed"] = report["gpu"]["passed"] and report["gpu"]["provenance"]["passed"]
        report["passed"] = report["cpu_passed"] and (args.preflight or report["sfc_numerical_gate_passed"])
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if not args.preflight:
            report["sfc_numerical_gate_passed"] = False
    finally:
        report["sources_before"], report["sources_after"] = before, source_snapshot()
        report["sources_unchanged"] = report["sources_before"] == report["sources_after"]
        report["passed"] = bool(report["passed"] and report["sources_unchanged"])
        if report["sfc_numerical_gate_passed"] is not None:
            report["sfc_numerical_gate_passed"] = bool(
                report["sfc_numerical_gate_passed"] and report["sources_unchanged"]
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **artifacts)
        with args.output.open("x") as handle:
            json.dump(json_safe(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "passed": report["passed"],
                    "sfc_numerical_gate_passed": report["sfc_numerical_gate_passed"],
                    "error": report.get("error"),
                },
                indent=2,
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
