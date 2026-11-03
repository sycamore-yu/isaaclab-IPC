"""Verify saved evidence and record the session's disclosed equal-IR2 M4 selection.

CPU-only: no solver execution, library replacement, or editing of source reports.
Original A0 failures and candidate-only reports retain their original meanings.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import check_acceptance as acceptance
import check_candidate_acceptance as candidate
import check_jerk_official as jerk
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/turbompc-port"
SELECTION_NOTE = "continued implementation with explicitly disclosed equal IR2 config"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": acceptance.sha256(path)}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def require_subset(actual, expected, label):
    for key, value in expected.items():
        if isinstance(value, dict):
            require_subset(actual[key], value, f"{label}.{key}")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            # Upstream YAML leaves scientific-notation settings such as
            # rho_max="1.0e6" as strings; compare their effective numeric value.
            require(float(actual[key]) == value, f"{label}.{key} differs")
        else:
            require(actual[key] == value, f"{label}.{key} differs")


def finite_array(arrays, key, shape, dtype=np.float64):
    value = arrays[key]
    require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), f"Invalid array {key}")
    return value


def validate_linear(document, arrays, provenance):
    require(document["qualification"] == candidate.QUALIFICATION, "Candidate qualification changed")
    require(
        document["candidate_suite_passed"] is True and document["m4_gate_passed"] is False, "Invalid candidate role"
    )
    require(document["mode"] == "gpu-candidate-experiment", "CPU preflight is not measurement evidence")
    require(document["thresholds"] == acceptance.THRESHOLDS, "Acceptance thresholds changed")
    require(document["solver_contract"] == acceptance.SOLVER, "Solver settings changed")
    require_subset(document, {"horizon": 15, "nx": 4, "nu": 2, "dt": 0.1, "dtype": "float64"}, "linear")
    require(document["provenance_before"] == document["provenance_after"] == provenance, "Source/build guard differs")
    for key in ("original_kkt_sha256_before", "original_kkt_sha256_after"):
        require(document[key] == candidate.ORIGINAL_HASH, "Original installed KKT changed")
    loaded = document["loaded_libraries"]
    require(loaded["passed"] is True and all(loaded["checks"].values()), "Loaded library validation failed")
    expected_solvers = {
        path for path in loaded["paths"] if Path(path).name.endswith("_ffi.so") or "_turbompc_native" in Path(path).name
    }
    require(set(loaded["solver_sha256"]) == expected_solvers, "Loaded solver hash inventory is incomplete")
    require(provenance["candidate_kkt"] in expected_solvers, "Candidate KKT was not loaded")
    require(provenance["installed_original_kkt"] not in expected_solvers, "Original KKT was loaded in candidate run")
    native_paths = [path for path in expected_solvers if "_turbompc_native" in Path(path).name]
    require(
        len(native_paths) == 1 and native_paths[0] in provenance["build_records"]["B"]["artifact_sha256"],
        "Loaded B is not the guarded native build",
    )
    for path in expected_solvers:
        if Path(path).name.endswith("_ffi.so"):
            require(provenance["selected_ffi"].get(Path(path).name) == path, "Loaded FFI path differs")
        require(
            loaded["solver_sha256"][path] == provenance["artifact_sha256"][path] == acceptance.sha256(path),
            f"Loaded artifact hash differs: {path}",
        )
    config_path = Path(provenance["candidate_package"]) / "problems/params/linear.yaml"
    constants = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in yaml.safe_load(config_path.read_text())["dynamics_state_dot_params"].items()
    }
    entries = document["batches"]
    require([entry["batch"] for entry in entries] == list(acceptance.REQUIRED_BATCHES), "Four ordered batches required")
    summaries = []
    for entry in entries:
        batch = entry["batch"]
        case = acceptance.case_bank(batch)
        for key, expected in case.items():
            value = arrays[f"b{batch}_input_{key}"]
            require(value.dtype == expected.dtype and np.array_equal(value, expected), f"B{batch} input {key} differs")
        require(entry["case_sha256"] == acceptance.array_hash(case), "Fixed input hash differs")
        require(entry["passed"] is True and all(entry["gates"].values()), f"B{batch} recorded gates failed")
        require(all(entry["configuration_checks"].values()), f"B{batch} configuration checks failed")
        require_subset(
            entry["effective_A_solver"],
            {
                key: value
                for key, value in acceptance.SOLVER.items()
                if key not in ("forward_backend", "backward_backend")
            },
            f"B{batch}.A",
        )
        require_subset(
            entry["effective_B_admm"],
            {("alpha" if key == "relaxation_parameter" else key): value for key, value in acceptance.ADMM.items()},
            f"B{batch}.B.admm",
        )
        require_subset(
            entry["effective_B_sqp"],
            {
                "max_iter": 15,
                "tolerance": acceptance.SOLVER["tol_convergence"],
                "linesearch": "merit",
                "alphas": acceptance.SOLVER["linesearch"]["alphas"],
                "armijo": acceptance.SOLVER["linesearch"]["merit"]["armijo_coefficient"],
            },
            f"B{batch}.B.sqp",
        )
        snapshots = {}
        for role in ("A", "B"):
            prefix = f"b{batch}_{role}_"
            shapes = {
                "states": (batch, 16, 4),
                "controls": (batch, 16, 2),
                "u0": (batch, 2),
                "convergence": (batch,),
                "cost_gradient": (batch, 10),
                "state_gradient": (batch, 4),
            }
            snapshot = {key: finite_array(arrays, prefix + key, shape) for key, shape in shapes.items()}
            for key in ("status", "valid", "iterations", "admm_iterations"):
                snapshot[key] = arrays[prefix + key]
                require(snapshot[key].shape == (batch,) and np.isfinite(snapshot[key]).all(), f"Invalid {prefix + key}")
            require(np.array_equal(snapshot["u0"], snapshot["controls"][:, 0]), "Saved u0 differs from the plan")
            snapshots[role] = snapshot
            require(
                len(entry["warmup_full_ms"][role]) == 10
                and np.isfinite(entry["warmup_full_ms"][role]).all()
                and min(entry["warmup_full_ms"][role]) > 0,
                "Ten finite positive warmups are required",
            )
            fd = entry["finite_difference"][role]
            require(
                fd["passed"] is True and [step["step"] for step in fd["steps"]] == [1e-3, 5e-4],
                "Two fixed FD steps are required",
            )
            for step in fd["steps"]:
                require(
                    all(
                        step[key] is True
                        for key in ("all_perturbations_solved", "finite", "all_perturbations_interior", "passed")
                    )
                    and np.isfinite(step["physical_maximum"])
                    and step["physical_maximum"] <= acceptance.THRESHOLDS["physical_violation"],
                    "FD perturbation feasibility/stability failed",
                )
                for key in ("cost_gradient", "state_gradient"):
                    numerical = finite_array(arrays, f"{prefix}fd_{step['step']}_{key}", shapes[key])
                    comparison = acceptance.gradient_error(snapshot[key], numerical)
                    require(comparison["passed"], f"{prefix}{key} fails saved FD")
                    np.testing.assert_allclose(
                        comparison["per_sample_relative"],
                        step["gradients"][key]["per_sample_relative"],
                        rtol=1e-12,
                        atol=1e-15,
                    )
        comparison = acceptance.compare(snapshots["A"], snapshots["B"], case, constants)
        require(comparison["passed"], f"B{batch} saved raw trajectory/gradient comparison failed")
        for suffix, report_key in (
            ("raw_", "A_raw_gradient_before_timing"),
            ("raw_after_", "A_raw_gradient_after_timing"),
        ):
            raw = entry[report_key]
            require(raw["raw_vjp_and_callbacks_finite"] is True, "Raw VJP/callback check failed")
            for key, callback in (("cost_gradient", "mpc_cost_adjoint"), ("state_gradient", "mpc_context_adjoint")):
                reference = snapshots["A"][key]
                value = finite_array(arrays, f"b{batch}_A_{suffix}{key}", reference.shape)
                callback_value = np.asarray(raw["callbacks"][callback], dtype=np.float64)
                require(
                    callback_value.shape == reference.shape
                    and acceptance.gradient_error(value, reference)["passed"]
                    and acceptance.gradient_error(callback_value, reference)["passed"],
                    "Raw VJP/callback does not match the saved gradient",
                )
        pairs = entry["pairs"]
        require(len(pairs) == 100, "Exactly 100 preserved pairs per batch required")
        limits = {
            "u0_max_abs": acceptance.THRESHOLDS["u0_max_abs"],
            "cost_gradient_max_relative": acceptance.THRESHOLDS["gradient_relative"],
            "state_gradient_max_relative": acceptance.THRESHOLDS["gradient_relative"],
            "physical_maximum": acceptance.THRESHOLDS["physical_violation"],
        }
        for index, pair in enumerate(pairs):
            require(
                pair["index"] == index and pair["order"] == (["A", "B"] if index % 2 == 0 else ["B", "A"]),
                "Pair ordering/index changed",
            )
            require(
                pair["numerical_passed"] is True
                and pair["all_solved"] is True
                and pair["start_time_ns"] < pair["end_time_ns"],
                "Pair status/timestamps failed",
            )
            require(
                all(np.isfinite(pair[key]) and 0 <= pair[key] <= limit for key, limit in limits.items()),
                "A timed pair exceeds numerical thresholds",
            )
        timing = acceptance.timing_summary(pairs)
        require(timing["passed"], f"B{batch} recomputed P50/P95 failed")
        require(acceptance.json_safe(timing) == entry["timing"], "Reported timing differs from the 100 raw pairs")
        summaries.append(
            {
                "batch": batch,
                "pairs": 100,
                "warmup_per_backend": 10,
                "case_sha256": entry["case_sha256"],
                "correctness": comparison,
                "timing": timing,
                "worst_timed_pair": {key: max(pair[key] for pair in pairs) for key in limits},
            }
        )
    replay = copy.deepcopy(document)
    candidate.finish_gates(SimpleNamespace(preflight_only=False, batches=list(acceptance.REQUIRED_BATCHES)), replay)
    require(
        replay["candidate_suite_passed"] is True and replay["candidate_gates"] == document["candidate_gates"],
        "Candidate gates do not recompute",
    )
    return summaries


def validate_jerk(document, arrays, provenance, reference):
    require(
        document["role"] == "official full-plan jerk accuracy diagnostic: ir2-candidate"
        and document["m4_result"] is False
        and document["diagnostic_checks_passed"] is True,
        "Wrong jerk evidence role",
    )
    require(
        all(document["checks"].get(key) is True for key in acceptance.OFFICIAL_REGRESSION_CHECKS),
        "Recorded official jerk checks failed",
    )
    saved = document["provenance"]
    require(
        saved["kkt_role"] == "ir2-candidate" and saved["upstream_commit"] == acceptance.REVISION,
        "Jerk uses another upstream or KKT role",
    )
    for key in ("expected_kkt_library", "actual_loaded_kkt_library"):
        require(Path(saved[key]).resolve() == Path(provenance["candidate_kkt"]), "Jerk candidate library path differs")
    for key in ("expected_kkt_library_sha256", "actual_loaded_kkt_sha256"):
        require(saved[key] == candidate.CANDIDATE_HASH, "Jerk candidate hash differs")
    for key in ("installed_baseline_sha256_before", "installed_baseline_sha256_after"):
        require(saved[key] == candidate.ORIGINAL_HASH, "Jerk changed the original KKT")
    require(
        Path(saved["installed_baseline_library"]).resolve() == Path(provenance["installed_original_kkt"]),
        "Jerk original library path differs",
    )
    expected_sources = acceptance.verify_original(reference, reference, Path(provenance["candidate_package"]))
    require(
        acceptance._upstream_source_index(saved["verified_upstream_files"])
        == acceptance._upstream_source_index(expected_sources)
        and all(acceptance.sha256(path) == digest for path, digest in saved["verified_upstream_files"].items()),
        "Jerk official Python/bridge sources differ",
    )
    require(
        saved["script_sha256"] == acceptance.sha256(ROOT / "scripts/turbompc/check_jerk_official.py"),
        "Jerk producer source changed",
    )
    require(saved["manifest_sha256"] == acceptance.sha256(saved["manifest"]), "Jerk patch manifest differs")
    helper_path = ROOT / "scripts/turbompc_benchmark_common.py"
    require(saved["legacy_case_helper_sha256"] == acceptance.sha256(helper_path), "Fixed jerk input helper differs")
    helper = acceptance.load_example("_finalize_fixed_jerk", helper_path)
    case = helper.make_case(1, 15, False)
    require(saved["same_saved_inputs"] is True, "Jerk fixed-input provenance is missing")
    for key, value in case.items():
        require(
            arrays[f"input_{key}"].dtype == np.asarray(value).dtype and np.array_equal(arrays[f"input_{key}"], value),
            f"Jerk fixed input {key} differs",
        )
    require_subset(
        document["case"],
        {
            "batch": 1,
            "horizon": 15,
            "nx": 9,
            "nu": 3,
            "dtype": "float64",
            "seed": 719,
            "loss": "sum(all 16 control nodes * original saved probe)",
        },
        "jerk.case",
    )
    implementation = document["implementation"]
    require_subset(
        implementation,
        {
            "backend_class": "_jerk_official_backend.MPCBackend",
            "bridge_class": "_jerk_official_bridge.MPC",
            "entry_point": "official MPC.forward (mean, full controls)",
        },
        "jerk.interface",
    )
    for key, suffix in (("backend_file", "examples/rl/util/mpc.py"), ("bridge_file", "examples/rl/util/jax_torch.py")):
        require(
            saved["verified_upstream_files"].get(implementation[key]) == acceptance.sha256(reference / suffix),
            "Jerk bridge/backend file differs",
        )
    require_subset(
        implementation["effective_solver_parameters"],
        {
            "constraint_handling": "projection",
            "log_barrier_kappa": 0.0,
            "tol_convergence": 1e-6,
            "convergence_criterion": "first_order",
            "forward_hessian_min_eig": 1e-12,
            "num_sqp_iteration_max": 15,
            "admm": acceptance.ADMM | {"eps_abs": 1e-8, "eps_rel": 1e-8},
            "linesearch": acceptance.SOLVER["linesearch"],
        },
        "jerk.solver",
    )

    def snapshot(prefix, context):
        values = {
            key: arrays[f"{prefix}_{key}"]
            for key in ("trajectory", "returned_controls", "mean", "status", "valid", "convergence", "iterations")
        }
        finite_array(arrays, f"{prefix}_trajectory", (1, 16, 12))
        finite_array(arrays, f"{prefix}_returned_controls", (1, 16, 3))
        metrics = jerk.snapshot_report(values, {**case, "x0": context}, helper.physical_residuals)
        require(
            metrics["solved"]
            and metrics["finite"]
            and metrics["physical_maximum"] <= 1e-5
            and metrics["returned_plan_max_abs"] <= 1e-8,
            f"{prefix} is not solved/finite/feasible",
        )
        for side in ("lower", "upper"):
            mask = finite_array(arrays, f"{prefix}_active_{side}", (1, 16, 12), np.bool_)
            require(np.array_equal(mask, arrays[f"base_active_{side}"]), f"{prefix} active {side} changed")
        require(
            np.isfinite(arrays[f"{prefix}_active_dual"]).all()
            and np.all(arrays[f"{prefix}_active_replay_status"] == 0)
            and np.max(arrays[f"{prefix}_active_replay_plan_max_abs"]) <= 1e-8,
            f"{prefix} actual KKT mask replay failed",
        )
        return values, metrics

    base, base_metrics = snapshot("base", case["x0"])
    native = finite_array(arrays, "native_trajectory", (1, 16, 12))
    require(
        np.all(arrays["native_status"] == 0) and np.max(np.abs(base["returned_controls"] - native[..., 9:])) <= 1e-4,
        "Saved native plan disagrees with official jerk",
    )
    gradient_checks = {}
    for key, width, callback in (
        ("cost_gradient", 12, "mpc_cost_adjoint"),
        ("state_gradient", 9, "mpc_context_adjoint"),
    ):
        analytic = finite_array(arrays, f"base_{key}", (1, width))
        gradient_checks[key] = {}
        for label in ("native", "raw_retained", "callback"):
            name = f"{label}_{callback}" if label == "callback" else f"{label}_{key}"
            value = finite_array(arrays, name, (1, width))
            comparison = acceptance.gradient_error(analytic, value)
            require(comparison["passed"], f"Jerk {key} fails {label} comparison")
            gradient_checks[key][label] = comparison
    fd = document["finite_difference"]
    require(
        [step["step"] for step in fd["steps"]] == [1e-4, 5e-5]
        and fd["forward_perturbations"] == len(fd["perturbations"]) == 88,
        "Incomplete jerk FD experiment",
    )
    require(
        all(
            fd[key] is True
            for key in ("all_solved", "all_finite", "all_active_masks_equal", "all_mask_replay_plans_match")
        ),
        "Jerk FD stability summary failed",
    )
    rows = {row["label"]: row for row in fd["perturbations"]}
    require(len(rows) == 88, "Duplicate jerk perturbation labels")
    seen, losses, worst_residual = set(), {}, base_metrics["physical_maximum"]
    direction = np.random.default_rng(45).normal(size=(1, 12))
    direction /= np.linalg.norm(direction)
    require(np.array_equal(arrays["old_cost_direction"], direction), "Old cost direction changed")
    directions = []
    for step_index, step in enumerate((1e-4, 5e-5)):
        for variable, width in (("cost", 12), ("state", 9), ("old_direction", 1)):
            numerical = np.empty((1, width))
            for coordinate in range(width):
                for sign, suffix in ((1.0, "plus"), (-1.0, "minus")):
                    stem = f"{variable}_{coordinate}" if variable != "old_direction" else variable
                    label = f"fd_h{step_index}_{stem}_{suffix}"
                    costs, context = case["costs"].copy(), case["x0"].copy()
                    if variable == "old_direction":
                        costs += sign * step * direction
                    else:
                        (costs if variable == "cost" else context)[0, coordinate] += sign * step
                    require(
                        np.array_equal(arrays[f"{label}_input_costs"], costs)
                        and np.array_equal(arrays[f"{label}_input_x0"], context),
                        f"{label} input changed",
                    )
                    value, metrics = snapshot(label, context)
                    row = rows[label]
                    require(
                        row["step"] == step
                        and row["sign"] == sign
                        and all(
                            row[key] is True
                            for key in (
                                "solved",
                                "finite",
                                "active_masks_equal",
                                "mask_replay_solved",
                                "mask_duals_finite",
                            )
                        ),
                        f"{label} recorded FD status differs",
                    )
                    loss = float(np.sum(value["returned_controls"] * case["probe"]))
                    require(
                        np.isfinite(row["loss"]) and abs(loss - row["loss"]) <= 1e-12,
                        f"{label} saved loss differs from its controls",
                    )
                    losses[label] = loss
                    seen.add(label)
                    worst_residual = max(worst_residual, metrics["physical_maximum"])
                prefix = f"fd_h{step_index}_{stem}"
                numerical[0, coordinate] = (losses[prefix + "_plus"] - losses[prefix + "_minus"]) / (2 * step)
            if variable == "old_direction":
                analytic = float(np.sum(arrays["base_cost_gradient"] * direction))
                relative = abs(analytic - numerical.item()) / max(abs(numerical.item()), 1e-12)
                require(relative <= 1e-3, "Old direction fails recomputed FD")
                directions.append(
                    {"step": step, "analytic": analytic, "numerical": numerical.item(), "relative_error": relative}
                )
            else:
                key = "cost_gradient" if variable == "cost" else "state_gradient"
                np.testing.assert_allclose(numerical, arrays[f"fd_h{step_index}_{key}"], rtol=1e-7, atol=1e-9)
                comparison = acceptance.gradient_error(arrays[f"base_{key}"], numerical)
                require(comparison["passed"], f"Jerk {key} fails reconstructed FD")
                gradient_checks[key][f"reconstructed_fd_{step}"] = comparison
    require(seen == rows.keys(), "Jerk perturbation inventory differs from fixed coordinate/direction design")
    return {
        "passed": True,
        "fixed_inputs_sha256": acceptance.array_hash({key: np.asarray(value) for key, value in case.items()}),
        "perturbations_reconstructed": 88,
        "actual_masks_unchanged": True,
        "physical_maximum": worst_residual,
        "gradient_checks": gradient_checks,
        "old_cost_direction": directions,
        "producer_sha256": saved["script_sha256"],
        "input_helper_sha256": saved["legacy_case_helper_sha256"],
        "legacy_benchmark_helper_note": "Recorded historically; never imported by the official jerk producer. "
        "Only its file hash was observed; its later output-preservation edit is not a solver change.",
    }


def finalize(args):
    reference = (ROOT / "build/reference").resolve()
    preflight_args = SimpleNamespace(
        manifest=(ROOT / "build/turbompc/kkt-reference-fix/cu128/manifest.json").resolve(),
        reference=reference,
        upstream_repo=reference,
    )
    current = candidate.preflight(preflight_args)
    documents = {
        name: json.loads(path.read_text())
        for name, path in (("candidate", args.candidate), ("jerk", args.jerk), ("original_gate", args.original_gate))
    }
    evidence = {}
    for name, path in (("candidate", args.candidate), ("jerk", args.jerk), ("original_jerk", args.original_jerk)):
        evidence[name] = file_evidence(path) | {"arrays": file_evidence(path.with_suffix(".npz"))}
    evidence["original_gate"] = file_evidence(args.original_gate)
    arrays = load_arrays(args.candidate.with_suffix(".npz"))
    linear = validate_linear(documents["candidate"], arrays, current)
    jerk_arrays = load_arrays(args.jerk.with_suffix(".npz"))
    regression = validate_jerk(documents["jerk"], jerk_arrays, current, reference)
    original_sources = acceptance.verify_original(reference, reference, Path(current["candidate_package"]))
    original = acceptance.validate_official_regression(
        args.original_jerk, {current["installed_original_kkt"]: candidate.ORIGINAL_HASH}, original_sources
    )
    require(original["passed"] is False and "error" not in original, "Original A0 failure evidence is invalid")
    require(
        all(
            original["checks"].get(key) is True
            for key in (
                "original_role",
                "same_measured_A_library",
                "same_pinned_official_sources",
                "exact_fixed_inputs",
                "raw_arrays_finite_and_complete",
                "complete_stable_fd_evidence",
            )
        ),
        "Original A0 failure lacks valid pinned, stable numerical evidence",
    )
    require(
        original["checks"]["raw_cost_gradient_matches_reference_and_fd"] is False
        and original["checks"]["old_direction_matches_both_fd_steps"] is False,
        "Expected original A0 gradient failure is absent",
    )
    original_gate = documents["original_gate"]
    require(original_gate["m4_gate_passed"] is False, "Original M4 report must retain failure")
    require(
        original_gate["official_regression"]["report_sha256"] == evidence["original_jerk"]["sha256"]
        and original_gate["official_regression"]["arrays_sha256"] == evidence["original_jerk"]["arrays"]["sha256"],
        "Original M4 report references different jerk evidence",
    )
    measurement = original_gate["measurement_report"]
    require(
        file_evidence(measurement["path"])["sha256"] == measurement["sha256"]
        and file_evidence(measurement["arrays_file"])["sha256"] == measurement["arrays_sha256"],
        "Original M4 measurement artifacts changed",
    )
    evidence["original_measurement"] = measurement
    original_measurement = json.loads(Path(measurement["path"]).read_text())
    require(original_measurement["thresholds"] == documents["candidate"]["thresholds"], "Original thresholds differ")
    original_batches = {entry["batch"]: entry for entry in original_measurement["batches"]}
    with np.load(measurement["arrays_file"], allow_pickle=False) as original_arrays:
        for entry in documents["candidate"]["batches"]:
            batch = entry["batch"]
            for key in ("state", "costs", "seed"):
                name = f"b{batch}_input_{key}"
                require(
                    original_arrays[name].dtype == arrays[name].dtype
                    and np.array_equal(original_arrays[name], arrays[name]),
                    "Original/candidate inputs differ",
                )
            for key in ("effective_A_solver", "effective_B_sqp", "effective_B_admm"):
                require(original_batches[batch][key] == entry[key], "Original/candidate effective settings differ")
    after = candidate.preflight(preflight_args)
    require(after == current, "Sources/builds changed during CPU finalization")
    for entry in evidence.values():
        require(acceptance.sha256(entry["path"]) == entry["sha256"], "An input report changed during validation")
        if "arrays" in entry:
            require(acceptance.sha256(entry["arrays"]["path"]) == entry["arrays"]["sha256"], "Input arrays changed")
    require("torch" not in sys.modules and "jax" not in sys.modules, "Finalization must remain CPU-only")
    return {
        "time_utc": datetime.now(UTC).isoformat(),
        "baseline_role": "upstream_7050a4f_plus_ir2",
        "selection_note": SELECTION_NOTE,
        "scope": "M4 for the explicitly disclosed equal-IR2 configuration; preserves original A0 and candidate reports",
        "m4_gate_passed": True,
        "original_baseline_passed": False,
        "numerical_and_performance_passed": True,
        "same_original_inputs_settings_and_thresholds": True,
        "gpu_measurements_rerun": False,
        "thresholds": acceptance.THRESHOLDS,
        "configuration": {
            "A": "upstream 7050a4f + exact three-line KKT IR_N_STEPS=2 patch",
            "B": "native TorchMPC with KKT IR_N_STEPS=2",
            "profile": "cu128",
            "horizon": 15,
            "dtype": "float64",
            "batches": [1, 16, 64, 256],
            "timed_pairs": 400,
            "warmup_per_backend_per_batch": 10,
            "shared_gpu": True,
        },
        "evidence": evidence,
        "exact_patch": current["exact_three_line_patch"],
        "provenance": {
            "candidate_tracked_files_checked": current["candidate_tracked_files_checked"],
            "source_file_count": len(current["source_sha256"]),
            "source_build_artifact_guards_passed": True,
            "candidate_kkt_sha256": candidate.CANDIDATE_HASH,
            "original_installed_kkt_sha256": candidate.ORIGINAL_HASH,
            "loaded_solver_sha256": documents["candidate"]["loaded_libraries"]["solver_sha256"],
            "build_records": current["build_records"],
            "finalizer": file_evidence(__file__),
            "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        },
        "linear_batches": linear,
        "official_jerk_stable_fd": regression,
        "original_A0_regression": original,
        "candidate_report_semantics_preserved": {
            "qualification": documents["candidate"]["qualification"],
            "candidate_suite_passed": True,
            "m4_gate_passed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=RESULTS / "candidate-ir2-official-linear-cu128-v1.json")
    parser.add_argument("--jerk", type=Path, default=RESULTS / "official-jerk-ir2-b1-v1.json")
    parser.add_argument("--original-jerk", type=Path, default=RESULTS / "official-jerk-original-b1-v1.json")
    parser.add_argument(
        "--original-gate", type=Path, default=RESULTS / "acceptance-linear-cu128-verified-v2-gate-reevaluation-v1.json"
    )
    parser.add_argument("--output", type=Path, default=RESULTS / "m4-ir2-accepted-v1.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        require(args.output.suffix == ".json", "Output must end in .json")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Preserve existing result: {args.output}")
    result = finalize(args)
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(acceptance.json_safe(result), handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output) if not args.check_only else None,
                "baseline_role": result["baseline_role"],
                "m4_gate_passed": result["m4_gate_passed"],
                "original_baseline_passed": result["original_baseline_passed"],
                "validated_pairs": 400,
                "reconstructed_jerk_perturbations": 88,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
