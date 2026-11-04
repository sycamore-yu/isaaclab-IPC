"""CPU evidence-gate tests; these fixtures do not simulate solver acceptance."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("turbompc_acceptance_gate", ROOT / "scripts/turbompc/check_acceptance.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def evidence(tmp_path):
    """A self-contained passing record with known data and trusted A identity."""
    case_helper = gate.load_example("_gate_test_case", ROOT / "scripts/turbompc_benchmark_common.py")
    case = case_helper.make_case(1, 15, False)
    trajectory = np.zeros((1, 16, 12))
    trajectory[:, 0, :9] = case["x0"]
    for k in range(15):
        old = trajectory[:, k, :9]
        new = old.copy()
        dt = case["dt"]
        new[:, :3] += dt * old[:, 3:6] + 0.5 * dt**2 * old[:, 6:9]
        new[:, 3:6] += dt * old[:, 6:9]
        trajectory[:, k + 1, :9] = new
    arrays = {f"input_{key}": np.asarray(value) for key, value in case.items()}
    arrays.update(
        base_trajectory=trajectory,
        native_trajectory=trajectory.copy(),
        base_returned_controls=trajectory[..., 9:].copy(),
        base_status=np.zeros(1, dtype=np.int32),
        base_valid=np.ones(1, dtype=bool),
        callback_mpc_cost_adjoint=np.ones((1, 12)),
        callback_mpc_context_adjoint=np.ones((1, 9)),
    )
    for prefix in ("base", "native", "raw_retained", "fd_h0", "fd_h1"):
        arrays[f"{prefix}_cost_gradient"] = np.ones((1, 12))
        arrays[f"{prefix}_state_gradient"] = np.ones((1, 9))
    library = str(tmp_path / "original/libcudss_sparse_kkt_ffi.so")
    library_hash = "a" * 64
    backend = str(tmp_path / "reference/examples/rl/util/mpc.py")
    bridge = str(tmp_path / "reference/examples/rl/util/jax_torch.py")
    sources = {backend: "b" * 64, bridge: "c" * 64}
    provenance = {
        "kkt_role": "original",
        "upstream_commit": gate.REVISION,
        "verified_upstream_files": sources,
        "same_saved_inputs": True,
    }
    for key in ("expected_kkt_library", "actual_loaded_kkt_library", "installed_baseline_library"):
        provenance[key] = library
    for key in (
        "expected_kkt_library_sha256", "actual_loaded_kkt_sha256",
        "installed_baseline_sha256_before", "installed_baseline_sha256_after",
    ):
        provenance[key] = library_hash
    settings = {
        "constraint_handling": "projection", "log_barrier_kappa": 0.0,
        "tol_convergence": 1e-6, "convergence_criterion": "first_order",
        "forward_hessian_min_eig": 1e-12, "num_sqp_iteration_max": 15,
        "admm": gate.ADMM | {"eps_abs": 1e-8, "eps_rel": 1e-8},
        "linesearch": deepcopy(gate.SOLVER["linesearch"]),
    }
    document = {
        "role": "official full-plan jerk accuracy diagnostic: original",
        "case": {
            "batch": 1, "horizon": 15, "nx": 9, "nu": 3, "dtype": "float64", "seed": 719,
            "loss": "sum(all 16 control nodes * original saved probe)",
        },
        "provenance": provenance,
        "implementation": {
            "backend_class": "_jerk_official_backend.MPCBackend",
            "bridge_class": "_jerk_official_bridge.MPC",
            "entry_point": "official MPC.forward (mean, full controls)",
            "backend_file": backend, "bridge_file": bridge,
            "effective_solver_parameters": settings,
        },
        "checks": dict.fromkeys(gate.OFFICIAL_REGRESSION_CHECKS, True),
        "diagnostic_checks_passed": True,
        "finite_difference": {
            "steps": [{"step": h, "old_direction": {"official_relative_error": 0.0}} for h in (1e-4, 5e-5)],
            "forward_perturbations": 88,
            "all_solved": True, "all_finite": True,
            "all_active_masks_equal": True, "all_mask_replay_plans_match": True,
            "perturbations": [{
                "solved": True, "finite": True, "active_masks_equal": True,
                "mask_replay_solved": True, "mask_duals_finite": True, "physical_maximum": 0.0,
            } for _ in range(88)],
        },
    }
    return {"path": tmp_path / "official.json", "document": document, "arrays": arrays,
            "libraries": {library: library_hash}, "sources": sources}


def evaluate(evidence):
    evidence["path"].write_text(json.dumps(evidence["document"]))
    np.savez_compressed(evidence["path"].with_suffix(".npz"), **evidence["arrays"])
    return gate.validate_official_regression(evidence["path"], evidence["libraries"], evidence["sources"])


@pytest.mark.parametrize("legacy", [{"unresolved": True}, {"incomplete": True, "read_errors": ["missing"]}])
def test_compliant_official_pass_is_not_permanently_blocked_by_legacy_failures(evidence, legacy):
    regression = evaluate(evidence)
    assert regression["passed"], regression
    report = {"selected_linear_suite_passed": True, "prior_extension_diagnostic": legacy,
              "prior_failure_blocks_m4": True}
    gate.apply_official_regression_gate(report, regression)
    assert report["m4_gate_passed"]
    assert not report["prior_failure_blocks_m4"]
    assert report["prior_extension_diagnostic"] == legacy


def test_failed_original_report_blocks_even_with_passing_linear_suite(evidence):
    evidence["document"]["checks"]["cost_gradient_vs_native"] = False
    evidence["document"]["diagnostic_checks_passed"] = False
    evidence["arrays"]["base_cost_gradient"] *= 1.01
    regression = evaluate(evidence)
    assert not regression["passed"]
    assert regression["checks"]["same_measured_A_library"]
    assert not regression["checks"]["raw_cost_gradient_matches_reference_and_fd"]
    report = {"selected_linear_suite_passed": True}
    gate.apply_official_regression_gate(report, regression)
    assert not report["m4_gate_passed"]


def test_candidate_role_cannot_pass_original_gate_even_with_all_matching_numbers(evidence):
    evidence["document"]["role"] = "official full-plan jerk accuracy diagnostic: ir2-candidate"
    evidence["document"]["provenance"]["kkt_role"] = "ir2-candidate"
    regression = evaluate(evidence)
    assert regression["checks"]["raw_cost_gradient_matches_reference_and_fd"]
    assert regression["checks"]["same_measured_A_library"]
    assert not regression["checks"]["original_role"]
    assert not regression["passed"]


@pytest.mark.parametrize("field", ["actual_loaded_kkt_sha256", "installed_baseline_sha256_after"])
def test_report_claiming_original_rejected_if_actual_library_does_not_match(evidence, field):
    evidence["document"]["provenance"][field] = "d" * 64
    result = evaluate(evidence)
    assert not result["checks"]["same_measured_A_library"]
    assert not result["passed"]


def test_matching_library_hash_at_unrelated_path_is_rejected(evidence):
    evidence["document"]["provenance"]["actual_loaded_kkt_library"] += ".candidate"
    result = evaluate(evidence)
    assert not result["checks"]["same_measured_A_library"]
    assert not result["passed"]


def test_wrong_inputs_rejected_despite_claim_of_same_saved_inputs(evidence):
    evidence["arrays"]["input_x0"] = evidence["arrays"]["input_x0"] + 1e-8
    result = evaluate(evidence)
    assert not result["checks"]["exact_fixed_inputs"]
    assert not result["passed"]


def test_changed_solve_tolerance_rejected_with_identical_inputs(evidence):
    evidence["document"]["implementation"]["effective_solver_parameters"]["tol_convergence"] = 1e-4
    result = evaluate(evidence)
    assert result["checks"]["exact_fixed_inputs"]
    assert not result["checks"]["fixed_solver_settings"]
    assert not result["passed"]


def test_raw_gradient_failure_cannot_be_overruled_by_report_booleans(evidence):
    evidence["arrays"]["base_cost_gradient"] *= 1.01
    result = evaluate(evidence)
    assert result["checks"]["reported_checks_passed"]
    assert not result["checks"]["raw_cost_gradient_matches_reference_and_fd"]
    assert not result["passed"]


@pytest.mark.parametrize("kind", ["report", "arrays", "checks", "fd_perturbations"])
def test_missing_evidence_fails_closed(evidence, kind):
    if kind == "report":
        result = gate.validate_official_regression(evidence["path"], evidence["libraries"], evidence["sources"])
    elif kind == "arrays":
        evidence["path"].write_text(json.dumps(evidence["document"]))
        result = gate.validate_official_regression(evidence["path"], evidence["libraries"], evidence["sources"])
    else:
        if kind == "checks":
            evidence["document"]["checks"] = {}
        else:
            evidence["document"]["finite_difference"]["perturbations"] = []
        result = evaluate(evidence)
    assert not result["passed"]


def test_passing_regression_does_not_override_failed_main_suite(evidence):
    regression = evaluate(evidence)
    assert regression["passed"]
    report = {"selected_linear_suite_passed": False}
    gate.apply_official_regression_gate(report, regression)
    assert not report["m4_gate_passed"]


def test_absent_legacy_files_are_diagnostic_only(tmp_path):
    result = gate.existing_diagnostics(tmp_path / "absent-history")
    assert result["incomplete"]
    assert not result["blocks_m4"]
