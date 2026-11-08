"""Small CPU contracts for the offline saved-reference analyzer."""

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_ipc.pathcover_validation import (
    ArchiveIntegrityError,
    PlanCache,
    SavedCurve,
    event_key,
    initial_boundary_contract,
    planning_attempts,
    polynomial_extrema,
    quality_failure_reasons,
    quality_of_record,
    select_records,
    summarize_quality,
)


def saved_hover(*, yaw_rate=0.0, duplicate_terminal=False):
    """Only stored arrays; no trajectory/front-end constructor is involved."""
    duration = 0.02
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 2, -1] = 1.0
    trajectory = SimpleNamespace(
        coefficients=coefficients, durations=np.array([duration]), _boundaries=np.array([0.0, duration]),
        _yaw_times=np.array([0.0, duration, duration] if duplicate_terminal else [0.0, duration]),
        _yaw_values=np.array([0.0, yaw_rate * duration, yaw_rate * duration]
                             if duplicate_terminal else [0.0, yaw_rate * duration]),
        _yaw_rates=np.array([yaw_rate, yaw_rate, 0.0] if duplicate_terminal else [yaw_rate, yaw_rate]),
    )
    faces = np.column_stack((np.r_[np.eye(3), -np.eye(3)], [-2.0, -2.0, -3.0, -2.0, -2.0, 0.0]))
    plan = SimpleNamespace(
        trajectory=trajectory, segments=[SimpleNamespace(corridor_index=0, duration=duration)], corridors=[faces],
        start_time=10.0, valid_until=10.0 + duration, local_goal=np.array([0.0, 0.0, 1.0]),
        metrics={"speed_acceptance_limit_m_s": 6.006, "optimizer_quadrature_feasible": True,
                 "optimizer_sfc_max_m": -1.0, "optimizer_speed_max_m_s": 0.0,
                 "optimizer_thrust_min_n": 0.75 * 9.81, "optimizer_thrust_max_n": 0.75 * 9.81,
                 "optimizer_tilt_max_rad": 0.0, "optimizer_zero_yaw_body_rate_max_rad_s": 0.0},
    )
    config = {"mass": 0.75, "gravity": 9.81, "thrust_min": 0.0, "nominal_thrust_min": 2.1,
              "thrust_max": 22.0725, "tilt_max": np.pi / 2, "max_velocity": 6.0,
              "body_rate_max": 3.0, "body_yaw_rate_max": 1.5}
    model = {"mass": 0.75, "gravity": 9.81, "specific_thrust_limits": [0.0, 29.43],
             "body_rate_limits": [3.0, 3.0, 1.5], "velocity_lower": [-6.0] * 3,
             "velocity_upper": [6.0] * 3, "cost_nodes": 5, "dt": 0.1}
    record = {"stamp": 10.0, "plan_sequence_id": 1, "state_time": 10.0, "frontend_plan": plan,
              "state": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              "acceleration": [0.0, 0.0, 0.0], "body_rates": [0.0, 0.0, 0.0],
              "goal": [0.0, 0.0, 1.0]}
    row = {"event": "solve", "stamp": 10.0, "plan_sequence_id": 1, "state_time": 10.0, "qp_ok": True}
    archive = {"schema_version": 1, "frontend_config": config, "records": [record]}
    metadata = {"parameters": {"frontend": "pathcover"}, "frontend_config": config, "solver": {"model": model}}
    return record, row, archive, metadata


def test_exact_join_keeps_failed_attempts_and_excludes_unreceived_plans():
    record, row, archive, metadata = saved_hover()
    row["qp_ok"] = False  # An unsolved MPC still returned a reference and must be analyzed.
    archive["records"].append({**record, "stamp": 10.1, "plan_sequence_id": 2})
    failure = {"event": "failure", "stamp": 10.05, "planning_ms": 109.4, "error": "MAX_ITER"}
    rows = [row, failure, {"event": "control", "stamp": 10.04}]
    selected, coverage = select_records(archive, rows, metadata)
    assert selected == [(record, row)]
    assert coverage["excluded_count"] == 1
    assert coverage["excluded_archive_records"][0]["after_last_evaluator_solve"] is True
    assert coverage["failed_attempts_without_returned_plan"] == 1
    attempts = planning_attempts(rows)
    assert attempts["attempt_count"] == 2
    assert attempts["returned_mpc_unsolved_count"] == 1
    assert attempts["failed_attempts"] == [failure]
    assert attempts["control_event_count"] == 1


@pytest.mark.parametrize("stamp,sequence", [(True, 1), (float("nan"), 1), (float("inf"), 1),
                                           (1.0, True), (1.0, 1.0), (1.0, 0)])
def test_event_identity_rejects_ambiguous_or_nonfinite_values(stamp, sequence):
    with pytest.raises(ArchiveIntegrityError):
        event_key({"stamp": stamp, "plan_sequence_id": sequence})


@pytest.mark.parametrize("corruption", ["missing", "duplicate_outside", "duplicate_evaluator", "state_time",
                                         "config", "missing_plan", "schema"])
def test_archive_coverage_rejects_incomplete_or_changed_evidence(corruption):
    record, row, archive, metadata = saved_hover()
    rows = [row]
    if corruption == "missing":
        row["stamp"] = np.nextafter(row["stamp"], np.inf)
    elif corruption == "duplicate_outside":
        archive["records"] += [{**record, "stamp": 11.0, "plan_sequence_id": 2},
                               {**record, "stamp": 12.0, "plan_sequence_id": 2}]
    elif corruption == "duplicate_evaluator":
        rows.append({**row, "stamp": 11.0})
    elif corruption == "state_time":
        row["state_time"] += 0.1
    elif corruption == "config":
        metadata["frontend_config"] = {**metadata["frontend_config"], "mass": 1.0}
    elif corruption == "missing_plan":
        record["frontend_plan"] = None
    else:
        archive["schema_version"] = 2
    with pytest.raises(ArchiveIntegrityError):
        select_records(archive, rows, metadata)


def test_polynomial_extrema_find_interior_sfc_peak():
    record, _, _, metadata = saved_hover()
    plan = record["frontend_plan"]
    plan.trajectory.durations[:] = 1.0
    plan.segments[0].duration = 1.0
    plan.trajectory._boundaries[:] = [0.0, 1.0]
    plan.trajectory._yaw_times[:] = [0.0, 1.0]
    plan.trajectory.coefficients[0, 0] = [0.0, 0.0, 0.0, -1.0, 1.0, 0.0]  # x=t-t^2
    plan.corridors[0][0, 3] = -0.24
    curve = SavedCurve(plan)
    result = polynomial_extrema(curve, metadata["frontend_config"]["mass"], 9.81)
    peak = result["extrema"]["sfc_max_m"]
    assert peak["value"] == pytest.approx(0.01, abs=1e-15)
    assert peak["relative_time_s"] == pytest.approx(0.5, abs=1e-15)
    assert peak["face"] == 0


def test_terminal_left_rate_and_duplicate_knot_survive_full_quality():
    record, row, _, metadata = saved_hover(yaw_rate=2.0, duplicate_terminal=True)
    result = quality_of_record(record, row, metadata["frontend_config"], metadata["solver"]["model"])
    rates = result["actual_saved_yaw"]
    assert rates["extrema"]["abs_body_r"]["value"] == pytest.approx(2.0, abs=1e-12)
    assert rates["terminal_curve_left_limit"]["world_yaw_rate_rad_s"] == pytest.approx(2.0)
    assert rates["deployed_sample_at_duration"]["world_yaw_rate_rad_s"] == 0.0
    assert rates["saved_yaw_metadata"]["zero_width_interval_indices"] == [1]
    assert quality_failure_reasons(result) == ["actual_body_axes"]


def test_free_collective_does_not_reinterpret_historical_fixed_acceleration():
    record, _, _, metadata = saved_hover()
    record["acceleration"] = [0.0, 0.0, 0.3]
    curve, config = SavedCurve(record["frontend_plan"]), metadata["frontend_config"]
    fixed = initial_boundary_contract(curve, record["state"], record["acceleration"], config)
    free = initial_boundary_contract(curve, record["state"], record["acceleration"], config,
                                     {"mode": "ctbr_collective"})
    assert fixed["mode"] == "fixed_pva"
    assert fixed["fixed_acceleration_max_abs_error_m_s2"] == pytest.approx(0.3)
    assert free["fixed_acceleration_max_abs_error_m_s2"] is None
    assert free["fixed_attitude_rotation_error_rad"] == pytest.approx(0.0)
    assert free["reference_minus_measured_initial_pva"][2, 2] == pytest.approx(-0.3)


def test_quality_allowances_and_proxy_nominal_diagnostics_are_separate():
    record, row, _, metadata = saved_hover()
    result = quality_of_record(record, row, metadata["frontend_config"], metadata["solver"]["model"])
    result["quadrature"]["recorded_feasible"] = False
    result["comparisons"]["reference_nominal_thrust_shortfall_n"] = 1.0
    assert quality_failure_reasons(result) == []
    result["comparisons"]["reference_thrust_upper_positive_excess_n"] = 1e-3
    assert quality_failure_reasons(result) == []
    result["comparisons"]["reference_thrust_upper_positive_excess_n"] = np.nextafter(1e-3, np.inf)
    assert quality_failure_reasons(result) == ["physical_thrust_high"]
    result["bounds"]["reference"]["recorded_state_conditioned_speed_threshold_m_s"] = None
    with pytest.raises(ArchiveIntegrityError, match="speed threshold"):
        quality_failure_reasons(result)


def test_empty_reference_coverage_has_no_quality_pass():
    summary = summarize_quality({"plans": [], "coverage": {
        "complete_for_evaluator": True, "evaluator_solve_count": 0, "matched_record_count": 0}})
    assert summary["reference_quality_passed"] is None
    assert summary["failure_count"] == 0


def test_cache_is_bound_to_inputs_code_and_exact_event(tmp_path):
    provenance = {"artifact_sha256": {"archive": "a"}, "analysis_source_sha256": {"module": "b"}}
    cache = PlanCache(tmp_path / "plans", provenance)
    record = {"stamp": 1.0, "plan_sequence_id": 1}
    cache.put(record, record)
    assert cache.get(record) == record
    with pytest.raises(ArchiveIntegrityError, match="identity"):
        cache.get({**record, "stamp": np.nextafter(1.0, np.inf)})
    with pytest.raises(ArchiveIntegrityError, match="changed"):
        PlanCache(tmp_path / "plans", {**provenance, "analysis_source_sha256": {"module": "c"}})


def test_saved_identity_validates_runtime_scope_and_end_artifact_hashes():
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_marsim_references.py"
    validate = runpy.run_path(str(script))["validate_saved_identity"]
    identity = {
        "schema_version": 2, "status": "identity_captured",
        "controller": {"git_head": "git", "source_sha256": "sources", "source_file_sha256": {"src/a.py": "a"},
                       "native": {"library": {"path": "/recorded/build/solver.so", "sha256": "solver"}}},
        "frontend": {"library": {"path": "/recorded/src/frontend.so", "sha256": "frontend"}},
        "executor": {"runtime_binary_sha256": {f"build/node{i}": str(i) for i in range(4)},
                     "validation_binary_sha256": {"build/check_allocator": "checker"}}, "training": {},
    }
    run = {"git_revision": "git", "source_sha256": {"src/a.py": "a"},
           "native_library_sha256": {"/recorded/build/solver.so": "solver", "/recorded/src/frontend.so": "frontend"},
           "executable_sha256": {f"/recorded/build/node{i}": str(i) for i in range(4)}}
    metadata = {"solver": {"native_extension_path": "/recorded/build/solver.so"}, "checkpoint_sha256": None}
    hashes = {"run.json": "run", "policy/metadata.json": "metadata", "metrics.json": "metrics"}
    start = {"phase": "start", "run": "arbitrary-run", "identity_matches": True, "identity_sha256": "identity"}
    end = {**start, "phase": "end", "run_evidence": {
        "git_revision": "git", "run_json": {"sha256": "run"},
        "metadata": {"sha256": "metadata"}, "metrics": {"sha256": "metrics"}}}
    result = validate(identity, "identity", start, end, run, metadata, hashes)
    assert result["identity_evidence_matches"] is True
    assert result["validation_only_binary_sha256"] == {"build/check_allocator": "checker"}
    bad = copy.deepcopy(run)
    bad["executable_sha256"]["/recorded/build/node0"] = "wrong"
    with pytest.raises(ArchiveIntegrityError, match="ROS executable"):
        validate(identity, "identity", start, end, bad, metadata, hashes)
    with pytest.raises(ArchiveIntegrityError, match="changed after"):
        validate(identity, "identity", start, end, run, metadata, {**hashes, "metrics.json": "changed"})


def test_cli_reads_cpu_archive_and_writes_failed_quality_with_failed_attempt(tmp_path):
    import torch

    _, row, archive, metadata = saved_hover(yaw_rate=2.0, duplicate_terminal=True)
    root = Path(__file__).resolve().parents[1]
    run, output = tmp_path / "arbitrary-run", tmp_path / "analysis"
    (run / "policy").mkdir(parents=True)
    metadata["checkpoint_sha256"] = None
    (run / "policy/metadata.json").write_text(json.dumps(metadata))
    (run / "metrics.json").write_text(json.dumps({"status": "failure", "reason": "fixture frontend failure"}))
    (run / "run.json").write_text(json.dumps({
        "git_revision": "fixture-only", "native_library_sha256": {},
        "executable_sha256": {}, "source_sha256": {},
    }))
    failure = {"event": "failure", "stamp": 10.1, "planning_ms": 109.4,
               "error": "PathCoverPlanningError: MAX_ITER"}
    (run / "policy_stats.jsonl").write_text("\n".join(json.dumps({"data": value}) for value in (row, failure)) + "\n")
    torch.save(archive, run / "policy/reference-plans.pt")
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": "",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/analyze_marsim_references.py"),
         "--run-dir", str(run), "--output-dir", str(output)],
        cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert completed.returncode == 0, completed.stdout
    report = json.loads((output / "references.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert report["coverage"]["matched_record_count"] == report["coverage"]["evaluator_solve_count"] == 1
    assert report["planning_attempts"]["attempt_count"] == 2
    assert report["planning_attempts"]["failed_attempt_count"] == 1
    assert report["planning_attempts"]["failed_attempts"] == [failure]
    assert report["quality"]["reference_quality_passed"] is False
    assert report["quality"]["failure_category_counts"] == {"actual_body_axes": 1}
    assert summary["quality"] == report["quality"]
    assert "plans" not in summary
    assert summary["provenance"]["recorded_git_revision"] == "fixture-only"
    assert summary["provenance"]["saved_identity"]["identity_evidence_matches"] is None
    for name, digest in report["provenance"]["artifact_sha256"].items():
        assert digest == hashlib.sha256((run / name).read_bytes()).hexdigest()
    assert len(report["provenance"]["artifact_sha256"]) == 5
    assert summary["per_plan_report"]["sha256"] == hashlib.sha256((output / "references.json").read_bytes()).hexdigest()
    assert (output / "summary.md").is_file()
    assert json.loads((output / "plan-cache/progress.json").read_text())["aggregate_available"] is True
