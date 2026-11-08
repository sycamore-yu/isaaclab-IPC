"""Frozen CTBR boundary/SFC regression with the current finite runtime budget.

The standalone CPU driver reuses the existing replay serializer and C++ oracle.
It runs no mapping, MPC, physics or GPU work. A historical 1000-step failure is
retained in fixture provenance; future implementations may converge sooner.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from isaaclab_ipc.benchmark_config import load_case
from isaaclab_ipc.pathcover_native import PathCoverPlan, PathCoverSegment, PathCoverTrajectory
from isaaclab_ipc.pathcover_validation import quality_failure_reasons, quality_of_record
from isaaclab_ipc.ros_ctbr_policy import benchmark_frontend_config
from isaaclab_ipc.turbompc.ctbr import CTBRConfig


def test_frozen_ctbr_problem_converges_with_runtime_budget(tmp_path, monkeypatch, record_property):
    pytest.importorskip("isaaclab_ipc._pathcover_native")
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native compiler is unavailable")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from replay_pathcover_gcopter import config_from_snapshot, serialized_problem
    from validate_pathcover_native import ORACLE_SOURCE

    fixture_path = root / "tests/data/pathcover_gcopter_dense_a2_seq914.json"
    fixture = json.loads(fixture_path.read_text())
    problem = fixture["gcopter_problem"]
    runtime = benchmark_frontend_config(load_case("dense"), CTBRConfig(**fixture["model"]))
    config = replace(config_from_snapshot(problem["config"]), max_iterations=runtime.max_iterations)

    source, binary = tmp_path / "replay.cpp", tmp_path / "replay"
    source.write_text(ORACLE_SOURCE)
    command = [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-fopenmp", "-pthread",
               "-DPATHCOVER_DIM=3", "-DPATHCOVER_PORTED", "-I/usr/include/eigen3",
               f"-I{root / 'third_party/pathcover_native/trajectory'}", str(source), "-o", str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run([str(binary), str(config.random_seed)],
                               input=serialized_problem(problem, config), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    values = np.fromstring(completed.stdout, sep=" ")
    pieces, objective = int(values[0]), float(values[1])
    assert np.isfinite(objective)
    assert pieces == len(problem["corridors"]) * config.pieces_per_polytope
    values = values[2:].reshape(pieces, 19)
    trajectory = PathCoverTrajectory(
        values[:, 1:].reshape(pieces, 3, 6), values[:, 0], initial_yaw=problem["initial_yaw"],
        body_yaw_rate_max=config.body_yaw_rate_max, yaw_sample_dt=config.yaw_sample_dt,
        yaw_low_speed=config.yaw_low_speed, gravity=config.gravity,
    )
    boundaries = np.r_[0.0, np.cumsum(trajectory.durations)]
    segments = tuple(
        PathCoverSegment(i // config.pieces_per_polytope, float(boundaries[i]),
                         float(boundaries[i + 1]), float(duration))
        for i, duration in enumerate(trajectory.durations)
    )
    start = float(problem["start_time"])
    initial, terminal = np.asarray(problem["initial_pva"]), np.asarray(problem["terminal_pva"])
    metrics = {"initial_boundary": problem["initial_boundary"],
               "speed_acceptance_limit_m_s": fixture["speed_acceptance_limit_m_s"]}
    plan = PathCoverPlan(
        trajectory, tuple(np.asarray(poly) for poly in problem["corridors"]), segments,
        start, start + trajectory.duration, np.stack((initial[:, 0], terminal[:, 0])),
        terminal[:, 0], metrics,
    )
    record = {**fixture["record"], "frontend_plan": plan}
    for key in ("state", "acceleration", "body_rates"):
        record[key] = np.asarray(record[key])
    row = {key: record[key] for key in ("stamp", "state_time", "plan_sequence_id")}
    quality = quality_of_record(record, row, config.native_options(), fixture["model"])
    failures = quality_failure_reasons(quality)
    (tmp_path / "reference-quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False) + "\n")
    record_property("finite_iteration_budget", config.max_iterations)
    record_property("objective", objective)
    record_property("reference_failures", json.dumps(failures))
    record_property("original_sfc_signed_max_m", quality["comparisons"]["original_sfc_signed_max_m"])
    record_property("actual_body_axis_peaks_rad_s", json.dumps([
        quality["actual_saved_yaw"]["extrema"][key]["value"]
        for key in ("abs_body_p", "abs_body_q", "abs_body_r")
    ]))
    assert not failures, json.dumps(quality, indent=2)
