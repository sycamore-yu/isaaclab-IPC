"""Check that independent planning and control rates remain separately observable."""

import pytest

from isaaclab_ipc.ros_evaluation import MissionRecorder


def test_reused_plan_does_not_repeat_solver_statistics(tmp_path):
    case = {
        "planner": {"super_planner": {"robot_r": 0.2}},
        "waypoints": [[10.0, 0.0, 1.0]],
        "waypoint_switch_distances": [1.0],
    }
    recorder = MissionRecorder(case, [[100.0, 100.0, 100.0]])
    recorder.observe(1.0, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    recorder.finish("timeout", "fixture_complete")
    solved = {
        "stamp": 1.0, "event": "solve", "solve_ms": 20.0, "frontend_ms": 50.0,
        "control_ms": 70.0, "qp_failed": True, "qp_failures": 1,
        "reanchor_rejected": False, "reanchor_rejections": 0, "acceptance_ms": 0.0,
    }
    recorder.stats.append({"elapsed_s": 0.0, "data": solved})
    recorder.stats.append({"elapsed_s": 0.005, "data": {
        **solved, "stamp": 1.005, "solve_ms": 8.0, "qp_failed": False,
        "reanchor_rejected": True, "reanchor_rejections": 1, "acceptance_ms": 0.5,
    }})
    for i in range(10):
        recorder.stats.append({"elapsed_s": 0.01 * i, "data": {
            **solved, "stamp": 1.01 + i * 0.01, "event": "control", "control_ms": 0.3,
            "state_age": 0.002, "solution_age": 0.01 + i * 0.01,
        }})
    metrics = recorder.save(
        tmp_path, mode="dynamic", benchmark="fixture", policy="acmpc",
        pcd="fixture.pcd", topics={}, timeout=1.0,
    )
    assert metrics["policy_solve_timing"]["count"] == 2
    assert metrics["policy_control_timing"]["count"] == 10
    assert metrics["policy_control_timing"]["rate_hz"] == pytest.approx(100.0)
    assert metrics["qp_solve_time_ms"]["over_10ms"] == 1
    assert metrics["frontend_time_ms"]["mean"] == 50.0
    assert metrics["computation_time_ms"]["mean"] == pytest.approx(0.3)
    assert metrics["policy_state_age_ms"]["mean"] == pytest.approx(2.0)
    assert metrics["qp_failures"] == 1
    assert metrics["reanchor_rejections"] == 1
    assert metrics["plan_acceptance_time_ms"]["mean"] == pytest.approx(0.25)


def test_failed_attempt_keeps_latency_and_final_sfc_violation(tmp_path):
    case = {
        "planner": {"super_planner": {"robot_r": 0.2}},
        "waypoints": [[10.0, 0.0, 1.0]],
        "waypoint_switch_distances": [1.0],
    }
    recorder = MissionRecorder(case, [[100.0, 100.0, 100.0]])
    recorder.observe(1.0, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    recorder.stats = [
        {"data": {"event": "solve", "stamp": 1.0, "solve_ms": 5.0, "planning_ms": 20.0}},
        {"data": {"event": "control", "actual_sfc_violation_m": 0.0}},
        {"data": {"event": "failure", "planning_ms": 200.0, "actual_sfc_violation_m": 0.021}},
    ]
    recorder.finish("failure", "frontend_failure")
    metrics = recorder.save(
        tmp_path, mode="dynamic", benchmark="fixture", policy="fixed-ctbr",
        pcd="fixture.pcd", topics={}, timeout=1.0,
    )
    assert metrics["planning_time_samples"] == 2
    assert metrics["planning_time_ms"]["p95"] == pytest.approx(191.0)
    assert metrics["planning_p95_under_100ms"] is False
    assert metrics["maximum_actual_sfc_violation_m"] == pytest.approx(0.021)
    assert metrics["policy_solve_timing"]["count"] == 1
    assert len(metrics["computational_failures"]) == 1
