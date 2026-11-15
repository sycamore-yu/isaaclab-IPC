"""Offline replay must preserve measured MARSIM/SUPER scene and flight evidence."""

import json

import numpy as np
import pytest

from scripts.export_marsim_super_rerun import _reference_snapshot, export


def test_reference_snapshot_retains_08s_history():
    commands = np.zeros((4, 16))
    commands[:, 0] = [1.0, 1.5, 2.0, 2.1]
    commands[:, 2] = [1, 2, 3, 4]
    np.testing.assert_array_equal(_reference_snapshot(commands, 4)[:, 0], [2, 3, 4])


def test_exports_actual_poses_and_original_pcd(tmp_path):
    pcd = tmp_path / "official.pcd"
    pcd.write_text(
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n0 0 0\n1 1 1\n2 2 2\n"
    )
    run = tmp_path / "run"
    run.mkdir()
    info = {
        "policy": "super", "status": "success", "benchmark": "dense", "mode": "ideal",
        "pcd": str(pcd), "pcd_points": 3, "robot_radius_m": .2,
        "waypoints": [[0, 1, 1], [0, 0, 1]],
        "waypoints_reached": 2,
        "waypoint_events": [
            {"index": 0, "stamp_s": 1.01, "position_m": [0, 1, 1]},
            {"index": 1, "stamp_s": 1.02, "position_m": [0, 0, 1]},
        ],
        "minimum_clearance_m": 0.3,
    }
    (run / "metrics.json").write_text(json.dumps(info))
    commands = np.zeros((2, 16))
    commands[:, 0] = [1.005, 1.015]
    np.savez_compressed(
        run / "trajectory.npz",
        stamp_s=np.array([1.0, 1.01, 1.02]),
        position_m=np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]]),
        quaternion_xyzw=np.array([[0, 0, 0, 1]] * 3),
        speed_mps=np.array([0, 1, 0]),
        clearance_m=np.array([.5, .4, .3]),
        position_commands=commands,
        lidar=np.empty((0, 4)),
    )
    output = tmp_path / "recording.rrd"
    result = export(run, output)
    assert output.stat().st_size > 1_000
    assert result["pcd_points"] == 3
    assert result["waypoints_reached"] == 2
    assert result["status"] == "success"
    with pytest.raises(FileExistsError):
        export(run, output)

    info["pcd_points"] = 4
    (run / "metrics.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="PCD differs"):
        export(run, tmp_path / "invalid.rrd")


def test_exports_native_ctbr_flight_without_inventing_position_commands(tmp_path):
    pcd = tmp_path / "dense.pcd"
    pcd.write_text(
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ascii\n0 0 0\n1 1 1\n2 2 2\n"
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({
        "policy": "fixed-ctbr", "status": "success", "benchmark": "dense",
        "mode": "dynamic", "pcd": str(pcd), "pcd_points": 3,
        "waypoints": [[0, 1, 1], [0, 0, 1]], "waypoints_reached": 2,
        "waypoint_events": [
            {"index": 0, "stamp_s": 1.01, "position_m": [0, 1, 1]},
            {"index": 1, "stamp_s": 1.02, "position_m": [0, 0, 1]},
        ],
        "minimum_clearance_m": 0.2,
    }))
    np.savez_compressed(
        run / "trajectory.npz",
        stamp_s=np.array([1., 1.01, 1.02]),
        position_m=np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]]),
        quaternion_xyzw=np.array([[0, 0, 0, 1]] * 3),
        speed_mps=np.array([0, 1, 0]),
        clearance_m=np.array([.3, .25, .2]),
        position_commands=np.empty((0, 16)),
        ctbr=np.array([[1., 0., 9.81, 0., .1, 0.], [1.01, .01, 10., 0., -.1, 0.]]),
        lidar=np.empty((0, 4)),
    )
    result = export(run, tmp_path / "dense.rrd")
    assert result["policy"] == "fixed-ctbr"
    assert result["control_samples"] == 2
    assert result["flight_samples"] == 3
    assert result["waypoints_reached"] == 2
