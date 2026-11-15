#!/usr/bin/env python3
"""Convert recorded MARSIM/SUPER or native CTBR flights to Rerun.

Actual odometry and published controls are replayed against the original PCD.
The evaluation saves no raw LiDAR frames or MPC candidate trajectories.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from isaaclab_ipc.scenes import read_pcd


def flight_blueprint(*, ctbr=False):
    charts = [
        rrb.TimeSeriesView(origin="/metrics/speed", name="Actual / reference speed (m/s)"),
        rrb.TimeSeriesView(origin="/metrics/clearance", name="Swept PCD / ground clearance (m)"),
    ]
    if ctbr:
        charts.extend([
            rrb.TimeSeriesView(origin="/metrics/ctbr/collective_thrust_mps2", name="Collective thrust (m/s²)"),
            rrb.TimeSeriesView(origin="/metrics/ctbr/body_rates", name="Body rates (rad/s)"),
        ])
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="Official PCD + measured flight" if ctbr else "Official PCD + actual flight / reference",
                eye_controls=rrb.EyeControls3D(
                    position=(70, -85, 70) if ctbr else (65, -5, 55),
                    look_target=(0, 0, 1.5),
                    eye_up=(0, 0, 1) if ctbr else (0, 1, 0),
                ),
                line_grid=False,
            ),
            rrb.Vertical(
                *charts,
                rrb.TextDocumentView(origin="/status/current", name="Mission progress"),
                row_shares=[2] * (len(charts) + 1),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="flight_time", expanded=True),
        collapse_panels=True,
    )


def _reference_snapshot(commands, last):
    """Return only published PositionCommand samples over the last 0.8 s."""
    selected = commands[:last]
    if not len(selected):
        return np.empty((0, 3))
    cutoff = selected[-1, 0] - 0.8
    return selected[selected[:, 0] >= cutoff, 2:5]


def export(source: Path, destination: Path, *, pcd_override: Path | None = None):
    if destination.exists():
        raise FileExistsError(destination)
    info = json.loads((source / "metrics.json").read_text())
    native_ctbr = info["policy"] in ("fixed-ctbr", "acmpc")
    if info["policy"] != "super" and not native_ctbr:
        raise ValueError(f"Unsupported MARSIM policy: {info['policy']}")
    with np.load(source / "trajectory.npz", allow_pickle=False) as data:
        states = {name: data[name] for name in data.files}
    stamps = states["stamp_s"]
    positions = states["position_m"]
    quaternions = states["quaternion_xyzw"]
    speed = states["speed_mps"]
    clearance = states["clearance_m"]
    reference = states["position_commands"]
    controls = states.get("ctbr", np.empty((0, 6)))
    if not len(stamps) or not np.all(np.diff(stamps) > 0):
        raise ValueError("Odometry timestamps must be non-empty and strictly increasing")
    if positions.shape != (len(stamps), 3) or quaternions.shape != (len(stamps), 4):
        raise ValueError("Missing actual 3D odometry / attitude")
    if reference.shape[1] != 16 or (len(reference) and np.any(np.diff(reference[:, 0]) < 0)):
        raise ValueError("Unexpected PositionCommand recording")
    if native_ctbr and (controls.ndim != 2 or controls.shape[1] != 6 or not len(controls)):
        raise ValueError("Native CTBR recording requires published CTBR samples [N,6]")

    pcd = pcd_override if pcd_override is not None else Path(info["pcd"])
    points = read_pcd(pcd, drop_nan=True)
    if len(points) != info["pcd_points"]:
        raise ValueError("PCD differs from the collision evaluator's original map")
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = {
        "super": "SUPER",
        "fixed-ctbr": "PathCover + CTBR (fixed cost)",
        "acmpc": "PathCover + AC-MPC",
    }[info["policy"]]
    control_description = ("CTBR controls in the time series; " if native_ctbr
                           else "magenta: published PositionCommand samples; ")
    stream = rr.RecordingStream(f"{label} · {info['benchmark']} · {info['status']}")
    try:
        stream.save(destination, default_blueprint=flight_blueprint(ctbr=native_ctbr))
        stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        colors = np.where((points[:, 2] >= 0.2)[:, None], [110, 161, 121], [77, 78, 79]).astype(np.uint8)
        stream.log("world/obstacles", rr.Points3D(
            points, colors=colors, radii=0.012 if native_ctbr else 0.021,
        ), static=True)

        waypoints = np.asarray(info["waypoints"], dtype=float)
        start = positions[0]
        stream.log(
            "world/waypoints",
            rr.Points3D(
                np.vstack((start, waypoints[0])),
                labels=["Start / return", "Outbound goal"],
                colors=[[255, 212, 50], [255, 212, 50]],
                radii=[0.27, 0.27],
            ),
            static=True,
        )
        stream.log(
            "world/actual_full_route",
            rr.LineStrips3D([positions], colors=[115, 135, 160],
                            radii=0.045 if native_ctbr else 0.018),
            static=True,
        )
        stream.log(
            "world/drone/body",
            rr.Boxes3D(half_sizes=[0.21, 0.15, 0.065], colors=[48, 147, 245]),
            static=True,
        )
        stream.log(
            "world/drone/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3)),
                vectors=np.eye(3) * 0.45,
                colors=[[255, 70, 70], [70, 245, 80], [80, 100, 255]],
            ),
            static=True,
        )
        stream.log(
            "status/recording",
            rr.TextDocument(
                f"{label} {info['benchmark']} / {info['mode']} | {info['status']}\n"
                f"Official waypoint mission (out and back), {len(waypoints)} targets.\n"
                f"Recorded odometry: {len(stamps)}, "
                f"{'CTBR commands' if native_ctbr else 'PositionCommand'}: "
                f"{len(controls) if native_ctbr else len(reference)}, "
                f"LiDAR messages: {len(states['lidar'])}.\n"
                f"Source: {source}\n"
                f"Map: original {pcd.name}, {len(points):,} valid PCD points.\n"
                "Blue: actual body and travelled trail; gray: complete actual route; "
                f"{control_description}yellow: official waypoints.\n"
                "Speed / clearance are recorded evaluation metrics. "
                "Raw LiDAR frames and internal optimizer reference/SFC geometry were not saved."
            ),
            static=True,
        )

        origin = stamps[0]
        command_i = 0
        last_command_i = 0
        ctbr_i = 0
        waypoint_i = 0
        events = sorted(info["waypoint_events"], key=lambda item: item["stamp_s"])
        for i, stamp in enumerate(stamps):
            stream.set_time("flight_time", duration=float(stamp - origin))
            stream.log("world/drone", rr.Transform3D(
                translation=positions[i], rotation=rr.Quaternion(xyzw=quaternions[i])
            ))
            stream.log("metrics/speed/actual_mps", rr.Scalars(float(speed[i])))
            stream.log("metrics/clearance/actual_m", rr.Scalars(float(clearance[i])))

            if i % 8 == 0 or i == len(stamps) - 1:
                stream.log("world/actual_trail", rr.LineStrips3D(
                    [positions[:i + 1]], colors=[50, 165, 245],
                    radii=0.065 if native_ctbr else 0.033,
                ))

            while command_i < len(reference) and reference[command_i, 0] <= stamp:
                command_i += 1
            if command_i != last_command_i and (i % 8 == 0 or i == len(stamps) - 1):
                current = reference[command_i - 1]
                stream.log("world/published_command", rr.Points3D(
                    [current[2:5]], colors=[234, 98, 224], radii=0.12,
                    labels=["Published PositionCommand"],
                ))
                recent = _reference_snapshot(reference, command_i)
                if len(recent) >= 2:
                    stream.log("world/recent_reference", rr.LineStrips3D(
                        [recent], colors=[234, 98, 224], radii=0.022
                    ))
                stream.log("metrics/speed/reference_mps", rr.Scalars(float(np.linalg.norm(current[5:8]))))
                last_command_i = command_i

            if native_ctbr:
                while ctbr_i < len(controls) and controls[ctbr_i, 0] <= stamp:
                    ctbr_i += 1
                if ctbr_i:
                    control = controls[ctbr_i - 1]
                    stream.log("metrics/ctbr/collective_thrust_mps2", rr.Scalars(float(control[2])))
                    for axis, value in zip(("wx", "wy", "wz"), control[3:6], strict=True):
                        stream.log(f"metrics/ctbr/body_rates/{axis}_radps", rr.Scalars(float(value)))

            while waypoint_i < len(events) and events[waypoint_i]["stamp_s"] <= stamp:
                reached = events[waypoint_i]
                stream.log("world/reached_waypoint", rr.Points3D(
                    [reached["position_m"]], colors=[87, 226, 102], radii=0.22,
                    labels=[f"Reached waypoint {reached['index'] + 1}"],
                ))
                waypoint_i += 1

            if i == 0 or i == len(stamps) - 1 or i % 100 == 0 or (
                waypoint_i and i > 0 and events[waypoint_i - 1]["stamp_s"] > stamps[i - 1]
            ):
                stream.log("status/current", rr.TextDocument(
                    f"{label} / {info['mode']} / {info['benchmark']}\n"
                    f"Flight recording time: {stamp - origin:.2f} s\n"
                    f"Waypoints reached: {waypoint_i}/{len(waypoints)}\n"
                    f"Speed: {speed[i]:.2f} m/s\n"
                    f"Swept clearance: {clearance[i]:.3f} m\n"
                    f"Result: {info['status'] if i == len(stamps) - 1 else 'in flight'}"
                ))

        stream.log("world/end", rr.Points3D(
            [positions[-1]], colors=[[60, 215, 95] if info["status"] == "success" else [235, 64, 50]],
            radii=0.3, labels=[info["status"].upper()],
        ))
        stream.flush()
    finally:
        stream.disconnect()

    return {
        "file": str(destination), "size_bytes": destination.stat().st_size,
        "source": str(source), "mode": info["mode"], "benchmark": info["benchmark"],
        "status": info["status"], "flight_samples": len(stamps), "pcd_points": len(points),
        "policy": info["policy"], "control_samples": len(controls) if native_ctbr else len(reference),
        "waypoints_reached": info["waypoints_reached"],
        "duration_s": round(float(stamps[-1] - stamps[0]), 3),
        "min_clearance_m": info["minimum_clearance_m"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing MARSIM evaluation directory")
    parser.add_argument("--output", type=Path, required=True, help="New .rrd path")
    parser.add_argument("--pcd", type=Path, help="Equivalent local copy of the exact evaluation PCD")
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output, pcd_override=args.pcd), indent=2), flush=True)


if __name__ == "__main__":
    main()
