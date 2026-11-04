#!/usr/bin/env python3
"""Exercise real Flightmare motion and MARSIM LiDAR in an existing ROS 2 graph.

Start flightmare_node with command_source:=position_command and the original
benchmark initial pose first. This script publishes only PositionCommand; all
measurements come from integrated odometry and the actual LiDAR renderer.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from isaaclab_ipc.ros_evaluation import timing_summary

START = np.array([0.0, -50.0, 1.5])
INITIAL_YAW = 1.5741
TURN = np.pi / 3
SPEED = 0.75
PHASE_DURATION = 2.0
PHASES = ("hover", "accelerate", "turn", "brake", "settle")
DURATION = PHASE_DURATION * len(PHASES)


def reference(elapsed):
    """Continuous p/v/a/jerk reference: 3 m forward, then stop, yaw +60 deg."""
    t = float(np.clip(elapsed, 0, DURATION))
    phase = min(int(t / PHASE_DURATION), len(PHASES) - 1)
    u = np.clip((t - phase * PHASE_DURATION) / PHASE_DURATION, 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = 30 * u**2 - 60 * u**3 + 30 * u**4
    dds = 60 * u - 180 * u**2 + 120 * u**3
    integral = 2.5 * u**4 - 3 * u**5 + u**6
    position = START.copy()
    velocity, acceleration, jerk = np.zeros((3, 3))
    yaw, yaw_rate = INITIAL_YAW, 0.0
    if phase == 1:
        position[1] += SPEED * PHASE_DURATION * integral
        velocity[1] = SPEED * s
        acceleration[1] = SPEED / PHASE_DURATION * ds
        jerk[1] = SPEED / PHASE_DURATION**2 * dds
    elif phase == 2:
        position[1] += SPEED * PHASE_DURATION * (0.5 + u)
        velocity[1] = SPEED
        yaw += TURN * s
        yaw_rate = TURN / PHASE_DURATION * ds
    elif phase == 3:
        position[1] += SPEED * PHASE_DURATION * (1.5 + u - integral)
        velocity[1] = SPEED * (1 - s)
        acceleration[1] = -SPEED / PHASE_DURATION * ds
        jerk[1] = -SPEED / PHASE_DURATION**2 * dds
        yaw += TURN
    elif phase == 4:
        position[1] += 2 * SPEED * PHASE_DURATION
        yaw += TURN
    return position, velocity, acceleration, jerk, yaw, yaw_rate


def angle_error(value):
    return np.arctan2(np.sin(value), np.cos(value))


def analyze(states, commands, accelerations, ctbr, lidar_poses, clouds, digests, *, completed, reason):
    phase_results = {}
    for index, name in enumerate(PHASES):
        phase = states[(states[:, 1] >= index * PHASE_DURATION) & (states[:, 1] < (index + 1) * PHASE_DURATION)]
        if not len(phase):
            phase_results[name] = {"samples": 0}
            continue
        refs = [reference(t) for t in phase[:, 1]]
        target_position = np.array([ref[0] for ref in refs])
        target_yaw = np.array([ref[4] for ref in refs])
        final = phase[phase[:, 1] >= (index + 1) * PHASE_DURATION - 0.5]
        speed = np.linalg.norm(phase[:, 5:8], axis=1)
        phase_results[name] = {
            "samples": len(phase),
            "final_window_samples": len(final),
            "final_mean_position_m": final[:, 2:5].mean(axis=0).tolist() if len(final) else None,
            "final_mean_velocity_world_mps": final[:, 5:8].mean(axis=0).tolist() if len(final) else None,
            "final_mean_speed_mps": float(np.linalg.norm(final[:, 5:8], axis=1).mean()) if len(final) else None,
            "final_max_speed_mps": float(np.linalg.norm(final[:, 5:8], axis=1).max()) if len(final) else None,
            "final_mean_yaw_rad": float(np.arctan2(np.sin(final[:, 12]).mean(), np.cos(final[:, 12]).mean()))
            if len(final)
            else None,
            "maximum_speed_mps": float(speed.max()),
            "maximum_height_error_m": float(np.abs(phase[:, 4] - START[2]).max()),
            "maximum_position_error_m": float(np.linalg.norm(phase[:, 2:5] - target_position, axis=1).max()),
            "maximum_yaw_error_rad": float(np.abs(angle_error(phase[:, 12] - target_yaw)).max()),
        }

    # Renderer metadata uses precisely the odometry timestamp it consumed.
    odometry_by_stamp = {row[0]: row for row in states}
    cloud_stamps = set(clouds[:, 0])
    position_errors, rotation_errors = [], []
    matching_clouds = 0
    for row in lidar_poses:
        if row[0] in cloud_stamps:
            matching_clouds += 1
        observed = odometry_by_stamp.get(row[0])
        if observed is not None:
            position_errors.append(float(np.linalg.norm(row[2:5] - observed[2:5])))
            rotation_errors.append(
                float((Rotation.from_quat(row[5:9]).inv() * Rotation.from_quat(observed[8:12])).magnitude())
            )
    moving = states[(states[:, 1] >= 2) & (states[:, 1] < 8)]
    published_target_distance = (
        cKDTree(commands[:, 2:5]).query(moving[:, 2:5])[0] if len(commands) and len(moving) else np.array([])
    )
    max_target_distance = float(published_target_distance.max()) if len(published_target_distance) else None

    def final(name, field):
        return phase_results[name].get(field)

    have_phases = all(phase_results[name].get("final_window_samples", 0) >= 10 for name in PHASES)
    limits = {
        "hover_position_error_m": 0.15,
        "hover_speed_mps": 0.15,
        "height_error_m": 0.25,
        "accelerated_forward_speed_mps_min": 0.5,
        "turn_angle_deg_min": 40.0,
        "turn_final_yaw_error_deg": 15.0,
        "brake_final_speed_mps": 0.25,
        "settled_speed_mps": 0.12,
        "settled_position_error_m": 0.2,
        "lidar_position_error_m": 1e-5,
        "lidar_orientation_error_rad": 1e-5,
        "minimum_distinct_state_error_m": 1e-4,
    }
    checks = {"completed": completed, "all_phases_measured": have_phases}
    if have_phases:
        hover_position = np.array(final("hover", "final_mean_position_m"))
        final_position = np.array(final("settle", "final_mean_position_m"))
        turn_yaw = final("turn", "final_mean_yaw_rad")
        checks.update(
            {
                "hover_stable": bool(
                    np.linalg.norm(hover_position - START) < limits["hover_position_error_m"]
                    and final("hover", "final_max_speed_mps") < limits["hover_speed_mps"]
                ),
                "accelerates": bool(
                    final("accelerate", "final_mean_velocity_world_mps")[1]
                    > limits["accelerated_forward_speed_mps_min"]
                ),
                "turns": bool(
                    abs(angle_error(turn_yaw - final("hover", "final_mean_yaw_rad")))
                    > np.deg2rad(limits["turn_angle_deg_min"])
                    and abs(angle_error(turn_yaw - INITIAL_YAW - TURN)) < np.deg2rad(limits["turn_final_yaw_error_deg"])
                ),
                "brakes": bool(
                    final("brake", "final_mean_speed_mps") < limits["brake_final_speed_mps"]
                    and final("settle", "final_max_speed_mps") < limits["settled_speed_mps"]
                ),
                "reaches_stop_position": bool(
                    np.linalg.norm(final_position - reference(DURATION)[0]) < limits["settled_position_error_m"]
                ),
                "holds_altitude": all(
                    phase_results[name]["maximum_height_error_m"] < limits["height_error_m"] for name in PHASES
                ),
            }
        )
    checks.update(
        {
            "integrated_state_differs_from_every_published_position": max_target_distance is not None
            and max_target_distance > limits["minimum_distinct_state_error_m"],
            "actual_acceleration_received": len(accelerations) >= 20,
            "generated_ctbr_received": len(ctbr) >= 20,
            "lidar_uses_actual_odometry_position": len(position_errors) >= 10
            and max(position_errors) < limits["lidar_position_error_m"],
            "lidar_uses_actual_odometry_orientation": len(rotation_errors) >= 10
            and max(rotation_errors) < limits["lidar_orientation_error_rad"],
            "lidar_matches_cloud_timestamps": matching_clouds >= 10 and matching_clouds >= 0.8 * len(lidar_poses),
            "lidar_moves_with_vehicle": bool(len(lidar_poses) and np.ptp(lidar_poses[:, 3]) > 2.0),
            "point_cloud_changes": len(set(digests)) >= 3,
        }
    )
    return {
        "passed": all(checks.values()),
        "reason": reason,
        "checks": checks,
        "thresholds": limits,
        "reference": {
            "start_m": START.tolist(),
            "end_m": reference(DURATION)[0].tolist(),
            "peak_speed_mps": SPEED,
            "yaw_change_deg": 60,
            "phase_duration_s": PHASE_DURATION,
        },
        "phases": phase_results,
        "ctbr_timing": timing_summary(ctbr[:, 0]),
        "odometry_timing": timing_summary(states[:, 0]),
        "position_command_timing": timing_summary(commands[:, 0]),
        "maximum_distance_from_any_published_position_m": max_target_distance,
        "lidar": {
            "pose_messages": len(lidar_poses),
            "cloud_messages": len(clouds),
            "cloud_pose_stamp_matches": matching_clouds,
            "odom_pose_stamp_matches": len(position_errors),
            "maximum_position_error_m": max(position_errors) if position_errors else None,
            "maximum_orientation_error_rad": max(rotation_errors) if rotation_errors else None,
            "distinct_cloud_payloads": len(set(digests)),
        },
        "array_columns": {
            "states": [
                "stamp_s",
                "scenario_s",
                "x",
                "y",
                "z",
                "vx_world",
                "vy_world",
                "vz_world",
                "qx",
                "qy",
                "qz",
                "qw",
                "yaw",
            ],
            "commands": [
                "stamp_s",
                "scenario_s",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "ax",
                "ay",
                "az",
                "jx",
                "jy",
                "jz",
                "yaw",
                "yaw_dot",
            ],
            "acceleration": ["stamp_s", "scenario_s", "ax_world", "ay_world", "az_world"],
            "ctbr": ["stamp_s", "scenario_s", "thrust_mps2", "wx_body", "wy_body", "wz_body"],
            "lidar_pose": ["stamp_s", "scenario_s", "x", "y", "z", "qx", "qy", "qz", "qw"],
            "clouds": ["stamp_s", "scenario_s", "point_count"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= DURATION:
        raise ValueError("--timeout must exceed the 10 s motion sequence")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("metrics.json", "trajectory.npz")):
        raise FileExistsError(f"Results already exist in {args.output}")

    import rclpy
    from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
    from mars_quadrotor_msgs.msg import PositionCommand
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2

    rclpy.init(args=[])
    node = rclpy.create_node("check_marsim_dynamics")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    states, commands, accelerations, ctbr, lidar_poses, clouds, digests = [], [], [], [], [], [], []
    sequence_start = None

    def elapsed():
        return time.monotonic() - sequence_start if sequence_start is not None else -1.0

    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def xyz(value):
        return [value.x, value.y, value.z]

    def odometry(message):
        q = message.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        rotation = Rotation.from_quat(quaternion)
        velocity = rotation.apply(xyz(message.twist.twist.linear))
        states.append(
            [
                stamp(message),
                elapsed(),
                *xyz(message.pose.pose.position),
                *velocity,
                *quaternion,
                rotation.as_euler("xyz")[2],
            ]
        )

    def acceleration(message):
        accelerations.append([stamp(message), elapsed(), *xyz(message.accel.linear)])

    def command(message):
        ctbr.append([stamp(message), elapsed(), message.twist.linear.z, *xyz(message.twist.angular)])

    def render_pose(message):
        q = message.pose.orientation
        lidar_poses.append([stamp(message), elapsed(), *xyz(message.pose.position), q.x, q.y, q.z, q.w])

    def cloud(message):
        clouds.append([stamp(message), elapsed(), message.width * message.height])
        digests.append(hashlib.blake2b(memoryview(message.data), digest_size=16).hexdigest())

    node.create_subscription(Odometry, "/marsim/odometry", odometry, qos)
    node.create_subscription(AccelStamped, "/marsim/acceleration", acceleration, qos)
    node.create_subscription(TwistStamped, "/marsim/ctbr", command, qos)
    node.create_subscription(PoseStamped, "/marsim/lidar_pose", render_pose, qos)
    node.create_subscription(PointCloud2, "/cloud_registered", cloud, qos)
    publisher = node.create_publisher(PositionCommand, "/planning/pos_cmd", qos)

    def publish():
        nonlocal sequence_start
        if sequence_start is None:
            if not (states and lidar_poses and clouds):
                return
            sequence_start = time.monotonic()
            node.get_logger().info(
                "Starting 10 s hover/accelerate/turn/brake/settle sequence; total reference travel 3 m"
            )
        t = elapsed()
        p, v, a, j, yaw, yaw_rate = reference(t)
        message = PositionCommand()
        message.header.frame_id = "world"
        message.header.stamp = node.get_clock().now().to_msg()
        for field, values in (
            (message.position, p),
            (message.velocity, v),
            (message.acceleration, a),
            (message.jerk, j),
        ):
            field.x, field.y, field.z = (float(value) for value in values)
        message.yaw, message.yaw_dot = float(yaw), float(yaw_rate)
        message.trajectory_flag = 1
        publisher.publish(message)
        commands.append([stamp(message), t, *p, *v, *a, *j, yaw, yaw_rate])

    node.create_timer(0.01, publish)
    deadline = time.monotonic() + args.timeout
    completed, reason = False, "timeout"
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if elapsed() >= DURATION:
                completed, reason = True, "sequence_completed"
                break
        if sequence_start is None:
            reason = "startup_timeout_waiting_for_odometry_and_lidar"
    except (KeyboardInterrupt, ExternalShutdownException):
        reason = "interrupted"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        arrays = {
            "states": np.asarray(states, dtype=float).reshape(-1, 13),
            "commands": np.asarray(commands, dtype=float).reshape(-1, 16),
            "acceleration": np.asarray(accelerations, dtype=float).reshape(-1, 5),
            "ctbr": np.asarray(ctbr, dtype=float).reshape(-1, 6),
            "lidar_pose": np.asarray(lidar_poses, dtype=float).reshape(-1, 9),
            "clouds": np.asarray(clouds, dtype=float).reshape(-1, 3),
            "cloud_digest": np.asarray(digests, dtype="U32"),
        }
        result = analyze(
            arrays["states"],
            arrays["commands"],
            arrays["acceleration"],
            arrays["ctbr"],
            arrays["lidar_pose"],
            arrays["clouds"],
            digests,
            completed=completed,
            reason=reason,
        )
        with (args.output / "trajectory.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with (args.output / "metrics.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
