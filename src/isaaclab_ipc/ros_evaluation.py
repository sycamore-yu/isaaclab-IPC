"""Independent ROS 2 evaluation of the complete official SUPER waypoint mission.

No Isaac Lab, Warp or training runtime is imported here. The collision geometry is
the full original PCD, with the same spherical robot and ground clearance used by
the training evaluator. Segment checks also cover motion between odometry samples.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from isaaclab_ipc.scenes import read_pcd

CONFIG = Path(__file__).parent / "config" / "super_benchmarks.yaml"


def timing_summary(stamps):
    """Observed stream frequency from all received messages, not timer settings."""
    stamps = np.asarray(stamps, dtype=float)
    intervals = np.diff(stamps)
    duration = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    return {
        "count": len(stamps),
        "duration_s": duration,
        "rate_hz": (len(stamps) - 1) / duration if duration > 0 else None,
        "median_interval_ms": float(np.median(intervals) * 1000) if len(intervals) else None,
        "p95_interval_ms": float(np.percentile(intervals, 95) * 1000) if len(intervals) else None,
        "maximum_interval_ms": float(intervals.max() * 1000) if len(intervals) else None,
    }


def duration_summary(values):
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "over_10ms": int(np.count_nonzero(np.asarray(values) > 10)),
    }


class MissionRecorder:
    """CPU recorder and ordered waypoint/collision evaluator."""

    def __init__(self, case, points, *, radius=None):
        self.case = case
        self.radius = float(radius if radius is not None else case["planner"]["super_planner"]["robot_r"])
        self.points = np.asarray(points, dtype=np.float64)
        self.tree = cKDTree(self.points)
        self.waypoints = np.asarray(case["waypoints"], dtype=float)
        self.tolerances = np.asarray(case["waypoint_switch_distances"], dtype=float)
        self.started = time.monotonic()
        self.rows = []
        self.clouds = []
        self.lidar_poses = []
        self.accelerations = []
        self.commands = []
        self.position_commands = []
        self.stats = []
        self.waypoint_events = []
        self.waypoint_index = 0
        self.status = None
        self.reason = None

    def elapsed(self):
        return time.monotonic() - self.started

    def clearance(self, position, previous=None):
        """Exact nearest PCD point to the segment, minus spherical robot radius."""
        distance = float(self.tree.query(position)[0])
        if previous is not None:
            delta = position - previous
            length = float(np.linalg.norm(delta))
            distance = min(distance, float(self.tree.query(previous)[0]))
            if length > 0:
                midpoint = (position + previous) * 0.5
                indices = self.tree.query_ball_point(midpoint, length * 0.5 + distance + 1e-9)
                if indices:
                    offsets = self.points[indices] - previous
                    fraction = np.clip(offsets @ delta / (length * length), 0, 1)
                    distance = min(distance, float(np.linalg.norm(offsets - fraction[:, None] * delta, axis=1).min()))
        point_clearance = distance - self.radius
        ground_clearance = min(position[2], previous[2] if previous is not None else position[2]) - self.radius
        return point_clearance, float(min(point_clearance, ground_clearance))

    def observe(self, stamp, position, velocity, quaternion_xyzw):
        if self.status is not None:
            return
        position, velocity, quaternion_xyzw = (
            np.asarray(value, dtype=float) for value in (position, velocity, quaternion_xyzw)
        )
        if not all(np.isfinite(value).all() for value in (position, velocity, quaternion_xyzw)):
            self.finish("timeout", "nonfinite_odometry")
            return
        previous = self.rows[-1][2:5] if self.rows else None
        point_clearance, clearance = self.clearance(position, previous)
        self.rows.append(np.r_[stamp, self.elapsed(), position, velocity, quaternion_xyzw, point_clearance, clearance])
        if clearance <= 0:
            self.finish("collision", "full_pcd_or_ground")
            return
        if np.linalg.norm(position - self.waypoints[self.waypoint_index]) < self.tolerances[self.waypoint_index]:
            self.waypoint_events.append(
                {
                    "index": self.waypoint_index,
                    "stamp_s": stamp,
                    "elapsed_s": self.elapsed(),
                    "position_m": position.tolist(),
                }
            )
            self.waypoint_index += 1
            if self.waypoint_index == len(self.waypoints):
                self.finish("success", "all_official_waypoints_reached")

    def finish(self, status, reason):
        if self.status is None:
            self.status = status
            self.reason = reason

    def save(self, output, *, mode, benchmark, policy, pcd, topics, timeout, time_mode="wall"):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        targets = [output / name for name in ("metrics.json", "trajectory.npz", "policy_stats.jsonl")]
        if any(path.exists() for path in targets):
            raise FileExistsError(f"Evaluation results already exist in {output}")
        rows = np.asarray(self.rows, dtype=float).reshape(-1, 14)
        stamps = rows[:, 0]
        speed = np.linalg.norm(rows[:, 5:8], axis=1)
        trajectory_length = float(np.linalg.norm(np.diff(rows[:, 2:5], axis=0), axis=1).sum())
        odometry_timing = timing_summary(stamps)
        durations = []
        solve_durations = []
        frontend_durations = []
        observation_durations = []
        planning_durations = []
        acceptance_durations = []
        control_stamps = []
        solve_stamps = []
        control_wall_stamps = []
        solve_wall_stamps = []
        state_ages = []
        solution_ages = []
        qp_failures = []
        cumulative_qp_failures = []
        reanchor_rejections = []
        cumulative_reanchor_rejections = []
        predicted_sfc_violations = []
        reference_sfc_violations = []
        actual_sfc_violations = []
        computational_failures = []
        for entry in self.stats:
            data = entry["data"]
            event = data.get("event")
            if event == "failure":
                computational_failures.append(data)
                if data.get("planning_ms") is not None:
                    planning_durations.append(float(data["planning_ms"]))
            if event in (None, "control", "failure"):
                if data.get("actual_sfc_violation_m") is not None:
                    actual_sfc_violations.append(float(data["actual_sfc_violation_m"]))
            if event in (None, "control"):
                value = data.get("control_ms", data.get("command_ms"))
                if value is not None:
                    durations.append(float(value))
                if "stamp" in data:
                    control_stamps.append(float(data["stamp"]))
                if "wall_stamp_s" in data:
                    control_wall_stamps.append(float(data["wall_stamp_s"]))
                for name, values in (("state_age", state_ages), ("solution_age", solution_ages)):
                    if data.get(name) is not None:
                        values.append(1000 * float(data[name]))
            if event in (None, "solve"):
                if "predicted_sfc_violation_m" in data:
                    predicted_sfc_violations.append(float(data["predicted_sfc_violation_m"]))
                if "reference_sfc_violation_m" in data:
                    reference_sfc_violations.append(float(data["reference_sfc_violation_m"]))
                if "solve_ms" in data:
                    solve_durations.append(float(data["solve_ms"]))
                    if "stamp" in data:
                        solve_stamps.append(float(data["stamp"]))
                    if "wall_stamp_s" in data:
                        solve_wall_stamps.append(float(data["wall_stamp_s"]))
                for name, values in (("frontend_ms", frontend_durations), ("observation_ms", observation_durations),
                                     ("planning_ms", planning_durations), ("acceptance_ms", acceptance_durations)):
                    if name in data:
                        values.append(float(data[name]))
                if "qp_failed" in data:
                    qp_failures.append(bool(data["qp_failed"]))
                if "reanchor_rejected" in data:
                    reanchor_rejections.append(bool(data["reanchor_rejected"]))
            if "qp_failures" in data:
                cumulative_qp_failures.append(int(data["qp_failures"]))
            if "reanchor_rejections" in data:
                cumulative_reanchor_rejections.append(int(data["reanchor_rejections"]))
        commands = np.asarray(self.commands, dtype=float).reshape(-1, 6)
        position_commands = np.asarray(self.position_commands, dtype=float).reshape(-1, 16)
        clouds = np.asarray(self.clouds, dtype=float).reshape(-1, 4)
        lidar_poses = np.asarray(self.lidar_poses, dtype=float).reshape(-1, 9)
        metrics = {
            "status": self.status,
            "reason": self.reason,
            "mode": mode,
            "time_mode": time_mode,
            "benchmark": benchmark,
            "policy": policy,
            "timeout_s": timeout,
            "elapsed_wall_s": self.elapsed(),
            "real_time_factor": (
                (stamps[-1] - stamps[0]) / self.elapsed() if len(rows) > 1 and self.elapsed() > 0 else 0.0),
            "trajectory_duration_s": float(stamps[-1] - stamps[0]) if len(rows) else 0.0,
            "trajectory_length_m": trajectory_length,
            "odometry_samples": len(rows),
            "odometry_rate_hz": odometry_timing["rate_hz"],
            "odometry_timing": odometry_timing,
            "maximum_odometry_gap_s": odometry_timing["maximum_interval_ms"] / 1000
            if odometry_timing["maximum_interval_ms"] is not None
            else None,
            "maximum_speed_mps": float(speed.max()) if len(speed) else None,
            "mean_speed_mps": float(speed.mean()) if len(speed) else None,
            "minimum_point_clearance_m": float(rows[:, 12].min()) if len(rows) else None,
            "minimum_clearance_m": float(rows[:, 13].min()) if len(rows) else None,
            "collision_geometry": "full_original_pcd_points_and_ground_swept_sphere",
            "robot_radius_m": self.radius,
            "pcd": str(pcd),
            "pcd_points": len(self.points),
            "waypoints": self.waypoints.tolist(),
            "waypoint_switch_distances_m": self.tolerances.tolist(),
            "waypoints_reached": self.waypoint_index,
            "waypoint_events": self.waypoint_events,
            "lidar_messages": len(self.clouds),
            "lidar_pose_messages": len(self.lidar_poses),
            "lidar_pose_cloud_stamp_matches": int(np.intersect1d(clouds[:, 0], lidar_poses[:, 0]).size),
            "lidar_pose_position_span_m": np.ptp(lidar_poses[:, 2:5], axis=0).tolist() if len(lidar_poses) else None,
            "acceleration_messages": len(self.accelerations),
            "ctbr_messages": len(self.commands),
            "position_command_messages": len(self.position_commands),
            "policy_stats_messages": len(self.stats),
            "qp_failures": max(cumulative_qp_failures)
            if cumulative_qp_failures
            else (sum(qp_failures) if qp_failures else None),
            "reanchor_rejections": max(cumulative_reanchor_rejections)
            if cumulative_reanchor_rejections
            else (sum(reanchor_rejections) if reanchor_rejections else None),
            "computation_time_ms": duration_summary(durations),
            "qp_solve_time_ms": duration_summary(solve_durations),
            "frontend_time_ms": duration_summary(frontend_durations),
            "observation_time_ms": duration_summary(observation_durations),
            "planning_time_ms": duration_summary(planning_durations),
            "planning_time_samples": len(planning_durations),
            "planning_time_scope": "completed and failed planning attempts; excludes diagnostic serialization",
            "planning_p95_under_100ms": bool(np.percentile(planning_durations, 95) < 100)
            if planning_durations else None,
            "maximum_predicted_sfc_violation_m": max(predicted_sfc_violations, default=None),
            "maximum_reference_sfc_violation_m": max(reference_sfc_violations, default=None),
            "maximum_actual_sfc_violation_m": max(actual_sfc_violations, default=None),
            "computational_failures": computational_failures,
            "plan_acceptance_time_ms": duration_summary(acceptance_durations),
            "policy_control_timing": timing_summary(control_stamps),
            "policy_solve_timing": timing_summary(solve_stamps),
            "policy_control_wall_timing": timing_summary(control_wall_stamps),
            "policy_solve_wall_timing": timing_summary(solve_wall_stamps),
            "policy_state_age_ms": duration_summary(state_ages),
            "policy_solution_age_ms": duration_summary(solution_ages),
            "ctbr_timing": timing_summary(commands[:, 0]),
            "ctbr_reception_timing": timing_summary(commands[:, 1]),
            "position_command_timing": timing_summary(position_commands[:, 0]),
            "array_columns": {
                "acceleration": ["stamp_s", "elapsed_s", "ax_world_mps2", "ay_world_mps2", "az_world_mps2"],
                "ctbr": [
                    "stamp_s",
                    "elapsed_s",
                    "collective_thrust_mps2",
                    "wx_body_radps",
                    "wy_body_radps",
                    "wz_body_radps",
                ],
                "lidar": ["stamp_s", "elapsed_s", "point_count", "data_bytes"],
                "lidar_pose": ["stamp_s", "elapsed_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"],
                "position_commands": [
                    "stamp_s",
                    "elapsed_s",
                    "x_world_m",
                    "y_world_m",
                    "z_world_m",
                    "vx_world_mps",
                    "vy_world_mps",
                    "vz_world_mps",
                    "ax_world_mps2",
                    "ay_world_mps2",
                    "az_world_mps2",
                    "jx_world_mps3",
                    "jy_world_mps3",
                    "jz_world_mps3",
                    "yaw_rad",
                    "yaw_dot_radps",
                ],
            },
            "topics": topics,
            "mode_outcomes": {mode: {name: int(self.status == name) for name in ("success", "collision", "timeout", "failure")}},
        }
        with targets[1].open("xb") as stream:
            np.savez_compressed(
                stream,
                stamp_s=stamps,
                elapsed_s=rows[:, 1],
                position_m=rows[:, 2:5],
                velocity_world_mps=rows[:, 5:8],
                quaternion_xyzw=rows[:, 8:12],
                speed_mps=speed,
                point_clearance_m=rows[:, 12],
                clearance_m=rows[:, 13],
                acceleration=np.asarray(self.accelerations, dtype=float).reshape(-1, 5),
                ctbr=commands,
                lidar=clouds,
                lidar_pose=lidar_poses,
                position_commands=position_commands,
            )
        with targets[2].open("x") as stream:
            for entry in self.stats:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        with targets[0].open("x") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("ideal", "dynamic"), required=True)
    result.add_argument("--time-mode", choices=("wall", "sim"), default="wall")
    result.add_argument("--benchmark", choices=("high_speed", "dense"), required=True)
    result.add_argument("--policy", choices=("super", "ipc", "fixed-ctbr", "acmpc"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--super-root", type=Path, default=Path("/home/tong/tongworkspace/reference_repos/SUPER"))
    result.add_argument("--pcd", type=Path)
    result.add_argument("--radius", type=float)
    result.add_argument("--odom-topic", default="/lidar_slam/odom")
    result.add_argument("--velocity-frame", choices=("world", "body", "auto"), default="auto")
    result.add_argument("--cloud-topic", default="/cloud_registered")
    result.add_argument("--lidar-pose-topic", default="/marsim/lidar_pose")
    result.add_argument("--accel-topic", default="/marsim/acceleration")
    result.add_argument("--ctbr-topic", default="/marsim/ctbr")
    result.add_argument("--position-command-topic", default="/planning/pos_cmd")
    result.add_argument("--stats-topic", default="/marsim/policy_stats")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("metrics.json", "trajectory.npz", "policy_stats.jsonl")):
        raise FileExistsError(f"Evaluation results already exist in {args.output}")
    case = yaml.safe_load(CONFIG.read_text())["cases"][args.benchmark]
    pcd = args.pcd or args.super_root / "mars_uav_sim" / "perfect_drone_sim" / "pcd" / case["pcd"]
    recorder = MissionRecorder(case, read_pcd(pcd, drop_nan=True), radius=args.radius)

    import rclpy
    from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
    from mars_quadrotor_msgs.msg import PositionCommand
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def vector(value):
        return [value.x, value.y, value.z]

    class EvaluationNode(Node):
        def __init__(self):
            super().__init__("marsim_evaluation")
            self.create_subscription(Odometry, args.odom_topic, self.odometry, qos)
            self.create_subscription(PointCloud2, args.cloud_topic, self.cloud, qos)
            self.create_subscription(PoseStamped, args.lidar_pose_topic, self.lidar_pose, qos)
            self.create_subscription(AccelStamped, args.accel_topic, self.acceleration, qos)
            self.create_subscription(TwistStamped, args.ctbr_topic, self.command, qos)
            self.create_subscription(PositionCommand, args.position_command_topic, self.position_command, qos)
            self.create_subscription(String, args.stats_topic, self.statistics, qos)

        def odometry(self, message):
            quaternion = message.pose.pose.orientation
            quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
            velocity = vector(message.twist.twist.linear)
            body = args.velocity_frame == "body" or (
                args.velocity_frame == "auto" and message.child_frame_id not in ("", "world", "map", "odom")
            )
            if body:
                velocity = Rotation.from_quat(quaternion).apply(velocity)
            recorder.observe(stamp(message), vector(message.pose.pose.position), velocity, quaternion)

        def cloud(self, message):
            recorder.clouds.append(
                [stamp(message), recorder.elapsed(), message.width * message.height, len(message.data)]
            )

        def acceleration(self, message):
            recorder.accelerations.append([stamp(message), recorder.elapsed(), *vector(message.accel.linear)])

        def lidar_pose(self, message):
            q = message.pose.orientation
            recorder.lidar_poses.append(
                [stamp(message), recorder.elapsed(), *vector(message.pose.position), q.x, q.y, q.z, q.w]
            )

        def command(self, message):
            recorder.commands.append(
                [stamp(message), recorder.elapsed(), message.twist.linear.z, *vector(message.twist.angular)]
            )

        def statistics(self, message):
            data = json.loads(message.data)
            recorder.stats.append({"elapsed_s": recorder.elapsed(), "data": data})
            if data.get("event") == "failure":
                recorder.finish("failure", data["error"])

        def position_command(self, message):
            recorder.position_commands.append(
                [
                    stamp(message),
                    recorder.elapsed(),
                    *vector(message.position),
                    *vector(message.velocity),
                    *vector(message.acceleration),
                    *vector(message.jerk),
                    message.yaw,
                    message.yaw_dot,
                ]
            )

    rclpy.init(args=[])
    node = EvaluationNode()
    node.get_logger().info(
        f"Monitoring {args.mode}/{args.policy}/{args.benchmark}: {len(recorder.points)} PCD points, "
        f"{len(recorder.waypoints)} ordered waypoints, timeout {args.timeout}s"
    )
    try:
        while rclpy.ok() and recorder.status is None:
            rclpy.spin_once(node, timeout_sec=0.05)
            simulated_elapsed = (
                recorder.rows[-1][0] - recorder.rows[0][0] if len(recorder.rows) > 1 else 0.0)
            if (simulated_elapsed if args.time_mode == "sim" else recorder.elapsed()) >= args.timeout:
                recorder.finish("timeout", "mission_deadline" if recorder.rows else "no_odometry")
            if args.time_mode == "sim" and recorder.elapsed() > args.timeout * 20:
                recorder.finish("timeout", "wall_safety_deadline")
    except (KeyboardInterrupt, ExternalShutdownException):
        recorder.finish("timeout", "interrupted")
    finally:
        recorder.finish("timeout", "ros_shutdown")
        metrics = recorder.save(
            args.output,
            mode=args.mode,
            benchmark=args.benchmark,
            policy=args.policy,
            pcd=pcd,
            topics={
                name: getattr(args, name + "_topic")
                for name in ("odom", "cloud", "lidar_pose", "accel", "ctbr", "position_command", "stats")
            },
            timeout=args.timeout,
            time_mode=args.time_mode,
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    return {"success": 0, "collision": 2, "timeout": 3, "failure": 4}[recorder.status]
