"""Check the unmodified PerfectDrone's actual ROS PositionCommand contract."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from mars_quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("check_perfect_drone")
    states = []
    node.create_subscription(Odometry, "/lidar_slam/odom", states.append, qos_profile_sensor_data)
    publisher = node.create_publisher(PositionCommand, "/planning/pos_cmd", qos_profile_sensor_data)
    position = np.array([0.1, -49.8, 1.6])
    velocity = np.array([0.2, 0.3, -0.1])
    acceleration = np.array([0.5, -0.25, 0.1])
    yaw = 0.4
    command = PositionCommand()
    command.header.frame_id = "world"
    for field, values in ((command.position, position), (command.velocity, velocity),
                          (command.acceleration, acceleration)):
        field.x, field.y, field.z = (float(x) for x in values)
    command.yaw = yaw
    start = time.monotonic()
    received = None
    try:
        while time.monotonic() - start < args.timeout:
            command.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=0.05)
            if states:
                state = states[-1]
                p = state.pose.pose.position
                if np.linalg.norm(np.array([p.x, p.y, p.z]) - position) < 1e-8:
                    received = state
                    break
        if received is None:
            raise RuntimeError("PerfectDrone did not execute PositionCommand within the deadline")
        p, v, q = received.pose.pose.position, received.twist.twist.linear, received.pose.pose.orientation
        observed_position = np.array([p.x, p.y, p.z])
        observed_velocity = np.array([v.x, v.y, v.z])
        observed_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        body_z = acceleration + np.array([0, 0, 9.80])
        body_z /= np.linalg.norm(body_z)
        body_y = np.cross(body_z, [np.cos(yaw), np.sin(yaw), 0])
        body_y /= np.linalg.norm(body_y)
        expected_rotation = np.column_stack([np.cross(body_y, body_z), body_y, body_z])
        errors = {
            "position_error_m": float(np.linalg.norm(observed_position - position)),
            "velocity_error_m_s": float(np.linalg.norm(observed_velocity - velocity)),
            "rotation_matrix_error": float(np.linalg.norm(observed_rotation - expected_rotation)),
        }
        result = {"passed": max(errors.values()) < 1e-7, **errors,
                  "position": observed_position.tolist(), "velocity": observed_velocity.tolist(),
                  "quaternion_xyzw": [q.x, q.y, q.z, q.w], "world_acceleration_command": acceleration.tolist(),
                  "yaw_command": yaw, "elapsed_wall_s": time.monotonic() - start}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result))
        if not result["passed"]:
            raise RuntimeError("PerfectDrone PositionCommand semantics differ from the upstream implementation")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
