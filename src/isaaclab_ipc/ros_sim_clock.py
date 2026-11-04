"""Lockstep 100 Hz simulation-time driver for the ROS 2 MARSIM dynamic executor.

An iteration advances simulated time by exactly 10 ms, waits for one policy
command and one Flightmare integration, and waits for each scheduled LiDAR
render. Real computation may take longer; it never silently skips physics.
"""

import json
import time


def expected_scan_step(step, stride=10):
    return step % stride == 0


def packed_step(step, commanded):
    return (int(step) << 1) | int(bool(commanded))


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import UInt64

    class ClockDriver(Node):
        def __init__(self):
            super().__init__("marsim_sim_clock")
            self.declare_parameter("frequency", 100.0)
            self.declare_parameter("scan_frequency", 10.0)
            self.declare_parameter("step_wall_timeout", 60.0)
            if self.get_parameter("frequency").value != 100.0:
                raise ValueError("Clock contract requires 100 Hz")
            scan_frequency = self.get_parameter("scan_frequency").value
            stride = 100 / scan_frequency
            if int(stride) != stride:
                raise ValueError("LiDAR frequency must divide 100 Hz")
            self.stride = int(stride)
            self.wall_timeout = float(self.get_parameter("step_wall_timeout").value)
            self.clock_pub = self.create_publisher(
                Clock, "/clock", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.tick_pub = self.create_publisher(UInt64, "/marsim/sim_tick", 10)
            self.step_pub = self.create_publisher(UInt64, "/marsim/sim_step", 10)
            self.control_done = self.step_done = self.scan_done = -1
            self.commanded = False
            self.create_subscription(UInt64, "/marsim/sim_policy_done", self.on_control, 10)
            self.create_subscription(UInt64, "/marsim/sim_step_done", self.on_step, 10)
            self.create_subscription(UInt64, "/marsim/sim_scan_done", self.on_scan, 10)

        def on_control(self, msg):
            self.control_done, self.commanded = msg.data >> 1, bool(msg.data & 1)

        def on_step(self, msg):
            self.step_done = msg.data

        def on_scan(self, msg):
            self.scan_done = msg.data

        def await_event(self, name, step):
            deadline = time.monotonic() + self.wall_timeout
            while rclpy.ok() and getattr(self, name) < step:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Lockstep {name} timed out at simulated step {step}")
                rclpy.spin_once(self, timeout_sec=0.025)
            if not rclpy.ok():
                raise RuntimeError("Lockstep interrupted")
            if getattr(self, name) != step:
                raise RuntimeError(f"Out-of-order {name} at simulated step {step}")

        def publish_clock(self, step):
            msg = Clock()
            nanoseconds = step * 10_000_000
            msg.clock.sec, msg.clock.nanosec = divmod(nanoseconds, 1_000_000_000)
            self.clock_pub.publish(msg)

        def run(self):
            while rclpy.ok() and (
                self.tick_pub.get_subscription_count() < 1
                or self.step_pub.get_subscription_count() < 1
                or self.clock_pub.get_subscription_count() < 2
            ):
                rclpy.spin_once(self, timeout_sec=0.05)

            self.publish_clock(0)
            self.step_pub.publish(UInt64(data=packed_step(0, False)))
            self.await_event("step_done", 0)
            self.await_event("scan_done", 0)
            started = time.monotonic()
            step = 0
            while rclpy.ok():
                step += 1
                self.publish_clock(step)
                # The /clock source is processed asynchronously by ROS nodes.
                # Wait for clock propagation before delivering this step.
                time.sleep(0.002)
                self.tick_pub.publish(UInt64(data=step))
                self.await_event("control_done", step)
                self.step_pub.publish(UInt64(data=packed_step(step, self.commanded)))
                self.await_event("step_done", step)
                if expected_scan_step(step, self.stride):
                    self.await_event("scan_done", step)
                if step % 500 == 0:
                    wall = time.monotonic() - started
                    self.get_logger().info(json.dumps({
                        "simulation_time_s": step / 100, "wall_time_s": wall,
                        "real_time_factor": (step / 100) / wall, "step_count": step,
                    }))

    rclpy.init()
    node = None
    try:
        node = ClockDriver()
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(f"Lockstep simulation failed: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
