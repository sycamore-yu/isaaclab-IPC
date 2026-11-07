"""Native AC-MPC / PathCover adapter for the existing MARSIM launch.

The prediction and action are CTBR throughout. Ten simulation ticks hold one
MPC action; there is no jerk conversion or second execution-trajectory solve.
"""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from .assets.quadrotor import MASS
from .benchmark_config import load_case
from .ctbr_navigation import PathCoverNavigation, position_sfc_violation
from .rl.ctbr_actor import CTBRMPCActor
from .rl.ctbr_observation import make_ctbr_observation
from .turbompc.ctbr import CTBRConfig


def load_ctbr_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor_cfg = copy.deepcopy(checkpoint.get("infos", {}).get("runner_cfg", {}).get("actor", {}))
    class_name = actor_cfg.pop("class_name", None)
    if class_name != "isaaclab_ipc.rl.ctbr_actor:CTBRMPCActor":
        raise ValueError(f"Expected a native 28N CTBR checkpoint; received {class_name!r}")
    model = checkpoint["actor_state_dict"]["_extra_state"]["model"]
    actor_cfg["model_config"] = model
    return checkpoint, actor_cfg, CTBRConfig(**model)


def goal_observation(state, body_rates, goal, config, *, max_planes=128, warm_start_controls=None):
    """A0 uses the goal directly and has no geometric frontend constraints."""
    batch, nodes = len(state), config.cost_nodes
    pva = state.new_zeros((batch, nodes, 9))
    pva[..., :3] = goal[:, None]
    direction = goal[:, :2] - state[:, :2]
    yaw = torch.atan2(direction[:, 1], direction[:, 0])[:, None].expand(-1, nodes)
    planes = state.new_zeros((batch, nodes, max_planes, 4))
    mask = torch.zeros(planes.shape[:-1], dtype=torch.bool, device=state.device)
    return make_ctbr_observation(state, body_rates, goal, pva, yaw, planes, mask,
                                 ctbr_config=config, warm_start_controls=warm_start_controls)


def benchmark_frontend_config(case, config, reference_speed=6.0):
    from .pathcover_native import PathCoverConfig

    mapping = case["planner"]["rog_map"]
    center, size = np.asarray(mapping["fix_map_origin"]), np.asarray(mapping["map_size"])
    lower, upper = center - size / 2, center + size / 2
    lower[2] = max(0.0, mapping["virtual_ground_height"])
    upper[2] = mapping["virtual_ceil_height"]
    return PathCoverConfig(
        bounds_lower=tuple(lower), bounds_upper=tuple(upper),
        resolution=max(0.1, mapping["resolution"]),
        search_resolution=0.25 if mapping["resolution"] < 0.25 else None,
        robot_radius=case["planner"]["super_planner"]["robot_r"],
        max_velocity=min(reference_speed, case["planner"]["traj_opt"]["boundary"]["max_vel"],
                         min(config.velocity_upper)),
        mass=config.mass, thrust_min=config.mass * config.specific_thrust_limits[0],
        thrust_max=config.mass * config.specific_thrust_limits[1],
        body_rate_max=min(config.body_rate_limits[:2]), body_yaw_rate_max=config.body_rate_limits[2],
        tilt_max=0.8, use_cuda=True,
    )


class CTBRController:
    """Shared actor and frontend contracts; no ROS objects or hidden PPO state."""

    def __init__(self, benchmark, *, policy, frontend="pathcover", checkpoint=None,
                 device="cuda:0", max_planes=128, reference_speed=6.0):
        started = perf_counter()
        self.device = torch.device(device)
        self.case = load_case(benchmark)
        self.frontend_name = frontend
        self.max_planes = 0 if frontend == "none" else max_planes
        saved, actor_cfg = None, {}
        self.config = CTBRConfig(mass=MASS)
        if checkpoint:
            saved, actor_cfg, self.config = load_ctbr_checkpoint(checkpoint)
        self.navigation = None
        if frontend == "pathcover":
            self.navigation = PathCoverNavigation(1, self.config,
                benchmark_frontend_config(self.case, self.config, reference_speed),
                device=self.device, max_planes=self.max_planes)
        elif frontend != "none":
            raise ValueError("frontend must be pathcover or none")
        state = torch.zeros((1, 10), dtype=torch.float64, device=self.device)
        state[:, :3] = state.new_tensor(self.case["start"])
        state[:, 3] = 1.
        rates = state.new_zeros((1, 3))
        warm_goal = state[:, :3] + state.new_tensor([.25, .1, .1])
        sample = TensorDict(goal_observation(state, rates, warm_goal, self.config,
                                            max_planes=self.max_planes), batch_size=[1])
        actor_cfg.setdefault("model_config", asdict(self.config))
        self.actor = CTBRMPCActor(sample, {"actor": ["policy"]}, "actor", 4, **actor_cfg).to(self.device)
        if saved is not None:
            self.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        if policy == "acmpc" and saved is None:
            raise ValueError("Learned AC-MPC requires a checkpoint")
        self.actor.fixed_cost = policy == "fixed-ctbr"
        self.actor.eval()
        self.warm_start = None
        self.last_observation = None
        self.last_result = None
        # Compile before subscribing to lockstep ticks, outside mission timing.
        with torch.no_grad():
            self.actor(sample)
        torch.cuda.synchronize(self.device)
        self.startup_ms = 1000 * (perf_counter() - started)

    @torch.no_grad()
    def solve(self, state, body_rates, acceleration, goal, cloud, now):
        started = perf_counter()
        physical = torch.as_tensor(np.r_[state, body_rates, acceleration, goal],
                                   dtype=torch.float64, device=self.device)[None]
        state_t, rates_t = physical[:, :10], physical[:, 10:13]
        acceleration_t, goal_t = physical[:, 13:16], physical[:, 16:19]
        if self.navigation is None:
            context = goal_observation(state_t, rates_t, goal_t, self.config,
                max_planes=self.max_planes, warm_start_controls=self.warm_start)
        else:
            context = self.navigation.observe(state_t, rates_t, acceleration_t, goal_t, {0: cloud}, now,
                                              warm_start_controls=self.warm_start)
        frontend_ms = 1000 * (perf_counter() - started)
        observation = TensorDict(context, batch_size=[1])
        solve_started = perf_counter()
        action = self.actor(observation)
        result = self.actor.last_result
        torch.cuda.synchronize(self.device)
        solve_ms = 1000 * (perf_counter() - solve_started)
        controls = result["controls"].detach()
        self.warm_start = torch.cat((controls[:, 1:], controls[:, -1:]), 1).clone()
        self.last_observation, self.last_result = observation, result
        trajectory = result["trajectory"]
        violation = 0.0
        if self.max_planes:
            residual = (context["planes"][..., :3] * trajectory[..., None, :3]).sum(-1) + context["planes"][..., 3]
            violation = float(residual.masked_fill(~context["plane_mask"], 0).clamp_min(0).amax())
        command = action[0].cpu().numpy()
        metrics = dict(
            qp_ok=bool(result["solved"][0]), qp_failed=not bool(result["solved"][0]),
            sfc_valid=bool(result["sfc_valid"][0]), qp_iterations=int(result["iterations"]),
            primal_residual=float(result["primal"][0]), dual_residual=float(result["dual"][0]),
            sqp_status=int(result["sqp_status"][0]), raw_qp_status=int(result["raw_qp_status"][0]),
            predicted_sfc_violation_m=violation,
            frontend_ms=frontend_ms if self.navigation is not None else 0.0,
            solve_ms=solve_ms,
            planning_ms=1000 * (perf_counter() - started),
        )
        if self.navigation is not None:
            metrics.update(self.navigation.last_reference_metrics[0])
        else:
            metrics["observation_ms"] = frontend_ms
        return command, metrics

    def close(self):
        self.actor.close()
        if self.navigation is not None:
            self.navigation.close()


def main(args=None):
    import rclpy
    from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from std_msgs.msg import String, UInt64

    class PolicyNode(Node):
        def __init__(self):
            super().__init__("native_acmpc_policy")
            defaults = dict(policy="fixed-ctbr", execution="dynamic", benchmark="high_speed",
                            checkpoint="", frontend="pathcover", device="cuda:0",
                            output_directory="results/marsim-ctbr", simulated_time=True,
                            frequency=100.0, replan_frequency=10.0, reference_speed=6.0)
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.options = p = {key: self.get_parameter(key).value for key in defaults}
            if p["execution"] != "dynamic":
                raise ValueError("The CTBR route uses MARSIM dynamic / Flightmare execution")
            if p["frequency"] != 100.0 or p["replan_frequency"] != 10.0:
                raise ValueError("The CTBR contract is 100 Hz actuation and 10 Hz replanning")
            torch.set_num_threads(1)
            self.controller = CTBRController(p["benchmark"], policy=p["policy"], frontend=p["frontend"],
                checkpoint=p["checkpoint"] or None, device=p["device"], reference_speed=p["reference_speed"])
            directory = Path(p["output_directory"])
            directory.mkdir(parents=True, exist_ok=True)
            self.directory = directory
            self.record = (directory / "policy.jsonl").open("w", buffering=1)
            checkpoint_sha = None
            if p["checkpoint"]:
                with Path(p["checkpoint"]).open("rb") as source:
                    checkpoint_sha = hashlib.file_digest(source, "sha256").hexdigest()
            metadata = dict(parameters=p, solver=self.controller.actor.solver_metadata(),
                frontend_config=asdict(self.controller.navigation.frontends[0].config)
                    if self.controller.navigation is not None else None,
                checkpoint_sha256=checkpoint_sha, startup_ms=self.controller.startup_ms,
                sfc_face_capacity=self.controller.max_planes,
                pipeline="native-pathcover-ctbr" if p["frontend"] == "pathcover" else "acmpc-ctbr-a0",
                command_step_s=.01, mpc_prediction_step_s=self.controller.config.dt,
                scheduling="100 Hz actuation; replan every 10 simulation ticks" if p["simulated_time"]
                    else "100 Hz ROS timer; synchronous 10 Hz MPC, measured wall cadence",
                state_order="world position, quaternion WXYZ, world velocity",
                control_units="collective specific thrust m/s^2, body angular rates rad/s",
                execution_acceptance="direct first MPC control; no duplicate trajectory integration")
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.state = self.body_rates = self.goal = self.cloud = self.action = None
            self.acceleration = np.zeros(3)
            self.have_cloud = False
            self.odom_stamp_ns = self.solve_stamp_ns = None
            self.sequence = self.controls = self.qp_failures = 0
            self.force_solve = True
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
            goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(Odometry, "/marsim/odometry", self.on_odom, sensor_qos)
            self.create_subscription(AccelStamped, "/marsim/acceleration", self.on_acceleration, sensor_qos)
            self.create_subscription(PointCloud2, "/cloud_registered", self.on_cloud, sensor_qos)
            self.create_subscription(PoseStamped, "/planning/click_goal", self.on_goal, goal_qos)
            self.command_pub = self.create_publisher(TwistStamped, "/marsim/ctbr", 10)
            self.stats_pub = self.create_publisher(String, "/marsim/policy_stats", 10)
            self.timer = None
            if p["simulated_time"]:
                self.done_pub = self.create_publisher(UInt64, "/marsim/sim_policy_done", 10)
                self.create_subscription(UInt64, "/marsim/sim_tick", self.on_sim_tick, 10)
            else:
                self.timer = self.create_timer(.01, self.control)
            self.get_logger().info(f"Ready: {p['policy']} / {p['frontend']}; N={self.controller.config.cost_nodes}")

        def on_odom(self, message):
            p, q = message.pose.pose.position, message.pose.pose.orientation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            v, w = message.twist.twist.linear, message.twist.twist.angular
            velocity = np.array([v.x, v.y, v.z])
            if message.child_frame_id not in ("", "world", "map", "odom"):
                velocity = rotation.apply(velocity)
            self.state = np.r_[[p.x, p.y, p.z], [q.w, q.x, q.y, q.z], velocity]
            self.body_rates = np.array([w.x, w.y, w.z])
            self.odom_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

        def on_acceleration(self, message):
            value = message.accel.linear
            self.acceleration = np.array([value.x, value.y, value.z])

        def on_cloud(self, message):
            if message.header.frame_id not in ("world", "map"):
                raise ValueError("Native PathCover requires a world-frame point cloud")
            self.cloud = read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
            self.have_cloud = True

        def on_goal(self, message):
            goal = message.pose.position
            self.goal = np.array([goal.x, goal.y, goal.z])
            self.force_solve = True
            self.controller.warm_start = None
            self.get_logger().info(f"Mission goal {self.goal.tolist()}")

        def on_sim_tick(self, message):
            prior = self.controls
            self.control(stamp_ns=message.data * 10_000_000)
            self.done_pub.publish(UInt64(data=(message.data << 1) | int(self.controls > prior)))

        def control(self, stamp_ns=None):
            if self.state is None or self.goal is None or not self.have_cloud:
                return
            started = perf_counter()
            stamp = Time(nanoseconds=stamp_ns) if stamp_ns is not None else self.get_clock().now()
            now = stamp.nanoseconds * 1e-9
            # The synchronous tick/command stamp names the end of the next
            # physics interval. The measured state and plan start at its beginning.
            state_time = self.odom_stamp_ns * 1e-9
            actual_violation = None
            if self.controller.navigation is not None:
                previous_plan = self.controller.navigation.last_plans[0]
                if previous_plan is not None:
                    # Compare execution with the plan that produced it, before replanning.
                    actual_violation = position_sfc_violation(previous_plan,
                        self.state[:3], state_time)
            if self.force_solve or self.solve_stamp_ns is None or stamp.nanoseconds - self.solve_stamp_ns >= 100_000_000:
                if self.sequence == 0:
                    np.savez_compressed(self.directory / "frontend-input.npz", cloud=self.cloud,
                        state=self.state, body_rates=self.body_rates, acceleration=self.acceleration,
                        goal=self.goal, now=state_time, command_stamp=now)
                attempt_started = perf_counter()
                try:
                    self.action, stats = self.controller.solve(self.state, self.body_rates, self.acceleration,
                                                               self.goal, self.cloud, state_time)
                except Exception as error:
                    # Record the failed attempt before serializing its diagnostic input.
                    failed_planning_ms = 1000 * (perf_counter() - attempt_started)
                    failure = dict(state=self.state, body_rates=self.body_rates,
                        acceleration=self.acceleration, goal=self.goal, cloud=self.cloud,
                        now=state_time, command_stamp=now,
                        warm_start_controls=None if self.controller.warm_start is None
                            else self.controller.warm_start.detach().cpu())
                    if self.controller.navigation is not None:
                        frontend = self.controller.navigation.frontends[0]
                        failure.update(frontend_config=asdict(frontend.config),
                            inflated_cloud=frontend.inflated_cloud(),
                            gcopter_problem=frontend.failed_problem(),
                            previous_plan=self.controller.navigation.last_plans[0])
                    torch.save(failure, self.directory / "failed-frontend-input.pt")
                    self.publish_stats(dict(event="failure", stamp=now,
                        planning_ms=failed_planning_ms,
                        actual_sfc_violation_m=actual_violation, error=f"{type(error).__name__}: {error}"))
                    raise
                self.cloud = None
                self.force_solve = False
                self.solve_stamp_ns = stamp.nanoseconds
                self.sequence += 1
                self.qp_failures += int(stats["qp_failed"])
                self.publish_stats(dict(event="solve", stamp=now, state_time=state_time,
                                        plan_sequence_id=self.sequence,
                                        qp_failures=self.qp_failures, **stats))
                if stats["qp_failed"] and not (self.directory / "first-unsolved.pt").exists():
                    torch.save(dict(observation=self.controller.last_observation.cpu(),
                                    solver_config=self.controller.actor.solver_metadata(),
                                    frontend_plan=self.controller.navigation.last_plans[0]
                                    if self.controller.navigation is not None else None,
                                    result={key: value.cpu() if isinstance(value, torch.Tensor) else value
                                            for key, value in self.controller.last_result.items()}),
                               self.directory / "first-unsolved.pt")
            command = TwistStamped()
            command.header.stamp = stamp.to_msg()
            command.header.frame_id = "body"
            command.twist.linear.z = float(self.action[0])
            command.twist.angular.x, command.twist.angular.y, command.twist.angular.z = map(float, self.action[1:])
            self.command_pub.publish(command)
            self.controls += 1
            self.publish_stats(dict(event="control", stamp=now, control_index=self.controls,
                plan_sequence_id=self.sequence, action=self.action.tolist(), braking=False,
                actual_sfc_violation_m=actual_violation,
                state_age=(stamp.nanoseconds - self.odom_stamp_ns) * 1e-9,
                solution_age=(stamp.nanoseconds - self.solve_stamp_ns) * 1e-9,
                control_ms=1000 * (perf_counter() - started)))

        def publish_stats(self, row):
            row.update(policy=self.options["policy"], execution="dynamic", wall_stamp_s=perf_counter())
            encoded = json.dumps(row, allow_nan=False)
            self.record.write(encoded + "\n")
            self.stats_pub.publish(String(data=encoded))

        def close(self):
            if self.timer is not None:
                self.timer.cancel()
            self.controller.close()
            self.record.close()

    rclpy.init(args=args)
    node = None
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
