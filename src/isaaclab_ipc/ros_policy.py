"""ROS 2 adapter for the existing IPC planner and 124-feature jerk MPC actor.

Both MARSIM executors supply measured velocity.  MARSIM ``ideal`` therefore
does not select the training package's legacy velocity-zeroing ideal profile.
CTBR uses collective specific thrust [m/s^2] and body rates [rad/s].
"""

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from isaaclab_ipc.benchmark_config import load_case, planner_params
from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.frontend import PlanningBatch, PlanningContext
from isaaclab_ipc.mpc import Config, dynamics
from isaaclab_ipc.planner import Planner
from isaaclab_ipc.rl.actor import MPCActor, flatness_torch


def observation(physical, goal, context, braking=False, yaw_target=0.0, climb_target=None, device="cpu"):
    """Match IPCMPCEnv._get_observations exactly, without importing Isaac Lab.

    ``physical`` is world p/v/a followed by the row-major body-to-world R.
    Solver inputs retain SI units; only the 124 policy features are scaled.
    """
    physical = torch.as_tensor(physical, dtype=torch.float32, device=device).reshape(1, 18)
    reference = torch.as_tensor(context.reference, dtype=torch.float64, device=device)[None]
    corridors = context.corridors or [np.empty((0, 4))] * 15
    planes, masks, indices = pack_corridors(corridors, 64)
    margins = np.full(15, 10.0)
    for i, faces in enumerate(context.corridors):
        if len(faces):
            faces = np.asarray(faces)[::-1]
            margins[i] = np.min(-(faces[:, :3] @ context.reference[i, :3] + faces[:, 3])
                                / np.maximum(np.linalg.norm(faces[:, :3], axis=-1), 1e-12))

    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)[None]

    features = torch.cat((
        physical[:, 3:6] / 10,
        physical[:, 6:9] / 20,
        physical[:, 9:],
        (torch.as_tensor(goal, dtype=physical.dtype, device=device)[None] - physical[:, :3]) / 10,
        (reference[:, :, :3] - physical[:, None, :3]).flatten(1) / 10,
        reference[:, :, 3:6].flatten(1) / 10,
        tensor(margins).clamp(-10, 10) / 10,
        tensor([braking]),
    ), -1).float()
    return TensorDict(dict(
        policy=features, mpc_state=tensor(context.state), reference=reference,
        planes=tensor(planes), plane_mask=tensor(masks, torch.bool),
        corridor_index=tensor(indices, torch.int64), rotation=physical[:, 9:].clone(),
        braking=tensor([braking], torch.bool), yaw_target=tensor([yaw_target]),
        sfc_valid=tensor([context.valid], torch.bool),
        climb_active=tensor([climb_target is not None], torch.bool),
        climb_target=tensor(np.zeros(3) if climb_target is None else climb_target),
    ), batch_size=[1])


def load_checkpoint(path):
    """Load a local RSL-RL actor and its saved configuration, with strict weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise ValueError("Expected an RSL-RL checkpoint containing actor_state_dict")
    actor_cfg = copy.deepcopy(checkpoint.get("infos", {}).get("runner_cfg", {}).get("actor", {}))
    class_name = actor_cfg.pop("class_name", None)
    if class_name != "isaaclab_ipc.rl.actor:MPCActor":
        raise ValueError(f"This adapter supports the 124-feature jerk MPCActor; received {class_name!r}")
    if checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1] != 124:
        raise ValueError("Checkpoint feature count differs from the jerk MPC training contract")
    config_path = Path(path).parent / "config.json"
    environment = json.loads(config_path.read_text()).get("environment", {}) if config_path.exists() else {}
    return checkpoint, actor_cfg, environment


def main(args=None):
    import rclpy
    from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
    from mars_quadrotor_msgs.msg import PositionCommand
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from std_msgs.msg import String

    class PolicyNode(Node):
        def __init__(self):
            super().__init__("ipc_policy")
            defaults = dict(policy="fixed", execution="dynamic", benchmark="high_speed", checkpoint="",
                            output_directory="results/marsim-policy", device="cpu", frequency=100.0,
                            odom_topic="", cloud_topic="/cloud_registered", goal_topic="/planning/click_goal",
                            acceleration_topic="/marsim/acceleration", ctbr_topic="/marsim/ctbr",
                            position_command_topic="/planning/pos_cmd", stats_topic="/marsim/policy_stats")
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            p = {key: self.get_parameter(key).value for key in defaults}
            if p["policy"] not in ("fixed", "acmpc") or p["execution"] not in ("ideal", "dynamic"):
                raise ValueError("policy must be fixed|acmpc; execution must be ideal|dynamic")
            if p["frequency"] <= 0:
                raise ValueError("frequency must be positive")
            self.options = p
            torch.set_num_threads(1)
            self.device = torch.device(p["device"])
            case, params = load_case(p["benchmark"]), planner_params(p["benchmark"])
            bounds = case["planner"]["traj_opt"]["boundary"]
            self.thrust_limits = (bounds["min_acc_thr"], bounds["max_acc_thr"])
            self.rate_limit = bounds["max_omg"]
            checkpoint, actor_cfg, environment = None, {}, {}
            checkpoint_sha256 = None
            if p["checkpoint"]:
                checkpoint, actor_cfg, environment = load_checkpoint(p["checkpoint"])
                with Path(p["checkpoint"]).open("rb") as source:
                    checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
                # Preserve the checkpoint's prediction model, bounds, costs and reference stride.
                # The frontend's workspace comes from the evaluated MARSIM scene.
                trained = copy.deepcopy(actor_cfg.get("planner_params") or Config.load(environment.get("scenario", "avoid")).params)
                trained["frontend"] = params["frontend"]
                if environment.get("reference_strides"):
                    trained["fsm"]["ref_dis"] = environment["reference_strides"][0]
                params = trained
                self.thrust_limits = actor_cfg.get("thrust_limits", (0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7))
                self.rate_limit = actor_cfg.get("rate_limit")
            if p["policy"] == "acmpc" and checkpoint is None:
                raise ValueError("policy:=acmpc requires checkpoint:=/absolute/path/checkpoint.pt")
            self.config = Config(params, profile="real")
            self.command_a, self.command_b = dynamics(1.0 / p["frequency"], [params["mpc"]["D_" + axis] for axis in "xyz"])
            self.planner = Planner(self.config, np.asarray(case["start"], dtype=float))
            self.batch = PlanningBatch(1)
            self.actor = None
            if p["policy"] == "acmpc":
                blank = PlanningContext(np.zeros(9), np.zeros((15, 9)), [np.empty((0, 4))] * 15, False)
                sample = observation(np.r_[np.zeros(9), np.eye(3).ravel()], np.zeros(3), blank, device=self.device)
                actor_cfg["planner_params"] = params
                self.actor = MPCActor(sample, {"actor": ["policy"]}, "actor", 4, **actor_cfg).to(self.device)
                self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                self.actor.eval()
            directory = Path(p["output_directory"]).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            self.record = (directory / "policy.jsonl").open("w", buffering=1)
            metadata = dict(parameters=p, planner_params=params, feature_count=124,
                            observation_order=["v/10", "a/20", "R_row_major", "goal_minus_p/10",
                                               "reference_p_minus_p/10", "reference_v/10", "margin_clipped/10", "braking"],
                            checkpoint_task=(checkpoint or {}).get("infos", {}).get("task"),
                            checkpoint_sha256=checkpoint_sha256,
                            checkpoint_environment={key: environment.get(key) for key in ("scenario", "reference_strides")},
                            thrust_limits=list(self.thrust_limits), rate_limit=self.rate_limit,
                            planner_profile="real", rotation="ROS xyzw to body-to-world R",
                            command_step_s=1.0 / p["frequency"], mpc_prediction_step_s=params["mpc"]["step"],
                            acceleration_source=p["acceleration_topic"] if p["execution"] == "dynamic" else "published_PositionCommand.acceleration",
                            solver="condensed-torch" if self.actor is not None else "osqp-fixed-ipc")
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.state = self.rotation = self.cloud = None
            self.acceleration = np.zeros(3)
            self.odom_stamp = self.accel_stamp = None
            self.last_tick = None
            self.qp_failures = self.controls = 0
            self.have_goal = False
            self.have_cloud = False
            odom_topic = p["odom_topic"] or ("/marsim/odometry" if p["execution"] == "dynamic" else "/lidar_slam/odom")
            self.body_twist = odom_topic == "/marsim/odometry"
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                    durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(Odometry, odom_topic, self.on_odom, sensor_qos)
            self.create_subscription(AccelStamped, p["acceleration_topic"], self.on_acceleration, sensor_qos)
            self.create_subscription(PointCloud2, p["cloud_topic"], self.on_cloud, sensor_qos)
            goal_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(PoseStamped, p["goal_topic"], self.on_goal, goal_qos)
            command_type = TwistStamped if p["execution"] == "dynamic" else PositionCommand
            command_topic = p["ctbr_topic"] if p["execution"] == "dynamic" else p["position_command_topic"]
            self.command_pub = self.create_publisher(command_type, command_topic, 10)
            self.stats_pub = self.create_publisher(String, p["stats_topic"], 10)
            self.timer = self.create_timer(1.0 / p["frequency"], self.control)
            self.get_logger().info(f"{p['policy']}/{p['execution']}: {odom_topic}, {p['cloud_topic']} -> {command_topic}; {p['frequency']} Hz requested")

        @staticmethod
        def seconds(stamp):
            return stamp.sec + stamp.nanosec * 1e-9

        def on_odom(self, message):
            q = message.pose.pose.orientation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            v = message.twist.twist.linear
            velocity = np.array([v.x, v.y, v.z])
            if self.body_twist:
                velocity = rotation @ velocity
            stamp = self.seconds(message.header.stamp)
            position = message.pose.pose.position
            self.state = np.r_[[position.x, position.y, position.z], velocity, self.acceleration]
            self.rotation, self.odom_stamp = rotation, stamp

        def on_acceleration(self, message):
            a = message.accel.linear
            self.acceleration = np.array([a.x, a.y, a.z])
            self.accel_stamp = self.seconds(message.header.stamp)
            if self.state is not None:
                self.state[6:] = self.acceleration

        def on_cloud(self, message):
            if message.header.frame_id not in ("world", "map"):
                raise ValueError(f"Expected world-frame MARSIM point cloud; received {message.header.frame_id!r}")
            self.cloud = read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
            self.have_cloud = True

        def on_goal(self, message):
            g = message.pose.position
            goal = np.array([g.x, g.y, g.z])
            if not np.isfinite(goal).all():
                raise ValueError("Received a nonfinite mission goal")
            self.planner.goal = goal
            self.planner.replan = True
            self.have_goal = True
            self.get_logger().info(f"Mission goal: {goal.tolist()}")

        def control(self):
            if self.state is None or not self.have_cloud or not self.have_goal:
                return
            if self.options["execution"] == "dynamic" and self.accel_stamp is None:
                return
            started = perf_counter()
            interval = None if self.last_tick is None else 1000 * (started - self.last_tick)
            self.last_tick = started
            stamp = self.get_clock().now()
            elapsed = stamp.nanoseconds * 1e-9
            cloud, self.cloud = self.cloud, None
            context = self.batch.contexts([self.planner], [self.state], {} if cloud is None else {0: cloud})[0]
            yaw = float(np.arctan2(self.planner.goal[1] - self.state[1], self.planner.goal[0] - self.state[0])) if self.config.params["yaw_ctrl_flag"] else 0.0
            if self.actor is not None:
                obs = observation(np.r_[self.state, self.rotation.ravel()], self.planner.goal, context,
                                  self.planner.braking, yaw, self.planner.recovery_target, self.device)
                # Keep every actual SFC face; absent trailing slots only add zero QP rows.
                count = max(1, int(obs["plane_mask"].sum(-1).max()))
                obs["planes"] = obs["planes"][:, :, :count, :]
                obs["plane_mask"] = obs["plane_mask"][:, :, :count]
            solve_started = perf_counter()
            with torch.inference_mode():
                if self.actor is not None:
                    command = self.actor(obs)[0].cpu().numpy()
                    result = self.actor.last_result
                    solved = bool(result["solved"][0])
                    solution = result["solution"][0].cpu().numpy()
                    state = result["state"][0].cpu().numpy()
                    iterations = int(result["iterations"])
                    primal = float(result["primal"][0])
                    status = "solved" if solved else "max iterations reached"
                elif context.valid:
                    problem = self.planner.mpc.problem(context.state, context.reference, context.corridors)
                    solution, status, primal, _ = self.planner.mpc.solve(problem)
                    state, iterations = problem[-1], None
                    solved = status in ("solved", "solved inaccurate")
                else:
                    state, solution, solved, iterations, primal, status = context.state, np.zeros(45), False, None, float("inf"), "invalid SFC"
                solve_ms = 1000 * (perf_counter() - solve_started)
                predicted, jerk, _ = self.planner.consume_state(state, solution, status if context.valid else "invalid SFC", elapsed)
                if self.actor is None:
                    command = flatness_torch(torch.as_tensor(predicted[None, 6:]), torch.as_tensor(jerk[None]),
                                             torch.as_tensor(self.rotation[None]), torch.as_tensor([yaw]),
                                             self.thrust_limits, self.rate_limit)[0].cpu().numpy()
            message = TwistStamped() if self.options["execution"] == "dynamic" else PositionCommand()
            published = self.get_clock().now()
            message.header.stamp = published.to_msg()
            message.header.frame_id = "base_link" if self.options["execution"] == "dynamic" else "world"
            if self.options["execution"] == "dynamic":
                message.twist.linear.z = float(command[0])
                message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = map(float, command[1:])
            else:
                # consume_state advances the 0.1 s MPC prediction. PerfectDrone
                # needs the same selected jerk advanced by one control period.
                predicted = self.command_a @ state + self.command_b @ jerk
                for field, values in ((message.position, predicted[:3]), (message.velocity, predicted[3:6]),
                                      (message.acceleration, predicted[6:]), (message.jerk, jerk)):
                    field.x, field.y, field.z = map(float, values)
                message.yaw = yaw
                message.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            self.command_pub.publish(message)
            if self.options["execution"] == "ideal":
                # PerfectDrone assigns this acceleration directly. Differencing its
                # independently assigned velocity would fabricate another state.
                self.acceleration = predicted[6:].copy()
            self.qp_failures += int(context.valid and not solved)
            self.controls += 1
            row = dict(stamp=published.nanoseconds * 1e-9, policy=self.options["policy"], execution=self.options["execution"],
                       control_index=self.controls, qp_ok=solved, qp_failed=bool(context.valid and not solved),
                       qp_failures=self.qp_failures, sfc_valid=context.valid, qp_iterations=iterations,
                       primal_residual=primal if np.isfinite(primal) else None, solve_ms=solve_ms,
                       control_ms=1000 * (perf_counter() - started), timer_interval_ms=interval,
                       command_age=published.nanoseconds * 1e-9 - self.odom_stamp, braking=self.planner.braking,
                       frontend_failures=self.planner.frontend_failures)
            encoded = json.dumps(row, allow_nan=False)
            self.record.write(encoded + "\n")
            self.stats_pub.publish(String(data=encoded))

        def close(self):
            self.timer.cancel()
            self.record.close()
            self.batch.close()
            if self.actor is not None:
                self.actor.close()

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
