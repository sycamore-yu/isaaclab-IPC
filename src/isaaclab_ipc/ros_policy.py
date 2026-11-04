"""ROS 2 adapter for the existing IPC planner and 124-feature jerk MPC actor.

Both MARSIM executors supply measured velocity.  MARSIM ``ideal`` therefore
does not select the training package's legacy velocity-zeroing ideal profile.
CTBR uses collective specific thrust [m/s^2] and body rates [rad/s].
"""

import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
from queue import Full
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from isaaclab_ipc.benchmark_config import load_case, planner_params
from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.frontend import PlanningBatch, PlanningContext
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.planner import Planner
from isaaclab_ipc.rl.actor import MPCActor
from isaaclab_ipc.ros_control import (
    IdealCommand,
    accept_plan,
    check_plan_constraints,
    ctbr_from_jerk,
    ctbr_from_reference,
    select_jerk,
    sequence_reference,
)


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


def planning_worker(requests, results, options, params, actor_cfg):
    """Own Planner and QP in a process so geometry cannot hold the ROS GIL."""
    torch.set_num_threads(1)
    config = Config(params, profile="real")
    planner = Planner(config, np.asarray(load_case(options["benchmark"])["start"], dtype=float))
    batch = PlanningBatch(1)
    actor = None
    device = torch.device(options["device"])
    if options["policy"] == "acmpc":
        checkpoint, _, _ = load_checkpoint(options["checkpoint"])
        blank = PlanningContext(np.zeros(9), np.zeros((15, 9)), [np.empty((0, 4))] * 15, False)
        sample = observation(np.r_[np.zeros(9), np.eye(3).ravel()], np.zeros(3), blank, device=device)
        actor_cfg = dict(actor_cfg, planner_params=params)
        actor = MPCActor(sample, {"actor": ["policy"]}, "actor", 4, **actor_cfg).to(device)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        actor.eval()
    goal_id = None
    sequence_id = qp_failures = 0
    try:
        while (request := requests.get()) is not None:
            started = perf_counter()
            state, rotation = request["state"], request["rotation"]
            if request["goal_id"] != goal_id:
                goal_id = request["goal_id"]
                planner.goal = request["goal"].copy()
                planner.replan = True
            if request["expired"] or request.get("replan", False):
                planner.braking = True
                planner.replan = True
            cloud = request["cloud"]
            context = batch.contexts([planner], [state], {} if cloud is None else {0: cloud})[0]
            frontend_ms = 1000 * (perf_counter() - started)
            observation_started = perf_counter()
            yaw = float(np.arctan2(planner.goal[1] - state[1], planner.goal[0] - state[0])) if params["yaw_ctrl_flag"] else 0.0
            if actor is not None:
                obs = observation(np.r_[state, rotation.ravel()], planner.goal, context,
                                  planner.braking, yaw, planner.recovery_target, device)
                # Preserve every actual SFC face; remove only absent trailing slots.
                count = max(1, int(obs["plane_mask"].sum(-1).max()))
                obs["planes"] = obs["planes"][:, :, :count, :]
                obs["plane_mask"] = obs["plane_mask"][:, :, :count]
            observation_ms = 1000 * (perf_counter() - observation_started)
            solve_started = perf_counter()
            with torch.inference_mode():
                if actor is not None:
                    actor(obs)
                    result = actor.last_result
                    solved = bool(result["solved"][0])
                    solution = result["solution"][0].cpu().numpy()
                    solver_state = result["state"][0].cpu().numpy()
                    iterations = int(result["iterations"])
                    primal = float(result["primal"][0])
                    status = "solved" if solved else "max iterations reached"
                elif context.valid:
                    problem = planner.mpc.problem(context.state, context.reference, context.corridors)
                    solution, status, primal, _ = planner.mpc.solve(problem)
                    solver_state, iterations = problem[-1], None
                    solved = status in ("solved", "solved inaccurate")
                else:
                    solver_state, solution, solved = context.state, np.zeros(3 * planner.mpc.n), False
                    iterations, primal, status = None, float("inf"), "invalid SFC"
            solve_ms = 1000 * (perf_counter() - solve_started)
            planner.consume_state(solver_state, solution, status if context.valid else "invalid SFC",
                                  request["source_stamp_ns"] * 1e-9)
            sequence_id += 1
            qp_failed = bool(context.valid and not solved)
            qp_failures += int(qp_failed)
            sequence = np.asarray(solution).reshape(planner.mpc.n, 3).copy()
            corridors = context.corridors or [np.empty((0, 4))] * planner.mpc.n
            faces, masks, schedule = pack_corridors(corridors, max(map(len, corridors)))
            # Retain the original face order for MPC.problem; pack_corridors reverses it.
            unique = [block[mask][::-1].copy() for block, mask in zip(faces, masks, strict=True)]
            # These are MPC.solve's OSQP primal tolerances. Solved-inaccurate
            # uses OSQP's existing tenfold termination tolerance.
            multiplier = 10 if status == "solved inaccurate" else 1
            eps_abs = actor.solver.eps_abs if actor is not None else multiplier * 1e-3
            eps_rel = 0.0 if actor is not None else multiplier * 1e-3
            plan = dict(sequence=sequence,
                        source_stamp_ns=request["source_stamp_ns"], plan_sequence_id=sequence_id,
                        goal_id=goal_id, braking=bool(planner.braking), sfc_valid=context.valid,
                        climb_target=None if planner.recovery_target is None else planner.recovery_target.copy(),
                        source_state=solver_state.copy(), source_measurement=state.copy(),
                        corridors=unique, corridor_index=schedule,
                        constraint_eps_abs=eps_abs, constraint_eps_rel=eps_rel, solver_status=status)
            stats = dict(event="solve", plan_sequence_id=sequence_id,
                         source_stamp=request["source_stamp_ns"] * 1e-9,
                         qp_ok=solved, qp_failed=qp_failed, qp_failures=qp_failures,
                         sfc_valid=context.valid, qp_iterations=iterations,
                         primal_residual=primal if np.isfinite(primal) else None,
                         frontend_ms=frontend_ms, observation_ms=observation_ms, solve_ms=solve_ms,
                         planning_ms=1000 * (perf_counter() - started),
                         request_wait_ms=1000 * (started - request["sent_monotonic"]),
                         braking=bool(planner.braking), frontend_failures=planner.frontend_failures)
            results.send((plan, stats))
    except KeyboardInterrupt:
        pass
    finally:
        batch.close()
        if actor is not None:
            actor.close()
        results.close()


class PlanningWorker:
    """At most one in-flight snapshot; the next request uses the latest sensors."""

    def __init__(self, options, params, actor_cfg):
        context = multiprocessing.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.results, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=planning_worker,
                                       args=(self.requests, sender, options, params, actor_cfg),
                                       name="marsim-mpc-planner", daemon=True)
        self.process.start()
        sender.close()
        self.busy = False

    def submit(self, request):
        if self.busy:
            return False
        self.requests.put_nowait(request)
        self.busy = True
        return True

    def poll(self):
        if self.results.poll():
            self.busy = False
            return self.results.recv()
        if not self.process.is_alive():
            raise RuntimeError(f"MPC planning worker exited with code {self.process.exitcode}")
        return None

    def close(self):
        if self.process.is_alive():
            try:
                self.requests.put(None, timeout=0.2)
            except Full:
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.requests.cancel_join_thread()
        self.requests.close()
        self.results.close()


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
            self.mpc = MPC(self.config)
            self.ideal_command = IdealCommand(
                [params["mpc"]["D_" + axis] for axis in "xyz"],
                history_size=max(2, int(np.ceil(self.mpc.n * self.mpc.dt * p["frequency"])) + 1),
            )
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
                            scheduling="one planning process; 100 Hz time-sampled jerk trajectory execution",
                            sequence_clock="acceptance timestamp; source age separately limits full-horizon validity",
                            reference_origin="execution PVA at acceptance (measured dynamic; current published ideal); original 0.1 s model knots",
                            acceptance="original jerk/state/SFC constraints rechecked at the execution PVA; rejected plans brake and replan",
                            ctbr_acceleration="time-sampled reference a; original one-step state feedback only during recovery",
                            ideal_integration="previous published PVA and jerk over actual publication interval",
                            statistics="event=solve records each QP once; event=control records each command",
                            acceleration_source=p["acceleration_topic"] if p["execution"] == "dynamic" else "PositionCommand.acceleration_matched_to_odometry_PV",
                            solver="condensed-torch" if p["policy"] == "acmpc" else "osqp-fixed-ipc")
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            self.state = self.rotation = self.cloud = None
            self.acceleration = np.zeros(3)
            self.odom_stamp_ns = self.accel_stamp_ns = None
            self.last_tick = None
            self.controls = self.goal_id = 0
            self.goal = None
            self.plan = None
            self.replan_required = False
            self.reanchor_rejections = 0
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
            self.worker = PlanningWorker(p, params, actor_cfg)
            self.timer = self.create_timer(1.0 / p["frequency"], self.control)
            self.get_logger().info(f"{p['policy']}/{p['execution']}: {odom_topic}, {p['cloud_topic']} -> {command_topic}; {p['frequency']} Hz control, asynchronous MPC")

        @staticmethod
        def nanoseconds(stamp):
            return stamp.sec * 1_000_000_000 + stamp.nanosec

        def on_odom(self, message):
            q = message.pose.pose.orientation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            v = message.twist.twist.linear
            velocity = np.array([v.x, v.y, v.z])
            if self.body_twist:
                velocity = rotation @ velocity
            stamp = self.nanoseconds(message.header.stamp)
            position = message.pose.pose.position
            position_velocity = np.r_[[position.x, position.y, position.z], velocity]
            if self.options["execution"] == "ideal" and self.ideal_command.state is not None:
                self.acceleration = self.ideal_command.observed_acceleration(position_velocity)
            self.state = np.r_[position_velocity, self.acceleration]
            self.rotation, self.odom_stamp_ns = rotation, stamp

        def on_acceleration(self, message):
            a = message.accel.linear
            self.acceleration = np.array([a.x, a.y, a.z])
            self.accel_stamp_ns = self.nanoseconds(message.header.stamp)
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
            self.goal = goal
            self.goal_id += 1
            self.plan = None
            self.have_goal = True
            self.get_logger().info(f"Mission goal: {goal.tolist()}")

        def control(self):
            started = perf_counter()
            completed = self.worker.poll()
            if self.state is None or not self.have_cloud or not self.have_goal:
                return
            if self.options["execution"] == "dynamic" and self.accel_stamp_ns is None:
                return
            interval = None if self.last_tick is None else 1000 * (started - self.last_tick)
            self.last_tick = started
            stamp = self.get_clock().now()
            execution_state = self.state
            if self.options["execution"] == "ideal":
                # Advance the previous command first. The gate must check the
                # PVA actually published this tick, not an older odometry sample.
                execution_state = self.ideal_command.advance(self.state, self.ideal_command.jerk, stamp.nanoseconds)
            if completed is not None:
                plan, stats = completed
                source_expired = stamp.nanoseconds - plan["source_stamp_ns"] >= round(self.mpc.n * self.mpc.dt * 1e9)
                accepted = plan["goal_id"] == self.goal_id and not source_expired
                check = None
                acceptance_ms = 0.0
                if accepted and not plan["braking"] and plan["sfc_valid"]:
                    check_started = perf_counter()
                    check = check_plan_constraints(plan, execution_state, self.mpc)
                    acceptance_ms = 1000 * (perf_counter() - check_started)
                rejected = check is not None and not check["feasible"]
                if rejected:
                    accepted = False
                    self.reanchor_rejections += 1
                    self.plan = None
                    self.replan_required = True
                if accepted:
                    self.plan = accept_plan(plan, execution_state, stamp.nanoseconds, self.mpc)
                stats.update(stamp=stamp.nanoseconds * 1e-9, accepted=accepted, source_expired=source_expired,
                             reanchor_rejected=rejected, reanchor_rejections=self.reanchor_rejections,
                             reanchor_violation=None if check is None else check["violation"],
                             reanchor_tolerance=None if check is None else check["tolerance"],
                             reanchor_constraint=None if check is None else check["constraint"],
                             acceptance_ms=acceptance_ms, acceptance_state=execution_state.tolist(),
                             sequence=plan["sequence"].tolist(), source_state=plan["source_state"].tolist(),
                             source_measurement=plan["source_measurement"].tolist(),
                             corridors=[faces.tolist() for faces in plan["corridors"]],
                             corridor_index=plan["corridor_index"].tolist(),
                             constraint_eps_abs=plan["constraint_eps_abs"], constraint_eps_rel=plan["constraint_eps_rel"],
                             solver_status=plan["solver_status"])
                self.publish_stats(stats)
            jerk, sequence_index, braking, expired = select_jerk(self.plan, execution_state, stamp.nanoseconds, self.mpc)
            yaw = float(np.arctan2(self.goal[1] - self.state[1], self.goal[0] - self.state[0])) if self.config.params["yaw_ctrl_flag"] else 0.0
            message = TwistStamped() if self.options["execution"] == "dynamic" else PositionCommand()
            message.header.stamp = stamp.to_msg()
            message.header.frame_id = "base_link" if self.options["execution"] == "dynamic" else "world"
            reference_acceleration = None
            if self.options["execution"] == "dynamic":
                if braking:
                    command = ctbr_from_jerk(self.state, self.rotation, jerk, yaw, self.mpc,
                                             self.thrust_limits, self.rate_limit)
                else:
                    reference_acceleration = sequence_reference(self.plan, stamp.nanoseconds, self.mpc)[6:]
                    command = ctbr_from_reference(reference_acceleration, self.rotation, jerk, yaw,
                                                 self.thrust_limits, self.rate_limit)
                message.twist.linear.z = float(command[0])
                message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = map(float, command[1:])
            else:
                predicted = execution_state
                self.ideal_command.jerk = jerk.copy()
                reference_acceleration = predicted[6:]
                for field, values in ((message.position, predicted[:3]), (message.velocity, predicted[3:6]),
                                      (message.acceleration, predicted[6:]), (message.jerk, jerk)):
                    field.x, field.y, field.z = map(float, values)
                message.yaw = yaw
                message.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            self.command_pub.publish(message)
            if not self.worker.busy:
                request = dict(state=self.state.copy(), rotation=self.rotation.copy(),
                               goal=self.goal.copy(), goal_id=self.goal_id, cloud=self.cloud,
                               source_stamp_ns=self.odom_stamp_ns, expired=expired, replan=self.replan_required,
                               sent_monotonic=perf_counter())
                if self.worker.submit(request):
                    self.cloud = None
                    self.replan_required = False
            self.controls += 1
            row = dict(event="control", stamp=stamp.nanoseconds * 1e-9,
                       control_index=self.controls,
                       plan_sequence_id=None if self.plan is None else self.plan["plan_sequence_id"],
                       sequence_index=sequence_index,
                       solution_age=None if self.plan is None else (stamp.nanoseconds - self.plan["source_stamp_ns"]) * 1e-9,
                       sequence_age=None if self.plan is None else (stamp.nanoseconds - self.plan["accepted_stamp_ns"]) * 1e-9,
                       state_age=(stamp.nanoseconds - self.odom_stamp_ns) * 1e-9,
                       command_age=(stamp.nanoseconds - self.odom_stamp_ns) * 1e-9,
                       jerk=jerk.tolist(), measured_acceleration=self.state[6:].tolist(),
                       reference_acceleration=None if reference_acceleration is None else reference_acceleration.tolist(),
                       sfc_valid=False if self.plan is None else self.plan["sfc_valid"],
                       control_ms=1000 * (perf_counter() - started), timer_interval_ms=interval,
                       braking=braking, plan_expired=expired)
            self.publish_stats(row)

        def publish_stats(self, row):
            row.update(policy=self.options["policy"], execution=self.options["execution"])
            encoded = json.dumps(row, allow_nan=False)
            self.record.write(encoded + "\n")
            self.stats_pub.publish(String(data=encoded))

        def close(self):
            self.timer.cancel()
            self.worker.close()
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
