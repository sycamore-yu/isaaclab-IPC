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
from .rl.ctbr_actor import DEFAULT_CTBR_QP_LINEAR_SOLVER, CTBRMPCActor
from .rl.ctbr_observation import add_progress_observation, goal_ctbr_observation
from .turbompc.ctbr import CTBRConfig


def load_ctbr_checkpoint(path, *, qp_linear_solver=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor_cfg = copy.deepcopy(checkpoint.get("infos", {}).get("runner_cfg", {}).get("actor", {}))
    class_name = actor_cfg.pop("class_name", None)
    if class_name != "isaaclab_ipc.rl.ctbr_actor:CTBRMPCActor":
        raise ValueError(f"Expected a native 28N CTBR checkpoint; received {class_name!r}")
    model = checkpoint["actor_state_dict"]["_extra_state"]["model"]
    actor_cfg["model_config"] = model
    if qp_linear_solver is not None:
        actor_cfg["qp_linear_solver"] = qp_linear_solver
    else:
        actor_cfg.setdefault("qp_linear_solver", DEFAULT_CTBR_QP_LINEAR_SOLVER)
    return checkpoint, actor_cfg, CTBRConfig(**model)


def goal_observation(
    state, body_rates, goal, config, *, reference_speed=3.0, max_planes=128, warm_start_controls=None,
):
    """A0 shares Newton's fixed-clock line reference and has no geometric frontend."""
    return goal_ctbr_observation(
        state, body_rates, goal, ctbr_config=config, reference_speed=reference_speed,
        max_planes=max_planes, warm_start_controls=warm_start_controls,
    )


def benchmark_frontend_config(case, config, reference_speed=3.0, *, nominal_thrust_min=None):
    """Match CTBR physical bounds with a separate positive reference cost target."""
    from .pathcover_native import PathCoverConfig

    mapping = case["planner"]["rog_map"]
    center, size = np.asarray(mapping["fix_map_origin"]), np.asarray(mapping["map_size"])
    lower, upper = center - size / 2, center + size / 2
    lower[2] = max(0.0, mapping["virtual_ground_height"])
    upper[2] = mapping["virtual_ceil_height"]
    physical_speed = min(case["planner"]["traj_opt"]["boundary"]["max_vel"],
                         *config.velocity_upper, *(-value for value in config.velocity_lower))
    return PathCoverConfig(
        bounds_lower=tuple(lower), bounds_upper=tuple(upper),
        resolution=max(0.1, mapping["resolution"]),
        search_resolution=0.25 if mapping["resolution"] < 0.25 else None,
        robot_radius=case["planner"]["super_planner"]["robot_r"],
        max_corridors=3, max_iterations=2000, safety_margin=0.0, position_penalty_aggregation="continuous_max",
        geometry_model="box_ellipsoid",
        risp_local_half_extent=8.0,
        max_velocity=physical_speed, nominal_velocity=min(reference_speed, physical_speed),
        mass=config.mass, gravity=config.gravity,
        thrust_min=config.mass * config.specific_thrust_limits[0],
        nominal_thrust_min=(max(2.0 / 0.716 * config.mass, config.mass * config.specific_thrust_limits[0])
                            if nominal_thrust_min is None else nominal_thrust_min),
        thrust_max=config.mass * config.specific_thrust_limits[1],
        # The chart's boundary is nominal; zero/non-upright force still errors.
        tilt_max=np.pi / 2,
        body_rate_max=min(config.body_rate_limits[:2]), body_yaw_rate_max=config.body_rate_limits[2],
        use_cuda=True,
    )


class CTBRController:
    """Shared actor and frontend contracts; no ROS objects or hidden PPO state."""

    def __init__(self, benchmark, *, policy, frontend="pathcover", checkpoint=None,
                 device="cuda:0", max_planes=128, reference_speed=3.0, qp_linear_solver=None,
                 sfc_constraint_mode="polytope", sfc_section_half_length_m=0.5):
        started = perf_counter()
        self.device = torch.device(device)
        self.reference_speed = float(reference_speed)
        self.case = load_case(benchmark)
        self.frontend_name = frontend
        self.max_planes = 0 if frontend == "none" else max_planes
        saved, actor_cfg = None, {}
        checkpoint_actor_config = None
        self.config = CTBRConfig(mass=MASS)
        if checkpoint:
            saved, actor_cfg, self.config = load_ctbr_checkpoint(checkpoint, qp_linear_solver=qp_linear_solver)
            checkpoint_actor_config = saved.get("infos", {}).get("runner_cfg", {}).get("actor", {})
        else:
            actor_cfg["qp_linear_solver"] = (
                DEFAULT_CTBR_QP_LINEAR_SOLVER if qp_linear_solver is None else qp_linear_solver
            )
        progress = sfc_constraint_mode == 'progress'
        if progress:
            from .turbompc.progress import ProgressConfig

            actor_cfg.setdefault('progress_config', asdict(ProgressConfig()))
        if bool(actor_cfg.get('progress_config') is not None) != progress:
            raise ValueError('Checkpoint progress mode must match sfc_constraint_mode')
        self.navigation = None
        if frontend == "pathcover":
            self.navigation = PathCoverNavigation(1, self.config,
                benchmark_frontend_config(self.case, self.config, reference_speed),
                device=self.device, max_planes=self.max_planes,
                constraint_mode=sfc_constraint_mode,
                section_half_length_m=sfc_section_half_length_m,
                progress_config=actor_cfg.get('progress_config'))
        elif frontend != "none":
            raise ValueError("frontend must be pathcover or none")
        elif sfc_constraint_mode != "polytope":
            raise ValueError("Section constraints require the PathCover frontend")
        state = torch.zeros((1, 10), dtype=torch.float64, device=self.device)
        state[:, :3] = state.new_tensor(self.case["start"])
        state[:, 3] = 1.
        rates = state.new_zeros((1, 3))
        warm_goal = state[:, :3] + state.new_tensor([.25, .1, .1])
        sample = goal_observation(state, rates, warm_goal, self.config,
                                  reference_speed=self.reference_speed, max_planes=self.max_planes)
        if progress:
            delta = warm_goal - state[:, :3]
            length = delta.norm(dim=-1, keepdim=True)
            tangent = (delta / length)[:, None].expand(-1, self.config.cost_nodes, -1)
            arc = (sample['reference'][..., :3] - state[:, None, :3]).norm(dim=-1, keepdim=True)
            geometry = torch.cat((arc, tangent, torch.zeros_like(arc),
                                  length[:, None].expand_as(arc), torch.zeros_like(arc),
                                  torch.full_like(arc, actor_cfg['progress_config'].get('max_speed', 4.0))), -1)
            sample = add_progress_observation(sample, geometry)
        sample = TensorDict(sample, batch_size=[1])
        actor_cfg.setdefault("model_config", asdict(self.config))
        self.actor = CTBRMPCActor(sample, {"actor": ["policy"]}, "actor", 4, **actor_cfg).to(self.device)
        if saved is not None:
            self.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        selection_source = (
            "runtime_override" if qp_linear_solver is not None else
            "checkpoint" if checkpoint_actor_config is not None and "qp_linear_solver" in checkpoint_actor_config else
            "ctbr_default"
        )
        self.actor.record_checkpoint_linear_solver(checkpoint_actor_config, resolution=selection_source)
        if policy == "acmpc" and saved is None:
            raise ValueError("Learned AC-MPC requires a checkpoint")
        self.actor.fixed_cost = policy == "fixed-ctbr"
        self.actor.eval()
        self.warm_start = None
        self.last_valid_command = np.array([self.config.gravity, 0., 0., 0.])
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
                reference_speed=self.reference_speed, max_planes=self.max_planes,
                warm_start_controls=self.warm_start)
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
        solved = bool(result["solved"][0])
        if solved:
            self.warm_start = torch.cat((controls[:, 1:], controls[:, -1:]), 1).clone()
        self.last_observation, self.last_result = observation, result
        trajectory = result["trajectory"]
        violation = 0.0
        if self.max_planes:
            residual = (context["planes"][..., :3] * trajectory[..., None, :3]).sum(-1) + context["planes"][..., 3]
            violation = float(residual.masked_fill(~context["plane_mask"], 0).clamp_min(0).amax())
        command = action[0].cpu().numpy()
        if solved:
            self.last_valid_command = command.copy()
        else:
            command = self.last_valid_command.copy()
        metrics = dict(
            qp_ok=solved, qp_failed=not solved,
            sfc_valid=bool(result["sfc_valid"][0]), qp_iterations=int(result["iterations"]),
            sqp_iterations=int(result["sqp_iterations"][0]),
            qp_iterations_total=int(result["qp_iterations_total"][0]),
            primal_residual=float(result["primal"][0]), dual_residual=float(result["dual"][0]),
            sqp_status=int(result["sqp_status"][0]), raw_qp_status=int(result["raw_qp_status"][0]),
            predicted_sfc_violation_m=violation,
            frontend_ms=frontend_ms if self.navigation is not None else 0.0,
            solve_ms=solve_ms,
            planning_ms=1000 * (perf_counter() - started),
        )
        if 'optimized_progress' in result:
            arc, speed = result['optimized_progress'][0], result['progress_speed'][0]
            reference = context['ctbr_reference'][0, :, :3]
            geometry = context['progress_reference'][0]
            lag = ((trajectory[0, :, :3] - reference) * geometry[:, 1:4]).sum(-1) - (arc - geometry[:, 0])
            metrics.update(virtual_progress_m=float(arc[-1] - arc[0]),
                           progress_speed_m_s=float(speed[0]),
                           maximum_lag_error_m=float(lag.abs().max()))
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
                            frequency=100.0, replan_frequency=10.0, reference_speed=3.0,
                            capture_reference_plans=False, qp_linear_solver="",
                            sfc_constraint_mode="polytope", sfc_section_half_length_m=0.5)
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            self.options = p = {key: self.get_parameter(key).value for key in defaults}
            if p["execution"] != "dynamic":
                raise ValueError("The CTBR route uses MARSIM dynamic / Flightmare execution")
            if p["frequency"] != 100.0 or p["replan_frequency"] != 10.0:
                raise ValueError("The CTBR contract is 100 Hz actuation and 10 Hz replanning")
            torch.set_num_threads(1)
            self.controller = CTBRController(p["benchmark"], policy=p["policy"], frontend=p["frontend"],
                checkpoint=p["checkpoint"] or None, device=p["device"], reference_speed=p["reference_speed"],
                qp_linear_solver=p["qp_linear_solver"] or None,
                sfc_constraint_mode=p["sfc_constraint_mode"],
                sfc_section_half_length_m=p["sfc_section_half_length_m"])
            directory = Path(p["output_directory"])
            directory.mkdir(parents=True, exist_ok=True)
            self.directory = directory
            self.record = (directory / "policy.jsonl").open("w", buffering=1)
            self.reference_records = [] if p["capture_reference_plans"] else None
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
                reference_archive_scope="Returned CPU plans retained after solve timing; serialized at shutdown; no extra integration"
                    if p["capture_reference_plans"] else None,
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
                    if self.options['sfc_constraint_mode'] == 'progress':
                        # Progress is not MINCO clock time. Compare with the
                        # executed MPC interval's selected original polytope.
                        context = self.controller.last_observation
                        k = min(max(int(np.ceil((state_time - previous_plan.start_time)
                                                 / self.controller.config.dt)), 1),
                                self.controller.config.cost_nodes - 1)
                        faces = context['planes'][0, k, context['plane_mask'][0, k]].cpu().numpy()
                        actual_violation = max(0., float(np.max(faces[:, :3] @ self.state[:3] + faces[:, 3])))
                    else:
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
                            geometry_obstacles=frontend.geometry_obstacles(),
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
                snapshot_name = ("first-unsolved.pt" if stats["qp_failed"] else
                                 "first-large-qp-work.pt" if stats["qp_iterations_total"] >= 4096 else None)
                # Capture the exact incoming warm start even when the final MPC
                # succeeds after expensive QPs. Serialization follows planning
                # timing and happens only once for each diagnostic category.
                if snapshot_name and not (self.directory / snapshot_name).exists():
                    torch.save(dict(observation=self.controller.last_observation.cpu(),
                                    stamp=now, state_time=state_time, stats=stats,
                                    policy=self.options["policy"], checkpoint=self.options["checkpoint"],
                                    solver_config=self.controller.actor.solver_metadata(),
                                    frontend_plan=self.controller.navigation.last_plans[0]
                                    if self.controller.navigation is not None else None,
                                    result={key: value.cpu() if isinstance(value, torch.Tensor) else value
                                            for key, value in self.controller.last_result.items()}),
                               self.directory / snapshot_name)
                reference_snapshot = self.directory / "first-reference-sfc.pt"
                if (self.controller.navigation is not None
                        and stats.get("optimizer_sfc_max_m", 0.0) > 1e-4
                        and not reference_snapshot.exists()):
                    # Reuse the returned polynomial and existing diagnostics.
                    # No map copy, additional integration or acceptance pass.
                    plan = self.controller.navigation.last_plans[0]
                    frontend = self.controller.navigation.frontends[0]
                    problem = dict(initial_pva=np.column_stack((self.state[:3], self.state[7:10], self.acceleration)),
                                   terminal_pva=np.column_stack((plan.local_goal, np.zeros(3), np.zeros(3))),
                                   corridors=plan.corridors, config=frontend.config.native_options(),
                                   optimizer_status=stats["optimizer_status"], initial_yaw=plan.trajectory.initial_yaw,
                                   initial_boundary=plan.metrics["initial_boundary"],
                                   initial_thrust_axis=plan.metrics["initial_boundary"]["thrust_axis"])
                    torch.save(dict(stamp=now, state_time=state_time, stats=stats,
                                    state=self.state, body_rates=self.body_rates,
                                    acceleration=self.acceleration, goal=self.goal,
                                    frontend_config=asdict(frontend.config),
                                    gcopter_problem=problem,
                                    frontend_plan=plan), reference_snapshot)
                if self.reference_records is not None and self.controller.navigation is not None:
                    # Plans own their CPU coefficient/SFC/yaw arrays. Keep the
                    # returned object; no map copy, GPU transfer or reevaluation.
                    self.reference_records.append(dict(
                        stamp=now, state_time=state_time, plan_sequence_id=self.sequence,
                        state=self.state.copy(), body_rates=self.body_rates.copy(),
                        acceleration=self.acceleration.copy(), goal=self.goal.copy(),
                        frontend_plan=self.controller.navigation.last_plans[0],
                    ))
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
            if self.reference_records is not None:
                torch.save(dict(
                    schema_version=1,
                    frontend_config=asdict(self.controller.navigation.frontends[0].config)
                        if self.controller.navigation is not None else None,
                    records=self.reference_records,
                    scope="Filter by evaluator stamp/sequence; shutdown plans may follow the last evaluated action",
                ), self.directory / "reference-plans.pt")
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
