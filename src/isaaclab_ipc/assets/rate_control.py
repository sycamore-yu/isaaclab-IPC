# Copyright (c) 2026, Isaac Lab Diff Flight Project Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Rate feedback, motor allocation, and body forces in Newton's physics step.

The Newton callback and body-wrench application follow the existing
isaaclab_mbrl_flight/controller.py implementation. This controller tracks body
rates rather than the source controller's world-acceleration/attitude command.
"""

import math

import torch
import warp as wp

from .quadrotor import INERTIA, MASS, MAX_THRUST


@wp.kernel
def rate_targets(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    ids: wp.array(dtype=wp.int32),
    command: wp.array2d(dtype=wp.float32),
    inverse: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    mass: float,
    inertia: float,
    max_thrust: float,
):
    i = wp.tid()
    q = wp.transform_get_rotation(body_q[ids[i]])
    twist = body_qd[ids[i]]
    omega = wp.quat_rotate_inv(q, wp.vec3(twist[3], twist[4], twist[5]))
    desired = wp.vec3(command[i, 1], command[i, 2], command[i, 3])
    torque = inertia * 45.0 * (desired - omega)
    collective = wp.clamp(command[i, 0] * mass, 0.0, 4.0 * max_thrust)
    baseline = collective * 0.25
    scale = 1.0
    for j in range(4):
        delta = inverse[j, 1] * torque[0] + inverse[j, 2] * torque[1] + inverse[j, 3] * torque[2]
        if delta > 0.0:
            scale = wp.min(scale, (max_thrust - baseline) / delta)
        elif delta < 0.0:
            scale = wp.min(scale, baseline / (-delta))
    for j in range(4):
        delta = inverse[j, 1] * torque[0] + inverse[j, 2] * torque[1] + inverse[j, 3] * torque[2]
        target[i, j] = wp.clamp(baseline + wp.clamp(scale, 0.0, 1.0) * delta, 0.0, max_thrust)


@wp.kernel
def compensate_motor_lag(
    target: wp.array2d(dtype=wp.float32),
    current: wp.array2d(dtype=wp.float32),
    tau_inc: wp.array2d(dtype=wp.float32),
    tau_dec: wp.array2d(dtype=wp.float32),
    thrust_const: wp.array2d(dtype=wp.float32),
    max_rate: wp.array2d(dtype=wp.float32),
    dt: float,
    min_thrust: float,
    max_thrust: float,
):
    """Allocate reachable motor forces and invert the native ESC dynamics.

    This motor-aware allocation is a native execution extension to IPC.
    The official actuator remains responsible for its RPS lag, saturation
    and rate limit. First project the allocated forces onto the reachable
    force box while preserving the requested collective whenever possible.
    Then invert the Euler/discrete dynamics for those reachable forces.
    An unreachable collective is clipped to the nearest reachable total.
    """
    i = wp.tid()
    desired = wp.vec4(0.0)
    lower = wp.vec4(0.0)
    upper = wp.vec4(0.0)
    requested_total = 0.0
    minimum_total = 0.0
    maximum_total = 0.0
    for j in range(4):
        desired[j] = wp.clamp(target[i, j], min_thrust, max_thrust)
        present = current[i, j]
        k = thrust_const[i, j]
        rps = wp.sqrt(wp.max(present / k, 0.0))
        lower_tau = tau_inc[i, j]
        if present > 0.0 and min_thrust < present:
            lower_tau = tau_dec[i, j]
        upper_tau = tau_inc[i, j]
        if present > 0.0 and max_thrust < present:
            upper_tau = tau_dec[i, j]
        # These are exactly the official Euler updates at its input bounds,
        # including max_rate's native RPS/s interpretation.
        lower_rate = wp.clamp(
            (wp.sqrt(min_thrust / k) - rps) / (dt + lower_tau),
            -max_rate[i, j],
            max_rate[i, j],
        )
        upper_rate = wp.clamp(
            (wp.sqrt(max_thrust / k) - rps) / (dt + upper_tau),
            -max_rate[i, j],
            max_rate[i, j],
        )
        lower_rps = rps + dt * lower_rate
        upper_rps = rps + dt * upper_rate
        lower[j] = wp.clamp(k * lower_rps * lower_rps, min_thrust, max_thrust)
        upper[j] = wp.clamp(k * upper_rps * upper_rps, min_thrust, max_thrust)
        requested_total += desired[j]
        minimum_total += lower[j]
        maximum_total += upper[j]

    reachable_total = wp.clamp(requested_total, minimum_total, maximum_total)
    lambda_lower = lower[0] - desired[0]
    lambda_upper = upper[0] - desired[0]
    for j in range(1, 4):
        lambda_lower = wp.min(lambda_lower, lower[j] - desired[j])
        lambda_upper = wp.max(lambda_upper, upper[j] - desired[j])
    # Euclidean projection onto box + sum: f_j = clip(desired_j + lambda).
    # This retains the original allocator's collective-first priority and
    # gives the closest feasible motor vector when torque is unreachable.
    for _ in range(24):
        shift = 0.5 * (lambda_lower + lambda_upper)
        total = 0.0
        for j in range(4):
            total += wp.clamp(desired[j] + shift, lower[j], upper[j])
        if total < reachable_total:
            lambda_lower = shift
        else:
            lambda_upper = shift
    shift = 0.5 * (lambda_lower + lambda_upper)

    for j in range(4):
        force = wp.clamp(desired[j] + shift, lower[j], upper[j])
        present = current[i, j]
        tau = tau_inc[i, j]
        if present > 0.0 and force < present:
            tau = tau_dec[i, j]
        alpha = dt / (dt + tau)
        k = thrust_const[i, j]
        current_rps = wp.sqrt(wp.max(present / k, 0.0))
        desired_rps = wp.sqrt(wp.max(force / k, 0.0))
        max_delta = max_rate[i, j] * dt
        delta = wp.clamp(desired_rps - current_rps, -max_delta, max_delta)
        command_rps = wp.clamp(
            current_rps + delta / alpha,
            wp.sqrt(min_thrust / k),
            wp.sqrt(max_thrust / k),
        )
        # Clamp RPS before squaring so a down-step cannot reflect a negative
        # inverse input into a positive force. The native model advances next.
        target[i, j] = wp.clamp(k * command_rps * command_rps, min_thrust, max_thrust)


@wp.kernel
def apply_wrench(
    body_q: wp.array(dtype=wp.transform),
    body_f: wp.array(dtype=wp.spatial_vector),
    ids: wp.array(dtype=wp.int32),
    wrench: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    body = ids[i]
    q = wp.transform_get_rotation(body_q[body])
    force = wp.quat_rotate(q, wp.vec3(wrench[i, 0], wrench[i, 1], wrench[i, 2]))
    torque = wp.quat_rotate(q, wp.vec3(wrench[i, 3], wrench[i, 4], wrench[i, 5]))
    # Newton applies gravity. This callback adds only the physical motor wrench.
    body_f[body] = body_f[body] + wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])


class RateController:
    def __init__(self, num_envs: int, device: str):
        self.num_envs, self.device = num_envs, device
        self.command = torch.zeros(num_envs, 4, device=device)
        self.command[:, 0] = 9.81
        self.ids = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.command_wp = wp.from_torch(self.command, dtype=wp.float32)
        self.ids_wp = wp.from_torch(self.ids, dtype=wp.int32)
        self.robot = None

    def bind(self, manager, robot):
        names = {name: i for i, name in enumerate(manager.backend.model.body_label)}
        self.ids.copy_(
            torch.tensor(
                [names[f"/World/envs/env_{i}/Robot/base"] for i in range(self.num_envs)],
                dtype=torch.int32,
                device=self.device,
            )
        )
        self.robot = robot
        self.inverse = torch.linalg.inv(robot.allocation_matrix[[2, 3, 4, 5], :]).contiguous()
        self.inverse_wp = wp.from_torch(self.inverse, dtype=wp.float32)
        motors = next(iter(robot.actuators.values()))
        if motors.cfg.integration_scheme != "euler" or not motors.cfg.use_discrete_approximation:
            raise ValueError("Motor inverse dynamics requires the official Euler/discrete Thruster")
        self.motor_dt = float(motors.cfg.dt)
        if not math.isfinite(self.motor_dt) or self.motor_dt <= 0.0:
            raise ValueError("Motor inverse dynamics requires a finite positive actuator dt")
        if motors.min_thrust < 0.0:
            raise ValueError("Motor inverse dynamics requires nonnegative thrust bounds")
        self.motor_min_thrust, self.motor_max_thrust = motors.min_thrust, motors.max_thrust
        # The official actuator resets these tensors in place. Keeping aliases
        # uses its current sampled parameters after partial resets as well.
        self.motor_tau_inc_wp = wp.from_torch(motors.tau_inc_s, dtype=wp.float32)
        self.motor_tau_dec_wp = wp.from_torch(motors.tau_dec_s, dtype=wp.float32)
        self.motor_thrust_const_wp = wp.from_torch(motors.thrust_const, dtype=wp.float32)
        self.motor_max_rate_wp = wp.from_torch(motors.max_rate, dtype=wp.float32)

    def apply(self, state):
        if self.robot is None:
            raise RuntimeError("Rate controller must bind before stepping physics")
        wp.launch(
            rate_targets,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                state.body_qd,
                self.ids_wp,
                self.command_wp,
                self.inverse_wp,
                self.robot.thrust_target_wp,
                MASS,
                INERTIA,
                MAX_THRUST,
            ],
            device=self.device,
        )
        wp.launch(
            compensate_motor_lag,
            dim=self.num_envs,
            inputs=[
                self.robot.thrust_target_wp,
                self.robot.motor_thrust_wp,
                self.motor_tau_inc_wp,
                self.motor_tau_dec_wp,
                self.motor_thrust_const_wp,
                self.motor_max_rate_wp,
                self.motor_dt,
                self.motor_min_thrust,
                self.motor_max_thrust,
            ],
            device=self.device,
        )
        self.robot.advance_motors()
        wp.launch(
            apply_wrench,
            dim=self.num_envs,
            inputs=[state.body_q, state.body_f, self.ids_wp, self.robot.body_wrench_wp],
            device=self.device,
        )
