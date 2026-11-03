"""Isaac Lab task adapting physical XYZW state to native CTBR MPC."""

import math

import gymnasium as gym
import numpy as np
import torch

from isaaclab_ipc.assets.quadrotor import MASS, MAX_THRUST
from isaaclab_ipc.turbompc.ctbr import CTBRConfig
from isaaclab_ipc.turbompc.quadrotor import body_rates_from_isaac, reference_from_ipc, state_from_isaac

from .mpc_env import FEATURES, IPCMPCEnv, make_forest_mpc_cfg, make_mpc_cfg

CTBR_FEATURES = FEATURES + 3


@torch.no_grad()
def checked_articulation_mass(robot, expected_mass: float) -> torch.Tensor:
    """Check each complete robot's mass, including its fixed rotor links.

    ArticulationData exposes masses in public body order as
    ``[num_instances, num_bodies]``. The controller's root-body index identifies
    only the base link; its mass alone is not the quadrotor's total mass.
    """
    link_masses = robot.data.body_mass.torch
    expected_shape = (robot.num_instances, robot.num_bodies)
    if tuple(link_masses.shape) != expected_shape:
        raise ValueError(
            f"Newton articulation mass shape: actual={tuple(link_masses.shape)}, expected={expected_shape}"
        )
    if not math.isfinite(expected_mass) or expected_mass <= 0:
        raise ValueError(f"Expected articulation mass must be finite and positive: expected={expected_mass!r} kg")
    masses = link_masses.sum(dim=-1)
    if not bool(torch.isfinite(link_masses).all()) or not torch.allclose(
        masses, torch.full_like(masses, expected_mass), atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Newton articulation, RateController and CTBR mass must agree: "
            f"actual={masses.cpu().tolist()} kg, expected={expected_mass} kg; "
            f"link_masses={link_masses.cpu().tolist()}"
        )
    return masses


def make_ctbr_cfg():
    cfg = make_mpc_cfg()
    cfg.observation_space = CTBR_FEATURES
    cfg.reference_strides = (10,)
    return cfg


def make_forest_ctbr_cfg():
    cfg = make_forest_mpc_cfg()
    cfg.observation_space = CTBR_FEATURES
    return cfg


class CTBRMPCEnv(IPCMPCEnv):
    """Retain IPC sensing/SFC/recovery; consume a distinct 15-by-4 CTBR plan."""

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        shapes = {"ctbr_state": (10,), "ctbr_reference": (15, 10), "body_rates": (3,)}
        for name, shape in shapes.items():
            self.single_observation_space[name] = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.ctbr_config = CTBRConfig(mass=MASS)
        gravity = tuple(float(value) for value in cfg.sim.gravity)
        if gravity != (0.0, 0.0, -self.ctbr_config.gravity):
            raise ValueError("CTBR task gravity must match the prediction model's explicit world gravity")
        if cfg.observation_space != CTBR_FEATURES:
            raise ValueError("CTBR task uses the 127-feature observation and a separate checkpoint")
        super().__init__(cfg, render_mode, **kwargs)
        try:
            masses = checked_articulation_mass(self.robot, self.ctbr_config.mass)
        except Exception:
            self.close()
            raise
        self.ctbr_runtime_contract = {
            "state_quaternion": "Isaac Lab XYZW -> CTBR WXYZ",
            "position_frame": "world axes, translated by environment origin",
            "linear_velocity_frame": "world",
            "angular_velocity_frame": "body",
            "newton_articulation_mass_kg": masses.cpu().tolist(),
            "body_names": list(self.robot.body_names),
            "link_masses_kg": self.robot.data.body_mass.torch.cpu().tolist(),
            "controller_mass_kg": MASS,
            "model_mass_kg": self.ctbr_config.mass,
            "gravity_m_s2": self.ctbr_config.gravity,
            "physical_total_thrust_capacity_N": 4 * MAX_THRUST,
            "prediction_dt": self.ctbr_config.dt,
            "control_dt": self.step_dt,
            "physics_dt": self.cfg.sim.dt,
            "model_body_rate_limits_rad_s": self.ctbr_config.body_rate_limits,
            "model_specific_thrust_limits_m_s2": self.ctbr_config.specific_thrust_limits,
        }

    def _get_observations(self):
        result = super()._get_observations()
        pose, twist = self.body_state()
        state = state_from_isaac(pose, twist, self.scene.env_origins)
        body_rates = body_rates_from_isaac(pose, twist)
        result.update(
            ctbr_state=state,
            ctbr_reference=reference_from_ipc(result["reference"], result["yaw_target"]),
            body_rates=body_rates,
        )
        rate_scale = body_rates.new_tensor(self.ctbr_config.body_rate_limits)
        result["policy"] = torch.cat((result["policy"], (body_rates / rate_scale).float()), -1)
        self.current_context = result
        return result

    def record_mpc_result(self, result):
        if result.get("model") != "ctbr" or result["state"].shape[-1] != 10 or result["solution"].shape[-1] != 60:
            raise ValueError("CTBR task requires the explicit 10-state, 15-by-4 CTBR result schema")
        super().record_mpc_result(result)

    def _consume_control_result(self, planner, data, index, time, action):
        # Never send a four-input plan to Planner.consume_state's jerk parser.
        solved = bool(data["solved"][index] and data["sfc_valid"][index])
        braking = planner.consume_recovery(data["navigation_state"][index], solved)
        if braking != bool(data["braking"][index]):
            raise RuntimeError("CTBR Actor and executed Planner recovery state disagree")
        cfg = self.ctbr_config
        action_lower = np.array([cfg.specific_thrust_limits[0], *(-v for v in cfg.body_rate_limits)])
        action_upper = np.array([cfg.specific_thrust_limits[1], *cfg.body_rate_limits])
        bad_action = np.any((action < action_lower - 1e-6) | (action > action_upper + 1e-6))
        # PPO may sample away from the optimized mean. Check the control that
        # will actually reach the motors, using the same raw Hamilton map.
        state = data["state"][index]
        qw, qx, qy, qz = state[3:7]
        body_z = np.array([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), qw * qw - qx * qx - qy * qy + qz * qz])
        force = float(action[0]) * cfg.mass
        acceleration = body_z * (force / cfg.mass) + np.array([0.0, 0.0, -cfg.gravity])
        velocity = state[7:10] + cfg.dt * acceleration
        bad_velocity = np.any(
            (velocity < np.asarray(cfg.velocity_lower) - 1e-5) | (velocity > np.asarray(cfg.velocity_upper) + 1e-5)
        )
        bad_acceleration = np.any(
            (acceleration < np.asarray(cfg.acceleration_lower) - 1e-5)
            | (acceleration > np.asarray(cfg.acceleration_upper) + 1e-5)
        )
        return bool(bad_action or bad_velocity or bad_acceleration)
