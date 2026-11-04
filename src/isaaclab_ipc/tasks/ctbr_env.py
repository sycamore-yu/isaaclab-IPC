"""Isaac Lab task adapting physical XYZW state to native CTBR MPC."""

import math
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch

from isaaclab_ipc.assets.quadrotor import MASS, MAX_THRUST
from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.ctbr_reference import (
    REFERENCE_DIAGNOSTIC_FIELDS as REFERENCE_DIAGNOSTICS,
)
from isaaclab_ipc.ctbr_reference import REFERENCE_GOVERNOR_VERSION, CTBRReferenceConfig, adapt_ctbr_reference
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
        shapes = {
            "ctbr_state": (10,),
            "ctbr_reference": (15, 10),
            "body_rates": (3,),
            "nominal_reference": (15, 9),
            "nominal_planes": (2, self.cfg.max_planes, 4),
            "nominal_plane_mask": (2, self.cfg.max_planes),
            "nominal_corridor_index": (15,),
            "ctbr_reference_diagnostics": (len(REFERENCE_DIAGNOSTICS),),
        }
        for name, shape in shapes.items():
            if name == "nominal_plane_mask":
                space = gym.spaces.Box(0, 1, shape=shape, dtype=np.bool_)
            elif name == "nominal_corridor_index":
                space = gym.spaces.Box(-1, 1, shape=shape, dtype=np.int64)
            else:
                space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
            self.single_observation_space[name] = space
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
            "reference_adaptation": {
                "version": REFERENCE_GOVERNOR_VERSION,
                "options": asdict(CTBRReferenceConfig()),
                "diagnostic_fields": list(REFERENCE_DIAGNOSTICS),
                "policy_features": CTBR_FEATURES,
                "scope": "Frozen rollout reference adaptation; actual measured state and original physical bounds are unchanged",
                "backup_scope": (
                    "Actual-state backups check the full original polytope. Virtual-speed backups additionally "
                    "stay within that same polytope's current horizontal clearance radius. These finite witnesses "
                    "do not certify the returned multi-polytope schedule, Newton tracking or future geometry."
                ),
                "reference_policy": "Conservative local horizontal maneuver reserve derived only from currently observed original SFC faces",
                "not_certified_meaning": "The particular backup was not certified, not a proof that the original problem is infeasible",
                "geometry_validity_unchanged": True,
            },
        }

    def _adapt_planning_contexts(self, contexts):
        """Freeze CTBR adaptation once during observation acquisition, before packing."""
        state, capture = self._ctbr_reference_capture
        actual = state.detach().cpu().numpy().copy()
        actual.flags.writeable = False
        nominal_planes = np.zeros((self.num_envs, 2, self.cfg.max_planes, 4))
        nominal_mask = np.zeros(nominal_planes.shape[:-1], dtype=bool)
        nominal_index = np.full((self.num_envs, self.ctbr_config.horizon), -1, dtype=np.int64)
        nominal_reference, adapted, diagnostics = [], [], []
        for i, (context, planner) in enumerate(zip(contexts, self.planners, strict=True)):
            original_state, original_valid = context.state.copy(), context.valid
            nominal_reference.append(context.reference.copy())
            if context.corridors:
                nominal_planes[i], nominal_mask[i], nominal_index[i] = pack_corridors(
                    context.corridors, self.cfg.max_planes
                )
            original_path = planner.path.copy()
            original_path.flags.writeable = False
            updated, audit = adapt_ctbr_reference(
                context, actual[i], self.ctbr_config, path=original_path, nearest=planner.nearest
            )
            if (
                not np.array_equal(context.state, original_state)
                or not np.array_equal(updated.state, original_state)
                or context.valid != original_valid
                or updated.valid != original_valid
            ):
                raise RuntimeError("CTBR reference adaptation must preserve measured state and geometry validity")
            adapted.append(updated)
            diagnostics.append(audit)
        capture.update(
            nominal_reference=torch.as_tensor(np.stack(nominal_reference), device=self.device),
            nominal_planes=torch.as_tensor(nominal_planes, device=self.device),
            nominal_plane_mask=torch.as_tensor(nominal_mask, device=self.device),
            nominal_corridor_index=torch.as_tensor(nominal_index, device=self.device),
            ctbr_reference_diagnostics=torch.tensor(
                [[item[key] for key in REFERENCE_DIAGNOSTICS] for item in diagnostics],
                dtype=torch.float64,
                device=self.device,
            ),
        )
        # Detailed local actual/virtual backup traces are last-call diagnostics,
        # not policy inputs or a replacement for the frozen observation fields.
        self.ctbr_reference_audit = diagnostics
        return adapted

    def _get_observations(self):
        pose, twist = self.body_state()
        state = state_from_isaac(pose, twist, self.scene.env_origins)
        body_rates = body_rates_from_isaac(pose, twist)
        capture = {}
        self._ctbr_reference_capture = (state, capture)
        try:
            result = super()._get_observations()
        finally:
            del self._ctbr_reference_capture
        result.update(capture)
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
