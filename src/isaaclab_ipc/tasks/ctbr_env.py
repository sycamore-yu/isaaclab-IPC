"""Newton CTBR training with frozen native PathCover observations.

The inherited IPCEnv supplies only the existing robot, physics and sensors.
The AC-MPC task has no IPC Planner, jerk model or recovery state machine.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from isaaclab_ipc.assets.quadrotor import MASS, MAX_THRUST
from isaaclab_ipc.rl.ctbr_observation import ctbr_feature_size, make_ctbr_observation, terminal_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig
from isaaclab_ipc.turbompc.quadrotor import body_rates_from_isaac, state_from_isaac

from .env import IPCEnv, make_cfg, make_forest_cfg
from .rewards import NavigationRewardCfg, navigation_reward


@torch.no_grad()
def checked_articulation_mass(robot, expected_mass: float) -> torch.Tensor:
    """Read the full articulated robot mass, including fixed rotor links."""
    link_masses = robot.data.body_mass.torch
    if tuple(link_masses.shape) != (robot.num_instances, robot.num_bodies):
        raise ValueError("Newton articulation masses must include every body")
    masses = link_masses.sum(dim=-1)
    if not math.isfinite(expected_mass) or expected_mass <= 0 or not torch.allclose(
        masses, torch.full_like(masses, expected_mass), atol=1e-6, rtol=0
    ):
        raise ValueError(f"Newton and CTBR masses differ: actual={masses.cpu().tolist()}, expected={expected_mass}")
    return masses


def _configure_ctbr(cfg, frontend):
    model = CTBRConfig(mass=MASS)
    cfg.ctbr_model = asdict(model)
    cfg.frontend = frontend
    cfg.pathcover_config = {}
    cfg.max_planes = 0 if frontend == "none" else 128
    cfg.reward = NavigationRewardCfg()
    cfg.observation_space = ctbr_feature_size(model.cost_nodes)
    cfg.thrust_limits = model.specific_thrust_limits
    # Policy/MPC at 10 Hz. Existing motor feedback runs every 300 Hz physics step.
    cfg.decimation = 30
    cfg.scene.num_envs = 4
    return cfg


def make_ctbr_cfg():
    """A0: standalone AC-MPC in an empty Newton scene."""
    return _configure_ctbr(make_cfg(num_envs=4, scenario="empty"), "none")


def make_avoid_ctbr_cfg():
    """A1/A2: the same CTBR actor with native PathCover and Mid-360."""
    cfg = _configure_ctbr(make_cfg(num_envs=4, scenario="avoid"), "pathcover")
    cfg.pathcover_config = {"bounds_lower": (-2.0, -4.0, 0.0), "bounds_upper": (12.0, 4.0, 4.0)}
    return cfg


def make_forest_ctbr_cfg():
    return _configure_ctbr(make_forest_cfg(), "pathcover")


def pathcover_config_for_ctbr(model, overrides):
    """Use the controller's physical model and optionally stricter planning limits."""
    from isaaclab_ipc.pathcover_native import PathCoverConfig

    options = dict(overrides)
    for name in ("mass", "gravity"):
        value = getattr(model, name)
        if name in options and not math.isclose(options[name], value, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"PathCover {name} must equal the CTBR model")
        options[name] = value
    upper_limits = {
        "body_rate_max": min(model.body_rate_limits[:2]),
        "body_yaw_rate_max": model.body_rate_limits[2],
        "max_velocity": min(*model.velocity_upper, *(-value for value in model.velocity_lower)),
    }
    for name, cap in upper_limits.items():
        value = options.get(name)
        if value is None:
            value = cap
        if value > cap:
            raise ValueError(f"PathCover {name} exceeds the CTBR limit {cap}")
        options[name] = value
    reference = PathCoverConfig(mass=model.mass, gravity=model.gravity, max_velocity=options["max_velocity"])
    options.setdefault("nominal_velocity", reference.nominal_velocity)
    floor = model.mass * model.specific_thrust_limits[0]
    cap = model.mass * model.specific_thrust_limits[1]
    if options.get("thrust_min") is None:
        options["thrust_min"] = max(reference.thrust_min, floor)
    if options.get("thrust_max") is None:
        options["thrust_max"] = cap
    if options["thrust_min"] < floor:
        raise ValueError("PathCover thrust_min is below the CTBR minimum")
    if options["thrust_max"] > cap:
        raise ValueError("PathCover thrust_max exceeds the CTBR maximum")
    return PathCoverConfig(**options)


class CTBRMPCEnv(IPCEnv):
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        nodes, faces = self.ctbr_config.cost_nodes, self.cfg.max_planes
        shapes = {
            "ctbr_state": (10,), "body_rates": (3,), "goal": (3,),
            "ctbr_reference": (nodes, 10), "ctbr_control_reference": (nodes, 4),
            "ctbr_warm_start": (nodes - 1, 4), "reference": (nodes, 9),
            "planes": (nodes, faces, 4), "plane_mask": (nodes, faces),
            "sfc_valid": (1,), "yaw_target": (1,),
        }
        for name, shape in shapes.items():
            boolean = name in ("plane_mask", "sfc_valid")
            self.single_observation_space[name] = gym.spaces.Box(
                0 if boolean else -np.inf, 1 if boolean else np.inf,
                shape=shape, dtype=np.bool_ if boolean else np.float64,
            )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Hydra parses an unquoted CLI `frontend=none` as Python None.
        if cfg.frontend is None:
            cfg.frontend = "none"
        self.ctbr_config = CTBRConfig(**cfg.ctbr_model)
        if tuple(cfg.sim.gravity) != (0.0, 0.0, -self.ctbr_config.gravity):
            raise ValueError("CTBR gravity must match Newton gravity")
        cfg.observation_space = ctbr_feature_size(self.ctbr_config.cost_nodes)
        super().__init__(cfg, render_mode, **kwargs)
        masses = checked_articulation_mass(self.robot, self.ctbr_config.mass)
        self.navigation = None
        frontend_config = None
        if cfg.frontend == "pathcover":
            from isaaclab_ipc.ctbr_navigation import PathCoverNavigation

            frontend_config = pathcover_config_for_ctbr(self.ctbr_config, cfg.pathcover_config)
            self.navigation = PathCoverNavigation(
                self.num_envs, self.ctbr_config, frontend_config,
                device=self.device, max_planes=cfg.max_planes,
            )
        elif cfg.frontend != "none":
            raise ValueError("CTBR frontend must be none or pathcover")
        self.records, self.observation_times = [], []
        self.frontend_times = [] if self.navigation is None else self.navigation.frontend_times_ms
        self.frontend_batch_times = [] if self.navigation is None else self.navigation.frontend_batch_times_ms
        self.current_context, self._pending, self.last_mpc_result = None, None, None
        self.reported = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._next_context = None
        self._next_context_valid = torch.zeros_like(self.reported)
        self._reset_context = torch.ones_like(self.reported)
        self.frontend_failure = torch.zeros_like(self.reported)
        self._frontend_errors = {}
        self.reward_active = torch.ones_like(self.reported)
        self.episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_record_index = np.full(self.num_envs, -1, dtype=np.int64)
        self.previous_action = self.controller.command.clone()
        self.reward_sums = torch.zeros((self.num_envs, 4), device=self.device)
        self.command_violations = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.qp_failures = torch.zeros_like(self.command_violations)
        self.qp_residual_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.sfc_violation_max = torch.zeros_like(self.qp_residual_max)
        self.speed_max = torch.zeros_like(self.qp_residual_max)
        self._warm_controls = torch.zeros(
            (self.num_envs, self.ctbr_config.horizon, 4), dtype=torch.float64, device=self.device
        )
        self._warm_controls[..., 0] = self.ctbr_config.mass * self.ctbr_config.gravity
        self._action_lower = torch.tensor(
            [self.ctbr_config.specific_thrust_limits[0], *(-v for v in self.ctbr_config.body_rate_limits)],
            device=self.device,
        )
        self._action_upper = torch.tensor(
            [self.ctbr_config.specific_thrust_limits[1], *self.ctbr_config.body_rate_limits], device=self.device
        )
        self.ctbr_runtime_contract = {
            "state_quaternion": "Isaac XYZW -> CTBR WXYZ",
            "position_frame": "world axes translated by environment origin",
            "velocity_frame": "world",
            "angular_velocity_frame": "body",
            "newton_articulation_mass_kg": masses.cpu().tolist(),
            "physical_total_thrust_capacity_N": 4 * MAX_THRUST,
            "model": asdict(self.ctbr_config),
            "policy_period_s": self.step_dt,
            "physics_period_s": self.cfg.sim.dt,
            "frontend": cfg.frontend,
            "frontend_config": None if frontend_config is None else asdict(frontend_config),
            "cost_outputs": self.ctbr_config.cost_dim,
        }

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.ids if env_ids is None else env_ids
        self.previous_action[ids] = self.controller.command[ids]
        self.reward_sums[ids] = 0
        self.reward_active[ids] = True
        self.reported[ids] = False
        self.command_violations[ids] = 0
        self.qp_failures[ids] = 0
        self.qp_residual_max[ids] = 0
        self.sfc_violation_max[ids] = 0
        self.speed_max[ids] = 0
        self._warm_controls[ids] = 0
        self._warm_controls[ids, :, 0] = self.ctbr_config.mass * self.ctbr_config.gravity
        host_ids = ids.cpu().numpy()
        self._reset_context[ids] = True
        self.episode_ids[host_ids] += 1
        self.episode_record_index[host_ids] = -1
        if self.navigation is not None:
            self.navigation.reset(host_ids)

    def _save_frontend_failure(self, state, rates, acceleration, goals, clouds, errors):
        if not self.cfg.log_dir:
            return
        path = Path(self.cfg.log_dir) / "failed-frontend-input.pt"
        # Keep the first exact problem; every later failure is retained in its
        # episode record without copying the accumulated map at every step.
        if path.exists():
            return
        snapshot = {
            "exception": "; ".join(f"env {index}: {error}" for index, error in errors.items()),
            "errors": {index: str(error) for index, error in errors.items()},
            "ctbr_state": state.cpu(), "body_rates": rates.cpu(),
            "acceleration": acceleration.cpu(), "goal": goals.cpu(),
            "time": self.elapsed.cpu(), "clouds": clouds,
            "warm_start_controls": self._warm_controls.cpu().clone(),
            "model_config": asdict(self.ctbr_config),
            "frontend_configs": [asdict(frontend.config) for frontend in self.navigation.frontends],
            "gcopter_problems": [frontend.failed_problem() for frontend in self.navigation.frontends],
            "inflated_clouds": [frontend.inflated_cloud() for frontend in self.navigation.frontends],
            "obstacle_boxes": [frontend.obstacle_boxes() for frontend in self.navigation.frontends],
            "previous_plans": self.navigation.last_plans,
        }
        torch.save(snapshot, path)

    def _observe_pathcover(self, env_ids=None, *, allow_failures=False):
        tick = perf_counter()
        pose, twist = self.body_state()
        state = state_from_isaac(pose, twist, self.scene.env_origins)
        rates = body_rates_from_isaac(pose, twist)
        goals = self.goal.expand(self.num_envs, 3) if self.goal.ndim == 1 else self.goal
        ids = self.ids if env_ids is None else env_ids
        host_ids = ids.cpu().tolist()
        clouds = dict(zip(host_ids, self.lidar_clouds(ids), strict=True))
        acceleration = self.acceleration()
        try:
            result = self.navigation.observe(
                state, rates, acceleration, goals, clouds, self.elapsed,
                warm_start_controls=self._warm_controls, env_ids=host_ids,
                allow_failures=allow_failures,
            )
        except (RuntimeError, ValueError) as error:
            self._save_frontend_failure(
                state, rates, acceleration, goals, clouds, {index: error for index in host_ids},
            )
            raise
        finally:
            self.observation_times.append(1000 * (perf_counter() - tick))
        if allow_failures and result[1]:
            self._save_frontend_failure(state, rates, acceleration, goals, clouds, result[1])
        return result

    def _get_observations(self):
        if self.navigation is None:
            tick = perf_counter()
            pose, twist = self.body_state()
            state = state_from_isaac(pose, twist, self.scene.env_origins)
            rates = body_rates_from_isaac(pose, twist)
            goals = self.goal.expand(self.num_envs, 3) if self.goal.ndim == 1 else self.goal
            nodes = self.ctbr_config.cost_nodes
            reference = state.new_zeros((self.num_envs, nodes, 9))
            reference[..., :3] = goals[:, None]
            delta = goals - state[:, :3]
            yaw = torch.atan2(delta[:, 1], delta[:, 0])[:, None]
            planes = state.new_zeros((self.num_envs, nodes, 0, 4))
            mask = torch.zeros(planes.shape[:-1], dtype=torch.bool, device=self.device)
            result = make_ctbr_observation(
                state, rates, goals, reference, yaw, planes, mask,
                ctbr_config=self.ctbr_config, warm_start_controls=self._warm_controls,
            )
            self.observation_times.append(1000 * (perf_counter() - tick))
        elif not self.cfg.automatic_reset and bool(self.reported.any()):
            # A completed episode needs a terminal measurement, not another
            # successful plan. Other evaluation rows still fail strictly.
            tick = perf_counter()
            pose, twist = self.body_state()
            result = terminal_ctbr_observation(
                self.current_context, state_from_isaac(pose, twist, self.scene.env_origins),
                body_rates_from_isaac(pose, twist), ctbr_config=self.ctbr_config,
            )
            self.observation_times.append(1000 * (perf_counter() - tick))
            active_ids = (~self.reported).nonzero().flatten()
            if len(active_ids):
                active_context = self._observe_pathcover(active_ids)
                for name, value in result.items():
                    value[active_ids] = active_context[name]
        elif self._next_context is None:
            # Initial/reset acquisition and fixed-cost evaluation are strict.
            result = self._observe_pathcover()
            self._reset_context.zero_()
        else:
            reset_ids = self._reset_context.nonzero().flatten()
            if len(reset_ids):
                reset_context = self._observe_pathcover(reset_ids)
                for name, value in self._next_context.items():
                    value[reset_ids] = reset_context[name]
                self._next_context_valid[reset_ids] = True
                self._reset_context[reset_ids] = False
            if not bool(self._next_context_valid.all()):
                raise RuntimeError("Reset failed planning rows before publishing the next CTBR observation")
            result, self._next_context = self._next_context, None
        self.current_context = result
        return result

    def record_mpc_result(self, result):
        if self._pending is not None:
            raise RuntimeError("Previous MPC result has not been consumed by env.step")
        if result.get("model") != "ctbr" or result["controls"].shape != self._warm_controls.shape:
            raise ValueError("CTBR task requires a matching physical control sequence")
        self._pending = result

    def _pre_physics_step(self, actions):
        if self._pending is None:
            raise RuntimeError("Supply actor.last_result before env.step")
        self.last_mpc_result, self._pending = self._pending, None
        result = self.last_mpc_result
        active = ~self.reported
        self.command_violations += active & (
            (actions < self._action_lower - 1e-5) | (actions > self._action_upper + 1e-5)
        ).any(-1)
        self.qp_failures += active & ~result["solved"]
        self.qp_residual_max = torch.maximum(
            self.qp_residual_max, torch.where(active, result["primal"], 0.0)
        )
        # Recede only the initializer; every new problem integrates from measurement.
        shift = self.step_dt / self.ctbr_config.dt
        offset = torch.arange(self.ctbr_config.horizon, device=self.device) + shift
        left = offset.floor().long().clamp_max(self.ctbr_config.horizon - 1)
        right = (left + 1).clamp_max(self.ctbr_config.horizon - 1)
        fraction = (offset - offset.floor())[None, :, None]
        controls = result["controls"]
        self._warm_controls.copy_(controls[:, left] * (1 - fraction) + controls[:, right] * fraction)
        super()._pre_physics_step(actions.clamp(self._action_lower, self._action_upper))

    def _get_dones(self):
        term, trunc = super()._get_dones()
        pose, twist = self.body_state()
        self.speed_max = torch.maximum(self.speed_max, twist[:, :3].double().norm(dim=-1))
        if self.current_context is not None and self.current_context["planes"].shape[-2]:
            index = min(round(self.step_dt / self.ctbr_config.dt), self.ctbr_config.cost_nodes - 1)
            planes = self.current_context["planes"][:, index]
            mask = self.current_context["plane_mask"][:, index]
            position = pose[:, :3] - self.scene.env_origins
            violation = (planes[..., :3] * position[:, None]).sum(-1) + planes[..., 3]
            violation = violation.masked_fill(~mask, 0).clamp_min(0).amax(-1)
            self.sfc_violation_max = torch.maximum(self.sfc_violation_max, violation)
        self.frontend_failure.zero_()
        self._frontend_errors = {}
        if self.navigation is not None and self.cfg.automatic_reset:
            # Compute the next planning problem before reward/done are returned.
            # Never mutate current_context: PPO still owns that pre-action batch.
            self._next_context = {name: torch.empty_like(value) for name, value in self.current_context.items()}
            self._next_context_valid.zero_()
            continuing = (~(term | trunc)).nonzero().flatten()
            if len(continuing):
                context, errors = self._observe_pathcover(continuing, allow_failures=True)
                for name, value in self._next_context.items():
                    value[continuing] = context[name]
                self._next_context_valid[continuing] = True
                if errors:
                    failed_ids = list(errors)
                    self.frontend_failure[failed_ids] = True
                    self._next_context_valid[failed_ids] = False
                    self._frontend_errors = {index: str(error) for index, error in errors.items()}
                term = term | self.frontend_failure
        self.extras["frontend_failure"] = self.frontend_failure.clone()
        self.extras["frontend_errors"] = dict(self._frontend_errors)
        for i in ((term | trunc) & ~self.reported).nonzero().flatten().cpu().tolist():
            self.reported[i] = True
            collision = bool(self.hit[i])
            reached = bool(self.extras["success"][i])
            self.episode_record_index[i] = len(self.records)
            self.records.append(dict(
                env_id=i, episode_id=int(self.episode_ids[i]), scenario=self.cfg.scenario,
                frontend=self.cfg.frontend, success=reached and not collision,
                goal_crossed=reached, collision=collision, timeout=bool(trunc[i]),
                frontend_failure=bool(self.frontend_failure[i]),
                frontend_failure_reason=self._frontend_errors.get(i),
                steps=int(self.episode_length_buf[i]), elapsed_s=float(self.elapsed[i]),
                violations=int(self.command_violations[i]), qp_failures=int(self.qp_failures[i]),
                maximum_qp_primal_residual=float(self.qp_residual_max[i]),
                maximum_sfc_violation_m=float(self.sfc_violation_max[i]),
                maximum_speed_m_s=float(self.speed_max[i]),
                minimum_physics_step_clearance_m=float(self.min_clearance[i]),
                final_position=self.extras["terminal_position"][i].cpu().tolist(),
            ))
        return term, trunc

    def _get_rewards(self):
        pose, _ = self.body_state()
        distance = (pose[:, :3] - self.scene.env_origins - self.goal).norm(dim=-1)
        collision = self.hit & self.reward_active
        success = self.extras["success"] & ~collision
        done = self.reset_terminated | self.reset_time_outs
        failure = done & ~success & ~collision
        terms = navigation_reward(
            self.cfg.reward, self.previous_distance, distance, self.controller.command, self.previous_action,
            self.step_dt, success, collision, failure,
        ) * self.reward_active[:, None]
        self.reward_sums += terms
        names = ("progress", "time", "smooth", "terminal")
        self.extras["reward_terms"] = dict(zip(names, terms.unbind(-1), strict=True))
        self.extras["log"] = {"Reward/" + name: value.mean() for name, value in self.extras["reward_terms"].items()}
        self.extras["log"]["Control/solved"] = self.last_mpc_result["solved"].float().mean()
        self.extras["log"]["Frontend/failure"] = self.frontend_failure.float().mean()
        completed = (done & self.reward_active).nonzero().flatten().cpu().tolist()
        for i in completed:
            record = self.records[self.episode_record_index[i]]
            record["reward_terms"] = dict(zip(names, self.reward_sums[i].cpu().tolist(), strict=True))
            record["episode_return"] = float(self.reward_sums[i].sum())
        if completed and self.cfg.log_dir:
            path = Path(self.cfg.log_dir) / "records.json"
            path.write_text(json.dumps(self.records, indent=2))
        self.previous_distance.copy_(distance)
        self.previous_action.copy_(self.controller.command)
        self.reward_active &= ~done
        self.recorder_manager.record_post_step()
        return terms.sum(-1)

    def close(self):
        if getattr(self, "navigation", None) is not None:
            self.navigation.close()
        super().close()
