"""Standard DirectRLEnv task exposing immutable IPC context to the MPC actor."""

import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from isaaclab_ipc.corridor_storage import pack_corridors
from isaaclab_ipc.frontend import PlanningBatch
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner

from .env import IPCEnv, make_cfg
from .rewards import NavigationRewardCfg, navigation_reward

FEATURES = 124
MAX_PLANES = 64


def make_mpc_cfg():
    cfg = make_cfg(num_envs=4)
    cfg.reference_strides = (10, 20)
    cfg.reward = NavigationRewardCfg()
    cfg.max_planes = MAX_PLANES
    cfg.observation_space = FEATURES
    return cfg


def make_forest_mpc_cfg():
    path = Path(__file__).resolve().parents[3] / "build/forest/forest.usdc"
    if not path.is_file():
        raise FileNotFoundError("Build the forest with scripts/build_forest.py first")
    cfg = make_cfg(num_envs=4, scenario="forest", forest_usd=path)
    cfg.reference_strides = (6,)
    cfg.reward = NavigationRewardCfg()
    cfg.max_planes = MAX_PLANES
    cfg.observation_space = FEATURES
    return cfg


class IPCMPCEnv(IPCEnv):
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        shapes = dict(
            mpc_state=(9,),
            reference=(15, 9),
            planes=(2, self.cfg.max_planes, 4),
            plane_mask=(2, self.cfg.max_planes),
            corridor_index=(15,),
            rotation=(9,),
            braking=(1,),
            yaw_target=(1,),
            sfc_valid=(1,),
        )
        for name, shape in shapes.items():
            if name in ("plane_mask", "braking", "sfc_valid"):
                space = gym.spaces.Box(0, 1, shape=shape, dtype=np.bool_)
            elif name == "corridor_index":
                space = gym.spaces.Box(-1, 1, shape=shape, dtype=np.int64)
            else:
                dtype = np.float32 if name == "rotation" else np.float64
                space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=dtype)
            self.single_observation_space[name] = space
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.planners = [None] * self.num_envs
        self.planning_batch = PlanningBatch(cfg.planner_workers)
        self.records = []
        self.frontend_times = []
        self._pending = None
        self.command_violations = np.zeros(self.num_envs, int)
        self.thrust_violations = np.zeros(self.num_envs, int)
        self.state_violations = np.zeros(self.num_envs, int)
        self.qp_residual_max = np.zeros(self.num_envs)
        self.qp_failures = np.zeros(self.num_envs, int)
        self.accepted_qp_residual_max = np.zeros(self.num_envs)
        self.accepted_qp_steps = np.zeros(self.num_envs, int)
        self.sfc_constrained_steps = np.zeros(self.num_envs, int)
        self.maximum_sfc_planes = np.zeros(self.num_envs, int)
        self.reported = np.zeros(self.num_envs, bool)
        self.episode_ids = np.zeros(self.num_envs, int)
        self.current_context = None
        weights = [getattr(cfg.reward, key) for key in ("progress", "time", "smooth", "goal", "collision", "failure")]
        if not np.isfinite(weights).all() or np.any(np.asarray(weights) < 0):
            raise ValueError("Reward weights must be finite and nonnegative")
        scales = np.asarray(cfg.reward.action_scale)
        if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Reward action_scale must contain four finite positive values")
        self.previous_action = self.controller.command.clone()
        self.reward_sums = torch.zeros(self.num_envs, 4, device=self.device)
        self.reward_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_record_index = np.full(self.num_envs, -1, dtype=int)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.ids if env_ids is None else env_ids
        self.previous_action[ids] = self.controller.command[ids]
        self.reward_sums[ids] = 0
        self.reward_active[ids] = True
        self.episode_record_index[ids.cpu().numpy()] = -1
        for i in ids.cpu().tolist():
            stride = self.cfg.reference_strides[i % len(self.cfg.reference_strides)]
            self.planners[i] = Planner(Config.load(self.cfg.scenario, stride, profile="real"), [self.cfg.goal_x, 0, 1])
            self.command_violations[i] = self.thrust_violations[i] = self.state_violations[i] = 0
            self.qp_residual_max[i] = 0
            self.qp_failures[i] = 0
            self.accepted_qp_residual_max[i] = 0
            self.accepted_qp_steps[i] = 0
            self.sfc_constrained_steps[i] = 0
            self.maximum_sfc_planes[i] = 0
            self.reported[i] = False
            self.episode_ids[i] += 1

    def _get_observations(self):
        tick = perf_counter()
        physical = super()._get_observations()["policy"]
        host_state = torch.cat((physical, self.elapsed[:, None]), -1).cpu().numpy()
        values, elapsed = host_state[:, :18], host_state[:, 18]
        clouds = {}
        ticks = np.rint(elapsed * 100).astype(int)
        ids = np.flatnonzero(ticks % 10 == 0)
        if len(ids):
            clouds.update(zip(ids, self.lidar_clouds(ids), strict=True))
        planes = np.zeros((self.num_envs, 2, self.cfg.max_planes, 4))
        valid = np.zeros(planes.shape[:-1], bool)
        corridor_indices = np.full((self.num_envs, 15), -1, dtype=np.int64)
        sfc_valid = np.ones((self.num_envs, 1), bool)
        refs, states, margins, braking, yaw = [], [], [], [], []
        contexts = self.planning_batch.contexts(self.planners, values, clouds)
        for i, (planner, state, context) in enumerate(zip(self.planners, values, contexts, strict=True)):
            x, ref, corridor = context.state, context.reference, context.corridors
            sfc_valid[i] = context.valid
            clearance = np.full(15, 10.0)
            if corridor:
                planes[i], valid[i], corridor_indices[i] = pack_corridors(corridor, self.cfg.max_planes)
            for k, faces in enumerate(corridor):
                if len(faces):
                    faces = np.asarray(faces)[::-1]
                    clearance[k] = np.min(
                        -(faces[:, :3] @ ref[k, :3] + faces[:, 3])
                        / np.maximum(np.linalg.norm(faces[:, :3], axis=-1), 1e-12)
                    )
            states.append(x)
            refs.append(ref)
            margins.append(clearance)
            braking.append([planner.braking])
            yaw.append(
                [
                    np.arctan2(planner.goal[1] - state[1], planner.goal[0] - state[0])
                    if planner.cfg.params["yaw_ctrl_flag"]
                    else 0.0
                ]
            )

        def tensor(v, dtype=torch.float64):
            return torch.as_tensor(np.asarray(v), device=self.device, dtype=dtype)

        reference = tensor(refs)
        # Fixed scales apply only to network features, never to solver inputs.
        features = torch.cat(
            (
                physical[:, 3:6] / 10,
                physical[:, 6:9] / 20,
                physical[:, 9:],
                (self.goal - physical[:, :3]) / 10,
                (reference[:, :, :3] - physical[:, None, :3]).flatten(1) / 10,
                reference[:, :, 3:6].flatten(1) / 10,
                tensor(margins).clamp(-10, 10) / 10,
                tensor(braking),
            ),
            -1,
        ).float()
        result = dict(
            policy=features,
            mpc_state=tensor(states),
            reference=reference,
            planes=tensor(planes),
            plane_mask=tensor(valid, torch.bool),
            corridor_index=tensor(corridor_indices, torch.int64),
            rotation=physical[:, 9:].clone(),
            braking=tensor(braking, torch.bool),
            yaw_target=tensor(yaw),
            sfc_valid=tensor(sfc_valid, torch.bool),
        )
        self.current_context = result
        self.frontend_times.append((perf_counter() - tick) * 1000)
        return result

    def close(self):
        if hasattr(self, "planning_batch"):
            self.planning_batch.close()
        super().close()

    def record_mpc_result(self, result):
        """Supply diagnostics for the next env.step; PPO updates never call this."""
        if self._pending is not None:
            raise RuntimeError("Previous MPC result has not been consumed by env.step")
        self._pending = result

    def _consume_control_result(self, planner, data, index, time, action):
        """Consume the IPC jerk schema and return its command-bound violation."""
        status = "solved" if data["solved"][index] else "max iterations reached"
        if not data["sfc_valid"][index]:
            status = "invalid SFC"
        planner.consume_state(data["state"][index], data["solution"][index], status, time)
        val = np.r_[data["predicted"][index, 3:], data["jerk"][index]]
        low = np.r_[planner.mpc.lower, planner.mpc.jlow[:3]]
        high = np.r_[planner.mpc.upper, planner.mpc.jhigh[:3]]
        return bool(np.any((val < low - 0.001 - 0.001 * abs(low)) | (val > high + 0.001 + 0.001 * abs(high))))

    def _pre_physics_step(self, actions):
        if self._pending is None:
            raise RuntimeError("Call record_mpc_result(actor.last_result) before env.step")
        result, self._pending = self._pending, None
        self.last_mpc_result = result
        data = {k: v.cpu().numpy() for k, v in result.items() if isinstance(v, torch.Tensor)}
        elapsed = self.elapsed.cpu().numpy()
        sampled = actions.cpu().numpy()
        schedule = self.current_context["corridor_index"]
        face_counts = self.current_context["plane_mask"].sum(-1).gather(1, schedule.clamp_min(0))
        max_faces = face_counts.masked_fill(schedule < 0, 0).amax(-1).cpu().numpy()
        for i, planner in enumerate(self.planners):
            violation = self._consume_control_result(planner, data, i, float(elapsed[i]), sampled[i])
            if not self.reported[i]:
                self.sfc_constrained_steps[i] += max_faces[i] > 0
                self.maximum_sfc_planes[i] = max(self.maximum_sfc_planes[i], int(max_faces[i]))
                self.command_violations[i] += violation
                self.thrust_violations[i] += not (0.08 * 9.81 / 0.7 <= sampled[i, 0] <= 0.9 * 9.81 / 0.7)
                if data["sfc_valid"][i]:
                    self.qp_failures[i] += not data["solved"][i]
                    self.qp_residual_max[i] = max(self.qp_residual_max[i], float(data["primal"][i]))
                    if data["solved"][i] and not data["braking"][i]:
                        self.accepted_qp_steps[i] += 1
                        self.accepted_qp_residual_max[i] = max(
                            self.accepted_qp_residual_max[i], float(data["primal"][i])
                        )
        super()._pre_physics_step(actions)

    def _get_rewards(self):
        pose, _ = self.body_state()
        distance = (pose[:, :3] - self.scene.env_origins - self.goal).norm(dim=-1)
        collision = self.hit & self.reward_active
        success = (
            self.extras["success"] & ~collision & torch.as_tensor(self.command_violations == 0, device=self.device)
        )
        done = self.reset_terminated | self.reset_time_outs
        failure = done & ~success & ~collision
        terms = (
            navigation_reward(
                self.cfg.reward,
                self.previous_distance,
                distance,
                self.controller.command,
                self.previous_action,
                self.step_dt,
                success,
                collision,
                failure,
            )
            * self.reward_active[:, None]
        )
        self.reward_sums += terms
        names = ("progress", "time", "smooth", "terminal")
        self.extras["reward_terms"] = dict(zip(names, terms.unbind(-1), strict=True))
        self.extras.setdefault("log", {}).update(
            {"Reward/" + name: value.mean() for name, value in self.extras["reward_terms"].items()}
        )
        self.extras["log"].update(
            {"Control/" + key: self.last_mpc_result[key].float().mean() for key in ("solved", "braking", "sfc_valid")}
        )
        for i in (done & self.reward_active).nonzero().flatten().cpu().tolist():
            record = self.records[self.episode_record_index[i]]
            record["reward_terms"] = dict(zip(names, self.reward_sums[i].cpu().tolist(), strict=True))
            record["episode_return"] = float(self.reward_sums[i].sum())
        if (done & self.reward_active).any() and self.cfg.log_dir:
            path = Path(self.cfg.log_dir) / "records.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.records, indent=2))
            temporary.replace(path)
        self.previous_distance.copy_(distance)
        self.previous_action.copy_(self.controller.command)
        self.reward_active &= ~done
        self.recorder_manager.record_post_step()
        return terms.sum(-1)

    def _get_dones(self):
        term, trunc = super()._get_dones()
        pose, twist = self.body_state()
        actual = torch.cat((twist[:, :3], self.acceleration()), -1).cpu().numpy()
        for i, planner in enumerate(self.planners):
            if self.reported[i]:
                continue
            self.state_violations[i] += bool(
                np.any((actual[i] < planner.mpc.lower - 0.02) | (actual[i] > planner.mpc.upper + 0.02))
            )
        for i in (term | trunc).nonzero().flatten().cpu().tolist():
            if self.reported[i]:
                continue
            self.reported[i] = True
            reached = bool(self.extras["success"][i])
            collision = bool(self.extras["collision"][i])
            planner = self.planners[i]
            self.episode_record_index[i] = len(self.records)
            self.records.append(
                dict(
                    env_id=i,
                    episode_id=int(self.episode_ids[i]),
                    stride=int(planner.cfg.params["fsm"]["ref_dis"]),
                    scenario=self.cfg.scenario,
                    success=bool(reached and not collision and self.command_violations[i] == 0),
                    goal_crossed=reached,
                    collision=collision,
                    timeout=bool(trunc[i]),
                    steps=int(self.episode_length_buf[i]),
                    violations=int(self.command_violations[i]),
                    sampled_thrust_limit_steps=int(self.thrust_violations[i]),
                    state_bound_steps=int(self.state_violations[i]),
                    qp_failures=int(self.qp_failures[i]),
                    frontend_failures=planner.frontend_failures,
                    recovery_events=planner.recovery_events,
                    brake_steps=planner.brake_steps,
                    maximum_qp_primal_residual=float(self.qp_residual_max[i]),
                    accepted_qp_steps=int(self.accepted_qp_steps[i]),
                    maximum_accepted_qp_primal_residual=float(self.accepted_qp_residual_max[i]),
                    sfc_constrained_steps=int(self.sfc_constrained_steps[i]),
                    maximum_sfc_planes=int(self.maximum_sfc_planes[i]),
                    minimum_physics_step_clearance_m=float(self.min_clearance[i]),
                    final_position=self.extras["terminal_position"][i].cpu().tolist(),
                )
            )
        if (term | trunc).any():
            recent = [r for r in self.records[-self.num_envs :] if r["success"]]
            self.extras["log"] = {"Episode/success_fraction": len(recent) / min(len(self.records), self.num_envs)}
        return term, trunc
