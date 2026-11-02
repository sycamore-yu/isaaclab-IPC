"""Standard DirectRLEnv task exposing immutable IPC context to the MPC actor."""

import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch

from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import InvalidCorridorError, Planner

from .env import IPCEnv, make_cfg

FEATURES = 124
MAX_PLANES = 32


def make_mpc_cfg():
    cfg = make_cfg(num_envs=4)
    cfg.reference_strides = (10, 20)
    cfg.max_planes = MAX_PLANES
    cfg.observation_space = FEATURES
    return cfg


def make_forest_mpc_cfg():
    path = Path(__file__).resolve().parents[3] / "build/forest/forest.usdc"
    if not path.is_file():
        raise FileNotFoundError("Build the forest with scripts/build_forest.py first")
    cfg = make_cfg(num_envs=4, scenario="forest", forest_usd=path)
    cfg.reference_strides = (6,)
    cfg.max_planes = MAX_PLANES
    cfg.observation_space = FEATURES
    return cfg


class IPCMPCEnv(IPCEnv):
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        shapes = dict(
            mpc_state=(9,),
            reference=(15, 9),
            planes=(15, self.cfg.max_planes, 4),
            plane_mask=(15, self.cfg.max_planes),
            rotation=(9,),
            braking=(1,),
            yaw_target=(1,),
            sfc_valid=(1,),
        )
        for name, shape in shapes.items():
            if name in ("plane_mask", "braking", "sfc_valid"):
                space = gym.spaces.Box(0, 1, shape=shape, dtype=np.bool_)
            else:
                dtype = np.float32 if name == "rotation" else np.float64
                space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=dtype)
            self.single_observation_space[name] = space
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.planners = [None] * self.num_envs
        self.records = []
        self.frontend_times = []
        self._pending = None
        self.command_violations = np.zeros(self.num_envs, int)
        self.thrust_violations = np.zeros(self.num_envs, int)
        self.state_violations = np.zeros(self.num_envs, int)
        self.qp_residual_max = np.zeros(self.num_envs)
        self.qp_failures = np.zeros(self.num_envs, int)
        self.reported = np.zeros(self.num_envs, bool)
        self.episode_ids = np.zeros(self.num_envs, int)
        self.current_context = None

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.ids if env_ids is None else env_ids
        for i in ids.cpu().tolist():
            stride = self.cfg.reference_strides[i % len(self.cfg.reference_strides)]
            self.planners[i] = Planner(Config.load(self.cfg.scenario, stride, profile="real"), [self.cfg.goal_x, 0, 1])
            self.command_violations[i] = self.thrust_violations[i] = self.state_violations[i] = 0
            self.qp_residual_max[i] = 0
            self.qp_failures[i] = 0
            self.reported[i] = False
            self.episode_ids[i] += 1

    def _get_observations(self):
        tick = perf_counter()
        physical = super()._get_observations()["policy"]
        values = physical.cpu().numpy()
        elapsed = self.elapsed.cpu().numpy()
        clouds = {}
        ticks = np.rint(elapsed * 100).astype(int)
        for tick_id in np.unique(ticks[ticks % 10 == 0]):
            ids = np.flatnonzero(ticks == tick_id)
            clouds.update(zip(ids, self.lidar_clouds(ids), strict=True))
        planes = np.zeros((self.num_envs, 15, self.cfg.max_planes, 4))
        valid = np.zeros(planes.shape[:-1], bool)
        sfc_valid = np.ones((self.num_envs, 1), bool)
        refs, states, margins, braking, yaw = [], [], [], [], []
        for i, (planner, state) in enumerate(zip(self.planners, values, strict=True)):
            if i in clouds:
                planner.update_cloud(clouds[i], state[:3])
            try:
                x, ref, corridor = planner.context(state[:9])
            except InvalidCorridorError:
                planner.frontend_failures += 1
                planner.replan = True
                sfc_valid[i] = False
                x = planner.cfg.profile.planner_state(state[:9])
                ref = np.zeros((15, 9))
                ref[:, :3] = x[:3]
                corridor = []  # Invalid contexts execute braking, never this placeholder QP.
            clearance = np.full(15, 10.0)
            for k, faces in enumerate(corridor):
                if len(faces) > self.cfg.max_planes:
                    raise ValueError(f"SFC has {len(faces)} planes; increase max_planes={self.cfg.max_planes}")
                if len(faces):
                    faces = np.asarray(faces)[::-1]
                    planes[i, k, : len(faces)] = faces
                    valid[i, k, : len(faces)] = True
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
            rotation=physical[:, 9:].clone(),
            braking=tensor(braking, torch.bool),
            yaw_target=tensor(yaw),
            sfc_valid=tensor(sfc_valid, torch.bool),
        )
        self.current_context = result
        self.frontend_times.append((perf_counter() - tick) * 1000)
        return result

    def record_mpc_result(self, result):
        """Supply diagnostics for the next env.step; PPO updates never call this."""
        if self._pending is not None:
            raise RuntimeError("Previous MPC result has not been consumed by env.step")
        self._pending = result

    def _pre_physics_step(self, actions):
        if self._pending is None:
            raise RuntimeError("Call record_mpc_result(actor.last_result) before env.step")
        result, self._pending = self._pending, None
        self.last_mpc_result = result
        data = {k: v.cpu().numpy() for k, v in result.items() if isinstance(v, torch.Tensor)}
        elapsed = self.elapsed.cpu().numpy()
        sampled = actions.cpu().numpy()
        for i, planner in enumerate(self.planners):
            status = "solved" if data["solved"][i] else "max iterations reached"
            if not data["sfc_valid"][i]:
                status = "invalid SFC"
            planner.consume_state(
                data["state"][i],
                data["solution"][i],
                status,
                float(elapsed[i]),
            )
            if not self.reported[i]:
                val = np.r_[data["predicted"][i, 3:], data["jerk"][i]]
                low = np.r_[planner.mpc.lower, planner.mpc.jlow[:3]]
                high = np.r_[planner.mpc.upper, planner.mpc.jhigh[:3]]
                self.command_violations[i] += bool(
                    np.any((val < low - 0.001 - 0.001 * abs(low)) | (val > high + 0.001 + 0.001 * abs(high)))
                )
                self.thrust_violations[i] += not (0.08 * 9.81 / 0.7 <= sampled[i, 0] <= 0.9 * 9.81 / 0.7)
                if data["sfc_valid"][i]:
                    self.qp_failures[i] += not data["solved"][i]
                    self.qp_residual_max[i] = max(self.qp_residual_max[i], float(data["primal"][i]))
        super()._pre_physics_step(actions)

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
                    minimum_physics_step_clearance_m=float(self.min_clearance[i]),
                    final_position=self.extras["terminal_position"][i].cpu().tolist(),
                )
            )
        if (term | trunc).any():
            if self.cfg.log_dir:
                path = Path(self.cfg.log_dir) / "records.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.records, indent=2))
                temporary.replace(path)
            recent = [r for r in self.records[-self.num_envs :] if r["success"]]
            self.extras["log"] = {"Episode/success_fraction": len(recent) / min(len(self.records), self.num_envs)}
        return term, trunc
