"""Batched Newton/MJWarp quadrotor task; actions are specific thrust and body rates."""

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_ipc.assets.quadrotor import QUADROTOR_CFG
from isaaclab_ipc.assets.rate_control import RateController


@configclass
class IPCSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = QUADROTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class IPCEnvCfg(DirectRLEnvCfg):
    decimation = 3
    episode_length_s = 35.0
    action_space = 4
    observation_space = 18
    state_space = 0
    seed = 0
    ui_window_class_type = None
    sim = sim_utils.SimulationCfg(
        dt=1 / 300, render_interval=30, use_newton_actuators=False, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())
    )
    scene = IPCSceneCfg(num_envs=1, env_spacing=50.0)
    scenario = "avoid"
    obstacle_distance = 1.0
    goal_x = 10.0
    collision_radius = 0.25
    automatic_reset = True


def make_cfg(num_envs=1, device="cuda:0", scenario="avoid", distance=1.0, forest_usd=None):
    cfg = IPCEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.scenario = scenario
    cfg.obstacle_distance = distance
    cfg.goal_x = 8.0 if scenario == "forest" else 10.0
    if scenario == "avoid":
        for name, pos, size in [
            ("wall", (5.0, 0.9, 1.5), (10.0, 0.2, 3.0)),
            ("obstacle", (7.0, 1.5, 2.0), (0.2, 0.2, 4.0)),
        ]:
            setattr(
                cfg.scene,
                name,
                RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + name,
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassCfg(mass=1.0),
                        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
                ),
            )
    elif scenario == "forest":
        if forest_usd is None:
            raise ValueError("forest_usd is required for the forest task")
        cfg.scene.forest = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Forest", spawn=sim_utils.UsdFileCfg(usd_path=str(forest_usd))
        )
    elif scenario != "empty":
        raise ValueError(scenario)
    return cfg


class IPCEnv(DirectRLEnv):
    def _setup_scene(self):
        self.controller = RateController(self.num_envs, self.device)
        self.sim.physics_manager.register_state_force_callback(self.controller.apply)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.controller.bind(self.sim.physics_manager, self.robot)
        self.sim.physics_manager.activate_newton_actuator_path()
        self.sim.physics_manager.set_decimation(cfg.decimation)
        self._physics_handles_decimation = self.sim.physics_manager.handles_decimation()
        self.ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.obstacle_active = self.triggered.clone()
        self.obstacle_y = torch.full((self.num_envs,), 1.5, device=self.device)
        self.target_y = torch.zeros(self.num_envs, device=self.device)
        self.min_clearance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.goal = torch.tensor([cfg.goal_x, 0, 1.0], device=self.device)
        self.previous_distance = torch.full((self.num_envs,), cfg.goal_x, device=self.device)
        self.forest_sensor = None
        if cfg.scenario == "forest":
            from pathlib import Path

            from isaaclab_ipc.sensing import ForestSensor

            self.forest_sensor = ForestSensor(Path(cfg.scene.forest.spawn.usd_path).parent, self.device)
        self.origin_wp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3)
        self.obstacle_y_wp = wp.from_torch(self.obstacle_y)
        self.minimum_wp = wp.from_torch(self.min_clearance)
        self.hit_wp = wp.from_torch(self.hit, dtype=wp.bool)
        self.collision_points = (
            self.forest_sensor.points if self.forest_sensor else wp.empty(1, dtype=wp.vec3, device=self.device)
        )
        self.sim.physics_manager.register_state_force_callback(self._record_clearance)

    def body_state(self):
        # Public data access resolves Newton's deferred FK after partial resets.
        _ = self.robot.data.body_link_pose_w
        active = self.sim.physics_manager.backend.state_0
        indices = self.controller.ids.long()
        return wp.to_torch(active.body_q)[indices], wp.to_torch(active.body_qd)[indices]

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite physical action")
        self.controller.command.copy_(actions)
        if self.cfg.scenario == "avoid":
            pose, _ = self.body_state()
            p = pose[:, :3] - self.scene.env_origins
            near = torch.abs(p[:, 0] - 6.9) <= self.cfg.obstacle_distance
            end = torch.round(self.elapsed * 1000).long()
            for offset in range(-9, 1):
                tick = end + offset
                valid = tick >= 0
                periodic = valid & (tick % 101 == 0)
                activate = periodic & near & ~self.obstacle_active
                self.target_y.copy_(torch.where(activate, p[:, 1], self.target_y))
                self.obstacle_y.copy_(torch.where(activate, p[:, 1] + 1.0, self.obstacle_y))
                self.obstacle_active |= activate
                self.obstacle_y.copy_(
                    torch.where(
                        periodic & self.obstacle_active,
                        torch.maximum(self.obstacle_y - 1.0, self.target_y),
                        self.obstacle_y,
                    )
                )
                first = valid & near & ~self.triggered
                self.obstacle_y.copy_(
                    torch.where(first, torch.maximum(self.obstacle_y - 1.0, self.target_y), self.obstacle_y)
                )
                self.triggered |= first
            obs = self.scene["obstacle"]
            obstacle_pose = obs.data.default_root_pose.torch.clone()
            obstacle_pose[:, :3] += self.scene.env_origins
            obstacle_pose[:, 1] = self.obstacle_y + self.scene.env_origins[:, 1]
            obs.write_root_pose_to_sim_index(root_pose=obstacle_pose, env_ids=self.ids)
        self.elapsed += self.step_dt

    def _apply_action(self):
        pass

    def _get_observations(self):
        pose, twist = self.body_state()
        observation = torch.cat(
            (
                pose[:, :3] - self.scene.env_origins,
                twist[:, :3],
                self.controller.accel,
                matrix_from_quat(pose[:, 3:]).reshape(self.num_envs, 9),
            ),
            dim=-1,
        )
        return {"policy": observation}

    def clearance(self, pose=None):
        if pose is None:
            pose, _ = self.body_state()
        p = pose[:, :3] - self.scene.env_origins
        clearance = p[:, 2] - self.cfg.collision_radius
        if self.forest_sensor is not None:
            clearance = torch.minimum(clearance, self.forest_sensor.clearance(p))
        if self.cfg.scenario == "avoid":
            for center, size in [
                (torch.tensor([5.0, 0.9, 1.5], device=self.device), torch.tensor([5.0, 0.1, 1.5], device=self.device)),
                (
                    torch.stack(
                        (torch.full_like(self.obstacle_y, 7.0), self.obstacle_y, torch.full_like(self.obstacle_y, 2.0)),
                        dim=-1,
                    ),
                    torch.tensor([0.1, 0.1, 2.0], device=self.device),
                ),
            ]:
                d = (p - center).abs() - size
                sdf = d.clamp_min(0).norm(dim=-1) + d.max(-1).values.clamp_max(0)
                clearance = torch.minimum(clearance, sdf - self.cfg.collision_radius)
        return clearance

    def _record_clearance(self, state):
        from isaaclab_ipc.collision import latch_clearance

        kind = {"empty": 0, "avoid": 1, "forest": 2}[self.cfg.scenario]
        wp.launch(
            latch_clearance,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.controller.ids_wp,
                self.origin_wp,
                self.obstacle_y_wp,
                self.collision_points,
                self.forest_sensor.grid.id if self.forest_sensor else 0,
                kind,
                self.cfg.collision_radius,
                self.minimum_wp,
                self.hit_wp,
            ],
            device=self.device,
        )

    def _get_dones(self):
        clearance = self.clearance()
        self.min_clearance.copy_(torch.minimum(self.min_clearance, clearance))
        self.hit |= clearance <= 0
        pose, _ = self.body_state()
        p = pose[:, :3] - self.scene.env_origins
        success = (p - self.goal).norm(dim=-1) < 0.5
        self.extras["success"] = success.clone()
        self.extras["collision"] = self.hit.clone()
        self.extras["terminal_position"] = p.clone()
        return self.hit | success, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        pose, _ = self.body_state()
        distance = (pose[:, :3] - self.scene.env_origins - self.goal).norm(dim=-1)
        reward = self.previous_distance - distance - 10 * self.hit.float() + 10 * (distance < 0.5).float()
        self.previous_distance = distance
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.ids
        super()._reset_idx(env_ids)
        ids = env_ids.to(torch.int32)
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = self.scene.env_origins[env_ids] + torch.tensor([0, 0, 1.0], device=self.device)
        pose[:, 3:] = torch.tensor([0, 0, 0, 1.0], device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=ids
        )
        self.robot.reset_motor_thrust(torch.full((len(ids), 4), 0.75 * 9.81 / 4, device=self.device), ids)
        self.controller.command[ids] = torch.tensor([9.81, 0, 0, 0], device=self.device)
        self.controller.accel[ids] = 0
        self.elapsed[ids] = 0
        self.triggered[ids] = False
        self.obstacle_active[ids] = False
        self.obstacle_y[ids] = 1.5
        self.target_y[ids] = 0
        self.hit[ids] = False
        self.min_clearance[ids] = float("inf")
        self.previous_distance[ids] = self.cfg.goal_x
        if self.cfg.scenario == "avoid":
            obs = self.scene["obstacle"]
            obstacle_pose = obs.data.default_root_pose.torch[ids].clone()
            obstacle_pose[:, :3] += self.scene.env_origins[ids]
            obs.write_root_pose_to_sim_index(root_pose=obstacle_pose, env_ids=ids)

    def _reset_envs_from_buffer(self):
        if self.cfg.automatic_reset:
            return super()._reset_envs_from_buffer()
        return self.ids[:0]


def make_forest_cfg():
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "build/forest/forest.usdc"
    if not path.is_file():
        raise FileNotFoundError("Prepare the original PCD with scripts/build_forest.py --output build/forest")
    return make_cfg(scenario="forest", forest_usd=path)
