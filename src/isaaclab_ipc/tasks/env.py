"""Batched Newton/MJWarp quadrotor task; actions are specific thrust and body rates."""

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, PvaCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_ipc.assets.quadrotor import QUADROTOR_CFG
from isaaclab_ipc.assets.rate_control import RateController

from .events import IPCEventsCfg


@configclass
class IPCSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = QUADROTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    pva = PvaCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base", update_period=0.0)


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
    initial_position = (0.0, 0.0, 1.0)
    initial_yaw = 0.0
    mission_start_delay = 0.0
    goal_tolerance = 0.5
    sensor_blind = 0.3
    sensor_return_max = float("inf")
    planner_params = None
    thrust_limits = (0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
    collision_radius = 0.25
    automatic_reset = True
    events = IPCEventsCfg()
    recorder_dir = None
    rerun_path = None
    rerun_label = "IPC"
    planner_workers = 1


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
    if scenario in ("forest", "avoid"):
        from isaaclab_ipc.mid360 import mid360_config

        cfg.scene.mid360 = mid360_config(scenario)
    return cfg


class IPCEnv(DirectRLEnv):
    def _setup_scene(self):
        if self.cfg.scenario == "forest":
            from isaaclab_ipc.mid360 import materialize_forest_targets

            materialize_forest_targets(self)
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
        from .recorders import FlightRecorderManager, recorder_config

        recorder_cfg = recorder_config(cfg)
        self.recorder_manager = FlightRecorderManager(recorder_cfg, self)
        self.mid360_rng = torch.Generator().manual_seed(cfg.seed)
        self.mid360_window_start = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)
        self.forest_geometry = None
        if cfg.scenario == "forest":
            from pathlib import Path

            from isaaclab_ipc.sensing import ForestGeometry

            self.forest_geometry = ForestGeometry(
                Path(cfg.scene.forest.spawn.usd_path).parent,
                self.device,
                radius=cfg.collision_radius,
            )
        self.origin_wp = wp.from_torch(self.scene.env_origins, dtype=wp.vec3)
        self.obstacle_y_wp = wp.from_torch(self.obstacle_y)
        self.minimum_wp = wp.from_torch(self.min_clearance)
        self.hit_wp = wp.from_torch(self.hit, dtype=wp.bool)
        self.collision_points = (
            self.forest_geometry.points if self.forest_geometry else wp.empty(1, dtype=wp.vec3, device=self.device)
        )
        self.sim.physics_manager.register_state_force_callback(self._record_clearance)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.mid360_rng.manual_seed(seed)
        return super().reset(seed=seed, options=options)

    def body_state(self):
        # ArticulationData owns state bindings and deferred FK across partial resets.
        _ = self.robot.data.body_link_pose_w
        return self.robot.data.root_link_pose_w.torch, self.robot.data.root_com_vel_w.torch

    def lidar_clouds(self, env_ids=None):
        from isaaclab_ipc.frontend import batch_clouds_to_host

        ids = slice(None) if env_ids is None else env_ids
        sensor = self.scene["mid360"]
        hits = sensor.data.ray_hits_w.torch[ids] - self.scene.env_origins[ids, None, :]
        position = self.body_state()[0][ids, :3] - self.scene.env_origins[ids]
        return batch_clouds_to_host(hits, position, self.cfg.sensor_blind, self.cfg.sensor_return_max)

    def acceleration(self):
        data = self.scene["pva"].data
        return quat_apply(data.quat_w.torch, data.lin_acc_b.torch)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite physical action")
        self.controller.command.copy_(actions)
        self.recorder_manager.record_pre_step()
        self.event_manager.apply(mode="ipc_control")

    def _apply_action(self):
        pass

    def _get_observations(self):
        pose, twist = self.body_state()
        observation = torch.cat(
            (
                pose[:, :3] - self.scene.env_origins,
                twist[:, :3],
                self.acceleration(),
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
        if self.forest_geometry is not None:
            clearance = torch.minimum(clearance, self.forest_geometry.clearance(p))
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
                self.forest_geometry.grid.id if self.forest_geometry else 0,
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
        success = (p - self.goal).norm(dim=-1) < self.cfg.goal_tolerance
        self.extras["success"] = success.clone()
        self.extras["collision"] = self.hit.clone()
        self.extras["terminal_position"] = p.clone()
        force = self.scene["contacts"].data.net_forces_w.torch
        self.extras["physical_contact"] = force.norm(dim=-1).amax(-1) > 1e-6
        return self.hit | success, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        pose, _ = self.body_state()
        distance = (pose[:, :3] - self.scene.env_origins - self.goal).norm(dim=-1)
        reward = self.previous_distance - distance - 10 * self.hit.float() + 10 * (distance < 0.5).float()
        # Keep normal persistent storage when steps run inside inference_mode.
        self.previous_distance.copy_(distance)
        self.recorder_manager.record_post_step()
        return reward

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.ids
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.record_pre_reset(env_ids)
        super()._reset_idx(env_ids)
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.reset(env_ids)
            self.recorder_manager.record_post_reset(env_ids)

    def _reset_envs_from_buffer(self):
        if self.cfg.automatic_reset:
            return super()._reset_envs_from_buffer()
        return self.ids[:0]

    def close(self):
        if hasattr(self, "recorder_manager"):
            self.recorder_manager.close()
        super().close()


def make_forest_cfg():
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "build/forest/forest.usdc"
    if not path.is_file():
        raise FileNotFoundError("Prepare the original PCD with scripts/build_forest.py --output build/forest")
    return make_cfg(scenario="forest", forest_usd=path)
