"""Mid-360 random scan windows on Isaac Lab 3.0's Newton mesh ray caster."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

TABLE = Path(__file__).resolve().parents[2] / "build/sensors/mid360_raydirs.npy"


def mid360_pattern(cfg, device):
    dirs = cfg.load_table(device)[: cfg.rays_per_scan : cfg.ray_decimation]
    return torch.zeros_like(dirs), dirs.clone()


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    func: Callable = mid360_pattern
    raydirs_path: str = str(TABLE)
    rays_per_scan: int = 20000
    ray_decimation: int = 1

    def load_table(self, device):
        if self.rays_per_scan <= 0 or self.ray_decimation <= 0:
            raise ValueError("Ray count and decimation must be positive")
        if not hasattr(self, "_all_dirs"):
            dirs = np.load(self.raydirs_path, allow_pickle=False)
            if dirs.ndim != 2 or dirs.shape[1] != 3 or len(dirs) < self.rays_per_scan:
                raise ValueError("Mid360 table must be [N,3], N >= rays_per_scan")
            dirs = torch.as_tensor(dirs.astype(np.float32))
            norms = dirs.norm(dim=-1, keepdim=True)
            if not torch.isfinite(dirs).all() or (norms < 1e-8).any():
                raise ValueError("Mid360 table contains invalid directions")
            self._all_dirs = dirs / norms
        self._all_dirs = self._all_dirs.to(device)
        return self._all_dirs


def mid360_config(scenario="forest", *, optimized=True):
    if scenario == "forest":
        targets = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Forest",
                is_shared=True,
                track_mesh_transforms=False,
            )
        ]
    elif scenario == "avoid":
        targets = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/" + name,
                is_shared=True,
                track_mesh_transforms=True,
            )
            for name in ("wall", "obstacle")
        ]
    else:
        raise ValueError(scenario)
    config = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        spawn=None,
        pattern_cfg=Mid360PatternCfg(),
        max_distance=40.0,
        update_period=0.1,
        ray_alignment="base",
        mesh_prim_paths=targets,
    )
    if optimized:
        from .compact_sensor import SharedMid360RayCaster

        config.class_type = SharedMid360RayCaster
    return config


def update_mid360_directions(env, env_ids):
    sensor = env.scene.sensors.get("mid360")
    if sensor is None or len(env_ids) == 0:
        return
    cfg = sensor.cfg.pattern_cfg
    table = cfg.load_table(env.device)
    start = int(torch.randint(len(table) - cfg.rays_per_scan + 1, (1,), generator=env.mid360_rng))
    if hasattr(sensor, "window_start"):
        sensor.window_start.torch[env_ids] = start
    else:
        # Explicit official-reference mode used by paired validation.
        window = table[start : start + cfg.rays_per_scan : cfg.ray_decimation]
        sensor.ray_directions.torch[env_ids] = window
    env.mid360_window_start[env_ids] = start


def materialize_forest_targets(env):
    from isaaclab.cloner import usd_replicate

    # Physics replication keeps only source USD assets; static mesh queries need each destination.
    usd_replicate(
        env.scene.stage,
        [env.scene.env_prim_paths[0] + "/Forest"],
        [env.cfg.scene.clone_cfg.clone_template + "/Forest"],
        np.arange(1, env.num_envs),
    )
