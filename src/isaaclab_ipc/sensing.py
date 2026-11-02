"""MiniCF-pattern Warp ray sensing and independent full-PCD collision queries."""

from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab.utils.warp import raycast_mesh


def minicf_directions(time):
    t = time + np.arange(20000) / 200000.0

    def cround(x):
        return np.copysign(np.floor(np.abs(x) + 0.5), x)

    x = -cround(-62050.63 * t + 3.11 * np.cos(314159.2 * t) * np.sin(628.318 * 2 * t)).astype(np.int64)
    x = (x % 360 / 0.2).astype(int)
    pi = 3.141519265357
    y = (
        cround(22.5 * np.cos(20 * pi * t) + 4 * np.cos(2 * pi / 0.006 * t) * np.cos(10000 * pi * t) + 22.5) / 0.2 + 170
    ).astype(int)
    pixels = np.unique(np.stack((np.clip(x, 0, 1799), np.clip(y, 0, 339)), axis=1), axis=0)
    angles = np.deg2rad((pixels - [900, 170]) * 0.2)
    az, el = angles.T
    return np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1).astype(np.float32)


@wp.kernel
def point_clearance(
    points: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    radius: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    p = positions[i]
    distance = float(1.0)
    query = wp.hash_grid_query(grid, p, 1.0)
    for j in query:
        distance = wp.min(distance, wp.length(p - points[j]))
    out[i] = distance - radius


class ForestSensor:
    def __init__(self, folder, device="cuda:0"):
        folder = Path(folder)
        self.device = device
        self.points = wp.array(np.load(folder / "points.npy"), dtype=wp.vec3, device=device)
        self.grid = wp.HashGrid(128, 128, 128, device=device)
        self.grid.build(self.points, 1.0)
        data = np.load(folder / "ray_mesh.npz")
        self.mesh = wp.Mesh(
            wp.array(data["vertices"], dtype=wp.vec3, device=device),
            wp.array(data["faces"].ravel(), dtype=int, device=device),
        )

    def sense(self, positions, rotations, time):
        directions = torch.as_tensor(minicf_directions(time), device=self.device)
        directions = torch.einsum("bij,rj->bri", rotations, directions).contiguous()
        origins = positions[:, None, :].expand_as(directions).contiguous()
        hits, _, _, _ = raycast_mesh(origins, directions, self.mesh, max_dist=30.0)
        good = torch.isfinite(hits).all(-1) & ((hits - origins).norm(dim=-1) >= 0.3)
        return [h[g].cpu().numpy() for h, g in zip(hits, good, strict=True)]

    def clearance(self, positions):
        positions = positions.contiguous()
        out = torch.empty(len(positions), device=self.device)
        wp.launch(
            point_clearance,
            dim=len(positions),
            inputs=[self.points, self.grid.id, wp.from_torch(positions, dtype=wp.vec3), 0.25, wp.from_torch(out)],
            device=self.device,
        )
        return out
