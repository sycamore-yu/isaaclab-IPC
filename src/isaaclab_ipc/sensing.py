"""Independent full-PCD collision queries; sensing belongs to MultiMeshRayCaster."""

from pathlib import Path

import numpy as np
import torch
import warp as wp


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


class ForestGeometry:
    def __init__(self, folder, device="cuda:0", radius=0.25):
        self.device = device
        self.radius = radius
        self.points = wp.array(np.load(Path(folder) / "points.npy"), dtype=wp.vec3, device=device)
        self.grid = wp.HashGrid(128, 128, 128, device=device)
        self.grid.build(self.points, 1.0)

    def clearance(self, positions):
        positions = positions.contiguous()
        out = torch.empty(len(positions), device=self.device)
        wp.launch(
            point_clearance,
            dim=len(positions),
            inputs=[self.points, self.grid.id, wp.from_torch(positions, dtype=wp.vec3), self.radius, wp.from_torch(out)],
            device=self.device,
        )
        return out
