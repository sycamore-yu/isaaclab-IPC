"""GPU incremental inflation with a packed map exported to native IPC A*.

Counts handle centroid replacement without retaining obsolete occupied cells.
The fixed SUPER world grid is represented as a bitset on the CPU side.
"""

import numpy as np
import torch
import warp as wp


@wp.kernel
def stamp_centroids(
    points: wp.array(dtype=wp.vec3d),
    signs: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    center: wp.vec3d,
    low: wp.vec3d,
    high: wp.vec3d,
    resolution: wp.float64,
    nx: int,
    ny: int,
    nz: int,
    n_offsets: int,
):
    i, j = wp.tid()
    point = points[i]
    local = point - center
    if not (
        local[0] > low[0] and local[0] < high[0]
        and local[1] > low[1] and local[1] < high[1]
        and local[2] > low[2] and local[2] < high[2]
    ):
        return
    ox = j // (n_offsets * n_offsets)
    oy = (j // n_offsets) % n_offsets
    oz = j % n_offsets
    x = point[0] + offsets[ox] - center[0]
    y = point[1] + offsets[oy] - center[1]
    z = point[2] + offsets[oz] - center[2]
    if x < low[0] or x >= high[0] or y < low[1] or y >= high[1] or z < low[2] or z >= high[2]:
        return
    ix = int((x - low[0]) / resolution)
    iy = int((y - low[1]) / resolution)
    iz = int((z - low[2]) / resolution)
    if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
        wp.atomic_add(counts, (ix * ny + iy) * nz + iz, signs[i])


@wp.kernel
def pack_grid_bits(counts: wp.array(dtype=wp.int32), words: wp.array(dtype=wp.int32), cells: int):
    word = wp.tid()
    start = word * 32
    bits = wp.int32(0)
    for shift in range(32):
        index = start + shift
        if index < cells and counts[index] > 0:
            bits = bits | (wp.int32(1) << shift)
    words[word] = bits


class WarpInflatedGrid:
    def __init__(self, resolution, size, center, radius, device="cuda:0"):
        if not str(device).startswith("cuda"):
            raise ValueError("Warp incremental grid requires CUDA")
        self.device = str(device)
        self.resolution = float(resolution)
        self.size = np.asarray(size, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.low, self.high = -self.size / 2, self.size / 2
        self.shape = tuple(np.ceil(self.size / self.resolution).astype(int))
        self.cells = int(np.prod(self.shape))
        offsets = []
        x = -float(radius)
        while x <= radius:
            offsets.append(x)
            x += self.resolution
        self.offsets = wp.array(np.asarray(offsets, dtype=np.float64), dtype=wp.float64, device=self.device)
        self.n_offsets = len(offsets)
        self.counts = wp.zeros(self.cells, dtype=wp.int32, device=self.device)
        self.words = wp.zeros((self.cells + 31) // 32, dtype=wp.int32, device=self.device)

    def update(self, removed, added, grid, reset=False):
        if reset:
            wp.launch(
                clear_counts,
                dim=self.cells,
                inputs=[self.counts],
                device=self.device,
            )
            removed = np.empty((0, 3), dtype=np.float64)
        removed = np.asarray(removed, dtype=np.float64).reshape(-1, 3)
        added = np.asarray(added, dtype=np.float64).reshape(-1, 3)
        if not len(removed) and not len(added) and not reset:
            return
        points = np.concatenate((removed, added))
        signs = np.concatenate((np.full(len(removed), -1, dtype=np.int32), np.ones(len(added), dtype=np.int32)))
        if len(points):
            points_gpu = torch.as_tensor(points, dtype=torch.float64, device=self.device)
            signs_gpu = torch.as_tensor(signs, dtype=torch.int32, device=self.device)
            wp.launch(
                stamp_centroids,
                dim=(len(points), self.n_offsets**3),
                inputs=[
                    wp.from_torch(points_gpu, dtype=wp.vec3d),
                    wp.from_torch(signs_gpu),
                    self.offsets, self.counts,
                    wp.vec3d(*self.center), wp.vec3d(*self.low), wp.vec3d(*self.high),
                    self.resolution, *self.shape, self.n_offsets,
                ],
                device=self.device,
            )
        wp.launch(pack_grid_bits, dim=len(self.words), inputs=[self.counts, self.words, self.cells], device=self.device)
        packed = wp.to_torch(self.words).cpu().numpy().view(np.uint32)
        grid.load_bits(packed)


@wp.kernel
def clear_counts(counts: wp.array(dtype=wp.int32)):
    counts[wp.tid()] = 0
