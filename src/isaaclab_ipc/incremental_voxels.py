"""Incremental, reproducible voxel centroids for SUPER's accumulated point cloud.

Each existing centroid represents one point in IPC's next voxel reduction, not
all historical hits. This preserves the previous online accumulation semantics.
"""

import numpy as np


class IncrementalVoxelCentroids:
    def __init__(self, leaf):
        self.leaf = np.float32(leaf)
        if not np.isfinite(self.leaf) or self.leaf <= 0:
            raise ValueError("Voxel leaf must be finite and positive")
        self.keys = np.empty(0, dtype=np.int64)
        self.points = np.empty((0, 3), dtype=np.float64)

    def _keys(self, points):
        indices = np.floor(np.asarray(points, dtype=np.float32) / self.leaf).astype(np.int64)
        bias = 1 << 20
        if ((indices < -bias) | (indices >= bias)).any():
            raise ValueError("Voxel position exceeds signed 21-bit coordinate range")
        return ((indices[:, 2] + bias) << 42) | ((indices[:, 1] + bias) << 21) | (indices[:, 0] + bias)

    def update(self, new_points):
        """Return (removed_centroids, added_centroids, rebuilt).

        Replaced centroids are removed and re-added for exact occupancy updates.
        The uncommon case where a rounded mean crosses a voxel edge is rebuilt.
        """
        incoming = np.asarray(new_points, dtype=np.float32).reshape(-1, 3)
        if not np.isfinite(incoming).all():
            raise ValueError("Nonfinite map point")
        if not len(incoming):
            return np.empty((0, 3)), np.empty((0, 3)), False
        new_keys, inverse, counts = np.unique(self._keys(incoming), return_inverse=True, return_counts=True)
        sums = np.zeros((len(new_keys), 3), dtype=np.float64)
        np.add.at(sums, inverse, incoming)
        indices = np.searchsorted(self.keys, new_keys)
        found = indices < len(self.keys)
        found[found] &= self.keys[indices[found]] == new_keys[found]
        new_means = sums / (counts[:, None] + found[:, None])
        previous = self.points[indices[found]].copy()
        new_means[found] = (sums[found] + previous) / (counts[found, None] + 1)
        new_means = new_means.astype(np.float32).astype(np.float64)
        if not np.array_equal(self._keys(new_means), new_keys):
            from .planner import voxel_centroids

            old = self.points
            updated = voxel_centroids(np.vstack((old, incoming)), float(self.leaf))
            self.keys = self._keys(updated)
            self.points = updated
            return old.copy(), updated.copy(), True

        old_keys = self.keys
        old_points = self.points
        changed = (new_means[found] != previous).any(axis=1)
        removed = previous[changed]
        added = np.vstack((new_means[found][changed], new_means[~found]))
        if (~found).any():
            keys = np.concatenate((old_keys, new_keys[~found]))
            points = np.concatenate((old_points, new_means[~found]))
            points[indices[found]] = new_means[found]
            order = np.argsort(keys, kind="stable")
            self.keys = keys[order]
            self.points = points[order]
        else:
            self.points = old_points.copy()
            self.points[indices[found]] = new_means[found]
        return removed, added, False
