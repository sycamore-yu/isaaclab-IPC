"""Observed-point-cloud swept-path validation for A* altitude recovery."""

import numpy as np


def vertical_segment_clear(cloud, start, end, clearance):
    """Check the swept ascent centerline, including small horizontal drift."""
    points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if not len(points) or end[2] <= start[2]:
        return False
    segment = end - start
    fraction = np.clip((points - start) @ segment / np.dot(segment, segment), 0, 1)
    nearest = start + fraction[:, None] * segment
    distances_squared = np.square(points - nearest).sum(axis=1)
    return bool(np.all(distances_squared > clearance**2))
