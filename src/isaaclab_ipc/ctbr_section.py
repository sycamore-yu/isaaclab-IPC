"""Fixed-progress SFC section slabs for CTBR MPC."""

import numpy as np


def section_tube_planes(faces, center, tangent, half_length):
    """Intersect the original SFC with two axial caps around a reference node.

    The original polytope already guarantees lateral clearance. Keeping its
    faces avoids projecting and tightening the same faces a second time.
    """
    faces = np.asarray(faces, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if not np.isfinite(half_length) or half_length <= 0:
        raise ValueError("Section half-length must be positive and finite")
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm <= 1e-12:
        raise ValueError("Section tangent cannot be zero")
    tangent = tangent / tangent_norm
    origin = float(np.dot(tangent, center))
    caps = np.array([
        [*tangent, -origin - half_length],
        [*-tangent, origin - half_length],
    ])
    return np.concatenate((faces, caps), axis=0)


def reference_tangent(pva, yaw, index):
    """Use reference velocity, a position secant, or its yaw at a full stop."""
    velocity = pva[index, 3:6]
    if np.linalg.norm(velocity) > 1e-6:
        return velocity / np.linalg.norm(velocity)
    previous = pva[max(index - 1, 0), :3]
    following = pva[min(index + 1, len(pva) - 1), :3]
    secant = following - previous
    if np.linalg.norm(secant) > 1e-6:
        return secant / np.linalg.norm(secant)
    return np.array([np.cos(yaw[index]), np.sin(yaw[index]), 0.0])
