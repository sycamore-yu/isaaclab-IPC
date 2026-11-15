"""Fixed-progress CMPCC-style SFC sections, safely bounded in the path direction."""

import numpy as np


def section_tube_planes(faces, center, tangent, half_length):
    """Return [normal, offset] planes for a bounded sweep of an SFC section.

    Given P={p | A p+c<=0}, intersect P with the plane through center
    normal to tangent. Project each face onto that plane, then sweep the
    resulting polygon along the tangent over [-backward, forward].
    Tightening lateral bounds by their worst axial displacement ensures the
    whole returned tube is a subset of P, not an infinite extrusion.
    The two axial caps enforce fixed reference progress.
    """
    faces = np.asarray(faces, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 4 or not len(faces):
        raise ValueError("A section requires nonempty SFC faces [F,4]")
    if center.shape != (3,) or tangent.shape != (3,):
        raise ValueError("Section center and tangent must be 3-vectors")
    if not np.isfinite(faces).all() or not np.isfinite(center).all() or not np.isfinite(tangent).all():
        raise ValueError("Section geometry must be finite")
    if not np.isfinite(half_length) or half_length <= 0:
        raise ValueError("Section half-length must be positive and finite")
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm <= 1e-12:
        raise ValueError("Section tangent cannot be zero")
    tangent = tangent / tangent_norm

    normals = faces[:, :3]
    axial = normals @ tangent
    lateral = normals - axial[:, None] * tangent
    slack = -(normals @ center + faces[:, 3])

    # Reserve half the available axial clearance for lateral motion. Taking
    # the full allowable sweep would tighten a limiting lateral face exactly
    # onto the reference point, causing observed CTBR QP convergence failures.
    # A negative slack is not silently converted into an enlarged corridor.
    forward = backward = float(half_length)
    positive, negative = axial > 1e-12, axial < -1e-12
    if positive.any():
        forward = min(forward, max(0.0, 0.5 * float(np.min(slack[positive] / axial[positive]))))
    if negative.any():
        backward = min(backward, max(0.0, 0.5 * float(np.min(slack[negative] / -axial[negative]))))

    worst_axial = np.maximum(axial * forward, -axial * backward)
    lateral_bound = slack - worst_axial
    lateral_norm = np.linalg.norm(lateral, axis=-1)
    regular = lateral_norm > 1e-10
    projected = np.column_stack((
        lateral[regular] / lateral_norm[regular, None],
        (-lateral_bound[regular] - lateral[regular] @ center) / lateral_norm[regular],
    ))
    # Never omit a small but nonzero projected normal: its original SFC face
    # remains active instead of being lost to the normalization threshold.
    # Exactly axial faces are redundant only when the caps satisfy them.
    axial_only = faces[~regular & ((lateral_norm > 0.0) | (lateral_bound < 0.0))]
    caps = np.array([
        [*tangent, -np.dot(tangent, center) - forward],
        [*-tangent, np.dot(tangent, center) - backward],
    ])
    return np.concatenate((projected, axial_only, caps), axis=0)


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
