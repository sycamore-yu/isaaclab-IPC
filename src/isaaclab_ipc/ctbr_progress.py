"""Arc-length geometry snapshots for free-progress CTBR MPC."""

import numpy as np


def sample_progress_plan(plan, state, cost_nodes, dt, max_planes=128, *, max_speed=6.0):
    """Sample a local geometric model; MINCO time only parameterizes its curve.

    Each returned row is [s_bar, tangent_xyz, s_lower, s_upper, nu_guess].
    A new local plan begins at the measured position, hence s0=0. Ordered
    corridor assignment is fixed for this solve and remains in PPO replay.
    """
    from .ctbr_navigation import SampledPlan

    # Equal normalized samples per polynomial make the spatial approximation
    # independent of a uniform rescaling of its original execution time.
    starts = np.r_[0., [segment.end for segment in plan.segments[:-1]]]
    ends = np.asarray([segment.end for segment in plan.segments])
    times = np.unique(np.concatenate([
        np.linspace(start, end, 65) for start, end in zip(starts, ends, strict=True)
    ]))
    raw = plan.trajectory.sample(times)
    positions = np.asarray(raw.pva)[..., :3]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    arc = np.r_[0., np.cumsum(distances)]
    boundaries = np.interp(np.r_[0., ends], times, arc)
    keep = np.r_[True, distances > 1e-10]
    positions, arc, times = positions[keep], arc[keep], times[keep]
    speed = min(float(np.linalg.norm(state[7:10])), max_speed)
    # At standstill, repeated s=0 tangents can extrapolate a tiny initial
    # vertical bend across the whole horizon. Preview at least 0.1 m per
    # 0.1 s node; this initializes geometry, not progress state or its bounds.
    preview_speed = max(speed, 1.0)
    seed = np.minimum(np.arange(cost_nodes) * dt * preview_speed, boundaries[-1])

    if len(arc) > 1:
        indices = np.searchsorted(arc[1:], seed, side="right").clip(0, len(arc) - 2)
        delta = positions[indices + 1] - positions[indices]
        lengths = arc[indices + 1] - arc[indices]
        tangent = delta / lengths[:, None]
        points = positions[indices] + (seed - arc[indices])[:, None] * tangent
    else:
        yaw0 = float(np.asarray(raw.yaw).flat[0])
        tangent = np.tile([np.cos(yaw0), np.sin(yaw0), 0.], (cost_nodes, 1))
        points = np.tile(positions[0], (cost_nodes, 1))
    # Upright heading is a soft attitude regularizer, not the old timed tilt
    # and acceleration reference. Optimized CTBR produces physical acceleration.
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    pva = np.column_stack((points, tangent * speed, np.zeros_like(points)))
    ownership = np.searchsorted(boundaries[1:], seed, side="right").clip(0, len(ends) - 1)
    planes = np.zeros((cost_nodes, max_planes, 4))
    masks = np.zeros((cost_nodes, max_planes), dtype=bool)
    progress = np.column_stack((seed, tangent, boundaries[ownership], boundaries[ownership + 1],
                                np.full(cost_nodes, speed)))
    violation = 0.0
    corridor_ids = np.zeros(cost_nodes, dtype=np.int64)
    for k, index in enumerate(ownership):
        corridor_ids[k] = plan.segments[index].corridor_index
        faces = np.asarray(plan.corridors[corridor_ids[k]])
        if index > 0 and abs(seed[k] - boundaries[index]) < 1e-9:
            previous = plan.segments[index - 1].corridor_index
            if previous != corridor_ids[k]:
                faces = np.concatenate((faces, plan.corridors[previous]))
        if len(faces) > max_planes:
            raise ValueError(f"Sample requires {len(faces)} SFC faces; configured capacity is {max_planes}")
        planes[k, :len(faces)] = faces
        masks[k, :len(faces)] = True
        if len(faces):
            violation = max(violation, float(np.max(faces[:, :3] @ points[k] + faces[:, 3])))
    return SampledPlan(pva, yaw, planes, masks, corridor_ids, violation, violation, progress)
