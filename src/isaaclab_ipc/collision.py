"""Collision latching inside Newton's captured physical substep."""

import warp as wp


@wp.func
def box_sdf(p: wp.vec3, center: wp.vec3, half: wp.vec3):
    d = wp.vec3(wp.abs(p[0] - center[0]), wp.abs(p[1] - center[1]), wp.abs(p[2] - center[2])) - half
    outside = wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))
    return wp.length(outside) + wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)


@wp.kernel
def latch_clearance(
    body_q: wp.array(dtype=wp.transform),
    ids: wp.array(dtype=wp.int32),
    origins: wp.array(dtype=wp.vec3),
    obstacle_y: wp.array(dtype=float),
    points: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    kind: int,
    radius: float,
    minimum: wp.array(dtype=float),
    hit: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    p = wp.transform_get_translation(body_q[ids[i]]) - origins[i]
    clearance = p[2] - radius
    if kind == 1:
        clearance = wp.min(clearance, box_sdf(p, wp.vec3(5.0, 0.9, 1.5), wp.vec3(5.0, 0.1, 1.5)) - radius)
        clearance = wp.min(clearance, box_sdf(p, wp.vec3(7.0, obstacle_y[i], 2.0), wp.vec3(0.1, 0.1, 2.0)) - radius)
    elif kind == 2:
        distance = float(1.0)
        query = wp.hash_grid_query(grid, p, 1.0)
        for j in query:
            distance = wp.min(distance, wp.length(p - points[j]))
        clearance = wp.min(clearance, distance - radius)
    minimum[i] = wp.min(minimum[i], clearance)
    if clearance <= 0.0:
        hit[i] = True
