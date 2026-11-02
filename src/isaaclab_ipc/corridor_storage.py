"""Lossless storage of IPC's two unique corridors and their prediction-step schedule."""

import numpy as np


def pack_corridors(corridors, max_planes):
    """Return owned float64 blocks, face masks and time-step indices (-1 means empty).

    Planner.context schedules at most two nonempty polytopes. Preserve face order (including
    the MPC's reversed rows) exactly; exceeding either capacity is an error, never truncation.
    """
    planes = np.zeros((2, max_planes, 4), dtype=np.float64)
    masks = np.zeros((2, max_planes), dtype=bool)
    indices = np.full(len(corridors), -1, dtype=np.int64)
    unique = []
    for step, faces in enumerate(corridors):
        faces = np.asarray(faces, dtype=np.float64)
        if not len(faces):
            continue
        if faces.ndim != 2 or faces.shape[1] != 4 or not np.isfinite(faces).all():
            raise ValueError("SFC planes must be finite [N,4]")
        if len(faces) > max_planes:
            raise ValueError(f"SFC has {len(faces)} planes; increase max_planes={max_planes}")
        index = next((k for k, prior in enumerate(unique) if np.array_equal(prior, faces)), len(unique))
        if index == len(unique):
            if index == 2:
                raise ValueError("IPC schedule exceeds two unique corridors")
            unique.append(faces)
            planes[index, : len(faces)] = faces[::-1]
            masks[index, : len(faces)] = True
        indices[step] = index
    return planes, masks, indices


def expand_corridors(obs):
    """Reconstruct the exact dense snapshot; old dense recordings remain readable."""
    planes, masks = obs["planes"], obs["plane_mask"]
    if "corridor_index" not in obs:
        return planes, masks
    indices = obs["corridor_index"].long()
    selected = indices.clamp_min(0)
    nfaces = planes.shape[-2]
    planes = planes.gather(1, selected[:, :, None, None].expand(-1, -1, nfaces, 4))
    masks = masks.gather(1, selected[:, :, None].expand(-1, -1, nfaces))
    present = indices >= 0
    return planes.masked_fill(~present[:, :, None, None], 0), masks & present[:, :, None]
