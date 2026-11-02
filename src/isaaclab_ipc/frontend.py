"""Bounded, ordered CPU planning and one host transfer per acquired cloud batch."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from .planner import InvalidCorridorError


def batch_clouds_to_host(hits, positions):
    valid = torch.isfinite(hits).all(-1) & ((hits - positions[:, None, :]).norm(dim=-1) >= 0.3)
    # One transfer, rather than one synchronizing boolean selection/transfer per environment.
    values = hits.masked_fill(~valid[..., None], torch.inf).cpu().numpy()
    mask = np.isfinite(values).all(-1)
    return [points[keep] for points, keep in zip(values, mask, strict=True)]


@dataclass
class PlanningContext:
    state: np.ndarray
    reference: np.ndarray
    corridors: list
    valid: bool


def _prepare_context(item):
    planner, state, cloud = item
    if cloud is not None:
        planner.update_cloud(cloud, state[:3])
    try:
        x, reference, corridors = planner.context(state[:9])
        return PlanningContext(x, reference, corridors, True)
    except InvalidCorridorError:
        planner.frontend_failures += 1
        planner.replan = True
        x = planner.cfg.profile.planner_state(state[:9])
        reference = np.zeros((15, 9))
        reference[:, :3] = x[:3]
        return PlanningContext(x, reference, [], False)


class PlanningBatch:
    """Each planner is owned by one task; all tasks finish before reset or action consumption."""

    def __init__(self, workers=1):
        if workers < 1:
            raise ValueError("planner_workers must be positive")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ipc-plan") if workers > 1 else None

    def contexts(self, planners, states, clouds):
        if len(planners) != len(states):
            raise ValueError("One state is required per planner")
        jobs = ((planner, states[i], clouds.get(i)) for i, planner in enumerate(planners))
        return list(self._pool.map(_prepare_context, jobs)) if self._pool else list(map(_prepare_context, jobs))

    def problems(self, planners, states, clouds):
        """Classical entry point: preserve Planner.prepare's exception and QP semantics."""
        if len(planners) != len(states):
            raise ValueError("One state is required per planner")
        jobs = ((planner, states[i], clouds.get(i)) for i, planner in enumerate(planners))
        return list(self._pool.map(_prepare_problem, jobs)) if self._pool else list(map(_prepare_problem, jobs))

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _prepare_problem(item):
    planner, state, cloud = item
    if cloud is not None:
        planner.update_cloud(cloud, state[:3])
    return planner.prepare(state[:9])[0]
