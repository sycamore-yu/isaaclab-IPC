"""Compare incremental SUPER maps with the original complete-cloud reduction."""

import numpy as np
import pytest
import torch

from isaaclab_ipc import _geometry
from isaaclab_ipc.incremental_voxels import IncrementalVoxelCentroids
from isaaclab_ipc.planner import voxel_centroids


@pytest.mark.parametrize('leaf', [0.05, 0.1, 0.3])
def test_incremental_centroids_match_original_repeated_reduction(leaf):
    rng = np.random.default_rng(302)
    cache = IncrementalVoxelCentroids(leaf)
    previous = np.empty((0, 3), dtype=np.float64)
    for _ in range(6):
        incoming = rng.uniform([-1.8, -50.9, 0.6], [1.8, -49.1, 2.4], (1800, 3))
        expected = voxel_centroids(np.concatenate((previous, incoming)), leaf)
        removed, added, rebuilt = cache.update(incoming)
        np.testing.assert_array_equal(cache.points, expected)
        assert not rebuilt
        assert len(removed) <= len(added)
        previous = expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp GPU grid requires CUDA')
def test_warp_inflation_reproduces_original_occupied_grid_and_updates():
    from isaaclab_ipc.warp_inflation import WarpInflatedGrid

    rng = np.random.default_rng(17)
    leaf = resolution = 0.1
    center = (0, -50, 1.5)
    size = (2, 4, 3)
    radius = 0.3
    cache = IncrementalVoxelCentroids(leaf)
    original = _geometry.Grid(resolution, [-1, -2, -1.5], [1, 2, 1.5])
    original.center(center)
    incremental = _geometry.Grid(resolution, [-1, -2, -1.5], [1, 2, 1.5])
    incremental.center(center)
    gpu = WarpInflatedGrid(resolution, size, center, radius)
    cloud = np.empty((0, 3))
    for i in range(4):
        new = rng.uniform([-.7, -51, .5], [.7, -49, 2.5], (250, 3))
        cloud = voxel_centroids(np.concatenate((cloud, new)), leaf)
        removed, added, rebuilt = cache.update(new)
        assert not rebuilt
        original.cloud_fast(cloud, radius)
        gpu.update(removed, added, incremental)
        assert original.occupancy_matches(incremental), f'GPU/CPU occupancy mismatch at scan {i}'


def test_super_frontend_no_free_endpoint_is_a_recordable_failure():
    from isaaclab_ipc.frontend import _prepare_context
    from isaaclab_ipc.mpc import Config
    from isaaclab_ipc.planner import Planner
    from isaaclab_ipc.super_benchmark import planner_params

    planner = Planner(Config(planner_params('high_speed')), [0, 50, 1.5])
    from types import SimpleNamespace

    planner.grid = SimpleNamespace(endpoint_free=lambda _: False, cloud_fast=lambda *_: None)
    result = _prepare_context((planner, np.array([0., -50, 1.5, 0, 0, 0, 0, 0, 0]), None))
    assert not result.valid
    assert planner.frontend_failures == 1
    assert planner.replan


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp GPU grid requires CUDA')
def test_warp_high_speed_grid_matches_original_a_star():
    from isaaclab_ipc.warp_inflation import WarpInflatedGrid

    size, center = (9, 12, 6), (0, -50, 1.5)
    low, high = [-4.5, -6, -3], [4.5, 6, 3]
    rng = np.random.default_rng(99)
    hits = rng.uniform([-1.4, -53.5, .6], [1.4, -46.5, 2.5], (180, 3))
    old = _geometry.Grid(.3, low, high)
    new = _geometry.Grid(.3, low, high)
    old.center(center)
    new.center(center)
    old.cloud_fast(hits, .6)
    gpu = WarpInflatedGrid(.3, size, center, .6)
    gpu.update([], hits, new)
    assert old.occupancy_matches(new)
    np.testing.assert_array_equal(
        old.search([3, -54, 1.5], [3, -46, 1.5]),
        new.search([3, -54, 1.5], [3, -46, 1.5]),
    )
