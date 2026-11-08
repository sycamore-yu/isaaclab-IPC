"""Actual native geometry modes, including the V28 one-box corner fixture."""

from dataclasses import replace

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
)

P0 = np.array([-.8190380334854126, 26.266569137573242, 2.998993158340454])
SOURCE_POINT = np.array([[-1.05, 26.05, 2.75]])


def config(model):
    return PathCoverConfig(
        bounds_lower=(-3.0, 24.0, 0.0), bounds_upper=(2.0, 30.0, 4.0),
        resolution=.1, search_resolution=.25, robot_radius=.2, robot_radius_z=.2,
        geometry_model=model, use_cuda=False, safety_margin=0.0,
    )


def native(options):
    module = pytest.importorskip("isaaclab_ipc._pathcover_native")
    assert hasattr(module.PlanResult, "geometry_model"), "load the box-ellipsoid candidate library"
    return module.NativeFrontend(options)


def test_native_box_ellipsoid_corner_fixture_and_actual_diagnostics():
    exact = native(config("box_ellipsoid").native_options())
    legacy_options = config("inflated_voxels").native_options()
    legacy_options.pop("geometry_model")  # Historical absence keeps the original model.
    legacy = native(legacy_options)
    for planner in (exact, legacy):
        planner.update_cloud(SOURCE_POINT, P0)
    np.testing.assert_array_equal(exact.inflated_cloud(), legacy.inflated_cloud())
    np.testing.assert_array_equal(exact.search_obstacle_centers(), legacy.search_obstacle_centers())
    assert len(exact.obstacle_centers()) == 125
    geometry = exact.geometry_obstacles()
    assert geometry["model"] == "box_ellipsoid" and geometry["body_radii"] == (.2, .2, .2)
    assert geometry["centers"].shape == geometry["halfwidths"].shape == (1, 3)
    lower = np.array([-11, 260, 27], dtype=float) * .1
    upper = np.array([-10, 261, 28], dtype=float) * .1
    assert np.all(geometry["centers"][0] - geometry["halfwidths"][0] <= lower)
    assert np.all(geometry["centers"][0] + geometry["halfwidths"][0] >= upper)
    assert exact.last_update.geometry_boxes == 1

    with pytest.raises(PathCoverPlanningError, match="occupancy"):
        legacy.plan(P0, np.zeros(3), np.zeros(3), P0)
    assert legacy.failed_problem()["geometry_model"] == "inflated_voxels"
    result = exact.plan(P0, np.zeros(3), np.zeros(3), P0)
    assert result.geometry_model == "box_ellipsoid"
    assert result.optimizer_status >= 0
    plane = result.corridors[0]
    normal = plane[:, :3]
    residual = (normal @ geometry["centers"][0] + plane[:, 3]
                - np.abs(normal) @ geometry["halfwidths"][0] - np.linalg.norm(normal * .2, axis=1))
    assert np.max(residual) > 0.0
    assert np.max(normal @ P0 + plane[:, 3]) <= 1e-10

    contact = np.array([-.8, 26.05, 2.75])
    with pytest.raises(PathCoverPlanningError, match="start intersects box-ellipsoid geometry"):
        exact.plan(contact, np.zeros(3), np.zeros(3), contact)
    assert exact.failed_problem()["geometry_model"] == "box_ellipsoid"
    exact.reset()
    assert exact.geometry_obstacles()["centers"].shape == (0, 3)
    assert exact.geometry_obstacles()["model"] == "box_ellipsoid"


def test_native_geometry_mode_validation_and_wrapper_actual_mode():
    options = config("box_ellipsoid").native_options()
    for changes in ({"geometry_model": "unknown"}, {"robot_radius_z": 0.0}, {"robot_radius": 0.0}):
        with pytest.raises(ValueError):
            native({**options, **changes})
    zero = native({**options, "robot_radius": 0.0, "robot_radius_z": 0.0})
    assert zero.geometry_obstacles()["body_radii"] == (0.0, 0.0, 0.0)
    planner = NativePathCoverFrontend(config("box_ellipsoid"))
    planner.update_cloud(SOURCE_POINT, P0)
    state = PathCoverState(P0, np.zeros(3), np.zeros(3))
    result = planner.plan(None, state, P0)
    assert result.metrics["geometry_model"] == "box_ellipsoid"
    assert planner.geometry_obstacles()["model"] == "box_ellipsoid"


def test_native_box_ellipsoid_cpu_cuda_geometry_match():
    module = pytest.importorskip("isaaclab_ipc._pathcover_native")
    torch = pytest.importorskip("torch")
    if not module.cuda_built or not torch.cuda.is_available():
        pytest.skip("native CUDA geometry comparison requires CUDA")
    planners = [native(replace(config("box_ellipsoid"), use_cuda=use_cuda,
                               cuda_min_grid_voxels=0).native_options()) for use_cuda in (False, True)]
    results = []
    for planner in planners:
        planner.update_cloud(SOURCE_POINT, P0)
        results.append(planner.plan(P0, np.zeros(3), np.zeros(3), [.5, 27.0, 3.1]))
    assert planners[0].last_update.gpu_grid_projection is False
    assert planners[1].last_update.gpu_grid_projection is True
    assert planners[1].last_update.gpu_box_compression is True
    for key in ("centers", "halfwidths"):
        np.testing.assert_array_equal(planners[0].geometry_obstacles()[key], planners[1].geometry_obstacles()[key])
    np.testing.assert_array_equal(planners[0].search_obstacle_centers(), planners[1].search_obstacle_centers())
    np.testing.assert_array_equal(results[0].durations, results[1].durations)
    np.testing.assert_array_equal(results[0].coefficients, results[1].coefficients)
    np.testing.assert_array_equal(results[0].path, results[1].path)
    assert results[0].endpoint_connection_visited_cells > 0
    for first, second in zip(results[0].corridors, results[1].corridors, strict=True):
        np.testing.assert_array_equal(first, second)
