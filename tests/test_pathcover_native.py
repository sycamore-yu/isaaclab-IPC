"""Behavioral checks for the complete native PathCover front end and its time contract."""

from dataclasses import replace

import numpy as np
import pytest

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
    PathCoverTrajectory,
    validate_pathcover_plan,
)


def configuration(**overrides):
    return replace(
        PathCoverConfig(bounds_lower=(-2.0, -3.0, 0.0), bounds_upper=(6.0, 3.0, 3.0),
                        resolution=0.2, robot_radius=0.2, max_velocity=2.0,
                        max_iterations=256, use_cuda=False),
        **overrides,
    )


def state():
    return PathCoverState(np.array([-1.0, 0.0, 1.5]), np.zeros(3), np.zeros(3), time=7.0)


def wall():
    y, z = np.meshgrid(np.arange(-0.6, 0.61, 0.1), np.arange(0.4, 2.61, 0.1), indexing="ij")
    return np.stack((np.full_like(y, 1.4), y, z), axis=-1).reshape(-1, 3)


def native_module():
    return pytest.importorskip("isaaclab_ipc._pathcover_native")


def test_polynomial_sampling_preserves_pva_and_holds_after_expiry():
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 0, 3:] = (0.5, 2.0, 1.0)  # x(t)=1+2t+0.5t²
    trajectory = PathCoverTrajectory(coefficients, np.array([2.0]))
    sample = trajectory.sample(np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(sample.position[:, 0], (1.0, 3.5, 7.0))
    np.testing.assert_allclose(sample.velocity[:, 0], (2.0, 3.0, 0.0))
    np.testing.assert_allclose(sample.acceleration[:, 0], (1.0, 1.0, 0.0))
    assert sample.pva.shape == (3, 9)
    assert trajectory.sample(0.5).position.shape == (3,)


def test_heading_limits_body_z_rate_instead_of_world_heading_derivative():
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, 0, 3:5] = (4.0, -3.0)  # constant horizontal acceleration 8 m/s²
    coefficients[0, 1, 4] = 0.2
    trajectory = PathCoverTrajectory(coefficients, np.array([0.3]), initial_yaw=np.pi / 2,
                                    body_yaw_rate_max=1.5)
    times = 0.5 * (trajectory._yaw_times[:-1] + trajectory._yaw_times[1:])
    z = np.array([8.0, 0.0, 9.81])
    z /= np.linalg.norm(z)

    def frames(t):
        yaw = trajectory.sample(t).yaw
        heading = np.stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)), axis=-1)
        y = np.cross(z, heading)
        y /= np.linalg.norm(y, axis=-1)[:, None]
        x = np.cross(y, z)
        return np.stack((x, y, np.broadcast_to(z, x.shape)), axis=-1)

    eps = 1e-6
    rotation = frames(times)
    derivative = (frames(times + eps) - frames(times - eps)) / (2 * eps)
    angular = np.swapaxes(rotation, -1, -2) @ derivative
    assert np.max(np.abs(angular[:, 1, 0])) <= 1.5 + 1e-4
    assert np.max(trajectory.sample(times).yaw_rate) > 1.5


def test_complete_frontend_obstacle_route_and_piece_corridor_contract():
    native_module()
    cfg = configuration()
    frontend = NativePathCoverFrontend(cfg)
    plan = frontend.plan(wall(), state(), np.array([5.0, 0.0, 1.5]))
    assert 1 < len(plan.corridors) <= cfg.max_corridors
    assert len(plan.segments) == len(plan.trajectory.durations)
    assert [segment.corridor_index for segment in plan.segments] == list(range(len(plan.corridors)))
    assert plan.start_time == 7.0
    assert plan.valid_until == pytest.approx(7.0 + plan.segments[-1].end)
    np.testing.assert_allclose(plan.sample(0.0).pva, np.concatenate((state().position, np.zeros(6))), atol=1e-7)
    np.testing.assert_allclose(plan.sample(plan.trajectory.duration).position, plan.local_goal, atol=1e-6)
    np.testing.assert_allclose(plan.sample(plan.trajectory.duration + 1).velocity, 0.0)
    metrics = validate_pathcover_plan(plan, cfg)
    assert metrics["finite"]
    # GCOPTER imposes soft penalties. These tolerances are measured, not a hard guarantee.
    assert metrics["sfc_max_violation_m"] < 0.02
    assert metrics["max_speed_m_s"] < cfg.max_velocity + 0.1
    assert metrics["max_thrust_n"] < cfg.native_options()["thrust_max"] + 0.2


def test_cpu_cuda_mapping_and_native_solution_agree():
    native = native_module()
    if not native.cuda_built:
        pytest.skip("explicit CPU-only extension")
    cfg = configuration()
    cpu = NativePathCoverFrontend(cfg)
    gpu = NativePathCoverFrontend(replace(cfg, use_cuda=True, cuda_min_grid_voxels=0))
    for shift in (0.0, 0.05, -0.04):
        cloud = wall() + np.array([shift, 0.0, 0.0])
        origin = state().position + np.array([0.03, shift, 0.07])
        cpu.update_cloud(cloud, origin)
        gpu.update_cloud(cloud, origin)
        assert cpu._native.map_fingerprint() == gpu._native.map_fingerprint()
    cpu_plan = cpu.plan(None, state(), np.array([5.0, 0.0, 1.5]))
    gpu_plan = gpu.plan(None, state(), np.array([5.0, 0.0, 1.5]))
    np.testing.assert_array_equal(cpu.inflated_cloud(), gpu.inflated_cloud())
    np.testing.assert_array_equal(cpu._native.potential_map(), gpu._native.potential_map())
    np.testing.assert_allclose(cpu_plan.path, gpu_plan.path, atol=1e-12)
    np.testing.assert_allclose(cpu_plan.trajectory.coefficients, gpu_plan.trajectory.coefficients, atol=1e-9)
    assert gpu._native.last_update.gpu_mapping
    assert gpu._native.last_update.gpu_inflation
    assert gpu_plan.metrics["gpu_distance_map"]


def test_coarse_search_preserves_fine_map_and_excludes_full_occupied_boxes():
    native_module()
    cfg = configuration(resolution=0.1, search_resolution=0.25)
    coarse = NativePathCoverFrontend(cfg)
    fine = NativePathCoverFrontend(replace(cfg, search_resolution=None))
    coarse_plan = coarse.plan(wall(), state(), np.array([5.0, 0.0, 1.5]))
    fine.update_cloud(wall(), state().position)
    np.testing.assert_array_equal(coarse.inflated_cloud(), fine.inflated_cloud())
    centres, half_extent = coarse.obstacle_boxes()
    np.testing.assert_allclose(centres / cfg.search_resolution - 0.5,
                               np.round(centres / cfg.search_resolution - 0.5), atol=1e-10)
    for corridor in coarse_plan.corridors:
        # Minimum residual over a box is c.a + d - half*|a|_1. At least one
        # complete separating face must exclude every occupied voxel volume.
        minimum_residual = (centres @ corridor[:, :3].T + corridor[:, 3]
                            - half_extent * np.abs(corridor[:, :3]).sum(axis=1))
        assert np.all(minimum_residual.max(axis=1) > 0.0)
    metrics = validate_pathcover_plan(coarse_plan, cfg)
    assert metrics["sfc_max_violation_m"] < 0.02
    assert metrics["max_tilt_rad"] < cfg.tilt_max + 0.02
    assert metrics["max_speed_m_s"] < cfg.max_velocity + 0.1


def test_body_radius_smaller_than_voxel_still_expands_occupied_volume():
    native_module()
    frontend = NativePathCoverFrontend(configuration(resolution=0.3, robot_radius=0.2))
    source = np.array([0.15, 0.15, 1.35])
    frontend.update_cloud(source[None], state().position)
    cloud = frontend.inflated_cloud()
    assert np.any(np.linalg.norm(cloud - (source + np.array([0.3, 0.0, 0.0])), axis=1) < 1e-6)


@pytest.mark.parametrize(("radius", "radius_z"), [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.25, 0.25)])
def test_inflation_excludes_tangent_cells_and_preserves_zero_radius_axes(radius, radius_z):
    native = native_module()
    cfg = configuration(resolution=0.25, robot_radius=radius, robot_radius_z=radius_z)
    source = np.array([0.125, 0.125, 1.125])
    # For radius=cell width, adjacent cells intersect the expanded volume;
    # cells two indices away only touch it and must not be filled wholesale.
    xy = (-1, 0, 1) if radius else (0,)
    zz = (-1, 0, 1) if radius_z else (0,)
    expected = {tuple(source + 0.25 * np.array([i, j, k])) for i in xy for j in xy for k in zz}
    frontends = [NativePathCoverFrontend(cfg)]
    if native.cuda_built:
        frontends.append(NativePathCoverFrontend(replace(cfg, use_cuda=True, cuda_min_grid_voxels=0)))
    for frontend in frontends:
        frontend.update_cloud(source[None], state().position)
        assert set(map(tuple, frontend.inflated_cloud())) == expected


def test_reset_isolates_observation_history_and_failure_is_explicit():
    native_module()
    frontend = NativePathCoverFrontend(configuration())
    frontend.update_cloud(wall(), state().position)
    assert len(frontend.inflated_cloud())
    with pytest.raises(PathCoverPlanningError, match="inflated occupancy"):
        frontend.plan(None, state(), np.array([1.5, 0.1, 1.5]))
    frontend.reset()
    assert len(frontend.inflated_cloud()) == 0
    plan = frontend.plan(None, state(), np.array([5.0, 0.0, 1.5]))
    assert len(plan.corridors) == 1
    assert plan.segments[0].corridor_index == 0


def test_virtual_workspace_clearance_and_iteration_limit_failure():
    native_module()
    cfg = configuration()
    frontend = NativePathCoverFrontend(cfg)
    below_ground = replace(state(), position=np.array([-1.0, 0.0, 0.1]))
    with pytest.raises(PathCoverPlanningError, match="workspace body clearance"):
        frontend.plan(None, below_ground, np.array([5.0, 0.0, 1.5]))
    plan = frontend.plan(None, state(), np.array([5.0, 0.0, 1.5]))
    floor = plan.corridors[0][np.all(plan.corridors[0][:, :3] == (0.0, 0.0, -1.0), axis=1)]
    assert floor[0, 3] == pytest.approx(cfg.bounds_lower[2] + cfg.robot_radius)
    limited = NativePathCoverFrontend(configuration(max_iterations=1))
    with pytest.raises(PathCoverPlanningError, match="GCOPTER optimization failed"):
        limited.plan(wall(), state(), np.array([5.0, 0.0, 1.5]))
    problem = limited.failed_problem()
    np.testing.assert_array_equal(problem["initial_pva"][:, 0], state().position)
    assert problem["stage"] == "optimize"
    assert problem["optimizer_status"] < 0
    assert len(problem["corridors"]) > 1
    assert problem["config"]["max_iterations"] == 1


def test_positive_optimizer_status_does_not_accept_speed_above_effective_limit():
    native_module()
    frontend = NativePathCoverFrontend(configuration(penalty_weights=(1e4, 0.0, 1e4, 1e4, 1e4)))
    with pytest.raises(PathCoverPlanningError, match="constraints violated at optimizer quadrature nodes"):
        frontend.plan(None, state(), np.array([5.0, 0.0, 1.5]))
    problem = frontend.failed_problem()
    assert problem["stage"] == "feasibility"
    assert problem["optimizer_status"] >= 0
    assert problem["constraint_metrics"][1] > 2.002
    assert problem["constraint_bounds"]["speed_max_m_s"] == 2.0
    assert problem["soft_constraint_allowed_error"]["speed_relative"] == 1e-3
    assert problem["soft_constraint_allowed_error"]["speed_m_s"] == 0.002
    assert problem["soft_constraint_allowed_error"]["sfc_m"] == 1e-4
    speed = problem["speed_feasibility"]
    assert speed["initial_speed_m_s"] == 0.0
    assert speed["nominal_limit_m_s"] == 2.0
    assert speed["effective_limit_m_s"] == 2.002
    assert speed["peak_speed_m_s"] == problem["constraint_metrics"][1]
    assert speed["nominal_excess_m_s"] == speed["peak_speed_m_s"] - 2.0
    candidate = problem["candidate"]
    assert np.isfinite(candidate["objective"]) and candidate["objective"] > 0.0
    trajectory = PathCoverTrajectory(np.stack(candidate["coefficients"]), np.asarray(candidate["durations"]))
    np.testing.assert_array_equal(candidate["durations"], problem["trial_durations"])
    np.testing.assert_array_equal(candidate["corridor_indices"], [0])
    np.testing.assert_allclose(trajectory.sample(0.0).pva, problem["initial_pva"].T.ravel(), atol=1e-9)
    np.testing.assert_allclose(trajectory.sample(trajectory.duration).pva, problem["terminal_pva"].T.ravel(), atol=1e-9)
    samples = trajectory.sample(np.linspace(0.0, trajectory.duration, frontend.config.integral_resolution + 1))
    assert np.linalg.norm(samples.velocity, axis=1).max() == pytest.approx(speed["peak_speed_m_s"], abs=1e-10)


def test_fixed_initial_overspeed_preserves_nominal_bound_and_real_excess():
    native_module()
    frontend = NativePathCoverFrontend(configuration())
    moving = replace(state(), velocity=np.array([3.0, 0.0, 0.0]))
    plan = frontend.plan(None, moving, np.array([5.0, 0.0, 1.5]))
    np.testing.assert_allclose(plan.sample(0.0).velocity, moving.velocity, atol=1e-9)
    assert plan.metrics["speed_nominal_limit_m_s"] == 2.0
    assert plan.metrics["speed_initial_m_s"] == 3.0
    assert plan.metrics["speed_effective_limit_m_s"] == 3.002
    peak = plan.metrics["optimizer_speed_max_m_s"]
    assert 3.0 - 1e-9 <= peak <= 3.002
    assert plan.metrics["speed_nominal_excess_m_s"] == peak - 2.0


def test_soft_speed_residual_and_original_corridor_metrics_are_preserved():
    native_module()
    cfg = configuration(safety_margin=0.01)
    plan = NativePathCoverFrontend(cfg).plan(None, state(), np.array([5.0, 0.0, 1.5]))
    # A finite soft penalty retains a true speed residual, rather than rounding
    # it away or claiming a strict continuous-time speed guarantee.
    speed = plan.metrics["optimizer_speed_max_m_s"]
    assert cfg.max_velocity + 1e-3 < speed <= cfg.max_velocity * 1.001
    assert speed <= validate_pathcover_plan(plan, cfg)["max_speed_m_s"] + 1e-9
    positions = plan.sample(np.linspace(0, plan.trajectory.duration, cfg.integral_resolution + 1)).position
    original = plan.corridors[0]
    expected = (positions @ original[:, :3].T + original[:, 3]).max()
    assert plan.metrics["optimizer_sfc_max_m"] == pytest.approx(expected, abs=1e-9)


def test_planning_failure_type_excludes_invalid_parameters_and_nonfinite_input():
    native = native_module()
    assert issubclass(native.PathCoverPlanningError, PathCoverPlanningError)
    with pytest.raises(ValueError, match="configuration"):
        NativePathCoverFrontend(configuration(position_penalty_smoothing_m=0.0))
    frontend = NativePathCoverFrontend(configuration())
    invalid = replace(state(), velocity=np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="must be finite"):
        frontend.plan(None, invalid, np.array([5.0, 0.0, 1.5]))


def test_dmp_cannot_escape_an_axially_enclosed_cell_through_its_corner():
    native_module()
    cfg = configuration(bounds_lower=(0.0, 0.0, 0.0), bounds_upper=(4.0, 4.0, 4.0),
                        resolution=1.0, robot_radius=0.0, robot_radius_z=0.0)
    start = PathCoverState(np.array([1.5, 1.5, 1.5]), np.zeros(3), np.zeros(3))
    occupied = start.position + np.concatenate((np.eye(3), -np.eye(3)))
    frontend = NativePathCoverFrontend(cfg)
    # The destination is free, but every edge leaving the start touches at least
    # one occupied side cell. A diagonal between free centres is not sufficient.
    with pytest.raises(PathCoverPlanningError, match="DMP did not find"):
        frontend.plan(occupied, start, np.array([2.5, 2.5, 2.5]))


def test_explicit_motion_seed_rejects_colliding_hull_and_malformed_input():
    native_module()
    frontend = NativePathCoverFrontend(configuration())
    start = state()
    seed = np.array([start.position, start.position + (0.1, 0.0, 0.0), (1.5, 0.1, 1.5)])
    with pytest.raises(PathCoverPlanningError, match="motion seed intersects"):
        frontend.plan(wall(), start, np.array([5.0, 0.0, 1.5]), initial_motion_seed=seed)
    with pytest.raises(ValueError, match="three finite points"):
        frontend.plan(None, start, np.array([5.0, 0.0, 1.5]), initial_motion_seed=seed[:2])
    seed[0, 0] += 0.01
    with pytest.raises(ValueError, match="starting at the current position"):
        frontend.plan(None, start, np.array([5.0, 0.0, 1.5]), initial_motion_seed=seed)
