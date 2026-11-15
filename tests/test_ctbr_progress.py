"""Tests for the new virtual-progress path, not the existing CTBR implementation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def straight_plan(duration=2.0):
    faces = np.array([[1., 0., 0., -10.], [-1., 0., 0., 0.],
                      [0., 1., 0., -2.], [0., -1., 0., -2.],
                      [0., 0., 1., -4.], [0., 0., -1., 0.]])

    def sample(times):
        t = np.asarray(times)
        pva = np.zeros(t.shape + (9,))
        pva[..., 0] = 10 * t / duration
        pva[..., 2] = 1.5
        pva[..., 3] = 10 / duration
        return SimpleNamespace(pva=pva, yaw=np.zeros_like(t))

    return SimpleNamespace(
        trajectory=SimpleNamespace(sample=sample), start_time=0.,
        segments=(SimpleNamespace(start=0., end=duration, corridor_index=0),),
        corridors=(faces,), local_goal=np.array([10., 0., 1.5]),
    )


def progress_observation(device='cpu'):
    from isaaclab_ipc.ctbr_progress import sample_progress_plan
    from isaaclab_ipc.rl.ctbr_observation import add_progress_observation, make_ctbr_observation
    from isaaclab_ipc.turbompc.ctbr import CTBRConfig

    config = CTBRConfig()
    state = torch.zeros((1, 10), dtype=torch.float64, device=device)
    state[:, 2], state[:, 3], state[:, 7] = 1.5, 1., 1.
    sample = sample_progress_plan(straight_plan(), state[0].cpu().numpy(), 5, .1, 16)
    obs = make_ctbr_observation(
        state, state.new_zeros((1, 3)), state.new_tensor([[10., 0., 1.5]]),
        torch.from_numpy(sample.pva[None]).to(device),
        torch.from_numpy(sample.yaw[None]).to(device),
        torch.from_numpy(sample.planes[None]).to(device),
        torch.from_numpy(sample.plane_mask[None]).to(device), ctbr_config=config,
    )
    return add_progress_observation(obs, state.new_tensor(sample.progress[None])), config


def test_path_progress_uses_geometry_not_minco_clock():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    state = np.array([0., 0., 1.5, 1., 0., 0., 0., 2., 0., 0.])
    fast = sample_progress_plan(straight_plan(2.), state, 5, .1, 16)
    slow = sample_progress_plan(straight_plan(7.), state, 5, .1, 16)
    np.testing.assert_allclose(fast.pva, slow.pva, atol=1e-10)
    np.testing.assert_allclose(fast.progress, slow.progress, atol=1e-10)
    assert fast.progress[0, 0] == 0.
    assert np.all(np.diff(fast.progress[:, 0]) > 0)
    assert fast.progress[1, 0] == pytest.approx(.2)
    np.testing.assert_allclose(fast.progress[:, 1:4], [[1., 0., 0.]] * 5)
    np.testing.assert_array_equal(fast.plane_mask.sum(-1), [6] * 5)
    np.testing.assert_allclose(fast.progress[:, 4:6], [[0., 10.]] * 5)
    assert np.all(fast.progress[:, 7] <= 4.)
    assert fast.local_path_length_m == pytest.approx(10.)


def test_reused_progress_reanchors_to_measured_position():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    state = np.array([2., 0., 1.5, 1., 0., 0., 0., 2., 0., 0.])
    snapshot = sample_progress_plan(straight_plan(), state, 5, .1, 16, reanchor=True)
    assert snapshot.pva[0, 0] == pytest.approx(2.)
    assert snapshot.progress[0, 0] == pytest.approx(0.)
    np.testing.assert_allclose(snapshot.progress[:, 4], -2.)
    np.testing.assert_allclose(snapshot.progress[:, 5], 8.)
    assert snapshot.local_path_length_m == pytest.approx(8.)
    assert np.all(np.diff(snapshot.progress[:, 0]) >= 0.)


def test_marsim_progress_limit_matches_native_super_scene():
    from isaaclab_ipc.benchmark_config import load_case
    from isaaclab_ipc.ros_ctbr_policy import (
        benchmark_frontend_config,
        benchmark_progress_speed,
        usable_progress_iterate,
    )
    from isaaclab_ipc.turbompc.ctbr import CTBRConfig

    assert benchmark_progress_speed(load_case('dense')) == 8.0
    assert benchmark_progress_speed(load_case('high_speed')) == 25.0
    assert benchmark_frontend_config(load_case('dense'), CTBRConfig()).max_corridors == 3
    assert benchmark_frontend_config(load_case('high_speed'), CTBRConfig()).max_corridors == 3
    assert usable_progress_iterate(False, 1, 5e-6, .006, True)
    assert not usable_progress_iterate(False, 1, 5e-6, .03, True)
    assert not usable_progress_iterate(False, 1, .05, 0., True)
    assert not usable_progress_iterate(False, 2, 1e-6, 0., True)


def test_standstill_previews_curve_instead_of_repeating_initial_tangent():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    plan = straight_plan()

    def curved(times):
        u = np.asarray(times) / 2.
        pva = np.zeros(u.shape + (9,))
        pva[..., 0] = 8 * u**3
        pva[..., 2] = 1.5 + .3 * u**2 * (1 - u)
        return SimpleNamespace(pva=pva, yaw=np.zeros_like(u))

    plan.trajectory.sample = curved
    state = np.array([0., 0., 1.5, 1., 0., 0., 0., 0., 0., 0.])
    sample = sample_progress_plan(plan, state, 5, .1, 16)
    assert sample.progress[-1, 0] > .1
    assert sample.progress[-1, 1] > .9
    # Progress and actual velocity are not replaced by this linearization seed.
    assert sample.progress[0, 0] == 0.
    np.testing.assert_array_equal(sample.progress[:, 6], 0.)
    assert .07 < sample.progress[1, 0] <= .1


def test_progress_dynamics_preserves_physics_and_integrates_virtual_speed():
    from isaaclab_ipc.turbompc.progress import ProgressDynamics
    from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics

    model = ProgressDynamics(dt=.1)
    z = torch.zeros((1, 16), dtype=torch.float64)
    z[:, 3], z[:, 10], z[:, 11], z[:, 15] = 1., 2., .75 * 9.81, 3.
    expected = CTBRDynamics()(torch.cat((z[:, :10], z[:, 11:15]), -1))
    actual, jac = model.linearize(z)
    torch.testing.assert_close(actual[:, :10], expected)
    assert actual[0, 10] == pytest.approx(2.3)
    assert jac[0, 10, 10] == pytest.approx(1.)
    assert jac[0, 10, 15] == pytest.approx(.1)
    lagrange = torch.ones((1, 11), dtype=z.dtype)
    hessian = model.lagrangian_hessian(z, lagrange)
    physical_hessian = model.physical.lagrangian_hessian(
        torch.cat((z[:, :10], z[:, 11:15]), -1), lagrange[:, :10])
    torch.testing.assert_close(hessian[0, :10, :10], physical_hessian[0, :10, :10])
    assert not hessian[0, 10].count_nonzero()
    assert not hessian[0, :, 15].count_nonzero()
    delta = torch.zeros_like(z)
    delta[:, 15] = 1e-5
    torch.testing.assert_close((model(z + delta) - model(z - delta)) / 2e-5, jac[:, :, 15])


def test_progress_cost_couples_position_velocity_and_progress():
    from isaaclab_ipc.turbompc.progress import ProgressQPBuilder

    obs, config = progress_observation()
    builder = ProgressQPBuilder(config)
    problem, guess = builder(obs)
    assert guess.shape == (1, 5, 16)
    assert problem.initial_state.shape == (1, 11)
    assert problem.D[0, 2, 0, 10] < 0  # p_x and s
    assert problem.D[0, 2, 7, 15] < 0  # v_x and virtual speed
    assert problem.q[0, -1, 10] < 0  # reward forward progress
    assert torch.linalg.eigvalsh(problem.D).min() > -1e-8
    assert torch.isfinite(problem.q).all()
    assert obs['policy'].shape == (1, 109)


def test_progress_hard_rows_preserve_source_sfc_and_progress_interval():
    from isaaclab_ipc.turbompc.progress import ProgressQPBuilder

    obs, config = progress_observation()
    builder = ProgressQPBuilder(config)
    data, _ = builder(obs)
    # Physical SFC rows keep their original world coordinates.
    active = obs['plane_mask'][0, 1]
    torch.testing.assert_close(data.G[0, 1, 9:9 + len(active), :3][active],
                               obs['planes'][0, 1, active, :3])
    assert data.lower[0, 1, 8] == pytest.approx(0.)
    assert data.upper[0, 1, 8] == pytest.approx(10.)


def test_progress_owns_corridor_by_arclength_at_junction():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    plan = straight_plan()
    first, second = plan.corridors[0].copy(), plan.corridors[0].copy()
    first[0, 3], second[1, 3] = -6., 4.
    plan.corridors = (first, second)
    plan.segments = (SimpleNamespace(start=0., end=1., corridor_index=0),
                     SimpleNamespace(start=1., end=2., corridor_index=1))
    state = np.array([0., 0., 1.5, 1., 0., 0., 0., 5., 0., 0.])
    sample = sample_progress_plan(plan, state, 5, .5, 16, max_speed=5.)
    np.testing.assert_allclose(sample.progress[:4, 0], [0., 2.5, 5., 7.5])
    assert 9. < sample.progress[-1, 0] < 10.
    np.testing.assert_array_equal(sample.corridor_index, [0, 0, 1, 1, 1])
    assert sample.plane_mask[2].sum() == 12
    np.testing.assert_allclose(sample.progress[2, 4:6], [5., 10.])


def test_progress_curvature_attitude_and_mission_goal_speed_limit():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    plan = straight_plan()

    def curve(times):
        u = np.asarray(times) / 2.
        pva = np.zeros(u.shape + (9,))
        pva[:, 0] = 10 * u
        pva[:, 1] = 2 * np.sin(np.pi * u)
        pva[:, 2] = 1.5
        return SimpleNamespace(pva=pva, yaw=np.zeros_like(u))

    plan.trajectory.sample = curve
    plan.local_goal = np.array([10., 0., 1.5])
    state = np.array([0., 0., 1.5, 1., 0., 0., 0., 3., 0., 0.])
    sample = sample_progress_plan(plan, state, 5, .1, 16, mission_goal=plan.local_goal)
    assert np.linalg.norm(sample.pva[:, 6:9], axis=-1).max() > .05
    assert np.all(sample.progress[:, 7] <= 4.)

    # Remaining distance, not the MINCO time clock, determines braking.
    close = straight_plan()
    close.local_goal = np.array([1., 0., 1.5])
    sample = sample_progress_plan(close, state, 5, .1, 16, mission_goal=close.local_goal)
    assert sample.progress[0, 7] == pytest.approx(np.sqrt(8.))
    assert sample.progress[-1, 7] < sample.progress[0, 7]


def test_progress_brakes_before_approaching_sharp_bend():
    from isaaclab_ipc.ctbr_progress import sample_progress_plan

    plan = straight_plan()

    def sharp_bend(times):
        u = np.asarray(times) / 2.
        x = 10. * u
        pva = np.zeros(u.shape + (9,))
        pva[:, 0] = x
        pva[:, 1] = .75 * (1. + np.tanh((x - 2.) / .2))
        pva[:, 2] = 1.5
        return SimpleNamespace(pva=pva, yaw=np.zeros_like(u))

    plan.trajectory.sample = sharp_bend
    state = np.array([0., 0., 1.5, 1., 0., 0., 0., 5., 0., 0.])
    result = sample_progress_plan(plan, state, 5, .1, 16)
    assert result.progress[0, 7] < 5.
    assert np.all(result.progress[:, 7] <= 4.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native CUDA MPC required')
def test_progress_changes_physical_control_and_cost_gradient():
    from dataclasses import asdict, replace

    from tensordict import TensorDict

    from isaaclab_ipc.rl.ctbr_actor import CTBRMPCActor

    values, config = progress_observation('cuda')
    obs = TensorDict(values, batch_size=[1])
    actor = CTBRMPCActor(obs, {'actor': ['policy']}, 'actor', 4,
                        model_config=asdict(config), progress_config={'reward': 0.8},
                        compile_kernels=False).to('cuda')
    try:
        actor.fixed_cost = True
        with torch.no_grad():
            driven = actor(obs).clone()
            assert actor.last_result['solved'][0]
            arc = actor.last_result['optimized_progress'].clone()
            actor.ctbr_problem.progress_config = replace(actor.ctbr_problem.progress_config, reward=0.)
            undriven = actor(obs).clone()
            assert actor.last_result['solved'][0]
            assert arc[0, -1] > actor.last_result['optimized_progress'][0, -1]
            assert (driven - undriven).abs().max() > 1e-4
            actor.ctbr_problem.progress_config = replace(actor.ctbr_problem.progress_config, reward=.8)

        costs = torch.zeros((1, config.cost_dim), device='cuda', dtype=torch.float64, requires_grad=True)
        data, guess = actor.ctbr_problem(obs, costs)
        result = actor.native_solver.solve(data, guess)
        assert result.status[0] == 0
        gradient = torch.autograd.grad(result.controls[0, 0, 2], costs)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-6
        direction = gradient.detach() / gradient.norm()
        outputs = []
        with torch.no_grad():
            for sign in (-1., 1.):
                problem, initial = actor.ctbr_problem(obs, costs.detach() + sign * 1e-3 * direction)
                perturbed = actor.native_solver.solve(problem, initial)
                assert perturbed.status[0] == 0
                outputs.append(float(perturbed.controls[0, 0, 2]))
        finite_difference = (outputs[1] - outputs[0]) / 2e-3
        assert finite_difference == pytest.approx(float((gradient * direction).sum()), rel=.03, abs=2e-3)
    finally:
        actor.close()
