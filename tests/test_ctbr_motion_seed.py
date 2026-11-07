"""One compiled physical endpoint batch and lazy native seed-failure capture."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_ipc.ctbr_navigation import PathCoverNavigation
from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverPlanningError,
    PathCoverState,
)
from isaaclab_ipc.turbompc.ctbr import CTBRConfig
from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled CUDA mathematical contract")
@pytest.mark.parametrize("batch", [1, 16])
def test_compiled_five_points_match_physical_rollout_and_preserve_snapshot(batch):
    generator = torch.Generator().manual_seed(610)
    state = torch.randn((batch, 10), generator=generator, dtype=torch.float64)
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    # Non-unit values must retain the production model's raw-q acceleration.
    state[:, 3:7] *= .999
    config = CTBRConfig(mass=.9, specific_thrust_limits=(1., 25.))
    navigation = object.__new__(PathCoverNavigation)
    navigation.config = config
    navigation._dynamics = CTBRDynamics(dt=config.dt, mass=config.mass, gravity=config.gravity)
    controls = torch.randn((batch, 2, 4), generator=generator, dtype=torch.float64)
    controls[:, 0, 0] = torch.linspace(2., 20., batch)
    acceleration = navigation._dynamics.acceleration(torch.cat((state, controls[:, 0]), -1))
    expected = navigation._dynamics.rollout(state, controls)[:, :, :3]
    endpoints = []
    for bound in config.specific_thrust_limits:
        alternative = controls.clone()
        alternative[:, 0, 0] = config.mass * bound
        endpoints.append(navigation._dynamics.rollout(state, alternative)[:, 2, :3])
    kernel = torch.compile(navigation._initial_motion_seed, fullgraph=True, dynamic=False)
    state_gpu, acceleration_gpu = state.cuda(), acceleration.cuda()
    for force in (controls[:, 0, 0].clone(memory_format=torch.contiguous_format).cuda(), None):
        result = kernel(state_gpu, acceleration_gpu, force)
        frozen = result.clone()
        torch.testing.assert_close(result[:, :3].cpu(), expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(result[:, 3:].cpu(), torch.stack(endpoints, 1), rtol=0, atol=1e-12)
        # Reusing the complete kernel with new inputs must not mutate old output.
        kernel(state_gpu + torch.tensor([.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device="cuda"),
               acceleration_gpu, force)
        torch.testing.assert_close(result, frozen, rtol=0, atol=0)


def test_native_seed_failure_retains_tip_segment_only_until_requested():
    seed = np.array([[0., 0., 1.], [.1, 0., 1.], [.2, 0., 1.]])
    segment = np.array([[.2, 0., .9], [.2, 0., 1.2]])
    received = []

    def fail(*args):
        received.append(args)
        raise PathCoverPlanningError("No collision-free motion tip")

    frontend = NativePathCoverFrontend.__new__(NativePathCoverFrontend)
    frontend.config = PathCoverConfig(use_cuda=False)
    frontend._native = SimpleNamespace(plan=fail, failed_problem=lambda: None, reset=lambda: None)
    frontend._yaw_failure = frontend._native_failure = None
    state = PathCoverState(seed[0], np.array([1., 0., 0.]), np.zeros(3))
    with pytest.raises(PathCoverPlanningError, match="motion tip"):
        frontend.plan(None, state, np.array([2., 0., 1.]),
                      initial_motion_seed=seed, initial_motion_tip_segment=segment)
    assert np.shares_memory(received[0][-2], seed)
    assert np.shares_memory(received[0][-1], segment)
    assert np.shares_memory(frontend._native_failure[0], seed)
    assert np.shares_memory(frontend._native_failure[1], segment)
    problem = frontend.failed_problem()
    np.testing.assert_array_equal(problem["initial_motion_seed"], seed)
    np.testing.assert_array_equal(problem["initial_motion_tip_segment"], segment)
    assert not np.shares_memory(problem["initial_motion_tip_segment"], segment)
    problem["initial_motion_tip_segment"][:] = 0.
    np.testing.assert_array_equal(frontend.failed_problem()["initial_motion_tip_segment"], segment)
    frontend.reset()
    assert frontend.failed_problem() is None

    def software_error(*args):
        raise RuntimeError("ABI mismatch")

    frontend._native.plan = software_error
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        frontend.plan(None, state, np.array([2., 0., 1.]),
                      initial_motion_seed=seed, initial_motion_tip_segment=segment)
    assert frontend._native_failure is None
