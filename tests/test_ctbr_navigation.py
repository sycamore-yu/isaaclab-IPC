"""The frontend's polynomial segment times determine the MPC's SFC schedule."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_ipc.ctbr_navigation import PathCoverNavigation, position_sfc_violation, sample_plan


def fixture_plan():
    first = np.array([[1., 0., 0., -1.], [-1., 0., 0., 0.]])
    second = np.array([[1., 0., 0., -2.], [-1., 0., 0., 1.]])

    def sample(times):
        pva = np.zeros((len(times), 9))
        pva[:, 0], pva[:, 3] = times, 1.
        return SimpleNamespace(pva=pva, yaw=np.full(len(times), .3))

    return SimpleNamespace(
        start_time=10., valid_until=12., trajectory=SimpleNamespace(sample=sample),
        corridors=(first, second), segments=(
            SimpleNamespace(corridor_index=0, start=0., end=1., duration=1.),
            SimpleNamespace(corridor_index=1, start=1., end=2., duration=1.),
        ),
    )


def test_native_segment_clock_and_intersection():
    value = sample_plan(fixture_plan(), 10.5, 4, .5, 4)
    np.testing.assert_array_equal(value.corridor_index, [0, 1, 1, 1])
    np.testing.assert_array_equal(value.plane_mask.sum(-1), [2, 4, 2, 2])
    np.testing.assert_allclose(value.pva[:, 0], [.5, 1., 1.5, 2.])
    assert value.reference_violation == 0.


def test_terminal_hold_does_not_extrapolate_polynomial():
    value = sample_plan(fixture_plan(), 11.9, 3, .1, 4)
    np.testing.assert_allclose(value.pva[:, 0], [1.9, 2., 2.])
    np.testing.assert_array_equal(value.pva[-1, 3:], 0.)


def test_face_capacity_never_truncates_geometry():
    with pytest.raises(ValueError, match="requires 4 SFC faces"):
        sample_plan(fixture_plan(), 11., 2, .1, 3)


def test_observed_position_uses_active_corridor_and_junction():
    plan = fixture_plan()
    assert position_sfc_violation(plan, [1.2, 0., 0.], 10.5) == pytest.approx(.2)
    assert position_sfc_violation(plan, [1.2, 0., 0.], 11.0) == pytest.approx(.2)
    assert position_sfc_violation(plan, [1.2, 0., 0.], 11.1) == 0.


def test_batch_selection_planning_failure_and_snapshot_ownership(monkeypatch):
    from isaaclab_ipc import pathcover_native
    from isaaclab_ipc.turbompc.ctbr import CTBRConfig

    instances = []
    failures = {}

    class Frontend:
        def __init__(self, config):
            self.config = config or pathcover_native.PathCoverConfig()
            self.index = len(instances)
            self.calls = self.resets = 0
            instances.append(self)

        def reset(self):
            self.resets += 1

        def plan(self, points, state, goal, *, initial_motion_seed):
            self.calls += 1
            assert points == self.index
            assert state.position[0] == self.index
            assert goal[0] == 20 + self.index
            np.testing.assert_allclose(initial_motion_seed,
                                       state.position + np.arange(3)[:, None] * .1 * state.velocity)
            if self.index in failures:
                raise failures[self.index]

            def sample(times):
                pva = np.zeros((len(times), 9))
                pva[:, :3] = state.position
                return SimpleNamespace(pva=pva, yaw=np.full(len(times), .1 * self.index))

            return SimpleNamespace(
                start_time=state.time, valid_until=state.time + 1., metrics={
                    "env_id": self.index,
                    "optimizer_quadrature_feasible": self.index != 2,
                    "optimizer_sfc_max_m": .003 if self.index == 2 else -.1,
                    "optimizer_speed_max_m_s": 7. if self.index == 2 else 4.,
                    "optimizer_thrust_min_n": -.1 if self.index == 2 else 2.,
                    "optimizer_thrust_max_n": 23. if self.index == 2 else 8.,
                    "optimizer_tilt_max_rad": 1.3 if self.index == 2 else .2,
                    "optimizer_zero_yaw_body_rate_max_rad_s": 3.01 if self.index == 2 else 1.,
                    "speed_nominal_excess_m_s": 1. if self.index == 2 else 0.,
                },
                trajectory=SimpleNamespace(sample=sample),
                corridors=(np.array([[1., 0., 0., -100.], [-1., 0., 0., -100.]]),),
                segments=(SimpleNamespace(corridor_index=0, start=0., end=1., duration=1.),),
            )

    monkeypatch.setattr(pathcover_native, "NativePathCoverFrontend", Frontend)
    navigation = PathCoverNavigation(3, CTBRConfig(cost_nodes=2), None, device="cpu", max_planes=4)
    state = torch.zeros((3, 10), dtype=torch.float64)
    state[:, 0], state[:, 2], state[:, 3] = torch.arange(3), 1., 1.
    state[:, 7] = torch.tensor([.3, .6, .9])
    goals = state[:, :3].clone()
    goals[:, 0] += 20
    rates = torch.zeros((3, 3), dtype=torch.float64)
    args = (state, rates, rates, goals, [0, 1, 2], torch.tensor([10., 11., 12.]))
    try:
        original = navigation.observe(*args, env_ids=[2, 0])
        saved = {name: value.clone() for name, value in original.items()}
        torch.testing.assert_close(original["ctbr_state"][:, 0], torch.tensor([2., 0.], dtype=torch.float64))
        assert [instance.calls for instance in instances] == [1, 0, 1]
        assert navigation.last_plans[1] is None
        assert navigation.reference_quality["returned_plans"] == 2
        assert navigation.reference_quality["optimizer_quadrature_infeasible_plans"] == 1

        failures[1] = pathcover_native.PathCoverPlanningError("No free path")
        result, errors = navigation.observe(*args, env_ids=[2, 1, 0], allow_failures=True)
        assert list(errors) == [1]
        assert result["sfc_valid"].flatten().tolist() == [True, False, True]
        torch.testing.assert_close(result["reference"][[0, 2], 0, 0],
                                   torch.tensor([2., 0.], dtype=torch.float64))
        for name in saved:
            torch.testing.assert_close(original[name], saved[name])
        navigation.reset([1])
        assert [instance.resets for instance in instances] == [0, 1, 0]
        assert navigation.last_plans[2] is not None
        quality = navigation.reference_quality
        assert quality["returned_plans"] == 4
        assert quality["optimizer_quadrature_infeasible_plans"] == 2
        assert quality["optimizer_extrema"]["sfc_max_m"] == .003
        assert quality["optimizer_extrema"]["thrust_min_n"] == -.1
        assert quality["maximum_positive_residual"] == pytest.approx({
            "sfc_m": .003, "speed_physical_m_s": 1., "speed_nominal_m_s": 1.,
            "thrust_lower_n": .1, "thrust_upper_n": 23. - 3. * .75 * 9.81,
            "tilt_rad": .1, "zero_yaw_body_rate_rad_s": .01,
        })
        assert quality["maximum_mpc_reference_sfc_violation_m"] == 0.

        with pytest.raises(pathcover_native.PathCoverPlanningError, match="No free path"):
            navigation.observe(*args, env_ids=[1])
        failures[1] = RuntimeError("Unexpected extension failure")
        with pytest.raises(RuntimeError, match="Unexpected extension failure"):
            navigation.observe(*args, env_ids=[1], allow_failures=True)
        assert quality["returned_plans"] == 4
        assert quality["optimizer_quadrature_infeasible_plans"] == 2
    finally:
        navigation.close()


def test_initial_motion_seed_matches_exact_ctbr_euler_positions():
    from isaaclab_ipc.turbompc.ctbr import CTBRConfig
    from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics

    generator = torch.Generator().manual_seed(53)
    state = torch.randn((16, 10), generator=generator, dtype=torch.float64)
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    controls = torch.randn((16, 2, 4), generator=generator, dtype=torch.float64)
    controls[..., 0] = torch.linspace(0., 22., 32).reshape(16, 2)
    config = CTBRConfig()
    navigation = object.__new__(PathCoverNavigation)
    navigation.config = config
    navigation._dynamics = CTBRDynamics(dt=config.dt, mass=config.mass, gravity=config.gravity)
    acceleration = navigation._dynamics.acceleration(torch.cat((state, controls[:, 0]), -1))
    expected = navigation._dynamics.rollout(state, controls)[..., :3]
    for force in (controls[:, 0, 0], None):
        actual = navigation._initial_motion_seed(state, acceleration, force)
        torch.testing.assert_close(actual, expected, rtol=0., atol=1e-12)
