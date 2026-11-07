"""Physical reference and rollout contracts, independent of the cost mapping."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from isaaclab_ipc.turbompc.quadrotor import CTBRDynamics, reference_from_pva


@pytest.mark.parametrize("yaw", [0.0, 0.7, np.pi, -2.8])
def test_reference_attitude_matches_thrust_direction_and_heading(yaw):
    pva = torch.tensor(
        [[[1, 2, 3, 0.4, -0.2, 0.1, 2.0, -1.0, 0.6], [-2, 3, 1, -0.3, 0.1, -0.1, -0.5, 3.0, -0.8]]],
        dtype=torch.float64,
    )
    result = reference_from_pva(pva, torch.tensor([yaw], dtype=torch.float64))
    rotations = Rotation.from_quat(result[0, :, [4, 5, 6, 3]].numpy()).as_matrix()
    # SciPy's independent conversion checks both axis alignment and handedness.
    thrust = pva[0, :, 6:9].numpy() + np.array([0, 0, 9.81])
    z = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
    y = np.cross(z, np.array([np.cos(yaw), np.sin(yaw), 0]))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    expected = np.stack((np.cross(y, z), y, z), axis=-1)
    np.testing.assert_allclose(rotations, expected, atol=1e-14)
    torch.testing.assert_close(result[..., :3], pva[..., :3])
    torch.testing.assert_close(result[..., 7:], pva[..., 3:6])
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-14)


def test_reference_singularities_have_explicit_unit_attitude_conventions():
    pva = torch.zeros((1, 3, 9), dtype=torch.float64)
    pva[0, :, 6:9] = torch.tensor([[0, 0, -9.81], [1, 0, -9.81], [0, 0, -19.62]], dtype=torch.float64)
    result = reference_from_pva(pva, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(result[..., 3:7].norm(dim=-1), torch.ones((1, 3), dtype=torch.float64))
    rotation = Rotation.from_quat(result[0, :, [4, 5, 6, 3]].numpy()).as_matrix()
    np.testing.assert_allclose(rotation[:, :, 2], [[0, 0, 1], [1, 0, 0], [0, 0, -1]], atol=1e-7)


def test_rollout_uses_measured_initial_state_and_matches_independent_rotation():
    rng = np.random.default_rng(302)
    initial = np.zeros((3, 10))
    initial[:, :3] = rng.normal(size=(3, 3))
    initial[:, 7:] = rng.normal(size=(3, 3))
    initial[:, 3] = 1
    controls = rng.normal(size=(3, 4, 4)) * 0.3
    controls[..., 0] += 0.75 * 9.81
    model = CTBRDynamics()
    actual = model.rollout(torch.tensor(initial), torch.tensor(controls)).numpy()
    expected = [initial.copy()]
    state = initial.copy()
    for k in range(controls.shape[1]):
        q_xyzw = state[:, [4, 5, 6, 3]]
        orientation = Rotation.from_quat(q_xyzw)
        next_q = (orientation * Rotation.from_rotvec(0.1 * controls[:, k, 1:])).as_quat()
        # The canonical discrete model divides by norm + 1e-8 each step.
        raw_norm = np.linalg.norm(q_xyzw, axis=1, keepdims=True)
        next_q *= raw_norm / (raw_norm + 1e-8)
        force = orientation.apply(np.array([0.0, 0.0, 1.0])) * raw_norm**2
        acc = force * (controls[:, k, :1] / 0.75) - np.array([0, 0, 9.81])
        state = np.concatenate(
            (state[:, :3] + 0.1 * state[:, 7:], next_q[:, [3, 0, 1, 2]], state[:, 7:] + 0.1 * acc), axis=-1
        )
        expected.append(state.copy())
    np.testing.assert_allclose(actual, np.stack(expected, axis=1), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual[:, 0], initial)


def test_rollout_and_reference_are_fullgraph_compilable():
    model = CTBRDynamics()
    state = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0.1, 0, 0]], dtype=torch.float64)
    controls = torch.tensor([[[7.4, 0.1, -0.2, 0.3]] * 4], dtype=torch.float64)
    compiled = torch.compile(model.rollout, fullgraph=True, dynamic=False, backend="eager")
    torch.testing.assert_close(compiled(state, controls), model.rollout(state, controls), rtol=0, atol=0)
    pva = torch.zeros((1, 5, 9), dtype=torch.float64)
    pva[..., 6] = 1.0
    yaw = torch.tensor([0.3], dtype=torch.float64)
    reference = torch.compile(reference_from_pva, fullgraph=True, dynamic=False, backend="eager")
    torch.testing.assert_close(reference(pva, yaw), reference_from_pva(pva, yaw), rtol=0, atol=0)
