import numpy as np
import pytest

from isaaclab_ipc import _geometry
from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import InvalidCorridorError, Planner


def test_metric_clearance_and_plane_format():
    cloud = np.array([[2, y, z] for y in np.linspace(-2, 2, 20) for z in np.linspace(0.5, 3, 10)])
    seed, end = np.array([0, 0, 1.0]), np.array([1, 0, 1.0])
    planes = _geometry.corridor(seed, end, cloud, 0.3)
    np.testing.assert_array_equal(planes, _geometry.corridor(seed, end, cloud, 0.3))
    a, b = planes[:, :3], -planes[:, 3]
    assert np.isfinite(planes).all()
    assert (a @ seed <= b + 1e-8).all() and (a @ end <= b + 1e-8).all()
    distance = (cloud @ a.T - b) / np.linalg.norm(a, axis=1)
    assert (distance.max(1) >= 0.3 - 1e-8).all()


def test_unsafe_seed_is_rejected_without_dropping_obstacles():
    with pytest.raises(_geometry.CorridorError):
        _geometry.corridor(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), [[0.1, 0, 1]], 0.3)


def test_planner_routes_invalid_geometry_to_frontend_recovery():
    planner = Planner(Config.load(), [8, 0, 1])
    planner.cloud = np.array([[0.1, 0, 1]])
    with pytest.raises(InvalidCorridorError):
        planner._corridor(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))
