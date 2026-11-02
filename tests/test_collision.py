import numpy as np
import warp as wp

from isaaclab_ipc.collision import latch_clearance


def test_full_point_collision_and_latching_cpu():
    points = wp.array([[0, 0, 1], [1, 0, 1]], dtype=wp.vec3, device="cpu")
    grid = wp.HashGrid(8, 8, 8, device="cpu")
    grid.build(points, 1.0)
    poses = wp.array([[0.2, 0, 1, 0, 0, 0, 1], [0.5, 0.5, 1, 0, 0, 0, 1]], dtype=wp.transform, device="cpu")
    ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    origins = wp.zeros(2, dtype=wp.vec3, device="cpu")
    y = wp.zeros(2, device="cpu")
    minimum = wp.full(2, 100.0, device="cpu")
    hit = wp.zeros(2, dtype=wp.bool, device="cpu")
    args = [poses, ids, origins, y, points, grid.id, 2, 0.25, minimum, hit]
    wp.launch(latch_clearance, dim=2, inputs=args, device="cpu")
    np.testing.assert_allclose(minimum.numpy(), [-0.05, np.sqrt(0.5) - 0.25], atol=1e-6)
    np.testing.assert_array_equal(hit.numpy(), [True, False])
    poses.assign(np.array([[0, 0, 2, 0, 0, 0, 1], [0, 0, 2, 0, 0, 0, 1]], dtype=np.float32))
    wp.launch(latch_clearance, dim=2, inputs=args, device="cpu")
    assert hit.numpy()[0]
