from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp


def cube_mesh(device="cpu"):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32)
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        np.int32,
    )
    return wp.Mesh(
        wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces.ravel(), dtype=wp.int32, device=device)
    )


def test_indexed_rays_use_environment_mesh_transform_and_mask():
    from isaaclab_ipc.compact_sensor import indexed_raycast

    mesh = cube_mesh()
    # Identical geometry, distinct world transforms. The second environment rotates 90 degrees.
    qz = np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)], np.float32)
    poses = wp.array([[0, 0, 0, 0, 0, 0, 1], [10, 0, 0, *qz]], dtype=wp.transform, device="cpu")
    table = wp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=wp.vec3, device="cpu")
    starts = wp.array([0, 0], dtype=wp.int32, device="cpu")
    mesh_ids = wp.array(np.full((2, 1), mesh.id, np.uint64), dtype=wp.uint64, device="cpu")
    positions = wp.array(np.array([[[3, 0, 0]], [[10, 3, 0]]], np.float32), dtype=wp.vec3, device="cpu")
    rotations = wp.array(np.array([[[0, 0, 0, 1]], [qz]], np.float32), dtype=wp.quat, device="cpu")
    mask = wp.array([True, True], dtype=wp.bool, device="cpu")
    out = wp.zeros((2, 3), dtype=wp.vec3, device="cpu")
    pos_out = wp.zeros(2, dtype=wp.vec3, device="cpu")
    quat_out = wp.zeros(2, dtype=wp.quat, device="cpu")
    drift = wp.zeros(2, dtype=wp.vec3, device="cpu")
    inputs = [
        mask,
        poses,
        table,
        starts,
        1,
        wp.vec3(0.0),
        wp.quat_identity(),
        drift,
        mesh_ids,
        positions,
        rotations,
        40.0,
    ]
    wp.launch(indexed_raycast, (2, 3), inputs=inputs, outputs=[pos_out, quat_out, out], device="cpu")
    result = out.numpy()
    np.testing.assert_allclose(result[0, 0], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result[1, 0], [10, 2, 0], atol=1e-5)
    assert np.isinf(result[:, 1:]).all()
    # A partial refresh must preserve the other environment's cached measurements.
    mask.assign(np.array([True, False]))
    starts.assign(np.array([1, 0], np.int32))
    inputs[4] = 0  # one repeated direction is intentional for this mask test
    wp.launch(indexed_raycast, (2, 3), inputs=inputs, outputs=[pos_out, quat_out, out], device="cpu")
    assert np.isinf(out.numpy()[0]).all()
    np.testing.assert_array_equal(out.numpy()[1], result[1])


def test_shared_mesh_loads_once_but_respects_scale(monkeypatch):
    from isaaclab_newton.sensors import LegacyMultiMeshRayCaster

    from isaaclab_ipc.compact_sensor import SharedMid360RayCaster

    loads = []
    monkeypatch.setattr(
        LegacyMultiMeshRayCaster,
        "_load_target_prim_warp_mesh",
        lambda self, prim, cfg, reference_prim=None: loads.append(prim) or len(loads),
    )
    monkeypatch.setattr("isaaclab_ipc.compact_sensor.sim_utils.resolve_prim_scale", lambda prim: prim.scale)
    sensor = SimpleNamespace(_shared_target_ids={}, cfg=SimpleNamespace(reference_meshes=True), _device="cpu")
    cfg = SimpleNamespace(prim_expr="/World/envs/env_.*/Forest", is_shared=True, merge_prim_meshes=True)
    # Call through an instance with the real inheritance chain, without starting a simulator.
    obj = object.__new__(SharedMid360RayCaster)
    obj.__dict__.update(vars(sensor))
    first = SimpleNamespace(scale=(1, 1, 1))
    second = SimpleNamespace(scale=(1, 1, 1))
    assert obj._load_target_prim_warp_mesh(first, cfg) == obj._load_target_prim_warp_mesh(second, cfg)
    assert len(loads) == 1
    obj._load_target_prim_warp_mesh(SimpleNamespace(scale=(2, 2, 2)), cfg)
    assert len(loads) == 2


def test_compact_buffers_have_no_environment_ray_copies(tmp_path):
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData

    from isaaclab_ipc.compact_sensor import SharedMid360RayCaster
    from isaaclab_ipc.mid360 import Mid360PatternCfg, mid360_config

    path = tmp_path / "table.npy"
    np.save(path, np.tile(np.array([[1, 0, 0]], np.float32), (64, 1)))
    cfg = mid360_config()
    cfg.pattern_cfg = Mid360PatternCfg(raydirs_path=str(path), rays_per_scan=20)
    stub = SimpleNamespace(cfg=cfg, _device="cpu", device="cpu", _view_count=256, _data=MultiMeshRayCasterData())
    SharedMid360RayCaster._initialize_rays_impl(stub)
    assert stub._data.ray_hits_w.torch.shape == (256, 20, 3)
    assert stub._direction_table.shape == (64,)
    assert stub.window_start.torch.shape == (256,)
    for name in ("ray_starts", "ray_directions", "_ray_starts_w", "_ray_directions_w", "_ray_distance_wp"):
        assert not hasattr(stub, name), name


@pytest.mark.parametrize("scenario", ["forest", "avoid"])
def test_compact_is_default_with_explicit_official_reference(scenario):
    from isaaclab_ipc.compact_sensor import SharedMid360RayCaster
    from isaaclab_ipc.mid360 import mid360_config

    assert mid360_config(scenario).class_type is SharedMid360RayCaster
    assert "SharedMid360RayCaster" not in str(mid360_config(scenario, optimized=False).class_type)


def test_query_reconstruction_preserves_official_voxel_boundaries():
    from isaaclab.utils.warp import raycast_dynamic_meshes

    from isaaclab_ipc.compact_sensor import indexed_raycast

    mesh = cube_mesh()
    rng = np.random.default_rng(42)
    table = np.column_stack((np.ones(4096), rng.uniform(-0.2, 0.2, (4096, 2)))).astype(np.float32)
    table /= np.linalg.norm(table, axis=-1, keepdims=True)
    position = torch.tensor([25.12345, 0.0732, 0.2456])
    mesh_positions = torch.tensor([[[28.0, 0, 0]]])
    mesh_rotations = torch.tensor([[[0.0, 0, 0, 1]]])
    ids = wp.array(np.array([[mesh.id]], np.uint64), dtype=wp.uint64, device="cpu")
    expected = raycast_dynamic_meshes(
        position.expand(1, len(table), 3),
        torch.from_numpy(table)[None],
        ids,
        mesh_positions,
        mesh_rotations,
        max_dist=40.0,
    )[0]
    out = wp.empty((1, len(table)), dtype=wp.vec3, device="cpu")
    wp.launch(
        indexed_raycast,
        (1, len(table)),
        inputs=[
            wp.array([True], dtype=wp.bool, device="cpu"),
            wp.array([[*position.tolist(), 0, 0, 0, 1]], dtype=wp.transform, device="cpu"),
            wp.array(table, dtype=wp.vec3, device="cpu"),
            wp.zeros(1, dtype=wp.int32, device="cpu"),
            1,
            wp.vec3(0.0),
            wp.quat_identity(),
            wp.zeros(1, dtype=wp.vec3, device="cpu"),
            ids,
            wp.from_torch(mesh_positions, dtype=wp.vec3),
            wp.from_torch(mesh_rotations, dtype=wp.quat),
            40.0,
        ],
        outputs=[wp.zeros(1, dtype=wp.vec3, device="cpu"), wp.zeros(1, dtype=wp.quat, device="cpu"), out],
        device="cpu",
    )
    actual = wp.to_torch(out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.floor(actual / 0.2), torch.floor(expected / 0.2), rtol=0, atol=0)
