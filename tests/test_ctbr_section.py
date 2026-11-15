"""CMPCC-style fixed-progress sections are bounded, safe, and use native CTBR rows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from isaaclab_ipc.ctbr_navigation import sample_plan
from isaaclab_ipc.ctbr_section import reference_tangent, section_tube_planes
from isaaclab_ipc.rl.ctbr_observation import make_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder


def box_faces():
    return np.array([
        [1., 0., 0., -2.], [-1., 0., 0., 0.],
        [0., 1., 0., -1.], [0., -1., 0., -1.],
        [0., 0., 1., -1.], [0., 0., -1., -1.],
    ])


def inside(planes, point, tolerance=1e-10):
    return np.all(planes[:, :3] @ point + planes[:, 3] <= tolerance)


def test_fixed_progress_section_is_a_bounded_subset():
    source = box_faces()
    section = section_tube_planes(source, [1., 0., 0.], [1., 0., 0.], 0.5)
    assert inside(section, [1.5, 0.75, 0.])
    assert inside(source, [1.6, 0., 0.])
    assert not inside(section, [1.6, 0., 0.])
    assert not inside(section, [1., 1.1, 0.])
    assert section.shape == (6, 4)
    assert np.isfinite(section).all()


def test_skewed_original_face_tightens_lateral_section():
    source = np.vstack((box_faces(), [1., 1., 0., -1.5]))
    section = section_tube_planes(source, [0.5, 0., 0.], [1., 0., 0.], 0.4)
    assert inside(section, [0.9, 0.6, 0.])
    assert inside(source, [0.5, 0.95, 0.])
    assert not inside(section, [0.5, 0.95, 0.])
    assert not inside(section, [1., 0., 0.])


def test_random_tilted_sections_are_contained_in_source_sfc():
    # Maximize each original plane over each new tube with an independent LP.
    source = box_faces()
    rng = np.random.default_rng(11)
    for _ in range(12):
        center = np.array([1., 0., 0.]) + rng.uniform(-0.3, 0.3, size=3)
        direction = rng.normal(size=3)
        tube = section_tube_planes(source, center, direction, 0.5)
        assert inside(tube, center)
        for face in source:
            result = linprog(
                -face[:3], A_ub=tube[:, :3], b_ub=-tube[:, 3],
                bounds=[(None, None)] * 3, method="highs",
            )
            assert result.success, result.message
            assert -result.fun + face[3] <= 1e-8


def test_reference_tangent_handles_hover_and_vertical_motion():
    pva = np.zeros((3, 9))
    yaw = np.full(3, np.pi / 2)
    np.testing.assert_allclose(reference_tangent(pva, yaw, 1), [0., 1., 0.], atol=1e-15)
    pva[2, 2] = 1.
    np.testing.assert_allclose(reference_tangent(pva, yaw, 1), [0., 0., 1.])
    pva[1, 3:6] = [3., 4., 0.]
    np.testing.assert_allclose(reference_tangent(pva, yaw, 1), [.6, .8, 0.])


def test_section_schedule_at_corridor_junction_and_terminal_hold():
    first = np.array([[1., 0., 0., -1.], [-1., 0., 0., 0.]])
    second = np.array([[1., 0., 0., -2.], [-1., 0., 0., 1.]])

    def sample(times):
        pva = np.zeros((len(times), 9))
        pva[:, 0], pva[:, 3] = times, 1.
        return SimpleNamespace(pva=pva, yaw=np.zeros(len(times)))

    plan = SimpleNamespace(
        start_time=10., trajectory=SimpleNamespace(sample=sample),
        corridors=(first, second),
        segments=(SimpleNamespace(corridor_index=0, end=1.),
                  SimpleNamespace(corridor_index=1, end=2.)),
    )
    sampled = sample_plan(plan, 10.5, 5, .25, 8, constraint_mode="section")
    assert sampled.corridor_index.tolist() == [0, 0, 1, 1, 1]
    assert sampled.reference_violation == 0
    for k, x in enumerate(sampled.pva[:, 0]):
        active = sampled.planes[k, sampled.plane_mask[k]]
        assert inside(active, [x, 0., 0.])
    # The exact junction enforces both neighboring corridors: only x=1 survives.
    junction = sampled.planes[2, sampled.plane_mask[2]]
    assert inside(junction, [1., 0., 0.])
    assert not inside(junction, [.999, 0., 0.])
    assert not inside(junction, [1.001, 0., 0.])


def test_section_rows_feed_existing_ctbr_builder_unchanged():
    source = box_faces()

    def sample(times):
        pva = np.zeros((len(times), 9))
        pva[:, 0] = 1. + 0.2 * times
        pva[:, 3] = .2
        return SimpleNamespace(pva=pva, yaw=np.zeros(len(times)))

    plan = SimpleNamespace(
        start_time=0., trajectory=SimpleNamespace(sample=sample),
        corridors=(source,), segments=(SimpleNamespace(corridor_index=0, end=10.),),
    )
    cfg = CTBRConfig()
    snapshot = sample_plan(plan, 0., cfg.cost_nodes, cfg.dt, 10, constraint_mode="section")
    state = torch.zeros((1, 10), dtype=torch.float64)
    state[:, 0], state[:, 3] = 1., 1.
    observation = make_ctbr_observation(
        state, torch.zeros((1, 3), dtype=torch.float64), torch.tensor([[2., 0., 0.]]),
        torch.from_numpy(snapshot.pva[None]), torch.from_numpy(snapshot.yaw[None]),
        torch.from_numpy(snapshot.planes[None]), torch.from_numpy(snapshot.plane_mask[None]),
        ctbr_config=cfg,
    )
    builder = CTBRQPBuilder(cfg)
    data, _ = builder(observation)
    active = snapshot.plane_mask[1]
    np.testing.assert_allclose(
        data.G[0, 1, 7:7 + len(active), :3].numpy()[active],
        snapshot.planes[1, active, :3],
    )
    np.testing.assert_allclose(
        data.upper[0, 1, 7:7 + len(active)].numpy()[active],
        -snapshot.planes[1, active, 3],
    )


def test_section_capacity_is_explicit_and_mode_is_validated():
    source = box_faces()
    with pytest.raises(ValueError, match="Section half-length"):
        section_tube_planes(source, [1., 0., 0.], [1., 0., 0.], 0.)
    with pytest.raises(ValueError, match="tangent"):
        section_tube_planes(source, [1., 0., 0.], [0., 0., 0.], .5)

    def sample(times):
        pva = np.zeros((len(times), 9))
        pva[:, 0], pva[:, 3] = 1., 1.
        return SimpleNamespace(pva=pva, yaw=np.zeros(len(times)))

    plan = SimpleNamespace(
        start_time=0., trajectory=SimpleNamespace(sample=sample),
        corridors=(source,), segments=(SimpleNamespace(corridor_index=0, end=1.),),
    )
    with pytest.raises(ValueError, match="requires 6 SFC faces"):
        sample_plan(plan, 0., 2, .1, 5, constraint_mode="section")
    with pytest.raises(ValueError, match="constraint mode"):
        sample_plan(plan, 0., 2, .1, 8, constraint_mode="invalid")
