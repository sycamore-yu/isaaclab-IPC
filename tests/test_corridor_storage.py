import numpy as np
import pytest
import torch


def test_corridor_storage_is_lossless_owned_and_time_indexed():
    from isaaclab_ipc.corridor_storage import expand_corridors, pack_corridors

    rng = np.random.default_rng(2)
    first, second = rng.normal(size=(7, 4)), rng.normal(size=(11, 4))
    schedule = [np.empty((0, 4))] + [first] * 7 + [second] * 7
    blocks, masks, indices = pack_corridors(schedule, 32)
    assert blocks.shape == (2, 32, 4) and blocks.dtype == np.float64
    assert masks.shape == (2, 32) and indices.shape == (15,)
    assert blocks.nbytes == 2048  # Dense float64 snapshot previously occupied 15,360 bytes.
    obs = {
        "planes": torch.from_numpy(blocks)[None],
        "plane_mask": torch.from_numpy(masks)[None],
        "corridor_index": torch.from_numpy(indices)[None],
    }
    planes, valid = expand_corridors(obs)
    for k, faces in enumerate(schedule):
        np.testing.assert_array_equal(planes[0, k, : len(faces)].numpy(), faces[::-1])
        assert int(valid[0, k].sum()) == len(faces)
    saved = planes.clone()
    first[:] = 0
    torch.testing.assert_close(expand_corridors(obs)[0], saved, rtol=0, atol=0)
    # Rollout storage may shuffle samples; the schedule must travel with each sample.
    shuffled = {key: value.repeat(3, *([1] * (value.ndim - 1)))[[2, 0, 1]] for key, value in obs.items()}
    torch.testing.assert_close(expand_corridors(shuffled)[0][2], saved[0], rtol=0, atol=0)


def test_corridor_storage_never_drops_constraints():
    from isaaclab_ipc.corridor_storage import pack_corridors

    with pytest.raises(ValueError, match="planes"):
        pack_corridors([np.ones((33, 4))], 32)
    with pytest.raises(ValueError, match="corridors"):
        pack_corridors([np.full((2, 4), n) for n in range(3)], 32)


def test_dense_saved_context_is_still_accepted():
    from isaaclab_ipc.corridor_storage import expand_corridors

    obs = {
        "planes": torch.zeros(2, 15, 32, 4, dtype=torch.float64),
        "plane_mask": torch.zeros(2, 15, 32, dtype=torch.bool),
    }
    planes, mask = expand_corridors(obs)
    assert planes is obs["planes"] and mask is obs["plane_mask"]


def test_actor_qp_assembly_is_identical_for_compact_snapshot():
    from tensordict import TensorDict

    from isaaclab_ipc.corridor_storage import pack_corridors
    from isaaclab_ipc.mpc import Config
    from isaaclab_ipc.planner import Planner
    from isaaclab_ipc.rl.actor import MPCActor

    planner = Planner(Config.load(), [8, 0, 1])
    state, ref, schedule = planner.context(np.array([0, 0, 1, 0.1, 0, 0, 0, 0, 0.0]))
    dense, valid = np.zeros((15, 32, 4)), np.zeros((15, 32), bool)
    for k, faces in enumerate(schedule):
        dense[k, : len(faces)] = faces[::-1]
        valid[k, : len(faces)] = True
    values = dict(
        policy=np.zeros(124, np.float32),
        mpc_state=state,
        reference=ref,
        planes=dense,
        plane_mask=valid,
        rotation=np.eye(3).reshape(-1),
        braking=np.array([False]),
        yaw_target=np.array([0.0]),
        sfc_valid=np.array([True]),
    )
    obs = TensorDict({k: torch.as_tensor(v)[None] for k, v in values.items()}, [1])
    actor = MPCActor(obs, {"actor": ["policy"]}, "actor", 4)
    packed = obs.clone()
    for key, value in zip(("planes", "plane_mask", "corridor_index"), pack_corridors(schedule, 32), strict=True):
        packed[key] = torch.as_tensor(value)[None]
    for actual, expected in zip(
        actor.problem(packed, torch.zeros(1, 7)), actor.problem(obs, torch.zeros(1, 7)), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(actor(packed), actor(obs), atol=1e-8, rtol=0)
