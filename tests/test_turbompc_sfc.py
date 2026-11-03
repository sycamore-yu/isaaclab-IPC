"""CPU checks of IPC's exact objective, time schedule, and complete SFC rows."""

import math

import numpy as np
import pytest
import torch

from isaaclab_ipc.corridor_storage import expand_corridors, pack_corridors
from isaaclab_ipc.mpc import MPC, Config
from isaaclab_ipc.turbompc.ipc import IPCQPBuilder


def context(capacity=4):
    first = np.array([[1, 0, 0, -0.5], [0, 1, 0, -0.8], [0, -1, 0, -0.8], [2, 0, 0, -1.0]])
    second = np.array([[-1, 0, 0, -0.5], [0, 0, 1, -2.0], [0, 0, -1, 0.2]])
    schedule = [np.empty((0, 4))] + [first] * 6 + [second] * 8
    state = np.array([[0.1, -0.2, 1, 11, -12, 0.2, 21, -22, -11]], dtype=float)
    reference = np.zeros((1, 15, 9))
    reference[0, :, :3] = np.linspace([0.1, -0.2, 1], [1, 0.4, 1.2], 15)
    planes, masks = np.zeros((1, 15, capacity, 4)), np.zeros((1, 15, capacity), bool)
    for k, faces in enumerate(schedule):
        planes[0, k, : len(faces)], masks[0, k, : len(faces)] = faces[::-1], True
    obs = {
        "mpc_state": torch.from_numpy(state),
        "reference": torch.from_numpy(reference),
        "planes": torch.from_numpy(planes),
        "plane_mask": torch.from_numpy(masks),
    }
    return obs, schedule


def condense(inputs, mpc, mask):
    """Eliminate states with the existing CPU MPC prediction, retaining row order."""
    n = mpc.n
    mapping = inputs.q.new_zeros((n + 1, 12, 3 * n))
    mapping[1:, :9] = torch.from_numpy(mpc.C.reshape(n, 9, 3 * n))
    for k in range(n):
        mapping[k, 9:, 3 * k : 3 * k + 3] = torch.eye(3, dtype=torch.float64)
    offset = inputs.q.new_zeros((1, n + 1, 12))
    offset[:, 0, :9] = inputs.initial_state
    offset[:, 1:, :9] = (inputs.initial_state @ torch.from_numpy(mpc.M).T).reshape(1, n, 9)
    h = torch.einsum("tni,btnm,tmj->bij", mapping, inputs.D, mapping)
    q = torch.einsum("tni,btn->bi", mapping, (inputs.D @ offset[..., None]).squeeze(-1) + inputs.q)
    a = torch.einsum("btmn,tnj->btmj", inputs.G, mapping)
    shift = (inputs.G @ offset[..., None]).squeeze(-1)

    def ordered(value):
        tail = value.shape[3:]
        return torch.cat(
            (
                value[:, :-1, 6:9].reshape(1, 3 * n, *tail),
                value[:, 1:, 3:6].reshape(1, 3 * n, *tail),
                value[:, 1:, :3].reshape(1, 3 * n, *tail),
                value[:, 1:, 9:].flatten(1, 2),
            ),
            1,
        )

    rows = torch.cat((torch.ones(9 * n, dtype=torch.bool), mask.flatten()))
    return (
        h[0],
        q[0],
        ordered(a)[0, rows],
        ordered(inputs.lower - shift)[0, rows],
        ordered(inputs.upper - shift)[0, rows],
    )


@pytest.mark.parametrize("learned", [False, True])
def test_full_structured_problem_matches_existing_condensed_ipc(learned):
    cfg = Config.load("avoid")
    mpc, builder = MPC(cfg), IPCQPBuilder(cfg)
    obs, schedule = context()
    costs = (
        torch.tensor([[0.2, -0.1, 0.3, 0.1, 0.1, -0.2, 0.05]], dtype=torch.float64) if learned else torch.zeros(1, 7)
    )
    inputs, guess = builder(obs, costs)
    original = list(mpc.problem(obs["mpc_state"][0].numpy(), obs["reference"][0].numpy(), schedule))
    if learned:
        c = costs[0].numpy()
        multipliers = np.exp(math.log(2) * np.tanh(c[:4]))
        weights = mpc.q * np.tile(np.repeat(multipliers[:3], 3), mpc.n)
        jerk = mpc.H - mpc.C.T @ (mpc.q[:, None] * mpc.C)
        original[0] = mpc.C.T @ (weights[:, None] * mpc.C) + multipliers[3] * jerk
        original[1] = mpc.C.T @ (weights * (mpc.M @ original[-1] - obs["reference"][0].numpy().ravel()))
        original[1] -= mpc.C[mpc.pi].T @ (mpc.q[mpc.pi] * np.tile(0.2 * np.tanh(c[4:]), mpc.n))
    actual = condense(inputs, mpc, obs["plane_mask"])
    for value, expected in zip(actual, original[:5], strict=True):
        np.testing.assert_allclose(value.numpy(), expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(inputs.initial_state[0].numpy(), original[-1])
    torch.testing.assert_close(builder.dynamics()(guess[:, :-1]), guess[:, 1:, :9], atol=1e-12, rtol=1e-12)
    assert not inputs.E.any()
    torch.testing.assert_close(inputs.D[0, -1, 9:, 9:], torch.eye(3, dtype=torch.float64))
    assert not inputs.q[0, -1, 9:].any() and not inputs.G[0, -1, :, 9:].any()


def test_compact_schedule_is_lossless_and_initial_stage_is_not_constrained():
    builder = IPCQPBuilder()
    obs, schedule = context()
    dense, _ = builder(obs)
    packed = dict(obs)
    for key, value in zip(("planes", "plane_mask", "corridor_index"), pack_corridors(schedule, 4), strict=True):
        packed[key] = torch.from_numpy(value)[None]
    compact, _ = builder(packed)
    for name in dense.__dataclass_fields__:
        torch.testing.assert_close(getattr(dense, name), getattr(compact, name), atol=0, rtol=0)
    assert not dense.G[0, :2, 9:].any()  # x0 and the empty first future slot.
    np.testing.assert_array_equal(dense.G[0, 2, 9:, :3].numpy(), schedule[1][::-1, :3])
    np.testing.assert_array_equal(dense.upper[0, 2, 9:].numpy(), -schedule[1][::-1, 3])
    assert int(packed["plane_mask"].sum()) == 7  # Repeated collinear faces remain stored.


def test_inactive_padding_is_inert_and_capacity_never_drops_a_face():
    builder = IPCQPBuilder()
    obs, _ = context()
    small, _ = builder(obs)
    bigger = dict(obs)
    bigger["planes"] = torch.nn.functional.pad(obs["planes"], (0, 0, 0, 65), value=float("nan"))
    bigger["plane_mask"] = torch.nn.functional.pad(obs["plane_mask"], (0, 65))
    padded, _ = builder(bigger)
    torch.testing.assert_close(padded.G[:, :, :13], small.G, atol=0, rtol=0)
    assert not padded.G[:, :, 13:].any()
    assert torch.isneginf(padded.lower[:, :, 13:]).all() and torch.isposinf(padded.upper[:, :, 13:]).all()
    bigger["planes"][0, 14, 68] = torch.tensor([3.0, 0.0, 0.0, -0.9])
    bigger["plane_mask"][0, 14, 68] = True
    retained, _ = builder(bigger)
    torch.testing.assert_close(retained.G[0, 15, 77, :3], torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))
    assert retained.upper[0, 15, 77] == -bigger["planes"][0, 14, 68, 3]


def test_invalid_active_geometry_and_schedule_are_rejected():
    builder = IPCQPBuilder()
    obs, schedule = context()
    obs["planes"][0, 1, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="active SFC"):
        builder(obs)
    for key, value in zip(("planes", "plane_mask", "corridor_index"), pack_corridors(schedule, 4), strict=True):
        obs[key] = torch.from_numpy(value)[None]
    obs["corridor_index"][0, 0] = -2
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        builder(obs)


def test_empty_corridors_and_cost_autograd_have_no_hidden_planner_dependency():
    builder = IPCQPBuilder()
    obs, _ = context()
    obs["plane_mask"].zero_()
    costs = torch.zeros((1, 7), dtype=torch.float64, requires_grad=True)
    inputs, _ = builder(obs, costs)
    assert not inputs.G[..., 9:, :].any()
    assert torch.isposinf(inputs.upper[..., 9:]).all()
    loss = inputs.D.sum() + inputs.q.sum()
    gradient = torch.autograd.grad(loss, costs)[0]
    assert torch.isfinite(gradient).all() and (gradient != 0).all()
    planes, masks = expand_corridors(obs)
    assert planes.shape[1] == 15 and not masks.any()
