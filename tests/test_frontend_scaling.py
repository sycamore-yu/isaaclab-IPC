import numpy as np
import torch

from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import Planner


def test_corridor_seed_cache_reuses_native_geometry_and_checks_each_endpoint(monkeypatch):
    calls = []
    bounds = np.array([[1, 0, 0, -1], [-1, 0, 0, -1], [0, 1, 0, -1], [0, -1, 0, -1], [0, 0, 1, -2], [0, 0, -1, 0.0]])

    def geometry(a, b, cloud, radius):
        calls.append((np.array(a), np.array(b)))
        return bounds.copy() if (bounds[:, :3] @ b + bounds[:, 3] <= 1e-8).all() else np.empty((0, 4))

    monkeypatch.setattr("isaaclab_ipc.planner._geometry.corridor", geometry)
    planner = Planner(Config.load(), [8, 0, 1])
    seed = np.array([0, 0, 1.0])
    first = planner._corridor(seed, seed)
    assert not len(planner._corridor(seed, [2, 0, 1]))
    np.testing.assert_array_equal(planner._corridor(seed, [0.5, 0, 1]), first)
    assert len(calls) == 1
    first[:] = 0  # Consumers must not mutate cached geometry.
    np.testing.assert_array_equal(planner._corridor(seed, seed), bounds)
    planner.update_cloud([], seed)
    planner._corridor(seed, seed)
    assert len(calls) == 2


def test_seed_cache_matches_original_native_calls():
    rng = np.random.default_rng(8)
    cloud = rng.uniform([2, -3, 0.2], [9, 3, 3], (180, 3))
    cached = Planner(Config.load(), [8, 0, 1])
    original = Planner(Config.load(), [8, 0, 1], cache_corridors=False)
    p = np.array([0, 0, 1.0])
    for planner in (cached, original):
        planner.update_cloud(cloud, p)
    for end in ([0.5, 0, 1], [1.5, 0.5, 1], [6, 1, 1], [9, 0, 1]):
        np.testing.assert_array_equal(cached._corridor(p, end), original._corridor(p, end))


def test_planner_batch_preserves_order_results_and_exception_isolation():
    from isaaclab_ipc.frontend import PlanningBatch
    from isaaclab_ipc.planner import InvalidCorridorError

    class Stub:
        def __init__(self, index):
            self.index = index
            self.cfg = Config.load()
            self.frontend_failures = 0
            self.replan = False
            self.updates = 0

        def update_cloud(self, cloud, position):
            self.updates += 1

        def context(self, state):
            if self.index == 1:
                raise InvalidCorridorError("seed blocked")
            return state.copy(), np.full((15, 9), self.index, dtype=float), []

    state = np.tile([0, 0, 1, 0, 0, 0, 0, 0, 0.0], (3, 1))
    planners = [Stub(i) for i in range(3)]
    with PlanningBatch(workers=2) as batch:
        results = batch.contexts(planners, state, {0: np.zeros((1, 3)), 2: np.ones((1, 3))})
    assert [r.valid for r in results] == [True, False, True]
    assert results[2].reference[0, 0] == 2
    assert planners[1].frontend_failures == 1 and planners[1].replan
    assert [p.updates for p in planners] == [1, 0, 1]


def test_batched_cloud_transfer_is_exact():
    from isaaclab_ipc.frontend import batch_clouds_to_host

    hits = torch.tensor(
        [[[1, 0, 0], [0.1, 0, 0], [torch.inf, torch.inf, torch.inf]], [[0, 2, 0], [0, 0.3, 0], [0, 0, 1.0]]]
    )
    positions = torch.zeros(2, 3)
    actual = batch_clouds_to_host(hits, positions)
    valid = torch.isfinite(hits).all(-1) & ((hits - positions[:, None]).norm(dim=-1) >= 0.3)
    for i in range(2):
        np.testing.assert_array_equal(actual[i], hits[i, valid[i]].numpy())


def test_batch_problem_preparation_matches_classical_ipc():
    from isaaclab_ipc.frontend import PlanningBatch

    planners = [Planner(Config.load(), [8, 0, 1]) for _ in range(2)]
    states = np.tile([0, 0, 1, 0.1, 0, 0, 0, 0, 0.0], (2, 1))
    with PlanningBatch(workers=2) as batch:
        problems = batch.problems(planners, states, {})
    for planner, state, actual in zip(planners, states, problems, strict=True):
        expected = planner.prepare(state)[0]
        for x, y in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(x, y)
