"""Packed observation fusion preserves rollout snapshots and partial resets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_ipc.rl.ctbr_observation import PackedCTBRObservation, make_ctbr_observation
from isaaclab_ipc.turbompc.ctbr import CTBRConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_full_batch_subsets_keep_one_graph_and_own_snapshots(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = CTBRConfig()
    batch, nodes, faces = 4, config.cost_nodes, 6
    rng = np.random.default_rng(17)
    physical = torch.tensor(rng.normal(size=(batch, 19)), dtype=torch.float64, device=device)
    state, rates, goal = physical[:, :10], physical[:, 10:13], physical[:, 16:19]
    state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
    warm = torch.tensor(rng.normal(size=(batch, nodes - 1, 4)), dtype=torch.float64, device=device)
    pva = rng.normal(size=(batch, nodes, 9))
    yaw = rng.normal(size=(batch, nodes))
    planes = rng.normal(size=(batch, nodes, faces, 4))
    mask = rng.random((batch, nodes, faces)) > .35
    planes[~mask] = np.nan
    samples = [SimpleNamespace(pva=pva[i], yaw=yaw[i], planes=planes[i], plane_mask=mask[i])
               for i in range(batch)]
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return torch._inductor.compile(graph, inputs)

    builder = PackedCTBRObservation(batch, config, faces, device, backend=backend)
    builder.prewarm()
    retained = []
    for ids, has_warm, inference in [([0, 1, 2, 3], True, False), ([3, 1], False, False),
                                     ([2], True, True), ([0, 1, 2, 3], False, True)]:
        indices = torch.tensor(ids, device=device)
        selected = [samples[i] for i in ids]
        valid = [i % 2 == 0 for i in ids]

        def tensor(name, dtype=torch.float64):
            return torch.as_tensor(np.stack([getattr(row, name) for row in selected]),
                                   dtype=dtype, device=device)

        expected = make_ctbr_observation(
            state[indices], rates[indices], goal[indices], tensor("pva"), tensor("yaw"),
            tensor("planes"), tensor("plane_mask", torch.bool), ctbr_config=config,
            reference_valid=torch.tensor(valid, device=device),
            warm_start_controls=warm[indices] if has_warm else None,
        )
        with torch.inference_mode(inference):
            actual = builder.build(
                state.clone(), rates.clone(), goal.clone(), selected, ids=ids, valid=valid,
                warm=warm.clone() if has_warm else None, device_ids=indices,
            )
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            assert value.shape == expected[name].shape
            assert value.dtype == expected[name].dtype
            assert value.stride() == expected[name].stride()
            assert not value.requires_grad
            tolerance = 5e-7 if value.dtype == torch.float32 else 5e-12
            torch.testing.assert_close(value, expected[name], rtol=0, atol=tolerance)
        retained.append((actual, {name: value.clone() for name, value in actual.items()}))
        builder.host.fill(123.)
        physical.add_(.01)
        warm.add_(.01)
        for previous, frozen in retained:
            for name, value in previous.items():
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    assert len(graphs) == 1


@pytest.mark.parametrize("faces", [0, 2])
def test_packed_single_environment_canonicalizes_physical_and_warm_slices(faces):
    config = CTBRConfig()
    nodes = config.cost_nodes
    physical = torch.zeros((1, 19), dtype=torch.float64)
    physical[:, 3] = 1.
    controls = torch.zeros((1, nodes, 4), dtype=torch.float64)
    controls[..., 0] = config.mass * config.gravity
    warm = controls[:, :-1]
    assert physical[:, :10].stride() == (19, 1)
    assert warm.stride() == (20, 4, 1)
    sample = SimpleNamespace(pva=np.zeros((nodes, 9)), yaw=np.zeros(nodes),
                             planes=np.zeros((nodes, faces, 4)),
                             plane_mask=np.zeros((nodes, faces), dtype=bool))
    builder = PackedCTBRObservation(1, config, faces, "cpu", compile_kernel=False)
    actual = builder.build(physical[:, :10], physical[:, 10:13], physical[:, 16:19], [sample], warm=warm)
    expected = make_ctbr_observation(
        physical[:, :10], physical[:, 10:13], physical[:, 16:19],
        torch.from_numpy(sample.pva[None]), torch.from_numpy(sample.yaw[None]),
        torch.from_numpy(sample.planes[None]), torch.from_numpy(sample.plane_mask[None]),
        ctbr_config=config, warm_start_controls=warm,
    )
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert value.stride() == torch.empty(value.shape).stride()
