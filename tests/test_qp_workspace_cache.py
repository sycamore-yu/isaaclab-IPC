import pytest
import torch

from isaaclab_ipc.rl.qp import DifferentiableQP


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_workspace_cache_is_bounded_and_old_gradients_survive_eviction(device):
    solver = DifferentiableQP(max_cached_shapes=2)
    gradients, outputs = [], []
    for batch in (1, 2, 3, 4, 1):
        h = 2 * torch.eye(2, dtype=torch.float64, device=device).expand(batch, -1, -1)
        g = torch.full((batch, 2), -0.5, dtype=torch.float64, device=device, requires_grad=True)
        a = torch.eye(2, dtype=torch.float64, device=device).expand(batch, -1, -1)
        bounds = torch.ones(batch, 2, dtype=torch.float64, device=device)
        result, solved, *_ = solver(h, g, a, -bounds, bounds)
        assert solved.all()
        assert len(solver._work) <= 2
        torch.testing.assert_close(result, torch.full_like(result, 0.25), atol=1e-7, rtol=0)
        gradients.append(g)
        outputs.append(result.sum())
    # Saved backward tensors must be independent of discarded/reused graph workspaces.
    for result, gradient in zip(outputs, gradients, strict=True):
        torch.testing.assert_close(torch.autograd.grad(result, gradient)[0], torch.full_like(gradient, -0.5))


def test_workspace_cache_capacity_must_be_positive():
    with pytest.raises(ValueError, match="max_cached_shapes"):
        DifferentiableQP(max_cached_shapes=0)
