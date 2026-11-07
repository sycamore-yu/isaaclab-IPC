"""The implicit backward reuses one graph across successful/failed batches."""

import pytest
import torch

from isaaclab_ipc.turbompc import ADMMConfig, MPCInputs, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.dynamics import LinearDynamics
from isaaclab_ipc.turbompc.solver import _ImplicitTrajectory


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA KKT required")
def test_saved_cost_metadata_does_not_recompile_mixed_and_successful_backward():
    kw = {"device": "cuda:0", "dtype": torch.float64}
    data = MPCInputs(
        initial_state=torch.tensor([[0.0], [0.2]], **kw),
        D=torch.diag(torch.tensor([1.0, 0.5], **kw)).repeat(2, 2, 1, 1).requires_grad_(),
        E=torch.zeros((2, 1, 2, 2), **kw),
        q=torch.tensor([-1.0, 0.0], **kw).repeat(2, 2, 1).requires_grad_(),
        G=torch.zeros((2, 2, 0, 2), **kw),
        lower=torch.zeros((2, 2, 0), **kw),
        upper=torch.zeros((2, 2, 0), **kw),
    )
    model = LinearDynamics(torch.tensor([[1.0, 1.0]], **kw), nx=1)
    solver = TorchMPC(model, SQPConfig(tolerance=1e-7), ADMMConfig(eps_abs=1e-10, eps_rel=0))
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    solver._adjoint_data = torch.compile(solver._adjoint_data, backend=backend, fullgraph=True, dynamic=False)
    weights = torch.tensor([0.3, 0.7], **kw)
    expected = torch.zeros_like(data.q)
    expected[:, 0, 1] = expected[:, 1, 0] = -weights / 1.5
    try:
        result = solver.solve(data)
        assert bool((result.status == 0).all())
        for failed in (True, False, True):
            status = result.status.clone()
            if failed:
                status[1] = 1
            trajectory = _ImplicitTrajectory.apply(
                data.initial_state,
                data.D,
                data.E,
                data.q,
                data.G,
                data.lower,
                data.upper,
                result.trajectory.detach(),
                result.admm.lambda0,
                result.admm.lambda_dyn,
                result.admm.y,
                status,
                solver,
            )
            gradient = torch.autograd.grad((trajectory[:, 0, 1] * weights).sum(), data.q)[0]
            target = expected.clone()
            if failed:
                target[1] = 0
            torch.testing.assert_close(gradient, target, atol=1e-9, rtol=1e-9)
        assert len(graphs) == 1
    finally:
        solver.close()
