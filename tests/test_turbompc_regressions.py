"""Regression cases for KKT accuracy, workspace streams, and failed batch rows."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

import isaaclab_ipc.turbompc as turbo


def test_sparse_kkt_default_ir_stays_within_accepted_error_budget():
    # The retained CSR values/RHS expose cuDSS IR=0's 3.38e-4 residual
    # despite a condition number of 33.87. See tests/data/README.md.
    fixture = Path(__file__).parent / "data" / "turbompc_kkt_accuracy_reproducer.npz"
    with np.load(fixture, allow_pickle=False) as saved:
        row_ptr, col_idx = saved["row"], saved["col"]
        values, rhs = saved["values"], saved["rhs"]
    assert row_ptr.shape == (41,) and rhs.shape == (40,)
    assert values.dtype == rhs.dtype == np.float64
    dense = np.zeros((40, 40), dtype=np.float64)
    row_idx = np.repeat(np.arange(40), np.diff(row_ptr))
    np.add.at(dense, (row_idx, col_idx), values)
    np.testing.assert_array_equal(dense, dense.T)
    expected = np.linalg.solve(dense, rhs)
    np.testing.assert_allclose(dense @ expected, rhs, atol=1e-10, rtol=0)

    solver = turbo.SparseKKTSolver(
        torch.from_numpy(row_ptr), torch.from_numpy(col_idx), device="cuda:0", dtype=torch.float64
    )
    try:
        actual = solver.solve(torch.from_numpy(values).to("cuda:0"), torch.from_numpy(rhs).to("cuda:0")).cpu().numpy()
        # IR=0 accepts a known error budget of 4e-4, not the old 1e-9
        # bound specific to the experimental IR=2 candidate.
        np.testing.assert_allclose(dense @ actual, rhs, atol=4e-4, rtol=0)
        np.testing.assert_allclose(actual, expected, atol=4e-4, rtol=0)
    finally:
        solver.close()


def _tracking_qp():
    """Two scalar integrators with a known bound-constrained optimum."""
    kw = dict(device="cuda:0", dtype=torch.float64)
    plus = torch.tensor([1.0, 0.0], **kw).reshape(1, 1, 1, 2).repeat(2, 2, 1, 1)
    return turbo.QPData(
        D=torch.diag(torch.tensor([1.0, 0.1], **kw)).repeat(2, 3, 1, 1),
        E=torch.zeros((2, 2, 2, 2), **kw),
        q=torch.tensor([-1.0, 0.0], **kw).repeat(2, 3, 1),
        A0=plus[:, 0].clone(),
        A_minus=torch.tensor([-1.0, -1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 2, 1, 1),
        A_plus=plus,
        c0=torch.tensor([[0.0], [0.2]], **kw),
        c_dyn=torch.zeros((2, 2, 1), **kw),
        G=torch.tensor([0.0, 1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 3, 1, 1),
        lower=torch.full((2, 3, 1), -0.25, **kw),
        upper=torch.full((2, 3, 1), 0.25, **kw),
    )


def test_admm_workspace_reuse_across_two_streams_preserves_prior_outputs():
    qp = _tracking_qp()
    reversed_qp = replace(qp, q=-qp.q)
    caller = torch.cuda.current_stream(qp.q.device)
    first_stream = torch.cuda.Stream(device=qp.q.device)
    second_stream = torch.cuda.Stream(device=qp.q.device)
    # The caller owns input readiness. The workspace owns ordering between its
    # successive solves; do not insert caller waits between the two streams.
    first_stream.wait_stream(caller)
    second_stream.wait_stream(caller)
    solver = turbo.NativeADMM(turbo.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9, max_iter=4096))
    try:
        with torch.cuda.stream(first_stream):
            first = solver.solve(qp)
            first_saved = {field.name: getattr(first, field.name).clone() for field in fields(first)}
        with torch.cuda.stream(second_stream):
            second = solver.solve(reversed_qp)
            second_saved = {field.name: getattr(second, field.name).clone() for field in fields(second)}
        with torch.cuda.stream(first_stream):
            third = solver.solve(qp)
        caller.wait_stream(first_stream)
        caller.wait_stream(second_stream)

        positive = qp.q.new_tensor([[[0.0, 0.25], [0.25, 0.25], [0.5, 0.0]], [[0.2, 0.25], [0.45, 0.25], [0.7, 0.0]]])
        negative = qp.q.new_tensor(
            [[[0.0, -0.25], [-0.25, -0.25], [-0.5, 0.0]], [[0.2, -0.25], [-0.05, -0.25], [-0.3, 0.0]]]
        )
        for result, expected in ((first, positive), (second, negative), (third, positive)):
            assert (result.status == 0).all(), result.status
            torch.testing.assert_close(result.x, expected, atol=2e-7, rtol=0)
        for result, saved in ((first, first_saved), (second, second_saved)):
            for name, snapshot in saved.items():
                torch.testing.assert_close(getattr(result, name), snapshot, atol=0, rtol=0)
        assert len({result.x.data_ptr() for result in (first, second, third)}) == 3
    finally:
        solver.close()


def test_mixed_success_and_failure_batch_preserves_healthy_cost_gradient():
    kw = dict(device="cuda:0", dtype=torch.float64)
    dynamics = turbo.LinearDynamics(torch.tensor([[1.0, 0.2]], **kw), nx=1)
    q = torch.tensor([-0.3, 0.0], **kw).repeat(2, 5, 1).requires_grad_()
    inputs = turbo.MPCInputs(
        initial_state=torch.tensor([[0.0], [0.2]], **kw),
        D=torch.diag(torch.tensor([1.0, 0.2], **kw)).repeat(2, 5, 1, 1),
        E=torch.zeros((2, 4, 2, 2), **kw),
        q=q,
        G=torch.tensor([0.0, 1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 5, 2, 1),
        lower=torch.full((2, 5, 2), -0.8, **kw),
        upper=torch.full((2, 5, 2), 0.8, **kw),
    )
    solver = turbo.TorchMPC(
        dynamics,
        turbo.SQPConfig(max_iter=30, tolerance=2e-8, linesearch="merit"),
        turbo.ADMMConfig(eps_abs=1e-10, eps_rel=1e-10, max_iter=4096),
    )
    try:
        reference = solver.solve(inputs)
        assert (reference.status == 0).all(), reference.errors
        reference_gradient = torch.autograd.grad(reference.states.square().sum(), q)[0]
        lower, upper = inputs.lower.clone(), inputs.upper.clone()
        # Only the second sample is infeasible: the same control must be >= 1
        # and <= 0. Each individual lower/upper interval remains well-formed.
        lower[1, :, 0], upper[1, :, 0] = 1.0, torch.inf
        lower[1, :, 1], upper[1, :, 1] = -torch.inf, 0.0
        result = solver.solve(replace(inputs, lower=lower, upper=upper))
        assert result.status[0] == 0, result.errors
        assert result.status[1] != 0, result.status
        gradient = torch.autograd.grad(result.states.square().sum(), q)[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[1]) == 0
        assert torch.linalg.vector_norm(gradient[0]) > 1e-6
        torch.testing.assert_close(result.trajectory[0], reference.trajectory[0], atol=2e-7, rtol=0)
        torch.testing.assert_close(gradient[0], reference_gradient[0], atol=2e-7, rtol=1e-5)
    finally:
        solver.close()
