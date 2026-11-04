"""Public SQP failure isolation, independent of the CUDA QP implementation."""

from dataclasses import fields

import pytest
import torch

from isaaclab_ipc.turbompc import ADMMConfig, ADMMResult, LinearDynamics, MPCInputs, SQPConfig, TorchMPC


class IndependentControlAdjoint:
    """Exact CPU adjoint for fixed x=0 and independent unit-Hessian controls."""

    def solve(self, qp, hessian, active, upstream):
        assert not active.any()
        assert all(torch.isfinite(value).all() for value in (qp.q, qp.c0, qp.c_dyn, hessian, upstream))
        torch.testing.assert_close(hessian, torch.eye(2, dtype=hessian.dtype).expand_as(hessian))
        # The loss uses controls only. Fixed states have zero sensitivity, and
        # the unconstrained unit control Hessian makes its adjoint equal to RHS.
        primal = torch.stack((torch.zeros_like(upstream[..., 0]), upstream[..., 1]), -1)
        batch, stages, _ = upstream.shape
        return primal, upstream.new_zeros((batch, stages)), torch.zeros_like(qp.lower)


@pytest.mark.parametrize("compile_kernels", [False, True], ids=["eager", "compiled"])
def test_public_solve_freezes_finished_rows_and_preserves_failure_diagnostics(monkeypatch, compile_kernels):
    """Seven fixed rows cover independent stop/failure contracts in two calls.

    The candidate stream is deliberate fault injection, not a numerical QP
    oracle. The actual objective has the analytic optimum x=0, u=1 at both
    nodes. Only raw-0 rows may certify a strict outer solution.
    """
    dtype = torch.float64
    batch, stages = 7, 2
    d = torch.eye(2, dtype=dtype).repeat(batch, stages, 1, 1).requires_grad_()
    q = torch.tensor([0.0, -1.0], dtype=dtype).repeat(batch, stages, 1).requires_grad_()
    inputs = MPCInputs(
        initial_state=torch.zeros((batch, 1), dtype=dtype),
        D=d,
        E=torch.zeros((batch, stages - 1, 2, 2), dtype=dtype),
        q=q,
        G=torch.zeros((batch, stages, 1, 2), dtype=dtype),
        lower=torch.full((batch, stages, 1), -torch.inf, dtype=dtype),
        upper=torch.full((batch, stages, 1), torch.inf, dtype=dtype),
    )
    solver = TorchMPC(
        LinearDynamics(torch.tensor([[1.0, 0.0]], dtype=dtype), nx=1),
        SQPConfig(max_iter=3, tolerance=1e-7, require_qp_convergence=True, compile_kernels=compile_kernels),
        ADMMConfig(max_iter=1, eps_abs=1e-8, eps_rel=0),
    )
    candidates, snapshots = [], []

    def candidate_stream(qp, warm_start):
        call = len(candidates)
        assert call < 2, "Every fixed row must finish within two outer iterations"
        if call == 0:
            assert warm_start is None
        else:
            # First-call fatal diagnostics are still genuine warm buffers,
            # even though their primal candidates were not accepted.
            for field in fields(warm_start):
                torch.testing.assert_close(
                    getattr(warm_start, field.name), snapshots[0][field.name], rtol=0, atol=0, equal_nan=True
                )
        x = -qp.q.clone()
        y = torch.zeros_like(qp.lower)
        lambda0 = x.new_zeros((batch, 1))
        lambda_dyn = x.new_zeros((batch, stages - 1, 1))
        status = torch.tensor([0, 2, 2, 1, 1, 4, 0] if call == 0 else [4, 0, 0, 2, 0, 0, 0], dtype=torch.int32)
        if call == 0:
            x[1, :, 1] = 11.0  # Initial fatal: retain this raw diagnostic, not its trajectory.
            y[2] = torch.nan  # Fatal plus nonfinite stationarity must report outer status 3.
            x[3, :, 1] = 0.25  # Continue, then reject a fatal second candidate.
            x[5, :, 1] = 15.0  # Raw NONFINITE must report 3 even with finite tensors.
            x[6, :, 1] = torch.nan  # Nonfatal raw code cannot certify a nonfinite trajectory.
        else:
            x[0] = torch.nan  # Already solved: later poisoned raw-4 output must be inert.
            y[0] = lambda0[0] = lambda_dyn[0] = torch.nan
            x[3, :, 1] = 13.0
        marker = x.new_tensor([10 * (call + 1) + row for row in range(batch)])
        candidate = ADMMResult(
            iterations=marker.to(torch.int32),
            x=x,
            z=marker[:, None, None].expand(batch, stages, 1).clone(),
            y=y,
            lambda0=lambda0,
            lambda_dyn=lambda_dyn,
            xi=(marker + 100)[:, None, None].expand(batch, stages, 1).clone(),
            rho=marker + 200,
            kernel_ns=marker + 300,
            status=status,
            residuals=(marker[:, None] + torch.arange(4, dtype=dtype)).clone(),
        )
        candidates.append(candidate)
        snapshots.append({field.name: getattr(candidate, field.name).clone() for field in fields(candidate)})
        return candidate

    adjoint = IndependentControlAdjoint()
    monkeypatch.setattr(solver.admm, "solve", candidate_stream)
    monkeypatch.setattr(solver, "_adjoint", lambda qp: adjoint)
    try:
        result = solver.solve(inputs)
        assert len(candidates) == 2
        torch.testing.assert_close(result.status, torch.tensor([0, 2, 3, 2, 0, 3, 3], dtype=torch.int32))
        torch.testing.assert_close(result.iterations, torch.tensor([1, 1, 1, 2, 2, 1, 1], dtype=torch.int32))
        expected_trajectory = torch.zeros_like(q)
        expected_trajectory[[0, 4], :, 1] = 1
        expected_trajectory[3, :, 1] = 0.25
        expected_trajectory[6, :, 1] = torch.nan
        torch.testing.assert_close(result.trajectory, expected_trajectory, rtol=0, atol=0, equal_nan=True)
        assert result.normal_cone_error is None
        torch.testing.assert_close(result.errors[[0, 4]], torch.zeros((2, 3), dtype=dtype), rtol=0, atol=0)
        torch.testing.assert_close(result.errors[[1, 3, 5], 0], q.new_tensor([1.0, 0.75, 1.0]))
        assert torch.isnan(result.errors[[2, 6], 0]).all()
        for field in fields(result.admm):
            expected = snapshots[0][field.name].clone()
            expected[4] = snapshots[1][field.name][4]
            torch.testing.assert_close(getattr(result.admm, field.name), expected, rtol=0, atol=0, equal_nan=True)
        # Failed rows keep their observable output, but must have exactly zero
        # derivatives; the two regular u=-q rows have known nonzero derivatives.
        q_gradient, d_gradient = torch.autograd.grad(result.controls.nan_to_num().sum(), (q, d))
        expected_q = torch.zeros_like(q)
        expected_q[[0, 4], :, 1] = -1
        expected_d = torch.zeros_like(d)
        expected_d[[0, 4], :, 1, 1] = -1
        torch.testing.assert_close(q_gradient, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(d_gradient, expected_d, rtol=0, atol=0)
        assert torch.isfinite(q_gradient).all() and torch.isfinite(d_gradient).all()
        # Fused selection must not mutate caller-owned raw results while
        # constructing the accepted solution or its saved autograd tensors.
        for candidate, snapshot in zip(candidates, snapshots, strict=True):
            for field in fields(candidate):
                torch.testing.assert_close(
                    getattr(candidate, field.name), snapshot[field.name], rtol=0, atol=0, equal_nan=True
                )
    finally:
        solver.close()
