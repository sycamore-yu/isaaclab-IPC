"""CPU checks for certified CTBR adjoints, with an independent full KKT oracle."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from isaaclab_ipc.turbompc import QPData
from isaaclab_ipc.turbompc.ctbr_polish import CTBRPolisher


def problem(batch=1):
    stages, nx, width, rows = 3, 10, 14, 2
    d = torch.diag(torch.linspace(1.5, 2.8, width, dtype=torch.float64)).repeat(batch, stages, 1, 1)
    e = torch.diag(torch.linspace(-0.06, 0.08, width, dtype=torch.float64)).repeat(batch, stages - 1, 1, 1)
    e[..., 3, 10] = 0.015
    a0 = torch.eye(nx, width, dtype=torch.float64).repeat(batch, 1, 1)
    am = a0[:, None].repeat(1, stages - 1, 1, 1)
    am[..., :4, 10:14] = torch.eye(4, dtype=torch.float64) * 0.2
    ap = -a0[:, None].repeat(1, stages - 1, 1, 1)
    g = torch.zeros(batch, stages, rows, width, dtype=torch.float64)
    g[..., 0, 10], g[..., 1, 12] = 1.0, 1.0
    active = torch.zeros(batch, stages, rows, dtype=torch.bool)
    active[:, 0, 0], active[:, 1, 1] = True, True
    qp = QPData(
        d,
        e,
        torch.full((batch, stages, width), 0.37, dtype=torch.float64),
        a0,
        am,
        ap,
        torch.full((batch, nx), 0.24, dtype=torch.float64),
        torch.full((batch, stages - 1, nx), -0.19, dtype=torch.float64),
        g,
        torch.full((batch, stages, rows), -1.0, dtype=torch.float64),
        torch.full((batch, stages, rows), 1.0, dtype=torch.float64),
    )
    hessian = d.clone()
    hessian[..., 11, 11] += 0.23  # Full curvature differs from the forward cost matrix.
    return qp, hessian, active


def dense_matrix(qp, hessian, active, row):
    """Explicit global block layout, independent of production defect helpers."""
    stages, width = qp.q.shape[1:]
    nx, rows = qp.A0.shape[-2], qp.G.shape[-2]
    primal, equalities = stages * width, stages * nx
    h = np.zeros((primal, primal))
    a = np.zeros((equalities, primal))
    g = np.zeros((stages * rows, primal))
    a[:nx, :width] = qp.A0[row].numpy()
    for stage in range(stages):
        block = slice(stage * width, (stage + 1) * width)
        h[block, block] = hessian[row, stage].numpy()
        g[stage * rows : (stage + 1) * rows, block] = qp.G[row, stage].numpy() * active[row, stage].numpy()[:, None]
        if stage:
            previous = slice((stage - 1) * width, stage * width)
            h[block, previous] = qp.E[row, stage - 1].numpy()
            h[previous, block] = qp.E[row, stage - 1].numpy().T
            eq = slice(stage * nx, (stage + 1) * nx)
            a[eq, previous] = qp.A_minus[row, stage - 1].numpy()
            a[eq, block] = qp.A_plus[row, stage - 1].numpy()
    constraints = np.concatenate((a, g))
    diagonal = np.r_[np.zeros(equalities), (~active[row]).numpy().astype(float).ravel()]
    return np.block([[h, constraints.T], [constraints, np.diag(diagonal)]])


class DenseBackend:
    """An approximate inverse with known contraction and optional faults."""

    def __init__(self, qp, hessian, active, contraction=0.0, *, settled_noise=False, fault=None):
        self.qp, self.hessian, self.active = qp, hessian, active
        self.matrices = [dense_matrix(qp, hessian, active, row) for row in range(len(qp.q))]
        self.contraction = np.broadcast_to(contraction, (len(qp.q),))
        self.settled_noise, self.fault = settled_noise, fault
        self.calls = []

    def _solve(self, qp, hessian, active, primal, equality, inequality):
        assert qp is self.qp and hessian is self.hessian and active is self.active
        for row in range(len(qp.q)):
            np.testing.assert_array_equal(dense_matrix(qp, hessian, active, row), self.matrices[row])
        rhs = torch.cat((primal.flatten(1), equality, inequality.flatten(1)), -1).detach().numpy()
        self.calls.append(rhs.copy())
        call_index = len(self.calls) - 1
        if self.fault == ("raise", call_index):
            raise RuntimeError("injected native error")
        solved = np.stack(
            [
                (1 - self.contraction[row]) * np.linalg.solve(matrix, rhs[row])
                for row, matrix in enumerate(self.matrices)
            ]
        )
        if call_index and self.settled_noise:
            solved[np.max(np.abs(rhs), axis=1) == 0] = 7.0
        if self.fault == ("nan", call_index):
            solved[-1, 0] = np.nan
        if self.fault == ("inf", call_index):
            solved[-1, -1] = np.inf
        batch, stages, width = primal.shape
        nx, rows = qp.A0.shape[-2], active.shape[-1]
        p, e = stages * width, stages * nx
        output = torch.from_numpy(solved)
        return (
            output[:, :p].reshape(batch, stages, width),
            output[:, p : p + e],
            output[:, p + e :].reshape(batch, stages, rows),
        )

    def solve(self, qp, hessian, active, upstream):
        return self._solve(
            qp,
            hessian,
            active,
            upstream,
            upstream.new_zeros((len(upstream), upstream.shape[1] * 10)),
            upstream.new_zeros(active.shape),
        )

    def solve_full(self, qp, hessian, active, primal, equality, inequality):
        return self._solve(qp, hessian, active, primal, equality, inequality)


def joined(result):
    return torch.cat([part.flatten(1) for part in result], -1).detach().numpy()


def rhs_for(upstream, matrix):
    value = np.zeros((len(upstream), len(matrix)))
    value[:, : upstream[0].numel()] = upstream.detach().numpy().reshape(len(upstream), -1)
    return value


def upstream(batch=1):
    return torch.linspace(-1, 1, 42, dtype=torch.float64).reshape(1, 3, 14).repeat(batch, 1, 1)


def assert_certified(polisher):
    diagnostics = polisher.last_adjoint_diagnostics
    assert diagnostics["passed"] and diagnostics["passed_rows"].all()
    assert diagnostics["error"] is None
    assert (diagnostics["normalized_residuals"] <= 1e-11).all()
    assert (diagnostics["scaled_actual_residuals"] <= 1e-11).all()


def test_rhs_homogeneity_and_full_original_kkt_oracle():
    qp, hessian, active = problem(3)
    factors = torch.tensor([1e-12, 1.0, 1e12], dtype=torch.float64)
    source = upstream(3) * factors[:, None, None]
    snapshots = [getattr(qp, field.name).clone() for field in fields(qp)] + [
        hessian.clone(),
        active.clone(),
        source.clone(),
    ]
    backend = DenseBackend(qp, hessian, active, contraction=0.003)
    polisher = CTBRPolisher()
    result = polisher.solve_adjoint(backend, qp, hessian, active, source)
    rhs = rhs_for(source, backend.matrices[0])
    actual = joined(result)
    expected = np.stack([np.linalg.solve(matrix, rhs[row]) for row, matrix in enumerate(backend.matrices)])
    np.testing.assert_allclose(
        actual / factors.numpy()[:, None], expected / factors.numpy()[:, None], atol=3e-12, rtol=3e-12
    )
    np.testing.assert_allclose(backend.calls[0][0], backend.calls[0][1], atol=0, rtol=2e-16)
    np.testing.assert_allclose(backend.calls[0][1], backend.calls[0][2], atol=0, rtol=2e-16)
    for row, matrix in enumerate(backend.matrices):
        assert np.max(np.abs(rhs[row] - matrix @ actual[row])) / factors[row] <= 1e-11
    assert polisher.last_adjoint_diagnostics["correction_counts"].tolist() == [4, 4, 4]
    assert_certified(polisher)
    after = [getattr(qp, field.name) for field in fields(qp)] + [hessian, active, source]
    for before, value in zip(snapshots, after, strict=True):
        torch.testing.assert_close(before, value, rtol=0, atol=0)


def test_zero_upstream_has_unit_scale_exact_zero_solution_and_no_corrections():
    qp, hessian, active = problem()
    source = torch.zeros_like(upstream())
    backend = DenseBackend(qp, hessian, active, contraction=0.9)
    polisher = CTBRPolisher()
    result = polisher.solve_adjoint(backend, qp, hessian, active, source)
    assert np.count_nonzero(joined(result)) == 0 and len(backend.calls) == 1
    assert polisher.last_adjoint_diagnostics["scales"].tolist() == [1]
    assert polisher.last_adjoint_diagnostics["upstream_inf_norm"].tolist() == [0]
    assert polisher.last_adjoint_diagnostics["correction_counts"].tolist() == [0]
    assert_certified(polisher)


def test_mixed_rows_freeze_certified_directions_and_zero_their_correction_rhs():
    qp, hessian, active = problem(3)
    source = upstream(3)
    backend = DenseBackend(qp, hessian, active, [0.0, 0.001, 0.003], settled_noise=True)
    polisher = CTBRPolisher()
    actual = joined(polisher.solve_adjoint(backend, qp, hessian, active, source))
    assert polisher.last_adjoint_diagnostics["correction_counts"].tolist() == [0, 3, 4]
    assert len(backend.calls) == 5
    assert all(np.count_nonzero(call[0]) == 0 for call in backend.calls[1:])
    assert np.count_nonzero(backend.calls[-1][1]) == 0
    rhs = rhs_for(source, backend.matrices[0])
    for row, matrix in enumerate(backend.matrices):
        np.testing.assert_allclose(actual[row], np.linalg.solve(matrix, rhs[row]), rtol=3e-12, atol=3e-12)
    assert_certified(polisher)


def test_budget_exhaustion_raises_and_retains_actual_unscaled_residual():
    qp, hessian, active = problem(2)
    source = upstream(2)
    backend = DenseBackend(qp, hessian, active, [0.0, 0.01])
    polisher = CTBRPolisher()
    with pytest.raises(RuntimeError, match="original-system residual exceeded.*rows=\\[1\\]"):
        polisher.solve_adjoint(backend, qp, hessian, active, source)
    diagnostics = polisher.last_adjoint_diagnostics
    assert diagnostics["passed_rows"].tolist() == [True, False]
    assert diagnostics["correction_counts"].tolist() == [0, 4]
    assert diagnostics["batched_correction_calls"] == 4
    assert diagnostics["scaled_actual_residuals"][1].max() > 1e-11
    assert diagnostics["actual_residual_history"].shape == (2, 5, 3)
    assert not diagnostics["passed"]


@pytest.mark.parametrize("fault", [("nan", 0), ("inf", 1), ("raise", 1)])
def test_nonfinite_or_native_error_raises_without_returning_a_gradient(fault):
    qp, hessian, active = problem()
    polisher = CTBRPolisher()
    backend = DenseBackend(qp, hessian, active, contraction=0.003, fault=fault)
    with pytest.raises(RuntimeError, match="CTBR adjoint failed"):
        polisher.solve_adjoint(backend, qp, hessian, active, upstream())
    diagnostics = polisher.last_adjoint_diagnostics
    assert diagnostics["error"] and not diagnostics["passed"]
    assert len(backend.calls) == fault[1] + 1
    assert diagnostics["correction_counts"].item() == fault[1]


def test_nonfinite_original_data_rejected_before_native_call():
    qp, hessian, active = problem()
    backend = DenseBackend(qp, hessian, active)
    source = upstream()
    source[0, 0, 0] = torch.nan
    polisher = CTBRPolisher()
    with pytest.raises(RuntimeError, match="nonfinite original matrix or upstream"):
        polisher.solve_adjoint(backend, qp, hessian, active, source)
    assert not backend.calls and not polisher.last_adjoint_diagnostics["passed"]


def test_diagnostics_are_detached_and_only_the_last_call_is_retained():
    qp, hessian, active = problem()
    backend = DenseBackend(qp, hessian, active)
    polisher = CTBRPolisher()
    source = upstream().requires_grad_()
    first_result = polisher.solve_adjoint(backend, qp, hessian, active, source)
    first = polisher.last_adjoint_diagnostics
    first_scale = first["scales"].clone()
    polisher.solve_adjoint(backend, qp, hessian, active, source * 2)
    assert polisher.last_adjoint_diagnostics is not first
    torch.testing.assert_close(first["scales"], first_scale, rtol=0, atol=0)
    assert all(not value.requires_grad and value.grad_fn is None for value in first_result)
    for value in polisher.last_adjoint_diagnostics.values():
        for tensor in value if isinstance(value, tuple) else (value,):
            if isinstance(tensor, torch.Tensor):
                assert not tensor.requires_grad and tensor.grad_fn is None


@pytest.mark.parametrize("budget", [-1, 5, True, 2.0])
def test_invalid_refinement_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="integer from 0 to 4"):
        CTBRPolisher(max_adjoint_refinements=budget)


def test_invalid_call_does_not_leave_previous_success_diagnostics_visible():
    qp, hessian, active = problem()
    backend = DenseBackend(qp, hessian, active)
    polisher = CTBRPolisher(max_adjoint_refinements=0)
    polisher.solve_adjoint(backend, qp, hessian, active, upstream())
    assert polisher.last_adjoint_diagnostics["passed"]
    with pytest.raises(ValueError, match="nonempty float64"):
        polisher.solve_adjoint(backend, qp, hessian, active, upstream().float())
    assert polisher.last_adjoint_diagnostics is None
    assert len(backend.calls) == 1
