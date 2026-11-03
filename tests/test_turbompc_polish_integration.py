"""CPU integration contracts for optional numerical solution refinement.

The ADMM output is deliberate approximate fault injection. The nonlinear
problem and dense KKT oracle are independent of the refinement implementation.
"""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_ipc.turbompc import ADMMConfig, ADMMResult, AutoDiffDynamics, MPCInputs, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr_polish import CTBRPolisher


def transition(z):
    return (z[0] + z[1].square()).unsqueeze(0)


class DenseKKTReference:
    """Eliminate inactive rows and solve an ordinary dense symmetric system."""

    def __init__(self):
        self.calls = []

    def solve(self, qp, hessian, active, upstream):
        self.calls.append((hessian.clone(), active.clone(), qp.A_minus.clone(), upstream.clone()))
        assert all(torch.isfinite(value).all() for value in (qp.q, qp.c0, qp.c_dyn, hessian, upstream))
        batch, stages, width = upstream.shape
        n0, nx = qp.A0.shape[-2], qp.A_minus.shape[-2]
        primal_size, equality_size = stages * width, n0 + (stages - 1) * nx
        primal, equality, inequality = [], [], []
        for row in range(batch):
            h = np.zeros((primal_size, primal_size))
            c = np.zeros((equality_size, primal_size))
            c[:n0, :width] = qp.A0[row].detach().numpy()
            for stage in range(stages):
                block = slice(stage * width, (stage + 1) * width)
                h[block, block] = hessian[row, stage].detach().numpy()
                if stage:
                    previous = slice((stage - 1) * width, stage * width)
                    h[block, previous] = qp.E[row, stage - 1].detach().numpy()
                    h[previous, block] = h[block, previous].T
                    eq = slice(n0 + (stage - 1) * nx, n0 + stage * nx)
                    c[eq, previous] = qp.A_minus[row, stage - 1].detach().numpy()
                    c[eq, block] = qp.A_plus[row, stage - 1].detach().numpy()
            selected = active[row].nonzero().tolist()
            g = np.zeros((len(selected), primal_size))
            for index, (stage, constraint) in enumerate(selected):
                g[index, stage * width : (stage + 1) * width] = qp.G[row, stage, constraint].detach().numpy()
            a = np.concatenate((c, g))
            kkt = np.block([[h, a.T], [a, np.zeros((len(a), len(a)))]])
            rhs = np.concatenate((upstream[row].detach().numpy().ravel(), np.zeros(len(a))))
            solved = np.linalg.solve(kkt, rhs)
            primal.append(torch.from_numpy(solved[:primal_size].reshape(stages, width)))
            equality.append(torch.from_numpy(solved[primal_size : primal_size + equality_size]))
            multipliers = torch.zeros_like(qp.lower[row])
            for index, (stage, constraint) in enumerate(selected):
                multipliers[stage, constraint] = solved[primal_size + equality_size + index]
            inequality.append(multipliers)
        return torch.stack(primal), torch.stack(equality), torch.stack(inequality)


class AnalyticPolisher:
    """Positive local minimizer of .5*x1² + q*x1 + .5*u0², x1=x0+u0²."""

    def __init__(self, fail_row=None):
        self.fail_row = fail_row
        self.validations = []
        self.accepted = []
        self.results = []

    def validate_inputs(self, inputs, dynamics):
        self.validations.append((inputs, dynamics))

    def __call__(self, solver, inputs, z, admm, accepted):
        assert self.validations and self.validations[-1][1] is solver.dynamics
        assert not any(getattr(inputs, field.name).requires_grad for field in fields(inputs))
        self.accepted.append(accepted.clone())
        trajectory = z.clone()
        u = (-inputs.q[:, 1, 0] - 0.5 - inputs.initial_state[:, 0]).sqrt()
        trajectory[accepted, 0, 0] = inputs.initial_state[accepted, 0]
        trajectory[accepted, 0, 1] = u[accepted]
        trajectory[accepted, 1, 0] = inputs.initial_state[accepted, 0] + u[accepted].square()
        trajectory[accepted, 1, 1] = 0
        lambda0, lambda_dyn, y = admm.lambda0.clone(), admm.lambda_dyn.clone(), admm.y.clone()
        lambda0[accepted], lambda_dyn[accepted], y[accepted] = 0.5, -0.5, 0
        active_mask = torch.zeros_like(y, dtype=torch.bool)
        status = torch.where(accepted, 0, 1).to(torch.int32)
        errors = z.new_zeros((len(z), 3))
        if self.fail_row is not None:
            row = self.fail_row
            assert accepted[row]
            status[row] = 2
            trajectory[row] = lambda0[row] = lambda_dyn[row] = y[row] = torch.nan
            errors[row] = torch.inf
            active_mask[row] = True
        result = SimpleNamespace(
            trajectory=trajectory,
            lambda0=lambda0,
            lambda_dyn=lambda_dyn,
            y=y,
            active_mask=active_mask,
            status=status,
            iterations=accepted.to(torch.int32),
            errors=errors,
        )
        self.results.append(result)
        return result


def scalar_case(monkeypatch, *, batch=1, polisher=None, explicit_none=False, compiled=False, near_bound=False):
    dtype = torch.float64
    d = torch.eye(2, dtype=dtype).repeat(batch, 2, 1, 1)
    d[:, 0, 0, 0] = 0
    q = torch.zeros((batch, 2, 2), dtype=dtype)
    q[:, 1, 0] = -1.5
    g = torch.zeros((batch, 2, 1, 2), dtype=dtype)
    upper = torch.full((batch, 2, 1), torch.inf, dtype=dtype)
    if near_bound:
        # Strictly inactive x0 <= 5e-9, close enough to fool tolerance touching.
        # x0 is already an equality: activating this row makes KKT singular.
        g[:, 0, 0, 0], upper[:, 0, 0] = 1, 5e-9
    inputs = MPCInputs(
        torch.zeros((batch, 1), dtype=dtype, requires_grad=True),
        d,
        torch.zeros((batch, 1, 2, 2), dtype=dtype),
        q.requires_grad_(),
        g,
        torch.full_like(upper, -torch.inf),
        upper.requires_grad_(),
    )
    options = {"polisher": polisher} if polisher is not None or explicit_none else {}
    solver = TorchMPC(
        AutoDiffDynamics(transition, nx=1),
        # This deliberately loose fixture accepts the supplied approximate QP;
        # it is not a change to production or acceptance tolerances.
        SQPConfig(max_iter=1, tolerance=0.25, require_qp_convergence=True, compile_kernels=compiled),
        ADMMConfig(eps_abs=1e-8, eps_rel=0),
        **options,
    )
    x = torch.tensor([[0.0, 0.99], [0.99**2, 0.0]], dtype=dtype).repeat(batch, 1, 1)
    raw = ADMMResult(
        iterations=torch.full((batch,), 25, dtype=torch.int32),
        x=x,
        z=torch.zeros_like(upper),
        y=torch.zeros_like(upper),
        lambda0=torch.full((batch, 1), 0.4, dtype=dtype),
        lambda_dyn=torch.full((batch, 1, 1), -0.4, dtype=dtype),
        xi=torch.zeros_like(upper),
        rho=torch.full((batch,), 0.1, dtype=dtype),
        kernel_ns=torch.arange(batch, dtype=dtype),
        status=torch.zeros(batch, dtype=torch.int32),
        residuals=torch.zeros((batch, 4), dtype=dtype),
    )
    calls = []

    def injected_admm(qp, warm_start):
        assert warm_start is None
        calls.append(qp)
        return raw

    reference = DenseKKTReference()
    monkeypatch.setattr(solver.admm, "solve", injected_admm)
    monkeypatch.setattr(solver, "_adjoint", lambda qp: reference)
    return SimpleNamespace(solver=solver, inputs=inputs, raw=raw, reference=reference, calls=calls)


def assert_raw_preserved(actual, expected):
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), expected[field.name], atol=0, rtol=0, equal_nan=True)


def test_default_none_keeps_original_unpolished_outputs_and_backward(monkeypatch):
    results = []
    for explicit in (False, True):
        case = scalar_case(monkeypatch, explicit_none=explicit)
        before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
        try:
            result = case.solver.solve(case.inputs, case.raw.x)
            assert result.polish is None and result.status.item() == 0
            torch.testing.assert_close(result.trajectory, before["x"], atol=0, rtol=0)
            (gradient,) = torch.autograd.grad(result.controls[:, 0].sum(), (case.inputs.q,))
            results.append(gradient)
            # The injected dual is approximate. This checks preservation of the
            # existing KKT semantics, not the quality of a real ADMM solution.
            expected = -1.98 / (0.2 + 1.98**2)
            assert gradient[0, 1, 0].item() == pytest.approx(expected, abs=1e-13)
            assert_raw_preserved(result.admm, before)
            assert_raw_preserved(case.raw, before)
        finally:
            case.solver.close()
    torch.testing.assert_close(*results, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_refined_trajectory_and_duals_control_full_hessian_vjp(monkeypatch, compiled):
    polisher = AnalyticPolisher()
    case = scalar_case(monkeypatch, polisher=polisher, compiled=compiled)
    before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert result.polish is polisher.results[0]
        assert result.status.item() == result.polish.status.item() == 0
        torch.testing.assert_close(
            result.trajectory, result.trajectory.new_tensor([[[0, 1], [1, 0]]]), atol=1e-14, rtol=0
        )
        torch.testing.assert_close(result.errors, torch.zeros_like(result.errors), atol=1e-14, rtol=0)
        q_gradient, x0_gradient = torch.autograd.grad(
            result.controls[:, 0].sum(), (case.inputs.q, case.inputs.initial_state)
        )
        assert q_gradient[0, 1, 0].item() == pytest.approx(-0.5, abs=1e-13)
        assert x0_gradient.item() == pytest.approx(-0.5, abs=1e-13)
        hessian, active, dynamics, _ = case.reference.calls[-1]
        assert hessian[0, 0, 1, 1].item() == pytest.approx(0, abs=1e-14)
        assert dynamics[0, 0, 0, 1].item() == pytest.approx(2, abs=1e-14)
        assert not active.any()
        # Both central differences exercise the public solve with changed q,
        # independently checking the positive sqrt branch's derivative.
        for step in (1e-4, 5e-5):
            values = []
            for sign in (-1, 1):
                q = case.inputs.q.detach().clone()
                q[:, 1, 0] += sign * step
                point = case.solver.solve(replace(case.inputs, q=q), case.raw.x)
                assert point.status.item() == 0
                values.append(point.controls[0, 0, 0].item())
            assert (values[1] - values[0]) / (2 * step) == pytest.approx(-0.5, abs=1e-8)
        assert_raw_preserved(result.admm, before)
        assert_raw_preserved(case.raw, before)
        assert polisher.validations[0][0] is case.inputs
    finally:
        case.solver.close()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_iteration_limit_finisher_preserves_sqp_diagnostics_and_refined_vjp(monkeypatch, compiled):
    polisher = AnalyticPolisher()
    polisher.finish_iteration_limit = True
    case = scalar_case(monkeypatch, polisher=polisher, compiled=compiled)
    case.solver.config = replace(case.solver.config, tolerance=1e-7)
    before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert result.sqp_status.tolist() == [1] and result.status.tolist() == [0]
        assert result.iterations.tolist() == [1] and result.admm.status.tolist() == [0]
        assert result.sqp_errors[0, 0].item() == pytest.approx(0.198, abs=1e-14)
        assert result.sqp_errors.max() > case.solver.config.tolerance
        assert result.errors.max() <= 1e-14
        assert result.sqp_errors.data_ptr() != result.errors.data_ptr()
        assert polisher.accepted[0].tolist() == [True]
        gradient = torch.autograd.grad(result.controls[:, 0].sum(), case.inputs.q)[0]
        assert gradient[0, 1, 0].item() == pytest.approx(-0.5, abs=1e-13)
        # The complete nonlinear Hessian must use the certified -0.5 dual,
        # not the raw approximate -0.4 dual from the exhausted SQP phase.
        hessian, active, _, _ = case.reference.calls[-1]
        assert hessian[0, 0, 1, 1].item() == pytest.approx(0, abs=1e-14)
        assert not active.any()
        assert_raw_preserved(result.admm, before)
        assert_raw_preserved(case.raw, before)
    finally:
        case.solver.close()


def test_iteration_limit_failed_certificate_keeps_finite_outputs_but_zero_gradient(monkeypatch):
    class RefusedFinisher(AnalyticPolisher):
        finish_iteration_limit = True

        def __call__(self, solver, inputs, z, admm, accepted):
            result = super().__call__(solver, inputs, z, admm, accepted)
            result.status[accepted] = 2
            result.errors[accepted] = 1
            return result

    polisher = RefusedFinisher()
    case = scalar_case(monkeypatch, polisher=polisher)
    case.solver.config = replace(case.solver.config, tolerance=1e-7)
    before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert polisher.accepted[0].tolist() == [True]
        assert result.sqp_status.tolist() == [1] and result.status.tolist() == [4]
        assert result.polish.status.tolist() == [2]
        assert torch.isfinite(result.trajectory).all()
        gradient = torch.autograd.grad(result.controls.sum(), case.inputs.q)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
        assert result.sqp_errors.max() > case.solver.config.tolerance
        assert_raw_preserved(result.admm, before)
    finally:
        case.solver.close()


def test_iteration_limit_mixed_rows_isolate_solved_limit_raw_limit_fatal_and_nonfinite(monkeypatch):
    polisher = AnalyticPolisher()
    polisher.finish_iteration_limit = True
    case = scalar_case(monkeypatch, batch=5, polisher=polisher)
    case.solver.config = replace(case.solver.config, tolerance=1e-7)
    # Row 0 is an exact original SQP solution, row 1 exhausts its SQP budget.
    case.raw.x[0] = case.raw.x.new_tensor([[0, 1], [1, 0]])
    case.raw.lambda0[0], case.raw.lambda_dyn[0] = 0.5, -0.5
    case.raw.status[2], case.raw.status[3] = 1, 2
    case.raw.x[4] = torch.nan
    before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert result.sqp_status.tolist() == [0, 1, 1, 2, 3]
        assert result.status.tolist() == [0, 0, 1, 2, 3]
        assert result.polish.status.tolist() == [0, 0, 1, 1, 1]
        assert polisher.accepted[0].tolist() == [True, True, False, False, False]
        gradient = torch.autograd.grad(result.controls.nan_to_num().sum(), case.inputs.q)[0]
        torch.testing.assert_close(gradient[:2, 1, 0], gradient.new_full((2,), -0.5), atol=1e-13, rtol=0)
        torch.testing.assert_close(gradient[2:], torch.zeros_like(gradient[2:]), atol=0, rtol=0)
        assert torch.isfinite(gradient).all()
        assert_raw_preserved(result.admm, before)
        assert_raw_preserved(case.raw, before)
    finally:
        case.solver.close()


def test_iteration_limit_finishing_requires_an_explicit_polisher_opt_in(monkeypatch):
    for mode in ("none", "undeclared", "disabled"):
        polisher = None if mode == "none" else AnalyticPolisher()
        if mode == "disabled":
            polisher.finish_iteration_limit = False
        case = scalar_case(monkeypatch, polisher=polisher)
        case.solver.config = replace(case.solver.config, tolerance=1e-7)
        try:
            result = case.solver.solve(case.inputs, case.raw.x)
            assert result.status.tolist() == [1]
            gradient = torch.autograd.grad(result.controls.sum(), case.inputs.q)[0]
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
            if polisher is None:
                assert result.sqp_status is None and result.sqp_errors is None and result.polish is None
            else:
                assert result.sqp_status.tolist() == [1] and result.polish.status.tolist() == [1]
                assert polisher.accepted[0].tolist() == [False]
        finally:
            case.solver.close()


def test_sqp_success_with_raw_iteration_limit_still_rejects_polish_route(monkeypatch):
    polisher = AnalyticPolisher()
    polisher.finish_iteration_limit = True
    case = scalar_case(monkeypatch, polisher=polisher)
    case.solver.config = replace(case.solver.config, require_qp_convergence=False)
    case.raw.status[:] = 1
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert result.sqp_status.tolist() == [0] and result.admm.status.tolist() == [1]
        assert result.status.tolist() == [4] and result.polish.status.tolist() == [1]
        assert polisher.accepted[0].tolist() == [False]
        gradient = torch.autograd.grad(result.controls.sum(), case.inputs.q)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)
    finally:
        case.solver.close()


def test_ctbr_iteration_limit_option_is_boolean_and_defaults_to_historical_behavior():
    assert CTBRPolisher().finish_iteration_limit is False
    assert CTBRPolisher(finish_iteration_limit=True).finish_iteration_limit is True
    for invalid in (0, 1, None, "true"):
        with pytest.raises(ValueError, match="finish_iteration_limit must be bool"):
            CTBRPolisher(finish_iteration_limit=invalid)


def test_certified_inactive_face_is_not_reactivated_by_near_bound_heuristic(monkeypatch):
    case = scalar_case(monkeypatch, polisher=AnalyticPolisher(), near_bound=True)
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        q_gradient, upper_gradient = torch.autograd.grad(
            result.controls[:, 0].sum(), (case.inputs.q, case.inputs.upper)
        )
        assert result.status.item() == 0 and not case.reference.calls[-1][1].any()
        assert q_gradient[0, 1, 0].item() == pytest.approx(-0.5, abs=1e-13)
        torch.testing.assert_close(upper_gradient, torch.zeros_like(upper_gradient), atol=0, rtol=0)
    finally:
        case.solver.close()


def test_failed_polish_and_not_attempted_rows_keep_zero_gradients_and_raw_buffers(monkeypatch):
    polisher = AnalyticPolisher(fail_row=1)
    case = scalar_case(monkeypatch, batch=3, polisher=polisher, near_bound=True)
    case.raw.status[2] = 2
    before = {field.name: getattr(case.raw, field.name).clone() for field in fields(case.raw)}
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert polisher.accepted[0].tolist() == [True, True, False]
        assert result.polish.status.tolist() == [0, 2, 1]
        assert result.status.tolist() == [0, 4, 2]
        gradients = torch.autograd.grad(
            result.controls.nan_to_num().sum(), (case.inputs.q, case.inputs.initial_state, case.inputs.upper)
        )
        for gradient in gradients:
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient[1:], torch.zeros_like(gradient[1:]), atol=0, rtol=0)
        assert gradients[0][0, 1, 0].item() == pytest.approx(-0.5, abs=1e-13)
        assert not case.reference.calls[-1][1].any()
        torch.testing.assert_close(result.trajectory[2], before["x"][2], atol=0, rtol=0)
        assert_raw_preserved(result.admm, before)
        assert_raw_preserved(case.raw, before)
    finally:
        case.solver.close()


def test_validator_sees_original_physical_requires_grad_before_numerical_detach(monkeypatch):
    class RejectPhysicalGradients(AnalyticPolisher):
        def validate_inputs(self, inputs, dynamics):
            super().validate_inputs(inputs, dynamics)
            assert inputs.initial_state.requires_grad and inputs.G.requires_grad
            raise ValueError("physical data requires gradients")

    polisher = RejectPhysicalGradients()
    case = scalar_case(monkeypatch, polisher=polisher)
    case.inputs.G.requires_grad_()
    try:
        with pytest.raises(ValueError, match="physical data requires gradients"):
            case.solver.solve(case.inputs, case.raw.x)
        assert polisher.validations[0][0] is case.inputs
        assert not case.calls and not polisher.results
    finally:
        case.solver.close()


@pytest.mark.parametrize("reject", [False, True], ids=["certified-vjp", "certificate-failure"])
def test_optional_certified_adjoint_controls_gradient_or_propagates_failure(monkeypatch, reject):
    """A successful forward cannot bypass a polisher's backward certificate."""

    class CertifiedAnalyticPolisher(AnalyticPolisher):
        def __init__(self):
            super().__init__()
            self.adjoint_calls = []
            self.exact = DenseKKTReference()

        def solve_adjoint(self, backend, qp, hessian, active, upstream):
            self.adjoint_calls.append((backend, upstream.clone()))
            if reject:
                raise RuntimeError("original KKT certificate failed")
            # Independent exact solve of the analytic nonlinear problem. The
            # supplied raw backend is deliberately inaccurate in this fixture.
            return self.exact.solve(qp, hessian, active, upstream)

    polisher = CertifiedAnalyticPolisher()
    case = scalar_case(monkeypatch, polisher=polisher)
    original_solve = case.reference.solve
    monkeypatch.setattr(case.reference, "solve", lambda *args: tuple(value * 0.5 for value in original_solve(*args)))
    try:
        result = case.solver.solve(case.inputs, case.raw.x)
        assert result.status.item() == result.polish.status.item() == 0
        if reject:
            with pytest.raises(RuntimeError, match="original KKT certificate failed"):
                torch.autograd.grad(result.controls[:, 0].sum(), (case.inputs.q,))
            assert case.inputs.q.grad is None and not polisher.exact.calls
        else:
            (gradient,) = torch.autograd.grad(result.controls[:, 0].sum(), (case.inputs.q,))
            assert gradient[0, 1, 0].item() == pytest.approx(-0.5, abs=1e-13)
            assert len(polisher.exact.calls) == 1
        assert len(polisher.adjoint_calls) == 1
        assert polisher.adjoint_calls[0][0] is case.reference
        assert not case.reference.calls
    finally:
        case.solver.close()
