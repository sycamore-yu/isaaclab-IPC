"""True nonlinear feasibility and regular implicit sensitivities of stage bounds."""

from dataclasses import replace

import pytest
import torch

from isaaclab_ipc.turbompc.constraints import PathConstraints
from isaaclab_ipc.turbompc.dynamics import LinearDynamics
from isaaclab_ipc.turbompc.native import ADMMConfig, ADMMResult
from isaaclab_ipc.turbompc.solver import MPCInputs, SQPConfig, TorchMPC

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native cuDSS requires CUDA")


def problem(device="cpu", batch=1):
    """Fixed scalar state and independent two-dimensional physical control."""
    kw = dict(dtype=torch.float64, device=device)
    dynamics = LinearDynamics(torch.tensor([[1.0, 0.0, 0.0]], **kw), nx=1)
    q = torch.zeros((batch, 2, 3), **kw)
    q[:, 0, 1:] = torch.tensor([-2.0, -0.6], **kw)
    inputs = MPCInputs(
        initial_state=torch.zeros((batch, 1), **kw),
        D=torch.eye(3, **kw).repeat(batch, 2, 1, 1),
        E=torch.zeros((batch, 1, 3, 3), **kw),
        q=q,
        G=torch.zeros((batch, 2, 1, 3), **kw),
        lower=torch.full((batch, 2, 1), -torch.inf, **kw),
        upper=torch.full((batch, 2, 1), torch.inf, **kw),
    )
    return dynamics, inputs


def circle(device="cpu", *, include_state=False):
    def function(z):
        norm = z[1:].square().sum()
        return torch.stack((norm, z[0].square())) if include_state else norm.unsqueeze(0)

    rows = 2 if include_state else 1
    return PathConstraints(
        function,
        lower=torch.full((rows,), -torch.inf, dtype=torch.float64, device=device),
        upper=torch.ones(rows, dtype=torch.float64, device=device),
        stage_mask=torch.tensor([[True] * rows, [False] * rows], device=device),
    )


def solver_for(dynamics, path, *, compile_kernels=False):
    return TorchMPC(
        dynamics,
        SQPConfig(max_iter=30, tolerance=1e-7, linesearch="merit", compile_kernels=compile_kernels),
        ADMMConfig(max_iter=16384, eps_abs=1e-8, eps_rel=0),
        constraints=path,
    )


def test_fixed_bounds_and_stage_mask_are_validated_and_owned():
    lower = torch.tensor([-1.0], dtype=torch.float64)
    upper = torch.tensor([1.0], dtype=torch.float64)
    mask = torch.tensor([[True], [False]])
    path = PathConstraints(lambda z: z[:1], lower=lower, upper=upper, stage_mask=mask)
    lower.fill_(-7)
    mask.fill_(True)
    assert path.lower.item() == -1 and not path.stage_mask[-1].item()
    with pytest.raises(ValueError, match="Learnable path bounds"):
        PathConstraints(lambda z: z[:1], lower=lower.requires_grad_(), upper=upper)
    with pytest.raises(ValueError, match="stage_mask"):
        PathConstraints(lambda z: z[:1], lower=lower.detach(), upper=upper, stage_mask=torch.ones(2, 1))
    with pytest.raises(ValueError, match="NaN"):
        PathConstraints(lambda z: z[:1], lower=torch.full_like(upper, torch.nan), upper=upper)
    with pytest.raises(ValueError, match="Learnable path physics"):
        PathConstraints(torch.nn.Linear(1, 1).double(), lower=lower.detach(), upper=upper)
    with pytest.raises(ValueError, match="node count"):
        path(torch.zeros((1, 3, 1), dtype=torch.float64))


def test_true_values_and_terminal_mask_are_independent_of_the_qp_linearization():
    dynamics, inputs = problem()
    path = circle()
    solver = solver_for(dynamics, path)
    z = torch.zeros_like(inputs.q)
    qp = solver._qp(inputs, z)
    trial = z.clone()
    trial[:, :, 1] = 2
    # At the zero-Jacobian expansion point the local row is vacuous, but the
    # original physical bound at u=(2,0) is violated by exactly three.
    assert (qp.G[..., -1, :] == 0).all()
    torch.testing.assert_close(path.violations(trial), torch.tensor([[[3.0], [0.0]]], dtype=z.dtype))
    _, physical = solver._violations(inputs, trial)
    assert physical[0, 0, -1] == 3 and physical[0, 1, -1] == 0
    assert torch.isneginf(qp.lower[0, 1, -1]) and torch.isposinf(qp.upper[0, 1, -1])
    trial[:, 0, 1] = 0.5
    shifted = solver._qp(inputs, trial)
    assert shifted.G[0, 0, -1, 1] == 1 and shifted.upper[0, 0, -1] == 1.25
    solver.close()


def test_stale_inactive_duals_do_not_add_curvature_or_pass_the_normal_cone_gate(monkeypatch):
    dynamics, inputs = problem()
    inputs = replace(inputs, q=torch.zeros_like(inputs.q, requires_grad=True))
    path = circle()
    solver = solver_for(dynamics, path)
    z = torch.zeros_like(inputs.q)
    path_y = torch.ones((1, 2, 1), dtype=z.dtype)
    active, _, _ = path.active_sets(z, path_y, 1e-7, 0)
    assert not active.any()
    assert path.lagrangian_hessian(z, path_y)[0, 0, 1, 1] == 2
    assert torch.count_nonzero(path.lagrangian_hessian(z, torch.where(active, path_y, 0))) == 0
    torch.testing.assert_close(path.normal_cone_residual(z, path_y), torch.tensor([[[1.0], [0.0]]], dtype=z.dtype))

    def invalid_candidate(qp, warm_start):
        del warm_start
        y = torch.zeros_like(qp.lower)
        y[..., -1] = 1
        return ADMMResult(
            iterations=torch.ones(1, dtype=torch.int32),
            x=z,
            z=torch.zeros_like(y),
            y=y,
            lambda0=z.new_zeros((1, 1)),
            lambda_dyn=z.new_zeros((1, 1, 1)),
            xi=torch.zeros_like(y),
            rho=z.new_ones(1),
            kernel_ns=z.new_zeros(1),
            status=torch.zeros(1, dtype=torch.int32),
            residuals=z.new_zeros((1, 2)),
        )

    monkeypatch.setattr(solver.admm, "solve", invalid_candidate)
    result = solver.solve(inputs)
    assert result.errors.shape == (1, 3) and torch.count_nonzero(result.errors) == 0
    assert result.normal_cone_error.item() == 1 and result.status.item() != 0
    assert torch.count_nonzero(torch.autograd.grad(result.trajectory.sum(), inputs.q)[0]) == 0
    solver.close()


@pytest.mark.parametrize("with_path", [False, True])
@pytest.mark.parametrize(
    ("require_qp", "budget", "expected_status", "expected_iterations"),
    [(False, 3, 0, 1), (True, 3, 0, 2), (True, 1, 1, 1)],
)
def test_inner_qp_budget_does_not_prematurely_certify_a_strict_outer_solution(
    monkeypatch, with_path, require_qp, budget, expected_status, expected_iterations
):
    dynamics, inputs = problem()
    inputs = replace(inputs, q=(0.1 * inputs.q).requires_grad_())
    solver = TorchMPC(
        dynamics,
        SQPConfig(max_iter=budget, tolerance=1e-6, require_qp_convergence=require_qp),
        ADMMConfig(max_iter=1, eps_abs=1e-8, eps_rel=0),
        constraints=circle() if with_path else None,
    )
    candidates = []

    def refine(qp, warm_start):
        # The first QP has an independently computed dual residual between the
        # inner and outer tolerances. A warm continuation reaches the exact
        # analytic solution of this unconstrained quadratic on the next call.
        x = -qp.q.clone()
        if not candidates:
            assert warm_start is None
            x[0, 0, 1] += 5e-7
        else:
            torch.testing.assert_close(warm_start.x, candidates[-1].x, rtol=0, atol=0)
            assert warm_start.status.item() == 1
        dual = (x + qp.q).abs().amax()
        candidate = ADMMResult(
            iterations=torch.ones(1, dtype=torch.int32),
            x=x,
            z=torch.zeros_like(qp.lower),
            y=torch.zeros_like(qp.lower),
            lambda0=x.new_zeros((1, 1)),
            lambda_dyn=x.new_zeros((1, 1, 1)),
            xi=torch.zeros_like(qp.lower),
            rho=x.new_ones(1),
            kernel_ns=x.new_zeros(1),
            status=(dual > solver.admm.config.eps_abs).to(torch.int32).reshape(1),
            residuals=torch.stack((dual * 0, dual, dual.new_tensor(1e-8), dual.new_tensor(1e-8)))[None],
        )
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(solver.admm, "solve", refine)
    try:
        assert SQPConfig().require_qp_convergence is False
        result = solver.solve(inputs)
        assert result.errors.amax() <= solver.config.tolerance
        assert len(candidates) == result.iterations.item() == expected_iterations
        assert result.status.item() == expected_status
        assert result.admm.status.item() == (expected_iterations == 1)
        if expected_iterations == 2:
            torch.testing.assert_close(result.trajectory, -inputs.q, rtol=0, atol=0)
        if expected_status != 0:
            gradient = torch.autograd.grad(result.trajectory.sum(), inputs.q)[0]
            assert torch.count_nonzero(gradient) == 0
    finally:
        solver.close()


@gpu
def test_curved_active_constraint_matches_dense_kkt_finite_difference_and_compilation():
    dynamics, inputs = problem("cuda")
    path = circle("cuda")
    q = inputs.q.clone().requires_grad_()
    learned = replace(inputs, q=q)
    eager, compiled = solver_for(dynamics, path), solver_for(dynamics, path, compile_kernels=True)
    try:
        result = eager.solve(learned)
        assert result.status.item() == 0, (result.errors, result.normal_cone_error)
        assert result.errors.shape == (1, 3) and result.normal_cone_error.max() <= 1e-7
        target = -q.detach()[0, 0, 1:]
        control = target / target.norm()
        torch.testing.assert_close(result.controls[0, 0], control, atol=2e-7, rtol=0)
        torch.testing.assert_close(result.controls[0, -1], torch.zeros_like(control), atol=1e-10, rtol=0)
        probe = torch.tensor([0.3, -0.7], dtype=q.dtype, device=q.device)
        gradient = torch.autograd.grad((result.controls[0, 0] * probe).sum(), q)[0]
        # Independent 3x3 KKT: H=I+2*lambda*I, J=2*u. Omitting
        # nonlinear constraint curvature gives the wrong tangent derivative.
        hessian = target.norm().cpu() * torch.eye(2, dtype=q.dtype)
        jacobian = (2 * control).cpu()
        kkt = torch.zeros((3, 3), dtype=q.dtype)
        kkt[:2, :2], kkt[:2, 2], kkt[2, :2] = hessian, jacobian, jacobian
        dense = torch.linalg.solve(kkt, torch.cat((probe.cpu(), torch.zeros(1, dtype=q.dtype))))
        torch.testing.assert_close(gradient[0, 0, 1:].cpu(), -dense[:2], atol=3e-7, rtol=1e-6)
        direction = torch.zeros_like(q)
        direction[0, 0, 1:] = torch.tensor([0.2, 0.4], dtype=q.dtype, device=q.device)
        analytic = (gradient * direction).sum()
        for h in (1e-3, 5e-4):
            losses = []
            for sign in (-1, 1):
                trial = eager.solve(replace(inputs, q=q.detach() + sign * h * direction))
                assert trial.status.item() == 0
                losses.append((trial.controls[0, 0] * probe).sum())
            torch.testing.assert_close(analytic, (losses[1] - losses[0]) / (2 * h), atol=3e-6, rtol=1e-4)
        actual = compiled.solve(learned)
        actual_gradient = torch.autograd.grad((actual.controls[0, 0] * probe).sum(), q)[0]
        assert actual.status.item() == 0
        torch.testing.assert_close(actual.trajectory, result.trajectory, atol=3e-7, rtol=1e-7)
        torch.testing.assert_close(actual_gradient, gradient, atol=3e-7, rtol=1e-6)
    finally:
        eager.close()
        compiled.close()


@gpu
def test_linear_geometry_gradients_keep_their_row_mapping_with_active_path_constraints():
    dynamics, inputs = problem("cuda")
    q, g, upper = inputs.q.clone(), inputs.G.clone(), inputs.upper.clone()
    q[0, 0, 1:] = -2
    g[0, 0, 0, 1], upper[0, 0, 0] = 1, 0.6
    g.requires_grad_()
    upper.requires_grad_()
    learned = replace(inputs, q=q, G=g, upper=upper)
    guess = torch.zeros_like(q)
    guess[0, 0, 1:] = torch.tensor([0.6, 0.8], dtype=q.dtype, device=q.device)
    solver = solver_for(dynamics, circle("cuda"))
    try:
        result = solver.solve(learned, guess)
        assert result.status.item() == 0, (result.errors, result.normal_cone_error)
        torch.testing.assert_close(result.controls[0, 0], guess[0, 0, 1:], atol=2e-7, rtol=0)
        grad_g, grad_upper = torch.autograd.grad(result.controls[0, 0, 1], (g, upper))
        torch.testing.assert_close(grad_upper[0, 0, 0], q.new_tensor(-0.75), atol=1e-6, rtol=0)
        torch.testing.assert_close(grad_g[0, 0, 0, 1:], q.new_tensor([0.45, 0.6]), atol=1e-6, rtol=0)
        h = 1e-3
        losses = []
        for sign in (-1, 1):
            perturbed = upper.detach().clone()
            perturbed[0, 0, 0] += sign * h
            trial = solver.solve(replace(learned, upper=perturbed), guess)
            assert trial.status.item() == 0
            losses.append(trial.controls[0, 0, 1])
        torch.testing.assert_close(grad_upper[0, 0, 0], (losses[1] - losses[0]) / (2 * h), atol=3e-5, rtol=1e-4)
    finally:
        solver.close()


@gpu
def test_failed_path_row_has_zero_gradient_without_contaminating_valid_batch():
    dynamics, inputs = problem("cuda", batch=2)
    inputs.initial_state[1, 0] = 2
    q = inputs.q.clone().requires_grad_()
    solver = solver_for(dynamics, circle("cuda", include_state=True))
    try:
        result = solver.solve(replace(inputs, q=q))
        assert result.status[0] == 0 and result.status[1] != 0, result.errors
        gradient = torch.autograd.grad(result.controls[..., 1].sum(), q)[0]
        assert torch.count_nonzero(gradient[1]) == 0 and torch.isfinite(gradient).all()
        _, single = problem("cuda")
        single_q = single.q.clone().requires_grad_()
        reference = solver.solve(replace(single, q=single_q))
        reference_gradient = torch.autograd.grad(reference.controls[..., 1].sum(), single_q)[0]
        assert reference.status.item() == 0
        torch.testing.assert_close(result.trajectory[:1], reference.trajectory, atol=2e-7, rtol=0)
        torch.testing.assert_close(gradient[:1], reference_gradient, atol=3e-7, rtol=1e-6)
    finally:
        solver.close()
