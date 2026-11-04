"""Local solution sensitivities, not derivatives of a finite ADMM rollout."""

from dataclasses import replace

import torch

import isaaclab_ipc.turbompc as turbo


def setup(nonlinear=False):
    assert hasattr(turbo, "TorchMPC"), "The native SQP and implicit-gradient layer is missing"
    kw = dict(device="cuda", dtype=torch.float64)
    matrix = torch.tensor([[1.0, 0.2]], **kw)
    if nonlinear:
        dynamics = turbo.AutoDiffDynamics(lambda z: z[:1] + 0.2 * (z[1:] - 0.1 * z[:1] ** 3), nx=1)
    else:
        dynamics = turbo.LinearDynamics(matrix, nx=1)
    d = torch.diag(torch.tensor([1.0, 0.2], **kw)).repeat(2, 5, 1, 1)
    q = torch.tensor([-0.3, 0.0], **kw).repeat(2, 5, 1)
    q[1, :, 0] = -2.0
    inputs = turbo.MPCInputs(
        initial_state=torch.tensor([[0.0], [0.2]], **kw),
        D=d,
        E=torch.zeros((2, 4, 2, 2), **kw),
        q=q,
        G=torch.tensor([0.0, 1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 5, 1, 1),
        lower=torch.full((2, 5, 1), -0.8, **kw),
        upper=torch.full((2, 5, 1), 0.8, **kw),
    )
    solver = turbo.TorchMPC(
        dynamics,
        turbo.SQPConfig(max_iter=30, tolerance=2e-8, linesearch="merit"),
        turbo.ADMMConfig(eps_abs=1e-10, eps_rel=1e-10, max_iter=4096),
    )
    return solver, inputs


def check_gradient(nonlinear):
    solver, inputs = setup(nonlinear)
    d = inputs.D.clone().requires_grad_()
    q = inputs.q.clone().requires_grad_()
    x0 = inputs.initial_state.clone().requires_grad_()
    result = solver.solve(replace(inputs, D=d, q=q, initial_state=x0))
    assert (result.status == 0).all(), result.errors
    assert result.errors.max() < 2e-8
    probe = torch.linspace(-0.3, 0.7, result.trajectory.numel(), device=q.device, dtype=q.dtype).reshape_as(
        result.trajectory
    )
    loss = (result.trajectory * probe).sum()
    gradients = torch.autograd.grad(loss, (d, q, x0))
    torch.manual_seed(17)
    directions = [torch.randn_like(x) for x in (d, q, x0)]
    directions[0] = 0.5 * (directions[0] + directions[0].transpose(-1, -2))
    for index, (gradient, direction) in enumerate(zip(gradients, directions, strict=True)):
        h = 1e-5
        values = []
        for sign in (-1, 1):
            changed = [d.detach(), q.detach(), x0.detach()]
            changed[index] = changed[index] + sign * h * direction
            trial = solver.solve(replace(inputs, D=changed[0], q=changed[1], initial_state=changed[2]))
            assert (trial.status == 0).all()
            values.append((trial.trajectory * probe).sum())
        numerical = (values[1] - values[0]) / (2 * h)
        analytic = (gradient * direction).sum()
        # IR=0 is the accepted vendor default. The observed worst-case
        # directional gradient error in this fixture is 0.426%.
        torch.testing.assert_close(analytic, numerical, atol=2e-5, rtol=5e-3)
    solver.close()


def test_linear_mpc_active_and_inactive_constraints_have_correct_cost_and_state_gradients():
    check_gradient(False)


def test_nonlinear_mpc_uses_the_full_lagrangian_hessian_in_backward():
    check_gradient(True)


def test_failed_samples_are_reported_and_have_zero_gradient():
    solver, inputs = setup(False)
    # A valid-looking pair of constraints, but an empty intersection.
    g = inputs.G.repeat(1, 1, 2, 1)
    lower = torch.cat((torch.full_like(inputs.lower, 1.0), torch.full_like(inputs.lower, -torch.inf)), -1)
    upper = torch.cat((torch.full_like(inputs.upper, torch.inf), torch.zeros_like(inputs.upper)), -1)
    q = inputs.q.clone().requires_grad_()
    result = solver.solve(replace(inputs, G=g, lower=lower, upper=upper, q=q))
    assert (result.status != 0).all()
    gradient = torch.autograd.grad(result.trajectory.sum(), q)[0]
    assert torch.count_nonzero(gradient) == 0
    solver.close()


def test_compiled_tensor_operations_preserve_solution_and_cost_gradient():
    eager, inputs = setup(True)
    compiled = turbo.TorchMPC(eager.dynamics, replace(eager.config, compile_kernels=True), eager.admm.config)
    q = inputs.q.clone().requires_grad_()
    result = eager.solve(replace(inputs, q=q))
    reference_grad = torch.autograd.grad(result.controls.square().sum(), q)[0]
    actual = compiled.solve(replace(inputs, q=q))
    actual_grad = torch.autograd.grad(actual.controls.square().sum(), q)[0]
    assert (actual.status == 0).all()
    torch.testing.assert_close(actual.trajectory, result.trajectory, atol=2e-8, rtol=1e-7)
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-7, rtol=1e-6)
    eager.close()
    compiled.close()


def test_duplicate_active_rows_preserve_cost_sensitivity():
    solver, inputs = setup(False)
    q = inputs.q.clone().requires_grad_()
    reference = solver.solve(replace(inputs, q=q))
    reference_gradient = torch.autograd.grad(reference.states.square().sum(), q)[0]
    duplicated = replace(
        inputs,
        q=q,
        G=inputs.G.repeat(1, 1, 2, 1),
        lower=inputs.lower.repeat(1, 1, 2),
        upper=inputs.upper.repeat(1, 1, 2),
    )
    actual = solver.solve(duplicated)
    actual_gradient = torch.autograd.grad(actual.states.square().sum(), q)[0]
    assert (actual.status == 0).all()
    torch.testing.assert_close(actual.trajectory, reference.trajectory, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-6, rtol=1e-4)
    solver.close()


def test_implicit_derivatives_include_control_rate_cost_and_moving_linear_bounds():
    solver, inputs = setup(False)
    e = inputs.E.clone()
    e[..., 1, 1] = -0.04
    e.requires_grad_()
    g = inputs.G.clone().requires_grad_()
    upper = inputs.upper.clone().requires_grad_()
    learned = replace(inputs, E=e, G=g, upper=upper)
    result = solver.solve(learned)
    assert (result.status == 0).all()
    gradient = torch.autograd.grad(result.states.square().sum(), (e, g, upper))
    torch.manual_seed(31)
    for name, value, grad in zip(("E", "G", "upper"), (e, g, upper), gradient, strict=True):
        direction = torch.randn_like(value)
        h = 1e-5
        with torch.no_grad():
            negative = solver.solve(replace(learned, **{name: value - h * direction}))
            positive = solver.solve(replace(learned, **{name: value + h * direction}))
            assert (negative.status == 0).all() and (positive.status == 0).all()
            numerical = (positive.states.square().sum() - negative.states.square().sum()) / (2 * h)
        torch.testing.assert_close((grad * direction).sum(), numerical, atol=2e-5, rtol=2e-3)
    solver.close()
