import numpy as np
import osqp
import pytest
import torch
from scipy import sparse
from tensordict import TensorDict

from isaaclab_ipc.mpc import Config
from isaaclab_ipc.planner import InvalidCorridorError, Planner, flatness
from isaaclab_ipc.rl.actor import MPCActor
from isaaclab_ipc.rl.qp import DifferentiableQP

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def context():
    planner = Planner(Config.load("avoid", 10), [10, 0, 1])
    state, reference, corridors = planner.context(np.array([0, 0, 1, 0.1, 0, 0, 0, 0, 0.0]))
    planes = np.zeros((15, 32, 4))
    mask = np.zeros((15, 32), bool)
    for i, faces in enumerate(corridors):
        planes[i, : len(faces)] = faces[::-1]
        mask[i, : len(faces)] = True
    data = dict(
        policy=np.zeros(124, np.float32),
        mpc_state=state,
        reference=reference,
        planes=planes,
        plane_mask=mask,
        rotation=np.eye(3).reshape(-1),
        braking=np.array([False]),
        yaw_target=np.array([0.0]),
        sfc_valid=np.array([True]),
    )
    obs = TensorDict({k: torch.as_tensor(v, device=DEVICE).unsqueeze(0) for k, v in data.items()}, [1])
    return planner, obs, (state, reference, corridors)


def test_qp_forward_and_cost_gradient_with_active_redundant_planes():
    def tensor(x):
        return torch.tensor(x, dtype=torch.float64, device=DEVICE)

    h = tensor([[[2.0, 0.0], [0.0, 3.0]]]).requires_grad_()
    g = tensor([[-4.0, -3.0]]).requires_grad_()
    a = tensor([[[1.0, 1.0], [2.0, 2.0], [1.0, 0.0], [0.0, 1.0]]])
    lo = tensor([[-float("inf"), -float("inf"), -2, -2]])
    hi = tensor([[1.0, 2.0, 2.0, 2.0]])
    solver = DifferentiableQP(eps_abs=1e-9, eps_rel=1e-9)
    x, solved, *_ = solver(h, g, a, lo, hi)
    assert solved.all()
    torch.testing.assert_close(x, tensor([[0.8, 0.2]]), atol=1e-7, rtol=0)
    assert torch.autograd.gradcheck(
        lambda hh, gg: solver(hh, gg, a, lo, hi)[0],
        (h, g),
        eps=1e-4,
        atol=2e-4,
        rtol=2e-3,
        check_undefined_grad=False,
        nondet_tol=1e-6,
        fast_mode=True,
    )
    # Contradictory rows exercise failure without passing inverted individual bounds.
    bad_a = tensor([[[1.0, 0.0], [1.0, 0.0]]])
    bad_lo, bad_hi = tensor([[2.0, -float("inf")]]), tensor([[float("inf"), 1.0]])
    x, good, *_ = solver(h, g, bad_a, bad_lo, bad_hi)
    assert not good.any()
    grad = torch.autograd.grad(x.sum(), g)[0]
    assert not grad.any()


def test_ipc_assembly_control_and_cost_network_gradient():
    planner, obs, inputs = context()
    actor = MPCActor(obs, {"actor": ["policy"]}, "actor", 4).to(DEVICE)
    problem = actor.problem(obs, torch.zeros((1, 7), device=DEVICE))
    original = planner.mpc.problem(*inputs)
    np.testing.assert_allclose(problem[0][0].cpu(), original[0], atol=1e-12)
    np.testing.assert_allclose(problem[1][0].cpu(), original[1], atol=1e-12)
    active_rows = torch.cat((torch.ones(135, dtype=torch.bool, device=DEVICE), obs["plane_mask"].flatten()))
    for value, expected in zip(problem[2:5], original[2:5], strict=True):
        np.testing.assert_allclose(value[0, active_rows].cpu(), expected, atol=1e-12)
    command = actor(obs)
    assert actor.last_result["solved"].all()
    oracle = osqp.OSQP()
    oracle.setup(
        P=sparse.csc_matrix(np.triu(original[0])),
        q=original[1],
        A=sparse.csc_matrix(original[2]),
        l=original[3],
        u=original[4],
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=20000,
        verbose=False,
        polishing=True,
    )
    result = oracle.solve(raise_error=False)
    assert result.info.status == "solved"
    np.testing.assert_allclose(actor.last_result["solution"][0].cpu(), result.x, atol=2e-4, rtol=2e-4)
    x, j, _ = planner.consume(original, actor.last_result["solution"][0].cpu().numpy(), "solved", 0)
    expected = flatness(x[6:], j, np.eye(3))
    expected[0] = np.clip(expected[0], 0.08 * 9.81 / 0.7, 0.9 * 9.81 / 0.7)
    np.testing.assert_allclose(command[0].detach().cpu(), expected, atol=2e-6)
    command[:, 2].sum().backward()
    assert torch.isfinite(actor.mlp[-1].bias.grad).all()
    assert actor.mlp[-1].bias.grad.abs().max() > 1e-6
    # A shuffled minibatch must evaluate the saved context, independently of replay buffers.
    saved = obs.clone()
    with torch.inference_mode():
        before = actor(obs).clone()
    obs["planes"].add_(0.01)
    with torch.no_grad():
        after = actor(saved)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=0)


def test_braking_is_a_deterministic_atom_with_no_cost_gradient():
    _, obs, _ = context()
    obs["braking"][:] = True
    obs["mpc_state"][:, 3] = 1.0
    actor = MPCActor(obs, {"actor": ["policy"]}, "actor", 4).to(DEVICE)
    action = actor(obs, stochastic_output=True)
    assert actor.last_result["braking"].all()
    torch.testing.assert_close(action, actor._backup)
    assert actor.get_output_log_prob(action).item() == 0
    assert actor.get_output_log_prob(action + 0.1).item() == -1e9
    actor.output_mean.sum().backward()
    assert not actor.mlp[-1].bias.grad.any()


def test_invalid_corridor_is_rejected_and_only_valid_batch_members_solve(monkeypatch):
    planner, obs, _ = context()
    monkeypatch.setattr("isaaclab_ipc.planner._geometry.corridor", lambda *args: np.full((1, 4), np.nan))
    # A new observed map invalidates the previously valid cached polytope.
    planner.update_cloud([], obs["mpc_state"][0, :3].cpu().numpy())
    with pytest.raises(InvalidCorridorError):
        planner.context(obs["mpc_state"][0].cpu().numpy())
    obs = torch.cat((obs, obs.clone()), dim=0)
    obs["sfc_valid"][1] = False
    obs["mpc_state"][1, 3] = 2.0
    actor = MPCActor(obs, {"actor": ["policy"]}, "actor", 4).to(DEVICE)
    original = actor.solver

    def solve_valid_only(*args):
        assert args[0].shape[0] == 1
        return original(*args)

    actor.solver = solve_valid_only
    actions = actor(obs, stochastic_output=True)
    assert actor.last_result["solved"].tolist() == [True, False]
    assert actor.last_result["braking"].tolist() == [False, True]
    torch.testing.assert_close(actions[1], actor._backup[1])
    actor.output_mean[0, 2].backward()
    assert actor.mlp[-1].bias.grad.abs().max() > 1e-6
    actor(obs[1:])  # Entirely invalid batches must skip the solver too.


def test_sfc_capacity_expansion_preserves_constraints_and_action():
    _, obs, _ = context()
    actor = MPCActor(obs, {"actor": ["policy"]}, "actor", 4).to(DEVICE)
    expanded = obs.clone()
    expanded["planes"] = torch.nn.functional.pad(obs["planes"], (0, 0, 0, 32))
    expanded["plane_mask"] = torch.nn.functional.pad(obs["plane_mask"], (0, 32))
    costs = torch.zeros((1, 7), device=DEVICE)
    original = actor.problem(obs, costs)
    larger = actor.problem(expanded, costs)
    for values, data in ((original, obs), (larger, expanded)):
        active = torch.cat((torch.ones(135, dtype=torch.bool, device=DEVICE), data["plane_mask"].flatten()))
        if values is original:
            reference = [v[0, active] for v in values[2:5]]
        else:
            for actual, expected in zip(values[2:5], reference, strict=True):
                torch.testing.assert_close(actual[0, active], expected, atol=0, rtol=0)
    with torch.no_grad():
        before = actor(obs)
        after = actor(expanded)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=0)


def test_polish_ignores_stale_duals_on_slack_constraints():
    from isaaclab_ipc.rl.qp import polish, residuals

    h = torch.eye(2, dtype=torch.float64, device=DEVICE)[None]
    g = torch.tensor([[-0.2, 0.0]], dtype=torch.float64, device=DEVICE)
    a = h.clone()
    lower = torch.full_like(g, -torch.inf)
    upper = torch.ones_like(g)
    stale_dual = torch.full_like(g, 0.001)
    x, dual, signs = polish(h, g, a, lower, upper, -g, stale_dual)
    primal, stationarity, _ = residuals(h, g, a, lower, upper, x, dual)
    assert signs.all() and (primal < 1e-12).all() and (stationarity < 1e-12).all()
    torch.testing.assert_close(x, -g)
