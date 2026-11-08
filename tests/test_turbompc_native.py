"""Behavioral checks for the reused TurboMPC CUDA kernels."""

import importlib
import importlib.util
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest
import torch


def api():
    assert importlib.util.find_spec("isaaclab_ipc.turbompc") is not None, "Native TurboMPC package is missing"
    return importlib.import_module("isaaclab_ipc.turbompc")


def tracking_qp(module, device="cuda:0"):
    """Two integrators whose optimal first two inputs saturate at 0.25."""
    kw = dict(device=device, dtype=torch.float64)
    d = torch.diag(torch.tensor([1.0, 0.1], **kw)).repeat(2, 3, 1, 1)
    q = torch.tensor([-1.0, 0.0], **kw).repeat(2, 3, 1)
    am = torch.tensor([-1.0, -1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 2, 1, 1)
    ap = torch.tensor([1.0, 0.0], **kw).reshape(1, 1, 1, 2).repeat(2, 2, 1, 1)
    return module.QPData(
        D=d,
        E=torch.zeros((2, 2, 2, 2), **kw),
        q=q,
        A0=ap[:, 0].clone(),
        A_minus=am,
        A_plus=ap,
        c0=torch.tensor([[0.0], [0.2]], **kw),
        c_dyn=torch.zeros((2, 2, 1), **kw),
        G=torch.tensor([0.0, 1.0], **kw).reshape(1, 1, 1, 2).repeat(2, 3, 1, 1),
        lower=torch.full((2, 3, 1), -0.25, **kw),
        upper=torch.full((2, 3, 1), 0.25, **kw),
    )


def raw_arguments(module, qp, config):
    """Cold-start raw binding inputs, used to exercise one workspace's mode key."""
    batch, nodes, width = qp.q.shape
    rho = qp.q.new_full((batch,), config.rho)
    return [
        value.contiguous()
        for value in (
            module.native.schur_matrix(qp, rho, config),
            qp.D, qp.E, qp.q, qp.A0, qp.A_minus, qp.A_plus, qp.G,
            qp.lower, qp.upper, qp.c0, qp.c_dyn,
            qp.q.new_zeros((batch, nodes, width)),
            torch.zeros_like(qp.lower).clamp(qp.lower, qp.upper),
            torch.zeros_like(qp.lower), torch.zeros_like(qp.c0), torch.zeros_like(qp.c_dyn),
            torch.zeros_like(qp.lower), rho, qp.q.new_full((batch,), config.slack_weight),
        )
    ]


def test_admm_linear_solver_names_are_explicit():
    module = api()
    assert module.ADMMConfig().linear_solver == "cudss_vector"
    for name in ("cudss_vector", "cudss_inverse_chunk"):
        assert module.ADMMConfig(linear_solver=name).linear_solver == name
    with pytest.raises(ValueError, match="Unknown ADMM linear solver"):
        module.ADMMConfig(linear_solver="automatic")


def test_legacy_extension_cannot_silently_enable_inverse(monkeypatch):
    module = api()
    from isaaclab_ipc.turbompc import native

    extension = native._extension()
    assert tuple(extension.supported_admm_linear_solvers) == ("cudss_vector", "cudss_inverse_chunk")
    monkeypatch.setattr(native, "_extension", lambda: SimpleNamespace(AdmmWorkspace=extension.AdmmWorkspace))
    qp = tracking_qp(module)
    vector = module.NativeADMM(module.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9))
    inverse = module.NativeADMM(module.ADMMConfig(linear_solver="cudss_inverse_chunk"))
    try:
        assert (vector.solve(qp).status == 0).all()
        with pytest.raises(RuntimeError, match="does not support cudss_inverse_chunk"):
            inverse.solve(qp)
        assert not inverse._workspaces
    finally:
        vector.close()
        inverse.close()


def test_raw_workspace_switches_mode_and_retains_independent_outputs():
    module = api()
    qp = tracking_qp(module)
    config = module.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9)
    arguments = raw_arguments(module, qp, config)
    workspace = module.native._extension().AdmmWorkspace(0)
    retained = []
    try:
        with pytest.raises(RuntimeError, match="Unknown ADMM linear solver"):
            workspace.solve(arguments, {**asdict(config), "linear_solver": "automatic"})
        for mode in (None, "cudss_inverse_chunk", "cudss_vector", "cudss_inverse_chunk"):
            options = asdict(config)
            if mode is None:
                del options["linear_solver"]  # Historical raw callers use the vector default.
            else:
                options["linear_solver"] = mode
            result = module.ADMMResult(*workspace.solve(arguments, options))
            assert (result.status == 0).all(), result.residuals
            if retained:
                torch.testing.assert_close(result.x, retained[0][0].x, atol=2e-7, rtol=0)
            retained.append((result, {field.name: getattr(result, field.name).clone() for field in fields(result)}))
    finally:
        workspace.close()
    for result, saved in retained:
        for name, snapshot in saved.items():
            torch.testing.assert_close(getattr(result, name), snapshot, atol=0, rtol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(max_iter=71, check_termination_every=7, check_infeasibility_every=11, adapt_rho_every=13),
        dict(max_iter=31, check_termination_every=37, check_infeasibility_every=41, adapt_rho_every=13),
        dict(max_iter=46, check_termination_every=25, check_infeasibility_every=25, adapt_rho_every=25),
        dict(max_iter=71, check_termination_every=7, check_infeasibility_every=11, adapt_rho_every=13,
             use_slack=True, slack_weight=1.0),
        dict(max_iter=71, check_termination_every=7, check_infeasibility_every=11, adapt_rho_every=13,
             constraint_handling=1, log_barrier_kappa=1e-3),
    ],
    ids=["unequal-events", "rho-only-interior", "tail-21", "slack", "barrier"],
)
def test_inverse_chunk_event_boundaries_match_vector(overrides):
    module = api()
    qp = tracking_qp(module)
    qp.q[0].zero_()  # This row can finish while the second row still runs.
    results = []
    for mode in ("cudss_vector", "cudss_inverse_chunk"):
        solver = module.NativeADMM(module.ADMMConfig(
            linear_solver=mode, eps_abs=1e-9, eps_rel=1e-9, **overrides
        ))
        try:
            results.append(solver.solve(qp))
        finally:
            solver.close()
    expected, actual = results
    # A deliberately truncated QP may be MAX_ITER; retain the actual status.
    torch.testing.assert_close(actual.status, expected.status, atol=0, rtol=0)
    torch.testing.assert_close(actual.iterations, expected.iterations, atol=0, rtol=0)
    for field in fields(actual):
        if field.name not in ("status", "iterations"):
            torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("linear_solver", ["cudss_vector", "cudss_inverse_chunk"])
def test_native_admm_constrained_batch_and_saved_outputs(linear_solver):
    module = api()
    qp = tracking_qp(module)
    solver = module.NativeADMM(
        module.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9, max_iter=4096, linear_solver=linear_solver)
    )
    result = solver.solve(qp)
    expected = torch.tensor(
        [[[0.0, 0.25], [0.25, 0.25], [0.5, 0.0]], [[0.2, 0.25], [0.45, 0.25], [0.7, 0.0]]],
        device=qp.q.device,
        dtype=qp.q.dtype,
    )
    assert (result.status == 0).all()
    torch.testing.assert_close(result.x, expected, atol=2e-7, rtol=0)
    saved = result.x.clone()
    warm = solver.solve(qp, warm_start=result)
    assert (warm.status == 0).all()
    torch.testing.assert_close(warm.x, result.x, atol=2e-7, rtol=0)
    solver.solve(replace(qp, q=-qp.q))
    torch.testing.assert_close(result.x, saved, atol=0, rtol=0)
    solver.close()


@pytest.mark.parametrize("linear_solver", ["cudss_vector", "cudss_inverse_chunk"])
def test_native_admm_uses_the_callers_cuda_stream(linear_solver):
    module = api()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qp = tracking_qp(module)
        solver = module.NativeADMM(module.ADMMConfig(eps_abs=1e-8, eps_rel=1e-8, linear_solver=linear_solver))
        result = solver.solve(qp)
        consumed = result.x.square().sum()
    torch.cuda.current_stream().wait_stream(stream)
    assert (result.status == 0).all()
    torch.testing.assert_close(
        consumed, torch.tensor(1.295, device=consumed.device, dtype=consumed.dtype), atol=1e-6, rtol=0
    )
    solver.close()


def test_sparse_kkt_solves_an_indefinite_system_and_checks_shapes():
    module = api()
    row_ptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    col_idx = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    solver = module.SparseKKTSolver(row_ptr, col_idx, device="cuda:0", dtype=torch.float64)
    values = torch.tensor([2.0, 1.0, 1.0, 0.0], device="cuda", dtype=torch.float64)
    rhs = torch.tensor([1.0, 0.0], device="cuda", dtype=torch.float64)
    answer = solver.solve(values, rhs)
    torch.testing.assert_close(answer, torch.tensor([0.0, 1.0], device="cuda", dtype=torch.float64), atol=1e-12, rtol=0)
    with pytest.raises((ValueError, RuntimeError), match="shape|size"):
        solver.solve(values, rhs[:1])
    solver.close()
