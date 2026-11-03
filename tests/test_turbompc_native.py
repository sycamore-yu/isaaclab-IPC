"""Behavioral checks for the reused TurboMPC CUDA kernels."""

import importlib
import importlib.util
from dataclasses import replace

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


def test_native_admm_constrained_batch_and_saved_outputs():
    module = api()
    qp = tracking_qp(module)
    solver = module.NativeADMM(module.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9, max_iter=4096))
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


def test_native_admm_uses_the_callers_cuda_stream():
    module = api()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qp = tracking_qp(module)
        solver = module.NativeADMM(module.ADMMConfig(eps_abs=1e-8, eps_rel=1e-8))
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
