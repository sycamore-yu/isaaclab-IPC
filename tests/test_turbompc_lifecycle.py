"""Native solve/close races, isolated so a GIL deadlock has a bounded timeout."""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from isaaclab_ipc.turbompc import native


def _admm_case():
    kw = dict(device="cuda:0", dtype=torch.float64)
    config = native.ADMMConfig(eps_abs=1e-9, eps_rel=1e-9, max_iter=4096)
    plus = torch.tensor([1.0, 0.0], **kw).reshape(1, 1, 1, 2).repeat(1, 2, 1, 1)
    qp = native.QPData(
        D=torch.diag(torch.tensor([1.0, 0.1], **kw)).repeat(1, 3, 1, 1),
        E=torch.zeros((1, 2, 2, 2), **kw),
        q=torch.tensor([-1.0, 0.0], **kw).repeat(1, 3, 1),
        A0=plus[:, 0].clone(),
        A_minus=torch.tensor([-1.0, -1.0], **kw).reshape(1, 1, 1, 2).repeat(1, 2, 1, 1),
        A_plus=plus,
        c0=torch.zeros((1, 1), **kw),
        c_dyn=torch.zeros((1, 2, 1), **kw),
        G=torch.tensor([0.0, 1.0], **kw).reshape(1, 1, 1, 2).repeat(1, 3, 1, 1),
        lower=torch.full((1, 3, 1), -0.25, **kw),
        upper=torch.full((1, 3, 1), 0.25, **kw),
    )
    rho = torch.full((1,), config.rho, **kw)
    args = [
        native.schur_matrix(qp, rho, config),
        qp.D,
        qp.E,
        qp.q,
        qp.A0,
        qp.A_minus,
        qp.A_plus,
        qp.G,
        qp.lower,
        qp.upper,
        qp.c0,
        qp.c_dyn,
        torch.zeros_like(qp.q),
        torch.zeros_like(qp.lower),
        torch.zeros_like(qp.lower),
        torch.zeros_like(qp.c0),
        torch.zeros_like(qp.c_dyn),
        torch.zeros_like(qp.lower),
        rho,
        torch.zeros((1,), **kw),
    ]
    workspace = native._extension().AdmmWorkspace(0)
    expected = torch.tensor([[[0.0, 0.25], [0.25, 0.25], [0.5, 0.0]]], **kw)

    def check(output):
        assert (output[9] == 0).all(), output[9]
        torch.testing.assert_close(output[1], expected, atol=2e-7, rtol=0)

    return workspace, ([item.contiguous() for item in args], asdict(config)), check


def _kkt_case():
    workspace = native._extension().KktWorkspace(
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        0,
    )
    kw = dict(device="cuda:0", dtype=torch.float64)
    values = torch.tensor([2.0, 1.0, 1.0, 0.0], **kw)
    rhs = torch.tensor([1.0, 0.0], **kw)
    expected = torch.tensor([0.0, 1.0], **kw)

    def check(output):
        torch.testing.assert_close(output, expected, atol=1e-10, rtol=0)

    return workspace, (values, rhs), check


def _run_worker(kind):
    factory = {"admm": _admm_case, "kkt": _kkt_case}[kind]
    completed = rejected = 0
    for _ in range(8):
        workspace, args, check = factory()
        stream = torch.cuda.Stream(device="cuda:0")
        stream.wait_stream(torch.cuda.current_stream("cuda:0"))
        # Exercise destruction of an initialized workspace, even when close
        # wins the following race. Keep every solve on this same CUDA stream.
        with torch.cuda.stream(stream):
            retained_output = workspace.solve(*args)
        gate = threading.Barrier(3)

        def solve(workspace=workspace, args=args, stream=stream, gate=gate):
            with torch.cuda.stream(stream):
                gate.wait()
                return workspace.solve(*args)

        def close(workspace=workspace, gate=gate):
            gate.wait()
            workspace.close()

        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                solving = pool.submit(solve)
                closing = [pool.submit(close), pool.submit(close)]
                for future in closing:
                    future.result()
                try:
                    output = solving.result()
                except RuntimeError as error:
                    # A close that wins must reject the solve at the binding
                    # boundary, rather than using an invalid registry handle.
                    assert "workspace is closed" in str(error), str(error)
                    rejected += 1
                else:
                    check(output)
                    completed += 1
            check(retained_output)
            with torch.cuda.stream(stream), pytest.raises(RuntimeError, match="workspace is closed"):
                workspace.solve(*args)
            workspace.close()
        finally:
            workspace.close()
    print(f"{kind}: {completed} concurrent solves completed, {rejected} rejected after close", flush=True)


@pytest.mark.parametrize("kind", ["admm", "kkt"])
def test_native_workspace_concurrent_solve_and_close(kind):
    # A thread timeout cannot recover from a GIL/lock inversion. A fresh child
    # bounds the failure and also ensures it loads the rebuilt extension.
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    paths = [str(root / "src")]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", kind],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is run by pytest or with --worker admm|kkt")
    _run_worker(sys.argv[2])
