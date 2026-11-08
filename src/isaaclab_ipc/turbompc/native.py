"""Structured QP solves on Torch-owned CUDA storage; no JAX runtime."""

import importlib.util
import os
from dataclasses import asdict, dataclass, fields
from functools import cache
from pathlib import Path
from threading import Lock

import torch


@cache
def _extension():
    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("The TurboMPC native extension requires a CUDA-enabled PyTorch build")
    major, minor = cuda_version.split(".")[:2]
    profile = f"cu{major}{minor}"
    default = Path(__file__).resolve().parents[3] / "build" / "turbompc" / "native" / profile
    build = Path(os.environ.get("TURBOMPC_NATIVE_BUILD_DIR", default)).resolve()
    libraries = list(build.glob("_turbompc_native*.so"))
    if not libraries:
        raise RuntimeError(
            f"No TurboMPC native extension for PyTorch CUDA {cuda_version} ({profile}) in {build}. "
            "Run scripts/build_turbompc.py with the matching PyTorch/CUDA environment first."
        )
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one TurboMPC native extension in {build}, found {len(libraries)}")
    spec = importlib.util.spec_from_file_location("_turbompc_native", libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class QPData:
    """Batched stage coordinates, with lower cost off-diagonal blocks E."""

    D: torch.Tensor
    E: torch.Tensor
    q: torch.Tensor
    A0: torch.Tensor
    A_minus: torch.Tensor
    A_plus: torch.Tensor
    c0: torch.Tensor
    c_dyn: torch.Tensor
    G: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def contiguous(self):
        return QPData(**{f.name: getattr(self, f.name).contiguous() for f in fields(self)})


@dataclass(frozen=True)
class ADMMConfig:
    """The upstream projection ADMM settings, with an explicit iteration budget."""

    max_iter: int = 4096
    check_termination_every: int = 25
    check_infeasibility_every: int = 25
    adapt_rho_every: int = 25
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    eps_prim_inf: float = 1e-4
    eps_dual_inf: float = 1e-4
    rho: float = 0.1
    sigma: float = 1e-6
    rho_f_factor: float = 1000.0
    alpha: float = 1.6
    adaptive_rho_tolerance: float = 5.0
    rho_update_max_factor: float = 1e6
    rho_min: float = 1e-6
    rho_max: float = 1e6
    constraint_handling: int = 0
    log_barrier_kappa: float = 0.0
    use_slack: bool = False
    slack_weight: float = 0.0
    linear_solver: str = "cudss_vector"

    def __post_init__(self):
        if self.linear_solver not in ("cudss_vector", "cudss_inverse_chunk"):
            raise ValueError(f"Unknown ADMM linear solver: {self.linear_solver!r}")


@dataclass(frozen=True)
class ADMMResult:
    """Independent output tensors, including all upstream warm-start buffers."""

    iterations: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    lambda0: torch.Tensor
    lambda_dyn: torch.Tensor
    xi: torch.Tensor
    rho: torch.Tensor
    kernel_ns: torch.Tensor
    status: torch.Tensor
    residuals: torch.Tensor


def schur_matrix(qp: QPData, rho: torch.Tensor, config: ADMMConfig) -> torch.Tensor:
    """Build P + sigma I + rho_f C.T C + rho G.T G in banded blocks."""
    b, t, n, _ = qp.D.shape
    rf = (rho * config.rho_f_factor)[:, None, None, None]
    diagonal = qp.D + config.sigma * torch.eye(n, dtype=qp.D.dtype, device=qp.D.device)
    diagonal = diagonal + rho[:, None, None, None] * (qp.G.transpose(-1, -2) @ qp.G)
    diagonal[:, 0] += rf[:, 0] * (qp.A0.transpose(-1, -2) @ qp.A0)
    diagonal[:, :-1] += rf * (qp.A_minus.transpose(-1, -2) @ qp.A_minus)
    diagonal[:, 1:] += rf * (qp.A_plus.transpose(-1, -2) @ qp.A_plus)
    lower = qp.E + rf * (qp.A_plus.transpose(-1, -2) @ qp.A_minus)
    matrix = qp.D.new_zeros((b, t, n, 3 * n))
    matrix[..., n : 2 * n] = diagonal
    matrix[:, 1:, :, :n] = lower
    matrix[:, :-1, :, 2 * n :] = lower.transpose(-1, -2)
    return matrix


class NativeADMM:
    """Reuse TurboMPC's cuDSS ADMM core on the caller's CUDA stream.

    A solver owns a device-local workspace for each CUDA stream. Factorization
    storage stays on its initializing stream. Results never alias that storage.
    This numerical forward interface does not itself register a derivative.
    """

    def __init__(self, config: ADMMConfig | None = None):
        self.config = config or ADMMConfig()
        self._config = asdict(self.config)
        self._workspaces = {}
        self._device = None
        self._closed = False
        self._lifecycle_lock = Lock()
        self._schur = schur_matrix

    @torch.no_grad()
    def solve(self, qp: QPData, warm_start: ADMMResult | None = None) -> ADMMResult:
        # Tensor preparation may release the GIL. Keep close() from clearing
        # the cache and allowing this call to recreate a workspace afterward.
        with self._lifecycle_lock:
            return self._solve(qp, warm_start)

    def _solve(self, qp: QPData, warm_start: ADMMResult | None) -> ADMMResult:
        if self._closed:
            raise RuntimeError("ADMM solver is closed")
        if qp.q.dtype != torch.float64 or not qp.q.is_cuda:
            raise ValueError("NativeADMM requires float64 CUDA QP tensors")
        qp = qp.contiguous()
        b, t, n = qp.q.shape
        m, nx, n0 = qp.G.shape[-2], qp.A_minus.shape[-2], qp.A0.shape[-2]
        if self._device is None:
            self._device = qp.q.device
        elif qp.q.device != self._device:
            raise ValueError("Use one NativeADMM instance per CUDA device")
        stream = torch.cuda.current_stream(self._device).cuda_stream
        if stream not in self._workspaces:
            extension = _extension()
            if self.config.linear_solver == "cudss_inverse_chunk" and self.config.linear_solver not in getattr(
                extension, "supported_admm_linear_solvers", ()
            ):
                raise RuntimeError("The loaded TurboMPC extension does not support cudss_inverse_chunk; rebuild it")
            self._workspaces[stream] = extension.AdmmWorkspace(self._device.index)
        workspace = self._workspaces[stream]
        if warm_start is None:
            warm = [
                qp.q.new_zeros((b, t, n)),
                qp.lower.new_zeros((b, t, m)).clamp(qp.lower, qp.upper),
                qp.q.new_zeros((b, t, m)),
                qp.q.new_zeros((b, n0)),
                qp.q.new_zeros((b, t - 1, nx)),
                qp.q.new_zeros((b, t, m)),
                qp.q.new_full((b,), self.config.rho),
            ]
        else:
            warm = [
                warm_start.x,
                warm_start.z,
                warm_start.y,
                warm_start.lambda0,
                warm_start.lambda_dyn,
                warm_start.xi,
                warm_start.rho,
            ]
        matrix = self._schur(qp, warm[-1], self.config)
        args = [
            matrix,
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
            *warm,
            qp.q.new_full((b,), self.config.slack_weight),
        ]
        output = workspace.solve([x.contiguous() for x in args], self._config)
        return ADMMResult(*output)

    def close(self):
        with self._lifecycle_lock:
            self._closed = True
            for workspace in self._workspaces.values():
                workspace.close()
            self._workspaces.clear()


class SparseKKTSolver:
    """Reusable symmetric sparse cuDSS solve for an immutable CSR pattern."""

    def __init__(self, row_ptr, col_idx, *, device, dtype=torch.float64):
        if dtype != torch.float64:
            raise ValueError("SparseKKTSolver currently requires float64")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SparseKKTSolver requires a CUDA device")
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self._workspace = _extension().KktWorkspace(row_ptr, col_idx, index)

    def solve(self, values, rhs):
        return self._workspace.solve(values, rhs)

    def close(self):
        self._workspace.close()
