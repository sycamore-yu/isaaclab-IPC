"""Batched CUDA ADMM forward and active-constraint implicit cost derivatives."""

import torch


def residuals(h, g, a, lower, upper, x, y):
    ax = (a @ x.unsqueeze(-1)).squeeze(-1)
    hx = (h @ x.unsqueeze(-1)).squeeze(-1)
    aty = (a.transpose(-1, -2) @ y.unsqueeze(-1)).squeeze(-1)
    primal = torch.maximum(ax - upper, lower - ax).clamp_min(0).amax(-1)
    dual = (hx + g + aty).abs().amax(-1)
    scale = torch.stack((hx.abs().amax(-1), g.abs().amax(-1), aty.abs().amax(-1))).amax(0)
    return primal, dual, scale


def constraint_basis(a, active):
    norm = a.norm(dim=-1).clamp_min(1e-12)
    rows = a / norm.unsqueeze(-1) * active.unsqueeze(-1)
    values, basis = torch.linalg.eigh(rows.transpose(-1, -2) @ rows)
    null = values <= 1e-9 * values.amax(-1, keepdim=True).clamp_min(1)
    projector = (basis * null.unsqueeze(-2)) @ basis.transpose(-1, -2)
    inverse = (basis * torch.where(null, 0.0, values.clamp_min(1e-12).reciprocal()).unsqueeze(-2)) @ basis.transpose(
        -1, -2
    )
    return rows, norm, projector, inverse


def polish(h, g, a, lower, upper, y):
    active = y.abs() > 1e-7
    rows, norm, projector, inverse = constraint_basis(a, active)
    bounds = torch.where(active, torch.where(y > 0, upper, lower), 0.0) / norm
    particular = inverse @ (rows.transpose(-1, -2) @ bounds.unsqueeze(-1))
    identity = torch.eye(h.shape[-1], device=h.device, dtype=h.dtype)
    system = projector @ h @ projector + identity - projector
    x = particular - projector @ torch.linalg.solve(system, projector @ (h @ particular + g.unsqueeze(-1)))
    dual = -(rows @ (inverse @ (h @ x + g.unsqueeze(-1)))).squeeze(-1) / norm
    signs_valid = ((dual * y) >= -1e-9).all(-1)
    return x.squeeze(-1), dual, signs_valid


class _CostGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, g, a, lower, upper, x, y, solved, active_tol):
        ctx.save_for_backward(h, a, lower, upper, x, y, solved)
        ctx.active_tol = active_tol
        return x.clone()

    @staticmethod
    def backward(ctx, upstream):
        h, a, lower, upper, x, y, solved = ctx.saved_tensors
        norm = a.norm(dim=-1).clamp_min(1e-12)
        ax = (a @ x.unsqueeze(-1)).squeeze(-1)
        slack = torch.minimum((ax - lower).abs(), (upper - ax).abs()) / norm
        # Zero-dual touching rows are nonsmooth; choose the inactive-side derivative.
        active = (slack < ctx.active_tol) & (y.abs() > 1e-7) & solved[:, None]
        _, _, projector, _ = constraint_basis(a, active)
        identity = torch.eye(h.shape[-1], device=h.device, dtype=h.dtype)
        system = projector @ h @ projector + identity - projector
        rhs = (projector @ upstream.unsqueeze(-1)) * solved[:, None, None]
        adjoint = projector @ torch.linalg.solve(system, rhs)
        gradient = adjoint @ x.unsqueeze(-2)
        return -0.5 * (gradient + gradient.transpose(-1, -2)), -adjoint.squeeze(-1), *([None] * 7)


class DifferentiableQP:
    """All solves stay on the input device. CUDA graphs cache 32 ADMM iterations.

    Costs must be positive definite; constraints are fixed inputs, with infinite bounds
    for absent sides. A failed solve returns solved=False and zero cost derivatives.
    """

    def __init__(self, max_iterations=4096, eps_abs=1e-5, eps_rel=1e-5):
        if max_iterations < 32 or max_iterations % 32:
            raise ValueError("max_iterations must be a positive multiple of 32")
        self.max_iterations = max_iterations
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self._work = {}

    @staticmethod
    def _iterate(w):
        inverse, at, g, lower, upper, x, z, y, a = w
        for _ in range(32):
            x.copy_(inverse @ (-g + at @ (0.1 * z - y)))
            ax = 1.6 * (a @ x) - 0.6 * z
            new_z = (ax + y / 0.1).clamp(lower, upper)
            y.add_(0.1 * (ax - new_z))
            z.copy_(new_z)

    def __call__(self, h, g, a, lower, upper):
        h = 0.5 * (h + h.transpose(-1, -2))
        if h.dtype != torch.float64:
            raise ValueError("IPC QP uses float64 for physical residuals and implicit derivatives")
        if (
            h.shape[:-1] != g.shape
            or a.shape[0] != g.shape[0]
            or lower.shape != a.shape[:2]
            or upper.shape != lower.shape
        ):
            raise ValueError("Expected H[B,n,n], g[B,n], A[B,m,n], lower/upper[B,m]")
        with torch.inference_mode(False), torch.no_grad():
            if not (torch.isfinite(h).all() & torch.isfinite(g).all() & torch.isfinite(a).all()):
                raise ValueError("Nonfinite QP cost or constraint matrix")
            if torch.isnan(lower).any() | torch.isnan(upper).any() | (lower > upper).any():
                raise ValueError("Invalid QP bounds")
            scale = h.diagonal(dim1=-2, dim2=-1).rsqrt()
            scaled_a = a * scale[:, None, :]
            row_scale = scaled_a.norm(dim=-1).clamp_min(1e-8).reciprocal()
            sa = scaled_a * row_scale.unsqueeze(-1)
            sh = h * scale.unsqueeze(-1) * scale.unsqueeze(-2)
            sg = (g * scale).unsqueeze(-1)
            slo = (lower * row_scale).unsqueeze(-1)
            shi = (upper * row_scale).unsqueeze(-1)
            factor = torch.linalg.cholesky(sh + 0.1 * (sa.transpose(-1, -2) @ sa))
            inverse = torch.cholesky_inverse(factor)
            key = (tuple(a.shape), str(a.device))
            if key not in self._work:
                w = [
                    inverse.clone(),
                    sa.transpose(-1, -2).contiguous(),
                    sg.clone(),
                    slo.clone(),
                    shi.clone(),
                    torch.zeros_like(sg),
                    torch.zeros_like(slo),
                    torch.zeros_like(slo),
                    sa.clone(),
                ]
                graph = None
                if a.is_cuda:
                    stream = torch.cuda.Stream(device=a.device)
                    stream.wait_stream(torch.cuda.current_stream(a.device))
                    with torch.cuda.stream(stream):
                        self._iterate(w)
                    torch.cuda.current_stream(a.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self._iterate(w)
                self._work[key] = (w, graph)
            w, graph = self._work[key]
            for dst, src in zip(w[:5], (inverse, sa.transpose(-1, -2), sg, slo, shi), strict=True):
                dst.copy_(src)
            w[8].copy_(sa)
            w[5].zero_()
            w[6].zero_().clamp_(slo, shi)
            w[7].zero_()
            for _iterations in range(32, self.max_iterations + 1, 32):
                if graph is None:
                    self._iterate(w)
                else:
                    graph.replay()
                x = w[5].squeeze(-1) * scale
                y = w[7].squeeze(-1) * row_scale
                primal, dual, dual_scale = residuals(h, g, a, lower, upper, x, y)
                solved = (primal <= self.eps_abs) & (dual <= self.eps_abs + self.eps_rel * dual_scale)
                if _iterations == 32 or _iterations % 256 == 0:
                    px, py, signs = polish(h, g, a, lower, upper, y)
                    pp, pd, ps = residuals(h, g, a, lower, upper, px, py)
                    good = signs & (pp <= self.eps_abs) & (pd <= self.eps_abs + self.eps_rel * ps)
                    x = torch.where(good[:, None], px, x)
                    y = torch.where(good[:, None], py, y)
                    primal = torch.where(good, pp, primal)
                    dual = torch.where(good, pd, dual)
                    solved |= good
                if solved.all():
                    break
            # Saved tensors must not alias the replay buffers or inference tensors.
            x, y, solved = x.clone(), y.clone(), solved.clone()
        result = _CostGradient.apply(h, g, a, lower, upper, x, y, solved, 5e-4)
        return result, solved, primal, dual, _iterations
