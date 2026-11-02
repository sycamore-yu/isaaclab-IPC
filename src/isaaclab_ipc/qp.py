"""Dense batched ADMM for IPC QPs; GPU iterations reuse a CUDA graph."""

import numpy as np
import torch


class BatchedQP:
    def __init__(
        self,
        batch,
        variables,
        constraints,
        device="cuda:0",
        dtype=torch.float64,
        max_iterations=4000,
        eps_abs=1e-3,
        eps_rel=1e-3,
    ):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.batch = batch
        self.device = device
        self.dtype = dtype
        self.max_iterations = max_iterations
        shape = (batch, constraints, variables)
        self.A = torch.zeros(shape, device=device, dtype=dtype)
        self.AT = torch.zeros((batch, variables, constraints), device=device, dtype=dtype)
        self.H = torch.zeros((batch, variables, variables), device=device, dtype=dtype)
        self.g = torch.zeros((batch, variables, 1), device=device, dtype=dtype)
        self.L = self.H.clone()
        self.inverse = self.H.clone()
        self.x = self.g.clone()
        self.z = torch.zeros((batch, constraints, 1), device=device, dtype=dtype)
        self.y = self.z.clone()
        self.lo = self.z.clone()
        self.hi = self.z.clone()
        self.graph = None
        self.rho = 0.1

    def _iterations(self):
        for _ in range(32):
            rhs = -self.g + torch.bmm(self.AT, self.rho * self.z - self.y)
            x = torch.bmm(self.inverse, rhs)
            ax = 1.6 * torch.bmm(self.A, x) - 0.6 * self.z
            z = torch.minimum(torch.maximum(ax + self.y / self.rho, self.lo), self.hi)
            self.y.add_(self.rho * (ax - z))
            self.z.copy_(z)
            self.x.copy_(x)

    def solve(self, problems):
        b, m, n = self.A.shape
        if len(problems) != b:
            raise ValueError("QP batch size mismatch")
        h = np.stack([p[0] for p in problems])
        g = np.stack([p[1] for p in problems])
        a = np.zeros((b, m, n))
        lo = np.full((b, m), -np.inf)
        hi = np.full((b, m), np.inf)
        for i, p in enumerate(problems):
            rows = len(p[3])
            if rows > m:
                raise ValueError(f"QP has {rows} constraints, capacity is {m}")
            a[i, :rows] = p[2]
            lo[i, :rows] = p[3]
            hi[i, :rows] = p[4]
        scale = 1 / np.sqrt(np.diagonal(h, axis1=1, axis2=2))
        scaled_a = a * scale[:, None, :]
        row_scale = 1 / np.maximum(np.linalg.norm(scaled_a, axis=-1), 1e-8)

        def tensor(x):
            return torch.as_tensor(x, device=self.device, dtype=self.dtype)

        self.A.copy_(tensor(scaled_a * row_scale[:, :, None]))
        self.AT.copy_(self.A.transpose(1, 2))
        self.H.copy_(tensor(h * scale[:, :, None] * scale[:, None, :]))
        self.g.copy_(tensor(g * scale).unsqueeze(-1))
        self.lo.copy_(tensor(lo * row_scale).unsqueeze(-1))
        self.hi.copy_(tensor(hi * row_scale).unsqueeze(-1))
        self.L.copy_(torch.linalg.cholesky(self.H + self.rho * torch.bmm(self.AT, self.A)))
        self.inverse.copy_(torch.cholesky_inverse(self.L))
        self.x.zero_()
        self.y.zero_()
        self.z.copy_(torch.minimum(torch.maximum(self.z.zero_(), self.lo), self.hi))
        if self.graph is None and str(self.device).startswith("cuda"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._iterations()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._iterations()
            self.x.zero_()
            self.y.zero_()
            self.z.zero_()
        scale_tensor = tensor(scale)
        row_scale_tensor = tensor(row_scale)
        for _iterations in range(32, self.max_iterations + 1, 32):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._iterations()
            solution = self.x.squeeze(-1) * scale_tensor
            # Check in original physical units, not scaled constraint units.
            hx_scaled = torch.bmm(self.H, self.x)
            aty_scaled = torch.bmm(self.AT, self.y)
            ax_scaled = torch.bmm(self.A, self.x)
            dual = (hx_scaled + self.g + aty_scaled).squeeze(-1) / scale_tensor
            primal = torch.maximum(ax_scaled - self.hi, self.lo - ax_scaled).clamp_min(0).squeeze(-1) / row_scale_tensor
            primal_norm = primal.amax(-1)
            dual_norm = dual.abs().amax(-1)
            hx = hx_scaled.squeeze(-1) / scale_tensor
            gradient = self.g.squeeze(-1) / scale_tensor
            aty = aty_scaled.squeeze(-1) / scale_tensor
            dual_scale = torch.stack((hx.abs().amax(-1), gradient.abs().amax(-1), aty.abs().amax(-1))).amax(0)
            ax_original = ax_scaled.squeeze(-1) / row_scale_tensor
            z_original = self.z.squeeze(-1) / row_scale_tensor
            primal_scale = torch.maximum(ax_original.abs().amax(-1), z_original.abs().amax(-1))
            good = (primal_norm <= self.eps_abs + self.eps_rel * primal_scale) & (
                dual_norm <= self.eps_abs + self.eps_rel * dual_scale
            )
            if bool(good.all()):
                break
        values = solution.cpu().numpy()
        ok = good.cpu().numpy()
        return (
            values,
            ["solved" if x else "max iterations reached" for x in ok],
            primal_norm.cpu().numpy(),
            dual_norm.cpu().numpy(),
            _iterations,
        )
