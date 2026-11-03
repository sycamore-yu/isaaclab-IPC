"""Fixed-pattern sparse KKT assembly for independent batched problems.

The block layout follows TurboMPC's backward_kkt_csr.py (MIT, pinned revision
in native/turbompc/upstream/provenance.json). Inactive inequality multiplier
rows get an identity diagonal. This removes vacuous zero equations without
regularizing the primal system or changing its solution sensitivity.
"""

import numpy as np
import torch

from .native import SparseKKTSolver


class SparseAdjoint:
    """Gather CSR values on device and call the reused cuDSS KKT kernel."""

    def __init__(self, batch, stages, n, nx, m, device, *, compile_kernels=False):
        self.batch, self.stages, self.n, self.nx, self.m = batch, stages, n, nx, m
        self.primal = stages * n
        self.equalities = stages * nx
        self.size = self.primal + self.equalities + stages * m
        shapes = [
            (stages, n, n),
            (stages - 1, n, n),
            (nx, n),
            (stages - 1, nx, n),
            (stages - 1, nx, n),
            (stages, m, n),
            (stages, m),
        ]
        offsets = np.cumsum([0, *[int(np.prod(s)) for s in shapes]])
        indexes = [np.arange(offsets[i], offsets[i + 1]).reshape(shape) for i, shape in enumerate(shapes)]
        di, ei, ai, ami, api, gi, ii = indexes
        rows, columns, sources = [], [], []

        def block(row, column, index):
            rr, cc = np.indices(index.shape)
            rows.append((rr + row).ravel())
            columns.append((cc + column).ravel())
            sources.append(index.ravel())

        for k in range(stages):
            block(k * n, k * n, di[k])
            if k:
                block(k * n, (k - 1) * n, ei[k - 1])
                block((k - 1) * n, k * n, ei[k - 1].T)
            if m:
                dual = self.primal + self.equalities + k * m
                block(dual, k * n, gi[k])
                block(k * n, dual, gi[k].T)
                rows.append(dual + np.arange(m))
                columns.append(dual + np.arange(m))
                sources.append(ii[k])
        block(self.primal, 0, ai)
        block(0, self.primal, ai.T)
        for k in range(stages - 1):
            dual = self.primal + nx + k * nx
            block(dual, k * n, ami[k])
            block(k * n, dual, ami[k].T)
            block(dual, (k + 1) * n, api[k])
            block((k + 1) * n, dual, api[k].T)
        row, column, source = np.concatenate(rows), np.concatenate(columns), np.concatenate(sources)
        order = np.lexsort((column, row))
        row, column, source = row[order], column[order], source[order]
        counts = np.bincount(row, minlength=self.size)
        ptr = np.concatenate(([0], np.cumsum(np.tile(counts, batch)))).astype(np.int32)
        columns = (column[None] + np.arange(batch)[:, None] * self.size).ravel().astype(np.int32)
        self.index = torch.as_tensor(source, device=device, dtype=torch.int64)
        self.solver = SparseKKTSolver(torch.from_numpy(ptr), torch.from_numpy(columns), device=device)
        if compile_kernels:
            self._assemble = torch.compile(self._assemble, fullgraph=True, dynamic=False)

    def _assemble(self, qp, hessian, active, upstream):
        b = self.batch
        active_g = qp.G * active[..., None]
        parts = [hessian, qp.E, qp.A0, qp.A_minus, qp.A_plus, active_g, (~active).to(qp.D.dtype)]
        storage = torch.cat([x.reshape(b, -1) for x in parts], dim=-1)
        values = storage.index_select(1, self.index).reshape(-1)
        rhs = torch.cat((upstream.reshape(b, -1), upstream.new_zeros((b, self.size - self.primal))), -1)
        return values, rhs.reshape(-1)

    def solve(self, qp, hessian, active, upstream):
        values, rhs = self._assemble(qp, hessian, active, upstream)
        solution = self.solver.solve(values, rhs).reshape(self.batch, self.size)
        primal = solution[:, : self.primal].reshape(self.batch, self.stages, self.n)
        dual = solution[:, self.primal :]
        return primal, dual[:, : self.equalities], dual[:, self.equalities :].reshape(self.batch, self.stages, self.m)

    def solve_full(self, qp, hessian, active, primal_rhs, equality_rhs, inequality_rhs):
        """Solve the same KKT matrix with explicit residuals in every block.

        This is used by numerical solution refinement. The caller supplies the
        signs of every residual, including inactive multiplier identity rows.
        The adjoint-only ``solve`` path above remains unchanged.
        """
        if primal_rhs.shape != (self.batch, self.stages, self.n):
            raise ValueError("KKT primal RHS shape differs from the fixed pattern")
        if equality_rhs.shape != (self.batch, self.equalities):
            raise ValueError("KKT equality RHS must contain initial then dynamics residuals")
        if inequality_rhs.shape != (self.batch, self.stages, self.m):
            raise ValueError("KKT inequality RHS shape differs from the fixed pattern")
        values, _ = self._assemble(qp, hessian, active, primal_rhs)
        rhs = torch.cat(
            (primal_rhs.reshape(self.batch, -1), equality_rhs, inequality_rhs.reshape(self.batch, -1)), -1
        ).reshape(-1)
        solution = self.solver.solve(values, rhs).reshape(self.batch, self.size)
        primal = solution[:, : self.primal].reshape(self.batch, self.stages, self.n)
        dual = solution[:, self.primal :]
        return primal, dual[:, : self.equalities], dual[:, self.equalities :].reshape(self.batch, self.stages, self.m)

    def close(self):
        self.solver.close()
