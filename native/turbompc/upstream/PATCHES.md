# Local differences from pinned TurboMPC

`provenance.json` records the SHA256 of each original upstream file, not the
patched contents. The original repository and commit are preserved there.

## Sparse KKT iterative refinement

`backward/cudss_sparse_kkt.cu`: enable two cuDSS iterative-refinement steps.
No ADMM update, QP objective, constraint, or primal regularization is changed.

Evidence: a 40-by-40 block-diagonal KKT system, condition number 33.87396,
had exactly matching independent dense and assembled CSR entries. With the
unmodified cuDSS configuration its infinity-norm residual was 3.3800e-4;
the independent float64 dense solve had residual 1.9429e-16. The resulting
cost derivative failed finite differences. Reproducer is retained at
`results/turbompc-port/kkt-accuracy-reproducer.npz`.

This numerical correction must be disclosed in comparisons with the unchanged
JAX baseline. Do not describe this file as byte-identical to upstream.
