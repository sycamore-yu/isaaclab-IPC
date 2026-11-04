# Local differences from pinned TurboMPC

`provenance.json` records the SHA256 of each original upstream file, not the
patched contents. The original repository and commit are preserved there.

## Sparse KKT iterative refinement

Current default: `backward/cudss_sparse_kkt.cu` leaves cuDSS iterative
refinement at its upstream/default value (IR_N_STEPS=0). The experimental
IR_N_STEPS=2 correction has been removed from the production source following
the decision to accept the original gradient error. No ADMM update, QP
objective, constraint, or primal regularization is changed.

Evidence: a 40-by-40 block-diagonal KKT system, condition number 33.87396,
had exactly matching independent dense and assembled CSR entries. With the
unmodified cuDSS configuration its infinity-norm residual was 3.3800e-4;
the independent float64 dense solve had residual 1.9429e-16. The resulting
cost derivative failed finite differences. Reproducer is retained at
`results/turbompc-port/kkt-accuracy-reproducer.npz`.

The historical IR2 benchmarks remain valid only for their recorded IR2 binaries;
they are not measurements of this current default. A default-IR binary must be
rebuilt before use. Do not describe this adapted Torch memory/stream binding
file as byte-identical to upstream.
