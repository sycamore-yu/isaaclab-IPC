# Local differences from pinned TurboMPC

`provenance.json` records the SHA256 of each original upstream file, not the
patched contents. The original repository and commit are preserved there.

## Sparse KKT iterative refinement

Current default: `backward/cudss_sparse_kkt.cu` explicitly selects cuDSS
`IR_N_STEPS=2`. This corrects the original KKT residual twice inside the native
solver using the existing factorization. No ADMM update, QP objective,
constraint, or primal regularization is changed. The former IR0 decision is
superseded by the gradient-accuracy gate for native AC-MPC (ADR-0008).

Evidence: a 40-by-40 block-diagonal KKT system, condition number 33.87396,
had exactly matching independent dense and assembled CSR entries. With the
unmodified cuDSS configuration its infinity-norm residual was 3.3800e-4;
the independent float64 dense solve had residual 1.9429e-16. The resulting
cost derivative failed finite differences. Reproducer is retained at
`results/turbompc-port/kkt-accuracy-reproducer.npz`.

On the 2026-11-07 AC-MPC cost-node fixture, IR0 reproduced a 4.46% worst-row
batch-gradient discrepancy despite matching primal trajectories. The generic
curved-constraint dense-KKT comparison also failed its gradient tolerance.
Results recorded before the new IR2 build remain IR0 evidence. Rebuild the
native extension before measuring the corrected path; build manifests record
the source hashes and binary used. Do not describe this adapted Torch
memory/stream binding file as byte-identical to upstream.
