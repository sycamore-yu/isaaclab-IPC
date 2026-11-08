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

## Explicit ADMM linear-solve backend

The public default remains `cudss_vector`: one GENERAL/FULL cuDSS vector
solve per ADMM iteration, with the original kernels, stopping tolerances,
certificate checks, rho adaptation and host synchronization order.

`cudss_inverse_chunk` is an explicit float64 option. For each QP factorization,
cuDSS solves a column-major stacked identity with `TN` right-hand sides. Each
independent batch matrix contributes a `TN`-by-`TN` inverse block. The same
descriptors and workspace-owned buffers live from analysis to destruction.
Every rho refactorization refreshes all inverse blocks before further ADMM
iterations. No inverse is reused across QPs, and no matrix-type change,
refinement step, tolerance change or fallback is added.

One batch block then fuses the existing Gamma, inverse GEMV and post-solve
device helpers. A chunk ends at the next termination, infeasibility, rho or
iteration-budget event. Certificate duals are saved before the final body of
the chunk; `x_old` is saved before every body. Completed rows preserve their
terminal values. Projection, slack and log-barrier semantics are retained;
log-barrier still has no hard-QP infeasibility certificate. The shared host
code performs the original residual/status check and then rho adaptation.

Workspace capacity includes the backend mode, dimensions and scalar type.
Only the inverse backend allocates its two `B*TN*TN` buffers. Mode or capacity
changes destroy the old descriptors before freeing their buffers. Python
requires float64 CUDA tensors and verifies the extension capability tuple
`supported_admm_linear_solvers` before creating an inverse workspace. A
legacy extension remains usable for vector solves, but cannot silently
accept an unsupported inverse request. The KKT backward solver and IR2 are
unchanged.

The isolated N5 prototype and event-boundary evidence are retained in
`results/acmpc-native-pathcover-v1/cudss-inverse-chunk-v35-study/`. The
production implementation uses generic `TN` formulas; the prototype's N5
whitelist is removed. Production tests cover backend switching, independent
outputs and streams, generic QPs, N2/N5/N10 CTBR problems and the actual CTBR
low-diagonal/high-rho linear system. Prototype performance is separate from
the production P0 and end-to-end acceptance gates.
