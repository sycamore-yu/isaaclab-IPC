# TurboMPC regression data

`turbompc_kkt_accuracy_reproducer.npz` is an unchanged copy of the synthetic
fixture retained at `results/turbompc-port/kkt-accuracy-reproducer.npz` during
the native-port investigation on 2026-11-03. It contains a 40-by-40 real
symmetric KKT system in CSR form: `row` (41 int32 entries), `col` (226 int32
entries), `values` (226 float64 entries), and `rhs` (40 float64 entries).

The recorded matrix condition number was 33.87396. The original cuDSS KKT
configuration from TurboMPC commit `7050a4f367d08504ecb6f5820d9161e704df5816`
produced an infinity-norm residual of 3.3800e-4; an independent float64 dense
solve produced 1.9429e-16. The local two-step iterative-refinement change is
documented in `native/turbompc/upstream/PATCHES.md`.

The regression reconstructs the dense matrix directly from this retained CSR
data and uses NumPy's dense solve as an independent reference. Both the native
solution error and its absolute residual must be at most 1e-9. CUDA 13 with
two refinement steps produced a residual of 1.00643105e-10; the tolerance
retains margin while rejecting the original 3.38e-4 error. This fixture
does not depend on ignored runtime results remaining in a checkout.

File size: 4176 bytes. SHA256:
`db7e966e056c8e00fbcde846403dcf1944f611c1794ce80f07c04d4630feaf42`.
