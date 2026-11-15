# Fixed-progress SFC sections for native AC-MPC

This experimental mode retains PathCover, the 10D CTBR prediction model,
TurboMPC, and the original RSL-RL Actor/Critic/PPO. It replaces only the
per-node SFC halfspaces passed into CTBR MPC. The V40 full-polytope baseline
remains the default and is not relabeled as a section result.

## Geometry

At fixed clock nodes `t_now + k * dt`, PathCover supplies reference position
`r_k`, velocity, yaw, and the owning convex polytope
`P_k = {p | A_k p + c_k <= 0}`. At an exact corridor junction, use the union
of both corridors' face sets, as in the original integration. The reference
tangent is the normalized velocity, or the position secant / yaw if stopped.
There is no optimized progress state.

For each source halfspace `a_i·p+c_i <= 0`, decompose
`alpha_i = a_i·t`, `n_i = a_i - alpha_i*t`, and
`slack_i = -(a_i·r+c_i)`. The normal-plane intersection gives
`n_i·(p-r) <= slack_i`. A finite sweep with axial progress
`s=t·(p-r) in [-h_back, h_forward]` uses the tightened inequality

```
n_i · (p-r) <= slack_i - max(-alpha_i*h_back, alpha_i*h_forward)
-h_back <= t · (p-r) <= h_forward
```

Both half-lengths start from `sfc_section_half_length_m` (default 0.5 m)
and are clipped to half the available axial source-polytope clearance at the
anchor. This leaves lateral slack instead of placing the reference exactly
on a tightened face. Axial-only
faces already satisfied by the caps are omitted. If such a face is not
satisfied due to a source reference inconsistency, the original face is
retained, rather than silently widening the source SFC.

This yields a **bounded inner approximation** of the CMPCC section sweep:
every point satisfying the section-tube inequalities also satisfies its
original SFC halfspaces, up to floating-point arithmetic. Unlike CMPCC's
unbounded polygon extrusion, it has explicit longitudinal caps. It is
generally more restrictive than the full polytope and may make a previously
feasible MPC problem infeasible.

The section is encoded directly as `planes: [B,N,F,4]` and
`plane_mask`, including the two axial caps. No new SQP/KKT implementation,
second solver, SO(3) trajectory follower, or new action variable is added.
The original node-0 state remains fixed. The MPC uses the new hard planes on
nodes 1 through N-1.

## Configuration

- `sfc_constraint_mode=polytope`: original complete PathCover SFC (default)
- `sfc_constraint_mode=section`: bounded fixed-progress sections
- `sfc_section_half_length_m=0.5`: requested axial half-length in metres

The face capacity stays at 128 slots per prediction node. If a section plus
its caps exceeds the configured capacity, the call fails explicitly rather
than dropping safety planes. Validated per-node reference and source-SFC
residuals are recorded separately.

### Newton fixed-cost evaluation

On the current workstation, the existing native cuDSS build requires its CUDA
13 wheel directory on the dynamic library search path:

```bash
export LD_LIBRARY_PATH=/home/tong/tongworkspace/projects/aerial/isaaclab-IPC/build/turbompc/envs/cu130/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PYTHONPATH=src TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-v35/cu130" \
  .venv/bin/python scripts/acmpc.py evaluate --fixed-cost \
  --task Isaac-ACMPC-PathCover-Forest-Direct --num-envs 1 \
  --output results/section-forest-eval \
  env.sfc_constraint_mode=section env.sfc_section_half_length_m=0.5
```

For learning, use the same overrides with `scripts/acmpc.py train` and
the existing RSL-RL options. PPO replay uses stored sections; it does not
re-run PathCover or backpropagate into the geometry.

### MARSIM dynamic evaluation

```bash
.venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --time-mode sim --benchmark dense \
  --policy fixed-ctbr --frontend pathcover \
  --sfc-constraint-mode section --sfc-section-half-length 0.5 \
  --output results/section-dense-eval
```

To run a learned Actor instead, select `--policy acmpc` and provide the
matching `--checkpoint`. Previously trained full-polytope checkpoints may
load because the tensor shapes are unchanged, but this is a constraint-domain
change and must not be presented as training or convergence under sections.

## Validation and interpretation

Unit tests verify that source-polytope inclusion holds for inclined sections
by independently maximizing every original face over the section using linear
programming. Additional cases check a skew face, terminal hover, exact
corridor junction, fixed-shape CTBR constraints, and the original sampling
path. Keep full-polytope and section evaluations separate.

The existing physical collision/SFC episode monitor still measures the
**original PathCover SFC**. MPC predicted constraints use the section tube.
Their residuals are not interchangeable. Node-based MPC constraints and a
0.4 s prediction horizon do not certify continuous-time collision freedom,
recursive feasibility, or local terminal braking. Evaluate these separately
before claiming safe or real-time flight.

## Recorded acceptance (2026-11-15)

The bounded inner-section geometry, its inclusion in the original SFC, the
original polytope sampling contract, and its CTBR hard-constraint interface
pass the targeted tests. The final combined section, navigation, and
observation suite passed 19 tests. Ruff, Python compilation and
`git diff --check` passed.

The first live fixed-cost Forest experiment (`results/section-forest-eval-20261115-v4/`)
arrived without collision, but encountered 6 unsolved MPC calls. The initial
section sweep exhausted an original face's axial clearance, leaving essentially
zero lateral slack at its reference anchor. Its first failing input, solver
statuses, residuals, and rollout remain under `first-unsolved-seed-0.pt`.
The corrected sweep reserves half the available axial clearance for lateral
motion, without relaxing the source SFC.

Corrected Newton Forest evaluation (seed 0, fixed costs, one episode):
`results/section-forest-eval-20261115-v5/`.

| Measurement | Result |
| --- | --- |
| Goal reached / collision / timeout | yes / no / no |
| Simulated flight | 3.499999 s, 35 policy steps |
| Solved MPC calls / QP failures | 35/35 / 0 |
| Maximum source-SFC sampled physical violation | 0 m |
| Minimum physical obstacle clearance | 0.416520 m |
| Maximum QP primal residual | 4.518e-6 |
| Returned PathCover references | 36 |

The real RSL-RL PPO smoke at `results/section-train-smoke-20261115/` used
two Newton environments, four rollout steps each, one PPO update with two
minibatches and one epoch. The 8 unique rollout samples, 8 PPO replays and
4 checkpoint-reload rows all solved. Both cost backpropagation calls yielded
finite nonzero MPC cost gradients; the cost-network maximum parameter change
was `2.0012719e-4`. Checkpoint action replay matched. The training audit
confirmed 8 PathCover calls during rollout and **zero** during update/reload.
This is training wiring/gradient acceptance, not policy convergence (zero
completed full episodes).

The new MARSIM parameters are connected through the CLI, launch, and ROS
controller; no full MARSIM Dense/High-speed section episode has been executed
in this change. The existing ROS overlay contains stale pre-migration
executable symlinks and must be rebuilt or repaired before live ROS evaluation.
