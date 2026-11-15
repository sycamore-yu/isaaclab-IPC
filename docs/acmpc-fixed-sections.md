# Fixed-progress SFC sections for native AC-MPC

The `section` mode retains PathCover, the 10D CTBR model, TurboMPC, and
RSL-RL PPO. It adds fixed-progress longitudinal bounds to each original
PathCover SFC. `polytope` remains the default baseline.

## Geometry

PathCover solves a new local DMP → RISP → GCOPTER/MINCO plan on every
10 Hz MPC planning tick. It returns up to three ordered SFC polytopes and
a complete local time-parametrized reference. At node `t_now+k*dt`,
sample `r_k` and tangent `t_k` and select the corresponding SFC
`P_k={p | A_k p+c_k <= 0}`. At a corridor junction, enforce both
adjacent SFCs.

The section constraint is the original SFC clipped by two axial planes:

```
A_k p + c_k <= 0
-h <= t_k · (p-r_k) <= h
```

The default half-length is `h=0.5 m`. The tangent comes from reference
velocity, a position secant at zero speed, or yaw for stationary segments.
The two added faces enforce fixed reference progress. Original lateral
faces remain unchanged, so the section is contained in the source SFC.
This avoids projecting and tightening lateral planes a second time.

The constraint array remains `planes: [B,N,F,4]` with `plane_mask`.
Node 0 is fixed by measurement; MPC applies hard SFC constraints at
nodes 1–4. The CTBR dynamics, SQP/KKT solver, Actor and PPO are unchanged.
The path progress is a fixed reference clock, not an optimized MPC state.

## Configuration

- `sfc_constraint_mode=polytope`: original complete PathCover SFC (default)
- `sfc_constraint_mode=section`: bounded fixed-progress sections
- `sfc_section_half_length_m=0.5`: requested axial half-length in metres

The face capacity stays at 128 slots per prediction node and includes the
two axial caps. Reference and original-SFC residuals remain available in
planning diagnostics.

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

Unit tests verify original-face preservation, axial caps, the skewed-face
case, terminal hover, corridor junctions, and the CTBR constraint interface.

Physical collision checks use the original obstacle geometry. MPC predicts
four 0.1 s CTBR transitions and constrains their endpoints.

## Historical tightened-sweep acceptance (2026-11-15)

The following Forest and MARSIM data precede the simplified axial-cap
implementation. They are the legacy-sweep comparison, not new-slab results.

The historical inner-section geometry, its inclusion in the original SFC, the
original polytope sampling contract, and its CTBR hard-constraint interface
pass the targeted tests. The final combined section, navigation, and
observation suite passed 19 tests. Ruff, Python compilation and
`git diff --check` passed.

The first live fixed-cost Forest experiment (`results/section-forest-eval-20261115-v4/`)
arrived without collision, but encountered 6 unsolved MPC calls. The initial
section sweep exhausted an original face's axial clearance, leaving essentially
zero lateral slack at its reference anchor. Its first failing input, solver
statuses, residuals, and rollout remain under `first-unsolved-seed-0.pt`.
The previous sweep reserved half the available axial clearance for lateral
motion. The current implementation retains the source faces instead.

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
controller. The initial environment had stale pre-migration executable
paths. The following section records their repair and subsequent live
Dense/High-speed evaluations.

## Rerun and MARSIM closed-loop acceptance (2026-11-15)

The recorded Newton Forest run at
`results/acmpc-section-forest-rerun-20261115/` reached its goal in 3.5 s
without collision or QP failure (35/35 successful solves, 0.416520 m
minimum obstacle clearance). Its complete Rerun file is
`recordings/acmpc-section-forest-20261115.rrd`, 8.5 MB.
`rerun rrd verify` passed; the Viewer screenshot was inspected and a
smaller preview saved at
`recordings/acmpc-section-forest-20261115-preview.jpg`.
The CTBR Rerun recorder now uses the actual N=5 prediction schedule and
status rather than the historical IPC N=15/planner fields.

The original ROS overlay was created at the old `free/isaaclab-IPC` paths.
Its absolute symlinks and generated environment scripts were repaired in
the ignored build directories (874 links, 20 scripts, with pre-change
copies); see `results/marsim-ros-path-repair-20261115.json`.
The legacy mission and LiDAR ELF binaries also embedded the old source
directory through CMake's `ROOT_DIR`, so a separate, clean ROS 2 overlay
was compiled from the existing pinned SUPER, Flightmare and corrected
MARSIM renderer sources at `build/marsim-relocated-20261115/`.
The first two Dense attempts failed *before collecting odometry* due to
the old embedded source paths and absent cuDSS dynamic-library search
path. They are not counted as navigation trials. The checked-in
`scripts/marsim_ros2_env.sh` now resolves the shared dependency directory
and cuDSS wheel from the real Git common directory, including worktrees.

The following fixed-cost **section** evaluations used identical CTBR
model, original mission waypoints, Flightmare dynamic execution,
original full-PCD collision evaluation, 100 Hz simulated control and
10 Hz MPC planning. Each scene had one complete episode, seed 0:

| Metric | Dense | High-speed |
| --- | ---: | ---: |
| Official waypoints | 2/2 | 2/2 |
| Collision / timeout | 0 / 0 | 0 / 0 |
| Simulated time (s) | 97.25 | 71.00 |
| Flight distance (m) | 227.057 | 199.315 |
| Minimum swept PCD clearance (m) | 0.164869 | 0.703578 |
| MPC QP failures | 0 | 0 |
| Maximum sampled actual SFC violation (m) | 0 | 0 |
| Maximum predicted SFC violation (m) | 9.789e-6 | 0 |
| Full planning p95 (ms) | 108.96 | 53.83 |
| Planning p95 <100 ms | **fail** | pass |
| Recorded wall-clock real-time factor | 0.591 | 0.712 |

Evidence:
`results/marsim-section-dense-run-20261115/{metrics.json,trajectory.npz,policy_stats.jsonl,launch.log}`
and
`results/marsim-section-high-speed-run-20261115/{metrics.json,trajectory.npz,policy_stats.jsonl,launch.log}`.
These traces contain successful episodes; saved startup failures remain
under their separate experiment directories.

Runtime selection after building a relocated overlay:

```bash
export IPC_MARSIM_RUNTIME="$PWD/build/marsim_ros2_relocated"
export TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-v35/cu130"
source scripts/marsim_ros2_env.sh
DISPLAY=:11 .venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --time-mode sim --benchmark dense \
  --policy fixed-ctbr --frontend pathcover \
  --sfc-constraint-mode section --sfc-section-half-length 0.5 \
  --domain 156 --render-prefix 'vglrun -d egl0' \
  --output results/section-dense-repeat --timeout 180
```

The `build/marsim_ros2_relocated` path is the independent ROS build in
the canonical `isaaclab-IPC` main checkout. The feature worktree uses
`build/marsim-relocated-20261115` for its separately validated build.
Both contain the same source-derived mission, LiDAR and Flightmare nodes;
their generated absolute paths differ with the build root.

The same call with `--benchmark high_speed`, a distinct output directory
and ROS domain reproduces the other evaluation. The compiled overlay,
native CUDA library, generated RRD and raw results are locally retained
artifacts, not tracked Git sources. These tests do not demonstrate
multi-seed robustness, an AC-MPC learned-cost advantage, or a
100 Hz wall-clock runtime guarantee.

## Simplified section and learned-cost comparison (2026-11-15)

The current `section` implementation keeps every original SFC face and adds
only two longitudinal caps. On the same saved Dense planning input, the
constraint counts were 23–30 (full SFC) versus 25–32 (both sections).
Geometric sampling p50 fell from 1.134 ms (historical tightened section)
to 0.569 ms (current slab). The old swept region was contained within the
new region at all five saved nodes.

The paired TurboMPC comparison used one frozen Dense state, reference,
warm start and fixed costs, with 15 repeated solves per arm:

| Constraints | MPC p50 / p95 (ms) | QP iterations p50 | Solved |
| --- | ---: | ---: | ---: |
| Full SFC | 43.55 / 58.01 | 75 | 15/15 |
| Historical tightened section | 50.71 / 62.72 | 125 | 15/15 |
| Current SFC + axial caps | 44.01 / 56.30 | 75 | 15/15 |

Evidence: `results/acmpc-simple-section-20261115/{frozen-dense-geometry.json,paired_solver.json}`.

The current section Actor was trained with 16 Newton environments,
32 PPO updates, 8192 interactions and 224 successful training episodes.
The final cost-network maximum parameter change was 0.002725;
checkpoint action reload matched. Its checkpoint is at
`results/acmpc-simple-section-20261115/train-eager-8192/checkpoint.pt`.
The Actor used `compile_kernels=False` to avoid a long first-use compile,
while retaining the native CUDA solver.

In five paired Newton Forest evaluation seeds (0–4), fixed and learned
costs both reached 5/5 goals in 3.5 s, with zero collisions and QP failures.
Mean episode rewards were 2.72136 / 2.72231 (fixed / learned).

The official MARSIM runs use the same simplified sections and the **same
checkpoint** for both policies. The fixed-cost arm sets `fixed_cost=True`
inside the loaded actor to retain exactly the learned arm's solver,
compilation mode and physical parameters.

| MARSIM Dense | Fixed cost | Learned cost |
| --- | ---: | ---: |
| Goals | 2/2 | 2/2 |
| Flight time (s) | 95.54 | 97.39 |
| Distance (m) | 226.57 | 231.11 |
| Minimum clearance (m) | 0.2021 | 0.2226 |
| QP failures | 0 | 0 |
| Planning p95 (ms) | 161.63 | 162.06 |

Evidence: `results/acmpc-simple-section-20261115/marsim-dense-{fixed,learned}/`.

| MARSIM High-speed | Fixed cost | Learned cost |
| --- | ---: | ---: |
| Goals | 2/2 | 2/2 |
| Flight time (s) | 70.99 | 70.99 |
| Distance (m) | 199.283 | 199.301 |
| Minimum clearance (m) | 0.6768 | 0.7058 |
| QP failures | 0 | 0 |
| Planning p95 (ms) | 104.72 | 107.93 |

Evidence: `results/acmpc-simple-section-20261115/marsim-high-speed-{fixed,learned}/`.
All four flights ended with collision=0 and sampled actual SFC violation=0.
Both cost modes use the same checkpoint SHA256
`3496515294ea2b0034e96b069c809aeec427eef84d964c1c627386b8eb6c3193`
and the same PathCover and native MPC configurations.

The paired fixed/learned trials use eager inference
(`compile_kernels=False`). This kept both arms identical and made training
practical, but their full planning p95 exceeds 100 ms on both maps. The
earlier V40 trials used compiled inference, so comparing those absolute
latencies would mix geometry and execution modes.

The learned 8192-step candidate slightly increased minimum PCD clearance
but did not reduce time to goal on either map. The simple section improved
frozen-input geometry construction and the matched MPC solver p50; the
complete online pipeline's timing remains dominated by PathCover and
inference execution.
