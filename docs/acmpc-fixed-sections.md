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
