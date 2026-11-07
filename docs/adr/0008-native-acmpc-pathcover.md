---
status: accepted
---

# AC-MPC uses native PathCover and 10D CTBR

The user approved this architecture on 2026-11-07. It supersedes the AC-MPC
model and frontend choices in ADR-0001, ADR-0002, and ADR-0004. Historical IPC
code and evidence remain available; this branch develops the new AC-MPC route.

## Contracts

- Isaac Lab / Newton supplies physical state, LiDAR and PPO training. MARSIM /
  Flightmare dynamic execution supplies the formal navigation evaluation.
- Native PathCover owns mapping, global DMP, RISP corridors and GCOPTER/MINCO
  trajectory timing. ROS and Gazebo are not frontend runtime dependencies.
  Production DMP searches the full available grid with the existing potential
  and legal-neighbor checks. The V12 Dense failure has no legal path inside its
  JPS guide region, although a legal full-grid path exists. Upstream JPS remains
  available for the independent original-chain comparison; `search_radius`
  belongs only to that comparison.
- The state is `[p_world(3), q_wxyz(4), v_world(3)]`. The control is collective
  force in N and body rates in rad/s. Execution divides force by vehicle mass.
- `N` counts cost nodes, including the measured initial state, as in AC-MPC.
  There are `N-1` physical transitions. The default `N=5`, `dt=0.1 s` covers
  0.4 s. The final node's control is not executed. The network emits `28N`
  parameters: a 14D diagonal quadratic cost and a 14D linear cost per node.
- Costs are learned about a reference. In absolute solver coordinates the
  linear coefficient is `q_abs = q_local - D * z_ref`. Scaling is selected for
  numerical conditioning and documented separately from the author baseline.
- At `t_now + k*dt`, sample the native polynomial and its owning corridor.
  Exact corridor junctions use both adjacent sets of faces. Use acceleration
  and yaw to construct reference attitude. Do not assume zero roll and pitch.
- Apply robot clearance once in the frontend. The MPC uses every returned SFC
  face as a hard constraint. GCOPTER's soft penalties alone are not a proof of
  constraint satisfaction; record sampled violations during validation.
- Request the official YAML's 0.15 m reference position margin in GCOPTER's penalty copy.
  It shifts the unit-normal position residual only, with the existing caps from
  polytope/intersection inscribed radii. Original SFCs, waypoint parameterization,
  map inflation and vehicle radius stay identical. Record both requested and
  applied margins. This remains a soft objective, not guaranteed clearance.
- Match the official frontend's reference acceptance semantics: optimization must
  produce a finite cost and a nonempty numerical trajectory. Interface conditions
  such as positive durations and a defined attitude/yaw remain necessary.
  Finite soft-constraint residuals are quality diagnostics, not an online veto.
  `optimizer_quadrature_feasible` retains the original sampled thresholds and
  zero-heading rate metric. It is not a full-curve feasibility certificate.
  Offline P1 quality checks keep their original tolerances; removing the veto
  does not make previously violating references pass those checks.
- Preserve measured initial PVA. `nominal_velocity` controls GCOPTER's existing
  speed cost; the independent `max_velocity` is the physical reference bound.
  The navigation factories derive that diagnostic bound from the scene and the
  most restrictive CTBR velocity component. Report nominal exceedance separately
  from physical violations. The physical reference bound stays fixed when the
  measured initial speed changes. Historical `speed_acceptance_limit_m_s` names
  now denote this quality threshold, not online rejection. CTBR hard velocity
  constraints remain unchanged.
- Use the upstream reference preset for the formal A1/A2 baseline: nominal speed
  3 m/s, tilt 0.7854 rad, penalty weights `(1e4, 100, 1e4, 1e4, 1e4)`.
  Scale upstream thrust bounds `(2, 10) N` by `mass / 0.716 kg` and intersect them
  with the CTBR physical force range. This defines the reference optimization
  limits; the MPC retains its physical input constraints. Preserve measured
  initial PVA even when it exceeds a reference limit. Prior 6 m/s runs remain
  separate stress results; do not combine them with the new matched baseline.
- Rollout observations freeze state, reference, SFC and warm-start controls.
  PPO replay does not mutate the frontend or use a hidden last-solve cache.
- The CTBR adapter supplies an optional three-point `initial_motion_seed` to
  PathCover. It contains p0, p1 and p2 from the exact Euler model, using a bounded
  warm-start thrust. The first RISP polytope contains their convex hull; later
  polytopes use the original point seeds. This prevents the initial corridor
  from cutting off that inertial prefix. Complete MPC feasibility is still
  determined by its dynamics and all node constraints.
- Compute this seed in one compiled GPU batch and include it in the measured
  context's existing host transfer. PathCover itself does not own the CTBR model
  or prediction time step, and its standalone point-seed API remains available.

## Numerical and execution decisions

Reuse native TurboMPC CUDA/cuDSS and its PyTorch adjoint. Keep dynamics and
warm-start rollout on the GPU, batch environments, and compile fixed-shape
tensor calculations. Do not recreate the JAX bridge or replace the solver.

Remove the old CTBR reference governor, jerk recovery coupling and duplicate
online execution-trajectory acceptance from the new route. Solver residuals,
hard SFC constraints and independent collision evaluation remain measurable.
Failure reports must not be turned into successful navigation statistics.

Flightmare retains feasible requested collective force during rotor saturation
and uniformly scales the torque vector. Feasible native allocations remain
unchanged. Motor lag and dynamic integration remain active; Newton's additional
motor lag compensation still differs from Flightmare's exponential response.

Convert inflated occupancy into search cells on the integer voxel lattice.
Only positive-volume overlap occupies a coarse cell. Recognize commensurate
resolution ratios at configuration time; do not route geometry through packed
float centres. Packed points remain a diagnostic representation.

Use authoritative fine occupancy for RISP and motion-seed geometry; retain the
conservatively coarsened map for DMP. Coalesce adjacent occupied integer voxels
into disjoint rectangular boxes with exactly the same union. RISP uses each
box's three half extents. Its randomized cuts can change with grouping, but the
obstacle volume must not change. Remove redundant half spaces using the existing
dual convex hull procedure. This preserves the polytope, not the sum of soft
penalties outside it.

If a physically free endpoint is blocked only by coarse occupancy, connect it
through a fine-grid local search within a 1 m axis-aligned half-width. Retain
all reachable coarse anchors in that fixed window. Include connector lengths
in the multi-source/multi-goal DMP objective and terminate only when its virtual
goal is removed from OPEN. Validate connector segments against fine voxel boxes.
Do not clear coarse occupancy or silently enlarge the local search window.
This finite hybrid graph does not claim completeness in continuous free space.
Use exact body-clearance workspace bounds for geometry; only the coarse search
rounds those bounds inward to complete grid cells.

Pack sampled PVA, yaw, SFC and validity into one host-to-device transfer and
compile the existing tensor observation equations for the fixed full batch.
Select partial-reset rows afterwards, preserving frozen snapshots and avoiding
one compilation per subset size. Prewarm through the existing constructor
synchronization point; do not synchronize each predicted node.

Construct heading with a native scalar ODE that bounds body yaw rate pointwise.
Use adaptive RK4 and cubic Hermite interpolation, and return the derivative of
the same interpolant. Fade heading feedback continuously below 0.1 m/s.
Use sine angular error to avoid a shortest-angle branch jump at plus/minus pi;
the antipodal heading remains an unstable equilibrium, not a global convergence
guarantee.
Local integration tolerances do not certify the full trajectory's roll, pitch,
thrust or corridor constraints. Preserve true undefined thrust-frame failures.
On a heading failure retain the existing native candidate; serialize it only
when failure diagnostics are requested, outside the measured planning call.

The 100 Hz actuator clock and 10 Hz MPC replanning are separate. Simulation
time execution is supported. Wall-time planning p95 below 100 ms is a measured
acceptance target, not an implication of a configured ROS timer.

## Validation

P0: frozen-input forward comparison, finite-difference cost gradients, complete
PPO update, checkpoint reload, N2/N5 and B1/B16/B64/B256 timings.

P1: native frontend build, ordered overlapping corridors, continuous polynomial
sampling, explicit geometry and dynamics violation measurements.

P2: identical reference/SFC contracts in Newton and MARSIM; fixed-cost task
execution before learned-policy claims.
Evaluate the actual solver status, hard-constraint residuals and physical
execution. Writing hard constraints into the MPC is not a guarantee that an
unconverged returned control satisfies them. The frontend's tilt and other soft
dynamic limits are not all duplicated as hard CTBR constraints.

P3: compare A0 (no frontend), A1 (PathCover/fixed costs), and A2
(PathCover/learned costs) with identical dynamics, sensing and task criteria.
Keep success, collision, timeout and computational failures separately.
