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
  polytope/intersection inscribed radii. In the first and last corridor, additionally
  cap each face's margin by the fixed endpoint's nonnegative original clearance.
  A fixed point on an original SFC face must not be required to move 0.15 m inward.
  Original SFCs, endpoint PVA, waypoint parameterization, map inflation and vehicle
  radius stay identical. Record requested margin, geometric caps and actual
  per-face margins separately. The per-corridor applied value is the minimum
  actual face margin. This remains a soft objective, not guaranteed clearance.
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
- Use the same reference preset for the formal A1/A2 baseline: nominal speed
  3 m/s, tilt 0.7854 rad. V22 raises only the position penalty from the upstream
  1e4 to 1e5 after an exactly reproduced SFC violation. Weights are
  `(1e5, 100, 1e4, 1e4, 1e4)`; increasing the rate weight as well reintroduced
  a 16.762 mm SFC violation and was rejected.
  Standalone PathCover retains the upstream thrust defaults `(2, 10) N` scaled by
  `mass / 0.716 kg`. The CTBR Newton and MARSIM factories use the physical CTBR
  upper force bound for reference optimization: `mass * specific_thrust_limits[1]`,
  currently 22.0725 N. Their lower reference bound still intersects the upstream
  lower bound with CTBR's lower force bound. This V24 change removes the earlier
  10.474860335 N reference upper bound; it does not change the MPC or executor
  limits. Preserve the old-model failures. A demonstrated feasible full curve for
  a saved descending state peaks at 13.59695 N; this is a witness under the
  actual force range, not proof that every old-model
  solution was infeasible. Preserve measured initial PVA. Prior 6 m/s runs remain
  separate stress results; do not combine them with the new matched baseline.
- Keep the physical roll/pitch rate bound separate from the reference target.
  `reference_body_rate_margin=0.01 rad/s` lowers only GCOPTER's zero-heading
  rate-norm target from 3 to 2.99 rad/s. Physical diagnostics and MPC bounds stay
  at 3 rad/s; body-yaw stays at 1.5 rad/s. This is a fixed optimization allowance,
  not a constraint guarantee or a changed acceptance tolerance. For zero drag,
  the zero-heading norm bounds the thrust-axis turning rate, which equals the
  actual roll/pitch rate norm for any valid heading. Its violations remain
  separate from actual per-axis violations. Serialize both limits and the margin;
  replay of old problems with no margin field must retain zero margin.
- Rollout observations freeze state, reference, SFC and warm-start controls.
  PPO replay does not mutate the frontend or use a hidden last-solve cache.
- The CTBR adapter supplies an optional three-point `initial_motion_seed` to
  PathCover, together with an optional two-endpoint `initial_motion_tip_segment`.
  The exact Euler model fixes p0 and p1. Its p2 still depends on the optimizable
  first collective force. The three-point seed uses bounded warm thrust only as
  a preference; the segment contains p2 at the physical CTBR force limits.
  A warm seed collision does not prove all admissible first forces infeasible.
- The native frontend queries fine occupied boxes in the swept seed hull's local
  bounding box. For each relevant box, two fixed-dimensional linear programs
  find the forbidden segment-parameter interval. Merge these intervals and select
  a point inside the nearest feasible component; preserve a feasible warm seed
  bit for bit. Contact is excluded, so an exact nearest point need not exist.
  Interior floating-point projection uses the existing separation predicate,
  without padding or clearing occupancy. A colliding fixed p0–p1 edge or an empty
  feasible parameter set remains a planning failure.
- The first RISP polytope contains the selected three-point convex hull; later
  polytopes use the original point seeds. The selected point is a geometric
  initialization witness, not an executed control or an equality fixing MPC F0.
  Complete MPC feasibility is still determined by all dynamics and node constraints.
- Compute the five points in one fixed full-batch GPU kernel and include them in
  the measured context's existing host transfer. Partial resets select rows after
  that full-batch computation. PathCover does not own the CTBR model or prediction
  period, and its standalone point-seed API remains available.

## Numerical and execution decisions

Reuse native TurboMPC CUDA/cuDSS and its PyTorch adjoint. Keep dynamics and
warm-start rollout on the GPU, batch environments, and compile fixed-shape
tensor calculations. Do not recreate the JAX bridge or replace the solver.

In 3D GCOPTER's main L-BFGS solve, retain the absolute gradient condition
`||g||_inf / max(1, ||x||_inf) < 1e-5` and also allow
`(||g||_inf / max(1, ||x||_inf)) / max(1, |J|) < 1e-7` for finite `J`.
The optional relative condition is disabled in every other L-BFGS use.
Apply it only at the initial or accepted iterate, before another line search.
Raw costs, Wolfe conditions, the existing past-improvement condition and physical
quality tolerances remain unchanged. This changes gradient stopping semantics;
it does not reinterpret a negative optimizer status as convergence. The fixed
positive GCOPTER objective has no added constant offset. The condition measures
scaled stationarity, not trajectory feasibility or an objective-error bound.
An independently solved one-piece failure was already within 19.4 ns of the
optimal duration; approximately 1e-12 native cost noise defeated its line search.
Retain the failed run and the high-precision comparison as evidence.

Use a history length of 32 in the 3D main solve. Keep the short-path and vertex
solves and other L-BFGS history lengths unchanged. The prior 256-entry history gave
973 objective evaluations on a measured slow input; 32 gave 486. The objective,
constraints, quadrature and stopping criteria remain the same, but the changed
search direction can select a different nonconvex solution. All 600 frozen V24
Dense references passed the original full-curve quality tolerances with this
configuration. Preserve objective regressions, including the 15.4674% maximum,
and validate complete task and latency separately. This is an explicit speed
and local-solution tradeoff, not a bitwise-equivalent optimization.

Initialize the common duration scale by a bracketed derivative root, using the
existing MINCO energy/time value and gradient. With fixed spatial points and
duration ratios, write `T_i = exp(ell) * base_T_i` and
`g = d log(energy + rho * sum(T_i)) / d ell`. Start at `ell = 0`, search in fixed
log-scale steps for negative/positive derivative endpoints, then bisect. Count
every actual objective/gradient evaluation against a total budget of 64. Success
requires finite values and `abs(g) < 1e-8`; a small interval or an exhausted budget
is not convergence. Require finite positive time weight and base durations. Zero
jerk energy remains valid when the total energy/time cost is positive.

This finds a stationary initialization, not a certified minimum: the scalar
objective can have multiple stationary points. Main optimization still owns all
spatial variables, individual durations and penalties. For one piece with no
active penalties, the existing positive-time transformation satisfies
`abs(dT/dtau)/T <= 1`; thus this scalar gradient bound is stricter than the main
objective-relative gradient threshold. Derivative evaluation resolves the known
cases even when remaining cost decrease is below double-precision cost noise.
Do not relabel a failed line search as success or add an optimizer fallback.
Retain the rejected uniform L-BFGS tolerance change and its 19 failures among
472 completed archive inputs as counterevidence.

Batch position-plane projections while preserving the original three-term dot
order and face-order cost/gradient accumulation. Reuse one buffer per piece;
do not allocate per quadrature node. This independent arithmetic optimization
retains the frozen numerical outputs. Keep the existing portable CPU build flags;
the measured native-instruction target did not improve the main workload.

Remove the old CTBR reference governor, jerk recovery coupling and duplicate
online execution-trajectory acceptance from the new route. Solver residuals,
hard SFC constraints and independent collision evaluation remain measurable.
Failure reports must not be turned into successful navigation statistics.

Newton and Flightmare retain feasible requested collective force during rotor
saturation. First maximize the common roll/pitch torque scale, then select the
closest feasible yaw torque. Solve the four-rotor affine bounds directly; do
not invoke another online optimizer. Yaw may reverse when maximal roll/pitch
requires it. Feasible native allocations remain unchanged. This replaces uniform
torque scaling, which allowed a weak yaw axis to suppress roll/pitch authority.
Motor lag and dynamic integration remain active; Newton's additional motor lag
compensation still differs from Flightmare's exponential response. All matched
A0/A1/A2 runs must identify the same executor contract. The priority principle
also appears in the [PX4 mixer documentation](https://px4.github.io/Firmware-Doxygen/d4/db1/_multirotor_mixer_8hpp_source.html);
this implementation keeps collective exact and does not copy airmode thrust
adjustment.

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

Reuse those exact integer boxes to project coarse occupancy. Merge matching X
intervals across consecutive rows before the remaining Z sort, preserving final
box order and RISP input. Compute 3D legal-neighbour masks together with the CUDA
distance field. The masks depend only on original occupancy and map boundaries;
regional search restrictions and CLOSED states stay in the search. Install the
potential and masks together, and clear masks when replacing a map through the
CPU interface. Retain direction order, edge costs and endpoint semantics.
The native chain consumes the raw DMP path; the public simplified-path API
remains available. Measure mask generation and transfer within the existing
distance-map stage, not as a second additive timing stage.

V22 computes exact box X runs and Y grouping on CUDA, then completes the original
Z merge on the CPU. Preserve every integer box boundary and final ordering.
The helper owns reusable buffers and a stream per frontend instance. Full-map
staging and host finishing count inside the existing grid stage. The incremental
inflater's scratch window is not a reusable copy of the accumulated map.
Limit NanoVoxMap's host merge workers using the existing frontend `num_threads`;
leave its CUDA update algorithm and public hardware-selected default unchanged.

Pass Nano's complete integer occupied-block snapshot directly to the inflater.
Replace the seed bit-grid on each update, including miss-cleared and empty maps;
do not accumulate a union of historical obstacles. Preserve the existing
inflation kernel, output clipping, index range and CPU/CUDA selection. Keep the
public point-cloud input for independent users. For coarse projection, precompute
integer axis endpoint ranges within the actual fine window, preserving box and
occupied-index order.

In the 3D GCOPTER objective, specialize the exact zero-drag, zero-yaw flatness
case while retaining the generic implementation. Preserve the surviving
arithmetic and reduction order. Compute snap only for a nonzero jerk adjoint.
Objective, quadrature, bounds and stopping rules do not change as part of this
performance optimization. Existing finite-cost and finite-quality rejection
remains. Finite-domain equality does not assert identical NaN payloads or signed
zeros outside that domain.

Strengthen 3D DMP's heuristic with exact reverse costs on a relaxed XY graph.
Each column uses its minimum free-cell potential; every actual 3D edge remains
subject to the original occupancy and legal-neighbour checks. Take the maximum
of the existing 26-neighbour lower bound and the XY bound, then apply the existing
1.5 heuristic weight. This intentionally changes search order and may change path
cost. Do not describe it as equivalent paths or globally optimal weighted search.
Cache only identical projected potentials, target connector costs, dimensions
and weights. Reproject changed maps and recheck targets independently. Account
for preprocessing inside DMP time, not as an additional timing leaf.

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
