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
- CTBR factories retain at most four ordered RISP corridors. Native PathCover
  selects the last corridor's DMP exit as the local terminal position and solves
  a complete new zero-terminal-velocity/acceleration MINCO trajectory. This is
  a shorter planning horizon, not truncation of an already optimized curve.
  Standalone PathCover retains six corridors and explicit Newton overrides persist.
  A matched 14-input study passes the same full-curve quality thresholds for both
  horizons. Its ten changed inputs reduce quadrature-node work by 34.373% and
  plane projection work by 25.371%, but seq578 increases work. The shortest new
  reference is 1.413821 s, longer than the N5 0.4 s window. Complete navigation
  and end-to-end latency still require physical evaluation.
- The state is `[p_world(3), q_wxyz(4), v_world(3)]`. The control is collective
  force in N and body rates in rad/s. Execution divides force by vehicle mass.
- `N` counts cost nodes, including the measured initial state, as in AC-MPC.
  There are `N-1` physical transitions. The default `N=5`, `dt=0.1 s` covers
  0.4 s. The final node's control is not executed. The network emits `28N`
  parameters: a 14D diagonal quadratic cost and a 14D linear cost per node.
- Costs are learned about a reference. In absolute solver coordinates the
  linear coefficient is `q_abs = q_local - D * z_ref`. Scaling is selected for
  numerical conditioning and documented separately from the author baseline.
- A0 uses a shared Newton/MARSIM fixed-clock line reference at `reference_speed`,
  initially 3 m/s. Sample N+1 positions from the current measured position toward
  the mission goal, stopping the position samples at that goal. Their forward
  differences give N velocity targets, including a partial arrival interval.
  Preserve the real mission goal feature, measured x0 and saved warm controls.
  The velocity changes, zero acceleration and heading are soft references, not
  a continuous dynamically feasible trajectory. This requires no geometric
  planner or free-time variables. The earlier repeated distant-goal reference
  failed nonlinear SQP even with fixed costs on the saved 100 m input. A shared
  reference contract changes A0 training inputs, so retrain its checkpoint.
- At `t_now + k*dt`, sample the native polynomial and its owning corridor.
  Exact corridor junctions use both adjacent sets of faces. Use acceleration
  and yaw to construct reference attitude. Do not assume zero roll and pitch.
- Apply robot clearance once in the frontend. The MPC uses every returned SFC
  face as a hard constraint. GCOPTER's soft penalties alone are not a proof of
  constraint satisfaction; record sampled violations during validation.
- CTBR factories use zero additional GCOPTER position margin. Robot clearance
  is represented by the occupied voxel boxes plus the body ellipsoid when
  constructing the returned SFCs. Inflated raster cells remain a conservative
  search representation. The earlier additional
  0.15 m inward penalty offset could conflict with braking near a corridor face;
  removing it does not enlarge those SFCs or change the physical collision radius.
  Standalone PathCover retains the official YAML's 0.15 m reference margin, and
  explicit Newton overrides remain supported. Nonzero margins retain the existing
  polytope/intersection caps and endpoint face-clearance caps. Record requested
  and applied margins separately. A soft position penalty never proves clearance.
- CTBR factories select `position_penalty_aggregation=continuous_max`. For each
  S3 polynomial piece, the position cost is the original smoothed L1 function
  applied to the largest plane residual over its complete time interval, multiplied
  by the existing position weight. There is no quadrature time weight on this term.
  Find its floating-point maximum from both endpoints and the real stationary
  roots of the normalized-time quintic plane projection. The derivative polynomial
  has degree at most four, so the existing root solver uses its analytic branch.
  Differentiate the selected face/time with the envelope formula; positive ties
  remain nonsmooth and select the first face, then the earliest candidate time.
  A conservative Bernstein upper bound can skip a face's roots only when it is
  strictly below the current maximum. Such a face cannot supply a winning endpoint
  either. Finite-input checks remain; other faces evaluate both endpoints and roots.
  This root calculation belongs to the objective, not a second acceptance pass.
  Dynamic penalties and the original quadrature diagnostics remain unchanged.
  Floating-point roots are not interval certificates. `integral` remains the
  standalone default, the behavior of historical snapshots without the field,
  and the S2/S4 path. The aggregation mode is independent of the initial-force
  boundary. Unmodified official oracles explicitly reject the new cost instead
  of silently solving a different objective.
- Match the official frontend's reference acceptance semantics: optimization must
  produce a finite cost and a nonempty numerical trajectory. Interface conditions
  such as positive durations and a defined attitude/yaw remain necessary.
  Finite soft-constraint residuals are quality diagnostics, not an online veto.
  `optimizer_quadrature_feasible` retains the original sampled thresholds and
  zero-heading rate metric. It is not a full-curve feasibility certificate.
  Offline P1 quality checks keep their original tolerances; removing the veto
  does not make previously violating references pass those checks.
- Preserve measured initial position, quaternion and velocity for CTBR. Acceleration
  is not a 10D CTBR state: it depends on the optimizable collective control.
  Pass the normalized measured quaternion's body-z axis to native PathCover and
  optimize one initial force scalar within the physical actuator bounds, with
  `a0 = F0 * body_z / mass - gravity * e_z`. The measured acceleration initializes
  F0 through its projection on body-z; it is not an equality boundary. Preserve
  both measured and reference acceleration and the raw/projected force guess in
  snapshots. Standalone calls without this optional axis keep fixed initial PVA.
  Historical captures without boundary metadata retain their fixed-PVA semantics.
- Construct the initial flatness heading as `atan2(-R01, R11)` from the measured
  rotation. For an upright body-z and positive force this reproduces the measured
  frame; Euler yaw generally does not. Zero or non-upright force still has no valid
  heading under this chart. Body rates remain controls, not additional boundary
  state constraints. Newton and Flightmare actuator lag is still present, so this
  reference boundary requires physical task validation as well as curve checks.
- `nominal_velocity` controls GCOPTER's existing
  speed cost; the independent `max_velocity` is the physical reference bound.
  The navigation factories derive that diagnostic bound from the scene and the
  most restrictive CTBR velocity component. Report nominal exceedance separately
  from physical violations. The physical reference bound stays fixed when the
  measured initial speed changes. Historical `speed_acceptance_limit_m_s` names
  now denote this quality threshold, not online rejection. CTBR hard velocity
  constraints remain unchanged.
- Use the same reference preset for the formal A1/A2 baseline: nominal speed
  3 m/s. V22 raises only the position penalty from the upstream
  1e4 to 1e5 after an exactly reproduced SFC violation. Weights are
  `(1e5, 100, 1e4, 1e4, 1e4)`; increasing the rate weight as well reintroduced
  a 16.762 mm SFC violation and was rejected.
  Standalone PathCover retains the upstream thrust defaults `(2, 10) N` scaled by
  `mass / 0.716 kg`. The CTBR Newton and MARSIM factories define physical force
  bounds from `mass * specific_thrust_limits`, currently 0 to 22.0725 N.
  Keep a separate `nominal_thrust_min`, normally `mass * 2 / 0.716`, as a positive
  soft planning target. The GCOPTER thrust penalty uses this nominal lower bound
  and the physical upper bound. Quadrature and physical quality diagnostics use
  the physical lower bound; report nominal shortfalls separately. Require
  `physical_min <= nominal_min < physical_max`. Standalone and historical
  configurations without the new field use their own recorded `thrust_min` as
  the nominal target. Explicit targets persist in snapshots and numerical replay.
  The V24 upper-bound change removes the earlier 10.474860335 N reference upper
  bound. Neither change alters the MPC or executor limits. A fixed measured
  initial force below the nominal target is a soft-target shortfall, not a
  physical impossibility. Positive soft cost does not guarantee force stays away
  from zero between quadrature nodes. For CTBR factories the default tilt is pi/2, the
  nominal boundary of the existing strictly upright heading chart; CTBR itself
  has no hard tilt bound. Preserve an explicitly requested stricter limit in
  the Newton factory. Standalone PathCover keeps its upstream 0.7854 rad tilt.
  Nonzero force and strictly positive vertical force remain actual chart-domain
  requirements. Zero-force or horizontal/inverted references must still report
  their existing explicit errors; no force clamp or invented heading is added.
  In the old fixed-PVA boundary, measured initial acceleration determined force
  and tilt. In V26, six thrust-floor and three tilt failures were already at that fixed
  boundary; changing duration or stopping rules cannot remove those conflicts.
  An isolated experiment also lowering the optimizer target to zero passed
  11 of 15 frozen failing references, leaving the SFC case and three body-rate
  cases. A later extra-piece experiment exposed a near-zero-force rate peak
  between quadrature nodes, so do not confuse a zero physical lower bound with
  a numerically useful flatness planning target. These are explicit model/cost
  decisions, not relaxed numerical quality tolerances.
  Preserve all old-model failures. A demonstrated feasible full curve for
  a saved descending state peaks at 13.59695 N; this is a witness under the
  actual force range, not proof that every old-model
  solution was infeasible. Prior 6 m/s runs remain
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

For the optional CTBR initial boundary, add one unconstrained scalar eta to the
existing S3NU GCOPTER variables. Map it through
`F0 = lower + (upper-lower) * (1+tanh(eta))/2`. Project only the initialization
fraction into a finite open interval before atanh, and record that projection.
Apply the same initial acceleration before duration initialization, each MINCO
objective evaluation and final trajectory construction. The MINCO adjoint row 2
is the derivative with respect to initial acceleration; contract it with body-z
divided by mass and the tanh derivative. No second optimizer, force clamp,
additional piece or changed stopping rule is introduced. Preserve generic S2/S4
and standalone fixed-PVA behavior. The bounded parametrization reaches physical
endpoints as a limit; the nominal positive force target remains a soft cost.

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
retains the frozen numerical outputs. Keep the existing floating-point CPU build
flags; the measured broad native-instruction target did not improve the main
workload. V32 separately enables integer POPCNT for the bitmap translation unit
on the verified x86 host and records that requirement in the build manifest.

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

CTBR factories select `geometry_model=box_ellipsoid`. Use the original occupied
NanoVoxMap voxel volumes, coalesced into boxes with exactly the same union, plus
the body ellipsoid for endpoints, motion seeds and RISP. The body clearance is a
single Minkowski addition. Keep the rasterized inflated map for conservative
coarse DMP search. Its occupied cells are not the continuous geometry predicate.
Standalone and historical configurations without the field retain
`geometry_model=inflated_voxels`.

The ellipsoid axes stay aligned with the world frame; the current CTBR body is
a 0.2 m sphere. For positive body semiaxes, scale space to a unit sphere. Point, segment,
triangle and tetrahedron distances to an axis-aligned box then give the relevant
intersection predicates. RISP subtracts the ellipsoid support from the nearest
box support when constructing a separating plane. Initial seed selection still
keeps p0/p1 fixed. For each raw box, the union of candidate triangles over an
interval of the thrust-tip parameter is their four-point convex hull. Prefix
and suffix intersection bisections enclose that box's complete forbidden
interval; merge the per-box intervals before selecting a free tip. Contact is
blocked. Preserve an already valid warm seed exactly. Endpoint virtual edges,
coarse anchors and visibility simplification use the same segment predicate.
Adjacent six-axis fine centers retain their original fast edge test: every
nonempty axis section of a raw lattice box plus an axis-aligned ellipsoid has
length at least one fine cell, so it cannot cross that edge without containing
an endpoint. This property does not apply to arbitrary edges. All-zero semiaxes
reduce to boxes; partially zero
semiaxes are explicitly unsupported in this mode, not replaced by epsilon radii.

This removes a demonstrated rasterization artifact in V28 Dense. At 44.8 s the
measured start was inside an inflated fine cell, although its distance to the
only contributing raw occupied voxel box was 0.316371 m for a 0.2 m body radius.
The old plan still contained the measured position; the newly observed cell
blocked the next planning call. No occupied input voxel is cleared by the new
geometry. V28's 418 returned references passed the complete curve checks, but
that run completed 0/2 waypoints and its all-attempt planning p95 was 104.857 ms.
New geometry therefore still requires task and performance validation.

RISP randomized cuts can change with box grouping, while the raw occupied union
must not change. Remove redundant half spaces using the existing dual convex
hull procedure. This preserves the polytope, not the sum of soft penalties
outside it. Record actual geometry mode and expose cached obstacle boxes plus
body radii for failure diagnostics outside the timed successful control path.

The historical geometry reuses its exact inflated boxes to project coarse
occupancy. In the new CUDA geometry path, project the inflated bit-grid directly:
one coarse-cell thread reads precisely the overlapping fine-cell intervals using
the same integer positive-volume rule. Preserve virtual boundary cells when
merging the returned cloud occupancy. This avoids constructing inflated boxes
that the new RISP no longer consumes; compress the raw boxes only once. The
explicit CPU path retains the same coarse-cell result. Record the actual
`gpu_grid_projection` observation. A frozen Dense grid with 1,832,342 fine cells
produced identical 369,600 coarse bytes with the direct CUDA, old CUDA-box, and
CPU volume paths. Direct projection reduced local p50 from 6.272604 to 2.388317 ms
in that comparison, excluding the new raw-box compression and other stages.

Merge matching X intervals across consecutive rows before the remaining Z sort,
preserving final box order and RISP input. Compute 3D legal-neighbour masks together with the CUDA
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

## Bounded corridors and reference convergence

CTBR uses an explicit `risp_local_half_extent=8.0` m in both Newton and MARSIM.
Each RISP call intersects the existing workspace with a local axis-aligned box.
The first box encloses the complete initial motion seed before expanding each
axis by 8 m; subsequent boxes surround their path seed. Prepare the original
obstacle boxes and outward-rounded axis bounds of their body-ellipsoid sum once
per plan. Omit a box only when a strict axis-support certificate puts that whole
sum outside the local box. Touching and numerically uncertain boxes remain.
The six local faces remain part of the raw corridor, so every omitted obstacle
is still excluded by the returned geometry. This is a new bounded corridor
configuration, not an equivalence claim about global RISP output. Keep global
DMP, the existing ordered path advancement and the fixed corridor count.
Zero retains global RISP; absent historical configuration resolves to zero.
Positive local extent requires `box_ellipsoid` in the native/Python adapters.
The full-table preparation and per-corridor linear filter remain in measured
planning time; this is not an algorithm independent of the global box count.

Use piecewise quadratic segment/AABB distance and the exact repeated-point
shortcut before producing the existing directed support certificate. Finite
distance witnesses may change at floating-point precision or at nonunique
minima. Collision acceptance continues to require a strict separating support
certificate; an approximate nearest distance alone cannot accept a segment.
Binary64 normal-adjacent steps and conservative cached support classification
reduce arithmetic cost without weakening that certificate.

Use GCOPTER relative cost tolerance `1e-5`. The previous `1e-4` stop produced
one actual initial pitch-rate violation among 944 complete Dense references.
All 944 frozen inputs pass the unchanged physical-quality comparison at the
selected tolerance. This is frozen-problem evidence, not a new task result.
In `continuous_max` mode, compute its pure quadrature SFC diagnostic once after
optimization from the final objective trial, before reconstructing the accepted
trajectory. Keep the continuous position objective, dynamic quadrature,
gradients and finite-value checks in every objective evaluation. Preserve
failure diagnostics over the same visited pieces. The integral position mode
retains its existing quadrature path.

In S3 continuous-position mode, evaluate P/V/A/J jointly by Horner and defer
derivative-basis construction past the existing exact-zero-contribution branch.
Use the original expression if a newly precomputed derivative coefficient or
the Horner result is nonfinite. This is an arithmetic optimization, not a new
acceptance threshold. Different floating-point reduction order can change the
nonconvex optimizer path and intermediate-overflow behavior. Preserve the actual
finite-value checks and independently validate the resulting complete references.
The integral mode and continuous SFC extremum calculation retain their old path.

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
