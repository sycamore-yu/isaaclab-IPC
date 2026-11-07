# PathCover native front end

Source: https://github.com/kunalnk123690/PathCover
Pinned commit: `8723a05d1993bf0199a453b47dc6ee167d493c23`.

Only ROS-independent algorithm files are vendored. Native code links system Eigen,
Boost headers and Qhull. It does not link ROS, PCL or Gazebo.

- `nanovoxmap/`: sparse log-odds mapping and its upstream CUDA insertion/ESDF kernels.
- `occupancy_inflation/`: bit-grid obstacle dilation and its upstream CUDA backend.
- `jps/`: upstream JPS3D and distance-map search. Native V13 uses global DMP;
  JPS and guide-region planning remain available to the independent upstream comparison.
- `corridor/`: upstream RISP and sequential PathCover geometry.
- `trajectory/`: PathCover's GCOPTER wrapper plus the vendored MINCO/flatness/L-BFGS code.

Licenses are retained in each directory. PathCover, NanoVoxMap, inflation and JPS
use BSD-3-Clause. The vendored GCOPTER files carry Zhepei Wang/Fei Gao's MIT
license; embedded geometry utilities retain their original notices.

## Explicit local changes

1. RISP compacts obstacle indices instead of copying and moving XYZ vectors.
   Point inputs retain the upstream cutting-plane equation. Native navigation
   supplies occupied voxel AABBs: each normal points from the seed to the closest
   point on a box, and the support function removes only boxes lying completely
   beyond the cut. Through V18 each separating cut was retained. V19 restores
   upstream redundancy removal after constructing the same obstacle-free polytope;
   direct one-face-per-box certificates are checked before that equivalent cleanup.
   A per-thread seeded generator replaces global `rand()`.
   The GCOPTER Seidel LP permutation stream is also per-thread and reseeded at
   each solve, so equal frozen inputs remain reproducible across CPU/CUDA maps.
2. PathCover uses the requested polytope count exactly, tests containment with
   all inequalities, finds outward segment intersections, and reports the
   covered endpoint when the corridor horizon truncates the path.
3. GCOPTER exposes the optimized piece-to-polytope indices and optimizer status.
   Negative optimizer status, including the iteration limit, fails planning.
   The default iteration cap matches the upstream drone configuration (1000).
4. GCOPTER preserves the measured initial velocity and acceleration. The upstream
   boundary projection is disabled because it otherwise changes the CTBR
   controller's actual initial state.
5. The native binding assembles the algorithm chain without ROS. Map inflation
   supplies body clearance once. V15 gives GCOPTER's reference position penalty
   the official drone YAML's 0.15 m inward offset, as detailed in item 24 below.
   Physical workspace bounds, including virtual ground/ceiling, are shrunk by
   the body radii once. The coarse search rounds these inward to complete free cells; V19
   geometry and local connections retain the exact physical bounds. This
   handles boundaries that do not appear in the observed cloud. Mapping and
   search grids are anchored at world zero rather than an arbitrary box origin.
   The native ellipsoid inflation mask covers the source voxel volume plus the
   body ellipsoid, testing the minimum distance between source/target voxel
   boxes. It marks strict interior intersections (`distance < radius`), not
   target cells that only touch the expanded boundary. Thus every point with
   clearance below the body radius remains covered without an extra layer at
   integer radius/resolution ratios. A zero-radius axis retains only the source
   cell along that axis; all-zero radii retain the source voxel itself.
   This fixes under-inflation when body radius is below map resolution.
   CPU and CUDA still consume the same run mask. The original centre-distance
   mask remains the default of the separately reusable upstream inflater.
6. DMP uses the exact obstacle-free 26-neighbour grid metric, rather than
   upstream's zero heuristic (Dijkstra). At weight 1 this metric is admissible
   and consistent. The current default weight 1.5 explicitly trades shortest-path
   optimality for fewer expansions. Closed nodes retain their settled cost and
   parent rather than being reopened under the weighted heuristic. Four frozen
   graphs measured cost increases of 0.085% to 8.94% against the prior weight-1.1
   reopening search. It retains the same nonnegative distance-potential
   edge costs. Set `dmp_heuristic_weight=0` for Dijkstra or 1
   for admissible A*. The binding caches the DMP potential map until occupancy changes.
   The DMP heap uses strict cost ordering instead of a nontransitive epsilon
   comparator. Relaxations ignore differences below 1e-12 cost units and update
   only open nodes. Successor buffers are reused per query.
   DMP additionally requires every nonempty axis-subset neighbour of a diagonal
   move to be physically free. This removes corner/edge contacts with occupied
   voxel AABBs that otherwise collapse RISP progress at a grid corner. It changes
   the feasible graph relative to upstream; path cost is not claimed unchanged.
   A constexpr direction mask reuses 26 occupancy probes per expanded 3D node;
   The heap comparator takes
   shared pointers by const reference rather than copying them on comparisons.
   The closed-list search removed a measured 150,000-reopening tail. Two slow
   diagnostic maps had DMP p95 16.44 ms over 20 paired runs; Dense p95 increased
   from 9.19 to 9.51 ms. Those slow cases combine historical PVA with the final
   captured map and are not exact online-frame replays. Complete MINCO checks
   passed all four candidate and four baseline problems, but one final-frame
   MINCO execution rose from 35.46 to 72.78 ms. End-to-end timing is measured
   separately; search speed alone does not establish the online latency budget.
7. A persistent CUDA workspace computes DMP's integer max-dilation with the
   upstream-generated mask. Integer atomic max preserves the CPU potential field
   exactly. One bulk upload/download replaces the CPU obstacle-by-mask loop;
   GPU kernels do not change the search cost or create a learned surrogate.
8. Optional `search_resolution` conservatively coarsens occupied fine voxels for
   DMP. Through V18, RISP also used those coarse AABBs. V19 uses the authoritative
   inflated fine volumes for RISP and explicit local endpoint connections, as
   detailed in item 30. NanoVoxMap and inflation retain `resolution`; no start
   cell is cleared and the robot is not inflated again. The coarse grid
   changes the discretized search problem, so its path/cost are not claimed
   equivalent to a finer search grid. The default preserves the mapping grid.
9. GCOPTER's initial time ratios use path-length / max(0.5 * nominal speed, measured initial
   speed), replacing upstream's 3 * maximum speed allocation. A subsequent single
   scalar energy/time initialization is described below. The default
   penalty quadrature uses 64 subdivisions per piece instead of 16. Both changes
   address measured nonconvex local-solution sensitivity: a fixed half-speed
   start failed high-speed frozen cases, while 16 subdivisions stagnated or
   converged to inverted trajectories in Dense/Forest cases. Their original
   initial polynomials satisfied SFC; spatial overshoot of the initial guess
   was not the established cause. Objective terms, weights, physical bounds,
   positive-time parameterization and the 1000-iteration cap are unchanged.
   L-BFGS now brackets and shrinks a nonfinite objective/gradient trial in its
   existing line search; finite accepted iterates still satisfy its Wolfe tests.
   No partial failed iterate or alternate planner is returned.
10. QuickHull explicitly calls `std::abs(T)`. The upstream unqualified `abs(T)`
    can select integer abs under a different include order and incorrectly label
    full-dimensional dual vertices as coplanar. The independent upstream driver
    includes `math.h` before that unmodified header to expose the floating overload,
    matching the native wrapper's original include context.
11. NanoVoxMap CUDA packs each ray-event key as an 8-voxel block coordinate plus
    the within-block index. GPU radix sorting now places all updates for a host
    block together, removing the second host comparison sort. Ray traversal,
    hit/miss counts, update order within a voxel, log-odds clamps and worker count
    are unchanged. Frozen multi-frame CPU/CUDA full-map fingerprints verify the
    log-odds state, not only the occupied threshold result.
12. The final objective evaluation aggregates its existing quadrature-node
    extrema for SFC, speed, thrust, tilt and zero-heading flatness body rate.
    Through V14, native planning rejected a positive optimizer status when those
    values were nonfinite or violated SFC by more than 1e-4 m, the configured
    physical speed bound, or another dynamic bound by more than 1e-3
    in its stated SI unit. These are explicit soft-constraint allowed errors,
    not floating-point roundoff or strict physical-limit guarantees. For example,
    valid simple 2 m/s cases retain measured speeds of 2.001714 and 2.001320 m/s;
    an absolute 1e-3 m/s gate rejected them. The CTBR hard bounds are unchanged.
    V15 retains these thresholds as diagnostics, as detailed in item 24. Failure
    snapshots retain actual extrema, bounds, allowed errors and initial PVA.
    Previously rejected finite candidates retain their existing polynomial
    coefficients, durations, piece-to-SFC indices and objective for exact offline
    replay. That copy occurs only on rejection. This does not reintegrate controls or resample a
    second execution trajectory, and is not a continuous-time certificate.
    The former 1e-6 m SFC gate rejected a High-speed trajectory with an exact
    polynomial maximum of 7.557e-6 m (quadrature maximum 7.173e-6 m), with all
    dynamic bounds met. The 0.1 mm numerical feasibility allowance is explicit;
    it does not alter the 1e-6 m position smoothing or label the earlier
    2.992 mm and 51.794 mm references as numerically feasible.
13. `body_yaw_rate_max` bounds the body z angular velocity. The Python heading
    constructor uses the analytic affine relation between that velocity and
    the world heading derivative, with two start/midpoint/end interval
    projections per 0.02 s step. `sample.yaw_rate` remains the world heading
    derivative and can exceed the body limit. Gravity comes from the same
    physical configuration. Independent rotation-matrix finite differences and
    finer offline sampling validate the construction; finite samples do not
    prove a continuous-time bound.
14. Position-penalty smoothing has its own explicit metre-valued setting,
    `position_penalty_smoothing_m=1e-6`. Dynamic penalties retain upstream's
    smoothing value 0.01. The previous shared 0.01 m width left a measured
    2.992 mm SFC violation even with successful L-BFGS stopping. Tighter stopping
    and 100 times larger position weight did not eliminate that residual.
    The smaller position width passed the original SFC for the seven previous
    frozen problems and this new Forest case, without moving the initial PVA,
    changing the corridor, or inflating the robot again. This remains a finite
    soft penalty, with feasibility checked separately at optimizer quadrature
    nodes and full-polynomial SFC extrema measured offline.
15. Expected planning failures use the native `PathCoverPlanningError`, which
    derives from an independently importable Python exception of that name.
    Missing binary interfaces, invalid parameters and nonfinite inputs retain
    their distinct exceptions; A0 imports do not load the native extension.
16. An optional `initial_motion_seed[3,3]` supplies three world points whose
    convex hull is preserved by the first RISP region. CTBR constructs its first
    exact Euler positions from the current state and a bounded first thrust.
    The geometry uses triangle/AABB separating axes and adds no obstacle padding.
    All later regions retain point RISP. The generic front end keeps point RISP
    when the argument is omitted, and does not infer dynamics or control timing.
    This fixes a measured Dense case where every original corridor excluded the
    unavoidable second Euler node by at least 45.49 mm; the replacement first
    region and unchanged MINCO objective admitted a full fixed-cost N5 solve.
    It is an initial geometry contract, not a certificate of complete MPC
    feasibility. A seed hull intersecting an occupied box fails explicitly.
17. DMP search-region construction fills contiguous X bit ranges for each row of
    its original extruded disk mask. The same `hypot` inclusion test and clipping
    retain the identical region. Four real maps and 588 boundary/configuration
    cases matched bitwise; this optimization changes neither graph nor cost.
18. Fine-to-coarse voxel projection caches repeated Y/Z coordinates and merges
    contiguous X index intervals with identical coarse Y/Z coverage. It retains
    the original floor/ceil boundary convention, occupied cells and sorted index
    order. Three recorded maps, 600 ordered/reversed/shuffled boundary cases and
    1,010,005 scalar boundary inputs matched the original projection. Reported
    projection-only timings exclude inflated-cloud serialization; `grid_ms`
    continues to include the complete native grid update.
19. The 3D GCOPTER wrapper restores the L-BFGS library's standard gradient
    convergence threshold, `g_epsilon=1e-5`. With the upstream override of zero,
    a real one-piece Forest problem exhausted 64 line-search trials near its
    stationary point. An independent 80-digit quintic jerk-energy calculation
    agreed with the native gradient within 4.53e-13. The next true objective
    improvement was 1.07e-14, below a double-precision ULP. The standard criterion
    returns convergence at the same accepted point; failed iterates are not
    relabeled as success. The previous 14 successful frozen problems retained
    identical coefficients and objectives.
20. `nominal_velocity` supplies the existing optimizer speed penalty and initial
    allocation; `None` resolves to `max_velocity`. `max_velocity` independently
    specifies the physical reference speed bound. The acceptance limit is
    `1.001 * max_velocity`, with no dependence on measured initial speed.
    The navigation factories use nominal 6 m/s and the existing scene/CTBR
    physical limits (High-speed 10, Dense 8, Newton 10 m/s). There is no added
    physical-speed objective term. Successful-plan metadata and failure snapshots
    retain nominal, physical and acceptance limits, initial speed, actual peak
    and nominal excess. V11/V12 instead used a state-conditioned allowance;
    those archived results retain their original meaning. A nominal exceedance
    is never labeled as compliance with the cruise target. The rule reuses the
    existing optimizer extrema without another trajectory traversal.
21. Before the full 3D optimization, one positive scalar rescales all initial
    durations while preserving waypoint positions and time ratios. Its smooth
    objective is the logarithm of MINCO control energy plus the existing time
    cost. This monotone transformation preserves the positive objective's minima
    and normalizes its gradient for a relative stopping criterion. It uses
    MINCO's energy and time gradients and does not evaluate flatness penalties.
    The log-scale L-BFGS solve has at most 64 iterations and is followed by one
    ordinary full-variable solve. Negative initialization status fails explicitly;
    there are no alternate starts or retry planner. In the initial raw-energy
    screen, the recorded Dense failure changed from an invalid 13,323,050
    objective to a feasible 423.226 objective, with 0.140 ms scalar work and 117
    full iterations. The longer constant-speed
    initialization needed 733 iterations but achieved a lower 204.496 objective;
    the different local-solution quality is retained in the report. Five further
    frozen cases passed full-polynomial SFC and sampled dynamic checks. An A2
    Forest case still took about 90 ms in the full optimizer alone. Initialization
    scale, status, evaluation count and time are exposed in diagnostics.
    The unnormalized scalar objective exhausted line-search precision in a
    synthetic wall case and one derived slow-frame diagnostic: final relative
    gradients were 9.67e-8 and 5.33e-8 while the absolute gradients exceeded
    1e-5. Normalization changes the stopping scale; negative status is never
    relabeled as convergence. The earlier raw-energy reports remain archived.
    The final log objective passed six frozen real problems and the wall
    regression; central finite differences agreed within 5.89e-10 relative error.
    A derived historical-PVA/final-map diagnostic still had a 0.2492 mm full-curve
    SFC violation. It is retained as a negative result, not an original online
    frame or a claim that finite soft penalties guarantee containment.
22. Native V13 runs the same legal DMP graph across the complete bounded map,
    without a preceding JPS guide or guide-region construction. A real Dense
    failure exhausted all 1,260 nodes of a disconnected guide region under both
    weights 1 and 1.5; the full graph had a valid path. The graph's occupied
    voxels, legal diagonal rules and distance-potential costs are unchanged,
    while the allowed search region changes intentionally. Four further frozen
    maps had valid full-graph paths and SFCs, with lower path costs and lower
    search time in that screen. This is the uniform search path, not a fallback.
    Metadata reports `search_backend=global_dmp`, `jps_ran=false`, and zero JPS
    runtime. Original JPS sources and the unmodified-chain oracle remain available.
23. V14 changes the existing `safety_margin` default from 0 to 0.01 m. Only the
    position-penalty copy receives the inward offset; the original SFC, waypoint
    parameterization, occupied map, robot radius and acceptance tolerance remain
    unchanged. The existing per-polytope/adjacent-overlap radius caps preserve
    interior in the penalty copy. This is reference standoff in a soft objective,
    not another application of physical body clearance or a hard guarantee.
    A recorded V13 Forest failure had a 2.998366 mm full-polynomial original-SFC
    violation. Changing only this parameter produced 1.684191 mm of original-SFC
    clearance with all sampled dynamic bounds met. The initialization scale was
    unchanged; the objective rose from 671.858 to 688.116. Single-run optimization
    timing does not establish online p95. The original candidate and its exact
    baseline reconstruction remain archived.
24. V15 restores the official drone YAML's `SafetyMargin=0.15 m`, retaining
    `weight_pos=1e4`, and removes the port's additional rejection of finite
    references solely for soft-constraint residuals. The official trajectory
    server accepts a finite optimized cost with a nonempty trajectory; this
    acceptance rule is distinct from proof of full-curve feasibility. Negative
    optimizer status, nonfinite costs/evaluations and empty candidates still fail.
    Polynomial and yaw construction retain their finite/defined checks.
    `optimizer_quadrature_feasible` records the previous numerical thresholds
    using the existing extrema, including the zero-heading rate. It neither
    certifies continuous containment nor the final heading's complete rate bound.
    The legacy `speed_acceptance_limit_m_s` field remains a quality-diagnostic
    threshold for historical reports. `reference_margin_requested_m` and
    `reference_margin_applied_per_corridor_m` expose the existing margin results;
    no new LP or trajectory traversal is introduced. MPC retains its own hard
    prediction-node constraints. Offline validation and its tolerances remain
    unchanged: the recorded 0.15 m comparison had original-SFC clearance but a
    sampled body-q peak of 3.001876 rad/s, above the 3.001 rad/s offline allowance.
    It remains a failed full-trajectory check. No previous-plan fallback or
    recovery state machine is added. The official YAML's margin is 0.15 m;
    the ROS node's separate missing-parameter default is 0.3 m.
25. V16 skips only exact zero-gradient work in the 3D penalty loop. With finite
    forward quantities and zero thrust/attitude/body-rate gradients, flatness
    backward reduces to passing the position/velocity gradients and zeroing the
    acceleration/jerk gradients. If those gradients and the penalty value are
    also exactly zero, their coefficient/time/cost increments are omitted.
    Forward flatness, penalties and every quadrature quality metric are retained;
    nonfinite forward results use the original backward path. The comparisons
    use exact zero, not a small-gradient threshold. The V14 Forest frozen problem
    reproduced all 224 objective/gradient evaluations and its final polynomial
    bitwise. In 20 alternating warm CPU pairs, optimizer medians changed from
    40.564 to 31.739 ms (21.75% less); this excludes setup and is not an online
    planning-p95 claim. Integration remains 64, reference margin remains 0.15 m,
    and all costs, diagnostics and optimizer stopping conditions are unchanged.
26. V17 projects the inflater's read-only logical voxel bit-grid directly into
    the search grid. Shared fine/coarse boundaries are constructed from global
    integer lattice indices; strict interval overlap is precomputed per axis.
    Commensurate resolution ratios are recovered once within four ULPs of the
    configuration division, with denominator at most 1,000,000; unsupported
    ratios report an invalid parameter. Projection then uses integer floor/ceil,
    including .1-to-.3 shared boundaries, without a world-coordinate epsilon.
    A touching face, edge or corner does not occupy the neighbouring coarse cell.
    The previous packed-float XYZ round trip moved a real Dense fine cell's two
    touching faces outward by 4.77e-8 m and falsely occupied an entire 0.25 m
    search cell. The same frozen map had 4,355 such extra coarse cells compared
    with an independent integer 2/5-resolution oracle. No radius, inflation
    kernel, physical workspace or collision-check threshold changes. Virtual
    boundary occupancy remains separate and persistent. Diagnostic
    `inflated_cloud()` retains the original float serialization on request;
    normal map updates no longer serialize or allocate/cast a full XYZ table.
27. V17 restores a coherent official drone reference preset: nominal speed
    3 m/s, tilt 0.7854 rad, velocity penalty weight 100 and reference thrust
    bounds (2, 10) N scaled by mass/0.716 kg. Newton and MARSIM factories
    intersect the reference thrust range with the unchanged CTBR physical range.
    Physical speed limits remain independent (High-speed 10, Dense 8, Newton
    10 m/s). CLI, ROS parameters and launch defaults use the same nominal speed.
    Integration remains 64, position smoothing 1e-6 m, reference margin 0.15 m
    and the existing time initialization is retained. Explicit historical
    pressure-test values remain configurable and stored inputs are not rewritten.
    Defaults resolve to concrete values in newly saved configurations. On the
    frozen V16 high-speed upward state, the preset removed the heading failure
    but still exceeded the z<=4.8 m workspace by 0.422056 m; that result does
    not establish complete-trajectory feasibility or performance from rest.
28. V17 constructs heading by clipping body-r pointwise and integrating the
    scalar yaw ODE in native C++. RK4 compares a full step with two half steps
    at 1e-7 rad local angle tolerance; cubic Hermite interpolation checks its
    local body-r derivative defect at 1e-4 rad/s. Base steps remain 20 ms and
    refinement is limited to 20 levels. Heading feedback fades continuously
    with min(horizontal_speed/0.1, 1), replacing the discontinuous low-speed
    switch and the previous shared constant-rate interval requirement. These
    local error checks do not certify full-curve dynamics or SFC containment.
    Nonfinite/downward force, undefined heading and integration nonconvergence
    remain explicit planning errors. Historical recorded linear-yaw profiles
    retain their original interpolation. If heading construction fails, the
    wrapper retains the existing native result; only an explicit diagnostic
    getter copies its candidate coefficients and corridor ownership. The old
    near-freefall reference remains dynamically invalid despite this heading
    repair: the isolated study measured body-p/q peaks of 161.76/33.61 rad/s.
29. V18 replaces the shortest-angle heading feedback by `sin(heading-yaw)`.
    The exact V17 Dense failure crossed the plus/minus-pi branch inside a
    smooth polynomial segment. Sine feedback removes this right-hand-side
    jump while retaining the small-error gain, low-speed fade, body-r clipping
    and integration tolerances. The antipodal equilibrium is unstable; this
    is not a global convergence claim. The exact candidate agrees with an
    independent DOP853 solution within 2.86e-7 rad and satisfies the unchanged
    per-axis rate tolerances. Its original zero-heading diagnostic is retained.
30. V19 separates coarse search from authoritative fine geometry. On each changed
    map, contiguous integer X runs are merged along Y only when their X intervals
    match, then along Z only when the entire XY rectangles match. The resulting
    boxes have exactly the same occupied union as the inflater's fine bit-grid;
    they never fill a gap. RISP receives their centers and independent three-axis
    half widths, including when search and mapping resolutions are equal.
    The cloud-only coarse list omits already occupied
    virtual-boundary cells and is therefore not reused as authoritative geometry.
    `obstacle_boxes()` exposes the fine SFC volumes; `search_obstacle_boxes()`
    separately exposes the coarsened cloud volumes. Exact physical bounds are
    `bounds_lower + body_radius` and `bounds_upper - body_radius`; only search
    retains the previous inward grid rounding and virtual occupancy.
    The real V18 Dense warm seed was 0.140239 m outside all inflated fine boxes,
    yet its p2 entered a valid coarse cover cell. That cell had seven fine boxes
    with positive-volume overlap, so this was not the earlier floating-point bug.
    The three-point initial seed and the original Euler CTBR model are retained.
    Fine-free endpoints inside occupied coarse cells connect through a fixed
    local fine-grid search. `endpoint_connection_radius=1.0` is the half-width
    in metres of an axis-aligned box about each endpoint, clipped to the exact
    physical workspace; it is not a Euclidean-ball radius. Six-axis Dijkstra
    exhausts that finite window, including routes through multiple free coarse
    components. Virtual endpoint links can have unequal lengths. Each reachable
    coarse anchor retains a shortest local-grid route, whose turns are simplified
    only when an analytic closed-fine-AABB slab test validates the replacement.
    Face, edge and corner contacts remain blocked. Coarse occupancy is unchanged.
    Multi-anchor DMP performs one weighted search, adding connector lengths in
    its existing grid-cost units and charging the original start-cell potential.
    A virtual goal must be popped before success; reaching the first goal anchor
    is insufficient. Normal coarse-free endpoints still use the original DMP
    entry point. The chosen two local links and coarse path form the RISP guide.
    No anchor within the fixed window or an exhausted global OPEN set produces an
    explicit planning failure. There is no radius expansion, search retry or
    previous-plan fallback. This finite mixed graph is not claimed complete for
    every continuous fine-map route, and weighted search is not claimed optimal.
    RISP's dual-Qhull redundancy removal uses a strict interior point, solving
    the existing interior LP only when the original seed is on a boundary.
    Removing redundant faces preserves the convex set, not the sum of soft face
    penalties at infeasible GCOPTER trial points. Box grouping can also change
    RISP's separating-plane choices. New corridors and optimized trajectories
    are therefore checked against the original fine occupancy; numerical
    equality with the prior coarse corridors is not claimed. No ROI boundary,
    larger MPC face capacity or extra body clearance is introduced.
31. Coarse occupancy reuses the exact integer boxes already constructed for RISP.
    Each upper-exclusive box is projected with the same rational-resolution
    integer floor/ceil bounds as the prior per-voxel projection. Positive-volume
    intersection distributes over an exact occupied union, so this skips a
    second fine-bit scan without changing any coarse occupied cell. Existing
    virtual occupancy and cloud-only reset ownership are unchanged. Sorting the
    coarse cloud indices and materializing their centers now happen only in the
    diagnostic getter, retaining its original sorted output. A frozen Dense V19
    map with 2,218,841 fine cells and 62,280 exact boxes produced identical coarse
    bytes, potential bytes, diagnostic centers, DMP raw path and path cost. The
    integer boundary regression also compares both implementations against an
    independent overlap oracle, including shared faces, negative indices and
    unequal commensurate resolutions. No change to inflation or padding is made.
32. DMP skips CLOSED successor candidates before computing unused transition
    work, stores the same 26 directions, step lengths and corner masks contiguously,
    and initializes the full search arrays once in `resetSearch`. Direction order,
    occupancy rules, weighted heuristic, edge costs and no-reopen policy are
    unchanged. The native chain calls explicit `computeGlobalRawPath` overloads
    for ordinary and multi-anchor endpoints, avoiding simplified-path work whose
    result it never consumes. These overloads clear `getPath` and fill only
    `getRawPath`; the original `computeGlobalPath` retains its public simplified
    output. Across 36 two-/three-dimensional global, regional and multi-anchor
    cases, path cost bits, raw waypoint order, selected anchors and all discovered
    state metadata match the pre-change implementation.
33. Packed-grid box compression matches each ordered X interval against the
    immediately preceding row with two cursors. Only identical intervals extend
    a Y rectangle; empty rows and new Z planes reset the active row. This removes
    the first global sort of X runs. The complete second sort and Z merge are
    unchanged, preserving every final box and its ordering, which also preserves
    RISP's input order. The unordered-cell overload retains the original sorted
    construction as an independent reference. Boundary and randomized-row tests
    compare exact unions and ordered boxes; the 2,218,841-cell Dense V19 map
    reproduces all 62,280 archived boxes, the fine bitmap and the coarse bitmap.
34. The CUDA distance-map update also derives one uint32 legal-move mask per
    coarse cell from the original occupancy on the same stream. A 3D diagonal
    remains legal only when every required axis-subset neighbour is free; map
    bounds and virtual occupied cells remain blocking. This preprocesses the
    existing geometry test, without using potential values or CLOSED state.
    Potential and masks are installed together; DMPlanner owns their storage.
    CPU `setMap` and the original two-argument potential setter clear old masks.
    GraphSearch reads mask bits in the original direction order, retaining
    regional restrictions, CLOSED handling, costs, heap order and the virtual
    goal rule. Two-dimensional and unmasked CPU searches retain the original
    geometry evaluation. CPU/GPU masks and complete discovered-state metadata
    agree on the isolated contracts and the frozen 369,600-cell Dense map.
    `gpu_legal_moves` reports the mask actually installed for a searched plan,
    including reused map frames. Preprocessing and transfer remain part of
    `distance_map_ms`; no extra kernel synchronization or timing leaf is added.

GCOPTER's continuous containment and dynamic bounds remain soft penalties.
The offline validation measures their actual violations. No duplicated online
execution-trajectory reintegration or recovery state machine is added.

NanoVoxMap keeps its authoritative map on the host. Its CUDA backend accelerates
ray-event generation/reduction; inflation accelerates dilation. The added DMP
potential field kernel also runs on CUDA. JPS/DMP graph search, RISP, Qhull and
MINCO remain native CPU algorithms. CUDA build/use and per-stage timing
are exposed in the result rather than inferred from configuration.
