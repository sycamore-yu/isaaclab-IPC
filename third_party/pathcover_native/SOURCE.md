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
   beyond the cut. Each separating cut is retained, providing a direct
   one-face-per-box exclusion certificate instead of a centre-only approximation.
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
   supplies body clearance once. V14 gives GCOPTER's reference position penalty
   a separate 0.01 m inward offset, as detailed in item 23 below.
   Physical workspace bounds, including virtual ground/ceiling, are shrunk by
   the body radii once and rounded inward to complete free search cells. This
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
   JPS/DMP and RISP AABB geometry. NanoVoxMap and inflation retain `resolution`.
   Search and SFC use identical occupied volumes, so a trajectory cannot enter
   a cell that the next search classifies as occupied while satisfying its SFC.
   No start cell is cleared and no fine-to-coarse connector is silently inserted.
   The coarse grid
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
    Native planning rejects a positive optimizer status when those values are
    nonfinite or violate SFC by more than 1e-4 m, the configured physical speed
    acceptance bound, or another dynamic bound by more than 1e-3
    in its stated SI unit. These are explicit soft-constraint allowed errors,
    not floating-point roundoff or strict physical-limit guarantees. For example,
    valid simple 2 m/s cases retain measured speeds of 2.001714 and 2.001320 m/s;
    an absolute 1e-3 m/s gate rejected them. The CTBR hard bounds are unchanged.
    Failure snapshots retain the actual extrema, bounds, allowed errors and
    initial PVA. A rejected finite candidate also retains its existing polynomial
    coefficients, durations, piece-to-SFC indices and objective for exact offline
    replay. That copy occurs only on rejection. This does not reintegrate controls or resample a
    second execution trajectory, and is not a continuous-time certificate.
    The former 1e-6 m SFC gate rejected a High-speed trajectory with an exact
    polynomial maximum of 7.557e-6 m (quadrature maximum 7.173e-6 m), with all
    dynamic bounds met. The 0.1 mm numerical feasibility allowance is explicit;
    it does not alter the 1e-6 m position smoothing or accept the earlier
    2.992 mm and 51.794 mm failures.
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

GCOPTER's continuous containment and dynamic bounds remain soft penalties.
The offline validation measures their actual violations. No duplicated online
execution-trajectory reintegration or recovery state machine is added.

NanoVoxMap keeps its authoritative map on the host. Its CUDA backend accelerates
ray-event generation/reduction; inflation accelerates dilation. The added DMP
potential field kernel also runs on CUDA. JPS/DMP graph search, RISP, Qhull and
MINCO remain native CPU algorithms. CUDA build/use and per-stage timing
are exposed in the result rather than inferred from configuration.
