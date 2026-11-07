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
- Native PathCover owns mapping, JPS/DMP, RISP corridors and GCOPTER/MINCO
  trajectory timing. ROS and Gazebo are not frontend runtime dependencies.
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
- Preserve measured initial PVA even when it exceeds the nominal reference
  speed. Keep the nominal bound in GCOPTER's objective; its speed feasibility
  allowance is `max(nominal_speed, initial_speed) + 0.001 * nominal_speed`.
  Report nominal exceedance separately. This state-conditioned allowance does
  not change the CTBR hard velocity bounds or guarantee immediate deceleration.
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

P3: compare A0 (no frontend), A1 (PathCover/fixed costs), and A2
(PathCover/learned costs) with identical dynamics, sensing and task criteria.
Keep success, collision, timeout and computational failures separately.
