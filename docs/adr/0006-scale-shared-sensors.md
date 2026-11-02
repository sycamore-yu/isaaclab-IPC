---
status: accepted
---

# Share read-only geometry and ray tables, not environment observations

The user approved the AME-2 scale optimization and validation before local merge.
This extends ADR-0005: a project-local Newton MultiMesh implementation may fuse ray transforms and
queries to avoid per-environment ray copies. Official target ownership, pose tracking, update/reset
and data interfaces remain authoritative. Installed Isaac Lab files are not patched.

Forest geometry/BVH and the scan table are immutable shared resources. Each environment keeps its
own pose, scan window, hit buffer, observed-map history, planning state and PPO context. Geometry
sharing does not supply unseen points to a planner or merge physics environments.

Full-PCD geometric collision, QP constraints/precision, control rates and sensor protocol are unchanged.
The ordinary MultiMesh path remains a diagnostic reference, not a silent fallback.

Implementation and validation: [scale optimization](../scale-optimization.md).
The default planner remains single-threaded with exact caching; bounded workers are opt-in after
observing increased warm-cache median latency despite lower aggregate/tail latency.
