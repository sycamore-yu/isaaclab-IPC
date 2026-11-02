# AME-2 scale optimization

The user approved implementation on a new feature branch, validation, and local merge into main.
Optimize the existing IPC task rather than changing its algorithm or sensor protocol.

## Chosen approach

Keep Isaac Lab's MultiMesh target discovery, per-environment transforms, sensor scheduling and reset.
Reuse one local-space Warp Mesh/BVH for targets explicitly declared identical across environments.
Replace repeated per-ray start/direction buffers with a shared Mid-360 table, per-environment window
indices, and a fused Warp query. This is the specifically approved exception to ADR-0005's original
no-custom-query rule. Do not modify installed packages or unrelated experiments.

Keep separate environments and separate observed maps; sharing geometry is not sharing observations.
Batch point-cloud transfers, parallelize independent CPU planners with bounded workers, and cache
identical corridor queries only within an unchanged observed map. Preserve deterministic ordering.
Store unique SFC blocks plus their time-step indices; reconstruct the exact float64 snapshot for MPC.

## Alternatives

Only changing `is_shared=True` does not work in the installed 3.0 implementation. Copying all of
AME-2 would bypass current MultiMesh target ownership and 3.0 APIs. Sharing one physical environment
or giving planners full-map truth changes task semantics. The chosen extension avoids those changes.

## Acceptance

- Mid-360: 20,000 rays, 10 Hz, 40 m, random contiguous windows, body-origin/axis-aligned mounting.
- Newton physics 300 Hz, commands 100 Hz; original QP precision, constraints and recovery unchanged.
- Full PCD collision test and 0.25 m sphere remain independent of sensing.
- Identical rays/poses yield equivalent official/optimized hits, including misses, rotations,
  dynamic targets, environment transforms and partial resets.
- Identical forest copies share one Mesh ID; raw memory allocation no longer scales as E copies.
- Snapshot compression is lossless and immutable, including shuffled PPO minibatches and recording.
- Measure small paired tests and scale capacity up to 256 when resources permit; report contention.
- Existing dirty code is separately snapshotted; experiments, records and user local skills survive.

Coplanar voxel merging and GPU voxel aggregation are candidates, not assumed wins: enable only with
geometry/map parity and end-to-end evidence. Preserve a measured official reference for comparison.
