# Independent full-chain PathCover reference

This CPU wrapper calls unmodified source at official commit
`8723a05d1993bf0199a453b47dc6ee167d493c23`.
It checks both the commit and a clean checkout before compiling.
The clone is read only.

The chain is NanoVoxMap, occupancy inflation, JPS, DMP, PathCover/RISP and
GCOPTER/MINCO. The wrapper follows the algorithm calls in the upstream ROS
planner without linking ROS, PCL or Gazebo. It does not execute the original
ROS yaw controller. Continuous PVA constraints are compared with a common
fixed world heading; the port's actual heading is reported separately.

Build dependencies are a C++17 compiler, Eigen, Boost headers, OpenMP and Qhull.
Python uses the project environment, NumPy, SciPy and PyTorch.

From the repository root, reproduce the recorded V9 comparison:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/compare_pathcover_upstream.py \
  --upstream build/pathcover-upstream \
  --validation results/acmpc-native-pathcover-v1/pathcover-v9-release-validation.json \
  --ported-plans 'results/acmpc-native-pathcover-v1/pathcover-v9-{case}-firstframe.pt' \
  --output results/pathcover-official-chain-replay
```

The input report identifies each nonempty observed cloud, initial PVA and scalar
configuration. The saved ported plan keeps its original trajectory and SFC.
The output retains serialized inputs, official stdout/stderr, mapping outputs,
per-case metadata and `comparison.json`.

The official kernels retain their centre-distance inflation, grid origin,
corner-permitting DMP, point RISP, original horizon behavior, time initialization
and position smoothing. The comparison reports those intentional differences.
Both versions receive the recorded physical bounds and quadrature setting;
these are not necessarily all default values from the official drone YAML.
Different covered endpoints make their objectives unsuitable as an optimality
ranking. Single CPU stage timings are not CUDA speedup measurements.

The published V9 result is
`results/acmpc-native-pathcover-v1/pathcover-v9-official-chain-comparison.json`.
All 3 inputs completed the original chain. The occupied NanoVoxMap coordinates
matched the port exactly. The report records actual path lengths, ordered
polytope overlaps, covered endpoints, polynomial SFC extrema and sampled
physical limits for both versions.
