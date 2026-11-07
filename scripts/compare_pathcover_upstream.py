"""Compare the full pinned official PathCover kernel chain on recorded first frames.

The reference uses unmodified official CPU sources. Intentional geometry and
optimization changes are reported separately; this is not an acceleration ratio.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linprog
from validate_pathcover_native import PIN

from isaaclab_ipc.pathcover_native import (
    NativePathCoverFrontend,
    PathCoverConfig,
    PathCoverSample,
    PathCoverSegment,
    PathCoverTrajectory,
    validate_pathcover_plan,
)


def build_reference(root, upstream, folder, jobs):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected official PathCover {PIN}, found {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("The independent official checkout must be unmodified")
    folder.mkdir(parents=True, exist_ok=True)
    wrapper = root / "scripts/pathcover_reference"
    includes = [
        upstream / "src/mapping/nanovoxmap_ros/nanovoxmap/include",
        upstream / "src/mapping/occupancy_inflation/include",
        upstream / "src/third_party/jps_lib/include",
        upstream / "src/corridor_planning/include",
        upstream / "src/trajectory_server/include",
        Path("/usr/include/eigen3"),
    ]
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-DPATHCOVER_DIM=3", "-fopenmp", "-pthread",
             *[f"-I{path}" for path in includes]]
    sources = [
        wrapper / "official_chain.cpp",
        *[upstream / "src/third_party/jps_lib/src" / sub / name
          for sub in ("jps_planner", "distance_map_planner")
          for name in ("graph_search.cpp", sub + ".cpp")],
        *[upstream / "src/mapping/occupancy_inflation/src" / name
          for name in ("point_cloud_inflater.cpp", "voxel_bit_grid.cpp")],
    ]

    def compile_source(source):
        target = folder / (hashlib.sha256(str(source).encode()).hexdigest()[:16] + ".o")
        if not target.exists() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            subprocess.run(["g++", *flags, "-c", str(source), "-o", str(target)], check=True)
        return target

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects = list(pool.map(compile_source, sources))
    oracle = folder / "official_chain"
    subprocess.run(["g++", "-fopenmp", "-pthread", *map(str, objects), "-lqhullcpp", "-lqhull_r",
                    "-o", str(oracle)], check=True)
    mapping = folder / "ported_mapping"
    source = wrapper / "ported_mapping.cpp"
    include = root / "third_party/pathcover_native/nanovoxmap/include"
    latest = max(source.stat().st_mtime_ns, *(path.stat().st_mtime_ns for path in include.rglob("*.hpp")))
    if not mapping.exists() or mapping.stat().st_mtime_ns < latest:
        subprocess.run(["g++", "-O3", "-DNDEBUG", "-std=c++17", "-fopenmp", "-pthread",
                        "-I/usr/include/eigen3", f"-I{include}", str(source), "-o", str(mapping)], check=True)
    return oracle, mapping


def captured_input(source):
    if source.suffix == ".npz":
        captured = np.load(source)
        state = captured["state"]
        initial = np.column_stack((state[:3], state[7:10], captured["acceleration"]))
        return captured["cloud"], initial, captured["goal"]
    captured = torch.load(source, weights_only=False, map_location="cpu")
    state = captured["current_state"]
    initial = np.column_stack((state["position"], state["velocity"], state["acceleration"]))
    return captured["points_world"], initial, captured["goal"]


def serialized_input(cloud, initial, goal, cfg):
    options = cfg.native_options()
    values = [
        *cfg.bounds_lower, *cfg.bounds_upper, cfg.resolution, options["search_resolution"],
        cfg.robot_radius, options["robot_radius_z"], cfg.max_corridors, cfg.risp_alpha,
        cfg.potential_radius, cfg.search_radius, cfg.random_seed,
        *initial.ravel(order="F"), *goal, len(cloud), *np.asarray(cloud).ravel(),
        cfg.time_weight, cfg.safety_margin, cfg.pieces_per_polytope, cfg.integral_resolution,
        cfg.relative_tolerance, cfg.max_iterations, cfg.native_options()["nominal_velocity"],
        cfg.body_rate_max - cfg.reference_body_rate_margin,
        cfg.tilt_max, cfg.nominal_thrust_min, options["thrust_max"], *cfg.penalty_weights,
        cfg.mass, cfg.gravity, 0.0, 0.0, 0.0, 0.001,
    ]
    return " ".join(map(str, values)) + "\n"


def sorted_rows(points):
    return points[np.lexsort(points.T[::-1])]


def raw_polynomial_plan(coefficients, durations, corridors, pieces):
    boundaries = np.r_[0.0, np.cumsum(durations)]
    trajectory = SimpleNamespace(coefficients=coefficients, durations=durations,
                                 duration=float(boundaries[-1]), _boundaries=boundaries)

    def sample(times):
        values = PathCoverTrajectory._derivatives(trajectory, times)
        return PathCoverSample(*values, np.zeros_like(times, dtype=float), np.zeros_like(times, dtype=float))

    segments = tuple(PathCoverSegment(i // pieces, float(boundaries[i]), float(boundaries[i + 1]), float(t))
                     for i, t in enumerate(durations))
    return SimpleNamespace(trajectory=trajectory, corridors=corridors, segments=segments,
                           sample=sample, metrics={"optimizer_status": 1})


def overlap_radius(a, b):
    h = np.vstack((a, b))
    result = linprog([0.0, 0.0, 0.0, -1.0], A_ub=np.column_stack((h[:, :3], np.ones(len(h)))),
                     b_ub=-h[:, 3], bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
    return float(result.x[-1]) if result.success else None


def minimum_overlap(corridors):
    values = [overlap_radius(a, b) for a, b in zip(corridors[:-1], corridors[1:], strict=True)]
    return min(values) if values and all(value is not None for value in values) else None


def closed_box_contacts(path, cells, origin, resolution):
    occupied = set(map(tuple, cells))
    contacts, penetrations = 0, 0
    for a, b in zip(path[:-1], path[1:], strict=True):
        delta = b - a
        if np.linalg.norm(delta) < 1e-12:
            continue
        lo = np.floor((np.minimum(a, b) - origin) / resolution - 1e-9).astype(int)
        hi = np.floor((np.maximum(a, b) - origin) / resolution + 1e-9).astype(int)
        for cell in itertools.product(*(range(x, y + 1) for x, y in zip(lo, hi, strict=True))):
            if cell not in occupied:
                continue
            lower = origin + resolution * np.asarray(cell)
            upper = lower + resolution
            enter, leave = 0.0, 1.0
            for axis in range(3):
                if abs(delta[axis]) < 1e-12:
                    if a[axis] < lower[axis] - 1e-9 or a[axis] > upper[axis] + 1e-9:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    t0, t1 = sorted(((lower[axis] - a[axis]) / delta[axis], (upper[axis] - a[axis]) / delta[axis]))
                    enter, leave = max(enter, t0), min(leave, t1)
            if enter <= leave + 1e-9:
                contacts += 1
                penetrations += leave - enter > 1e-8
    return {"closed_aabb_contacts": contacts, "positive_parameter_interval_contacts": int(penetrations)}


def parse_original(path):
    tagged, faces = {}, {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key, *values = parts
        if key == "FACE":
            faces.setdefault(int(values[0]), []).append(list(map(float, values[1:])))
        elif key in ("OCCUPIED", "INFLATED", "PATH", "SEED", "PIECE", "OBJECTIVE"):
            tagged.setdefault(key, []).append(list(map(float, values)))
    return ({key: np.asarray(value) for key, value in tagged.items()},
            tuple(np.asarray(faces[i]) for i in sorted(faces)))


def analyze_case(folder, name, metadata, ported, cloud, cfg):
    original, corridors = parse_original(folder / (name + ".stdout"))
    options = cfg.native_options()
    native = NativePathCoverFrontend(replace(cfg, use_cuda=False))
    initial = np.asarray(metadata["initial_pva"])
    native.update_cloud(cloud, initial[:, 0])
    ported_map = np.loadtxt(folder / (name + ".ported-mapping.txt")).reshape(-1, 3)
    equal = np.array_equal(sorted_rows(original["OCCUPIED"]), sorted_rows(ported_map))
    row = {"input": metadata, "mapping_occupied_coordinates_bitwise_equal": equal,
           "occupied_voxels": len(ported_map), "original_returncode": metadata["returncode"],
           "original_inflated_voxels": len(original["INFLATED"]),
           "ported_inflated_voxels": len(native.inflated_cloud())}
    if metadata["returncode"]:
        row["original_diagnostic"] = (folder / (name + ".stderr")).read_text()
        return row
    path, seeds = original["PATH"], original["SEED"]
    row["original"] = {
        "path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "corridors": len(corridors), "faces": [len(h) for h in corridors],
        "covered_distance_m": float(np.linalg.norm(seeds[-1] - path[0])),
        "minimum_overlap_radius_m": minimum_overlap(corridors), "stage_ms": {},
    }
    for line in (folder / (name + ".stderr")).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "TIME":
            row["original"]["stage_ms"][parts[1]] = float(parts[2])
    resolution = options["search_resolution"]
    origin = np.asarray(cfg.bounds_lower)
    cells = np.unique(np.floor((original["INFLATED"] - origin) / resolution).astype(int), axis=0)
    row["original"]["path_volume_contacts"] = closed_box_contacts(path, cells, origin, resolution)
    pieces = original["PIECE"]
    reference = raw_polynomial_plan(pieces[:, 2:].reshape(-1, 3, 6), pieces[:, 1], corridors, cfg.pieces_per_polytope)
    numerics = validate_pathcover_plan(reference, cfg)
    numerics.pop("optimizer_status")  # The unmodified wrapper exposes only cost and trajectory.
    row["original"].update(
        objective=float(original["OBJECTIVE"][0, 0]), numerics_fixed_heading=numerics,
        initial_pva_max_error=float(np.max(np.abs(reference.sample(np.asarray(0.0)).pva - initial.ravel(order="F")))),
    )
    centers, _ = native.obstacle_boxes()
    origin = np.floor(np.asarray(cfg.bounds_lower) / resolution) * resolution
    cells = np.rint((centers - origin) / resolution - 0.5).astype(int)
    reference = raw_polynomial_plan(ported.trajectory.coefficients, ported.trajectory.durations,
                                    ported.corridors, cfg.pieces_per_polytope)
    row["ported"] = {
        "path_length_m": float(np.linalg.norm(np.diff(ported.path, axis=0), axis=1).sum()),
        "corridors": len(ported.corridors), "faces": [len(h) for h in ported.corridors],
        "covered_distance_m": float(np.linalg.norm(ported.local_goal - ported.path[0])),
        "minimum_overlap_radius_m": minimum_overlap(ported.corridors),
        "path_volume_contacts": closed_box_contacts(ported.path, cells, origin, resolution),
        "objective": ported.metrics["objective"], "numerics_fixed_heading": validate_pathcover_plan(reference, cfg),
        "numerics_native_heading": validate_pathcover_plan(ported, cfg),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, help="Release report containing first_frames")
    parser.add_argument("--ported-plans", required=True, help="Saved plan path pattern with {case}")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    oracle, mapping = build_reference(root, args.upstream.resolve(), folder / "_build", args.jobs)
    validation = json.loads(args.validation.read_text())
    report = {
        "upstream_commit": PIN, "original_kernels_unmodified": True,
        "scope": "Full ROS-free original kernel chain on identical first-frame cloud, PVA and stated scalar parameters",
        "ported_validation": str(args.validation.resolve()),
        "reference_yaw": "Common fixed world heading zero for PVA dynamics; the original ROS yaw controller is not run",
        "performance": "One CPU reference execution per case; no matched CUDA acceleration claim",
        "intentional_differences": [
            "Original centre-distance inflation versus occupied voxel-volume inflation",
            "Original lower-bound grid versus world-zero grid, workspace body clearance and conservative coarsening",
            "Original corner-permitting Dijkstra graph versus legal axis-subset graph and configured weighted A*",
            "Original point RISP/global rand/redundancy removal versus occupied AABB cuts with thread-local RNG",
            "Original off-by-one horizon versus the configured exact corridor count",
            "Original time initialization and shared smoothing versus measured-speed initialization and metre position width",
            "The port may receive an explicit first motion seed; its recorded validation input states that choice",
            "Both receive the recorded quadrature and physical bounds, not necessarily the official YAML defaults",
        ],
        "cases": {},
    }
    for name, record in validation["first_frames"].items():
        source = Path(record["source_input"])
        cloud, initial, goal = captured_input(source)
        config_values = dict(record["config"])
        config_values.setdefault("reference_body_rate_margin", 0.0)
        cfg = PathCoverConfig(**config_values)
        text = serialized_input(cloud, initial, goal, cfg)
        (folder / (name + ".input.txt")).write_text(text)
        with (folder / (name + ".stdout")).open("w") as output, (folder / (name + ".stderr")).open("w") as error:
            result = subprocess.run([str(oracle)], input=text, text=True, stdout=output, stderr=error, timeout=args.timeout)
        metadata = {"source": str(source), "config": record["config"], "returncode": result.returncode,
                    "initial_pva": initial.tolist(), "goal": np.asarray(goal).tolist(), "raw_cloud_points": len(cloud)}
        (folder / (name + ".metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        text = " ".join(map(str, [cfg.resolution, *initial[:, 0], len(cloud), *np.asarray(cloud).ravel()])) + "\n"
        with (folder / (name + ".ported-mapping.txt")).open("w") as output:
            subprocess.run([str(mapping)], input=text, text=True, stdout=output, check=True, timeout=args.timeout)
        plan_path = Path(args.ported_plans.format(case=name))
        ported = torch.load(plan_path, weights_only=False, map_location="cpu")["plan"]
        row = analyze_case(folder, name, metadata, ported, cloud, cfg)
        row["ported_plan"] = str(plan_path.resolve())
        report["cases"][name] = row
        (folder / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, "official return code", result.returncode, "occupied coordinates equal",
              row["mapping_occupied_coordinates_bitwise_equal"], flush=True)


if __name__ == "__main__":
    main()
