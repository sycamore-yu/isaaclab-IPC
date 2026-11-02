"""Independent all-point clearance bounds for the original forest trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from isaaclab_ipc.scenes import read_pcd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcd", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    a = p.parse_args()
    points = read_pcd(a.pcd)
    tree = cKDTree(points)
    results = []
    for record in json.loads((a.campaign / "records.json").read_text()):
        if record["case"] != "forest":
            continue
        folder = Path(record["evidence"])
        trace = np.loadtxt(folder / "trajectory.csv", delimiter=",", skiprows=1)
        distances, _ = tree.query(trace[:, 1:4])
        max_step = np.linalg.norm(np.diff(trace[:, 1:4], axis=0), axis=1).max()
        result = {
            "repeat": record["repeat"],
            "points": len(points),
            "samples": len(trace),
            "minimum_distance_m": float(distances.min()),
            "sample_clearance_m": float(distances.min() - 0.25),
            "continuous_polyline_clearance_lower_bound_m": float(distances.min() - 0.25 - max_step / 2),
            "goal_crossed": record["metrics"]["goal_crossed"],
        }
        result["success"] = bool(result["goal_crossed"] and result["continuous_polyline_clearance_lower_bound_m"] > 0)
        results.append(result)
    (a.campaign / "forest-clearance.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
