"""Prepare the original PCD as a native point map and a ray-only voxel mesh."""

import argparse
import json
from pathlib import Path

import numpy as np

from isaaclab_ipc.scenes import read_pcd


def surface(voxels, leaf=0.1):
    # Remove internal voxel faces; no change to the occupied voxel union.
    v = np.asarray(voxels, dtype=np.int32)
    origin = v.min(0)
    dims = v.max(0) - origin + 3

    def keys(p):
        q = p - origin + 1
        return (q[:, 0].astype(np.int64) * dims[1] + q[:, 1]) * dims[2] + q[:, 2]

    occupied = keys(v)
    vertices = []
    faces = []
    offset = 0
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for direction in [-1, 1]:
            delta = np.eye(3, dtype=int)[axis] * direction
            exposed = v[~np.isin(keys(v + delta), occupied)]
            corner = np.zeros((4, 3), int)
            corner[:, axis] = int(direction > 0)
            corner[:, others] = [[0, 0], [1, 0], [1, 1], [0, 1]]
            points = (exposed[:, None, :] + corner) * leaf
            tri = np.array([[0, 1, 2], [0, 2, 3]])
            vertices.append(points.reshape(-1, 3))
            faces.append((np.arange(len(exposed))[:, None, None] * 4 + tri + offset).reshape(-1, 3))
            offset += len(exposed) * 4
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcd", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    points = read_pcd(a.pcd)
    # Entire original map remains the independent collision authority.
    np.save(a.output / "points.npy", points)
    voxels = np.unique(np.floor(points / 0.1).astype(np.int32), axis=0)
    vertices, faces = surface(voxels)
    from pxr import Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(a.output / "forest.usdc"))
    root = UsdGeom.Xform.Define(stage, "/Forest")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Forest/surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([(0.2, 0.38, 0.17)])
    stage.GetRootLayer().Save()
    (a.output / "source.json").write_text(
        json.dumps(
            {
                "pcd": str(a.pcd.resolve()),
                "points": len(points),
                "voxel_leaf_m": 0.1,
                "vertices": len(vertices),
                "triangles": len(faces),
                "collision": "point-distance termination; no rigid contact response against foliage",
                "sensor": "Mid-360 directions, MultiMesh first ray hit of 0.1 m occupied voxels",
            },
            indent=2,
        )
    )
    print("points", len(points), "triangles", len(faces), flush=True)


if __name__ == "__main__":
    main()
