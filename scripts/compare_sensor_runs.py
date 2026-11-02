"""Compare saved sensor runs, including voxel assignments (not just small point errors)."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old, new = np.load(a.reference / "hits.npz"), np.load(a.candidate / "hits.npz")
    origins = new["env_origins"]
    result = {}
    for key in ("initial", "after"):
        x, y = old[key], new[key]
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y))
        valid = np.isfinite(x).all(-1)
        max_error = float(np.abs(x[valid] - y[valid]).max())
        ix = np.floor(((x - origins[:, None, :])[valid]).astype(np.float32) / np.float32(0.2))
        iy = np.floor(((y - origins[:, None, :])[valid]).astype(np.float32) / np.float32(0.2))
        result[key] = dict(max_abs_error_m=max_error, voxel_bin_mismatches=int(np.any(ix != iy, -1).sum()))
        np.testing.assert_allclose(x, y, atol=1e-4, rtol=1e-5)
        np.testing.assert_array_equal(ix, iy)
    np.testing.assert_array_equal(old["pose"], new["pose"])
    np.testing.assert_array_equal(old["windows"], new["windows"])
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
