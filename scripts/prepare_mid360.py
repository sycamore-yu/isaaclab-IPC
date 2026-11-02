"""Download AME-2's pinned Mid-360 direction table and its GPL-3.0 license."""

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

REVISION = "34b6fe9eae6ff2c599ca6ecffa24c76e9bce8148"
BASE = f"https://raw.githubusercontent.com/leggedrobotics/ame2_minimal/{REVISION}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/sensors"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    urlretrieve(BASE + "/ame2/ame2/sensors/mid360_raydirs.npy", args.output / "mid360_raydirs.npy")
    urlretrieve(BASE + "/LICENSE", args.output / "LICENSE.ame2")
    table = np.load(args.output / "mid360_raydirs.npy", allow_pickle=False)
    if table.ndim != 2 or table.shape[1] != 3 or len(table) < 20000 or not np.isfinite(table).all():
        raise ValueError("Downloaded Mid360 table is not a finite [N,3] table")
    (args.output / "source.json").write_text(
        json.dumps(
            dict(
                repository="leggedrobotics/ame2_minimal", revision=REVISION, shape=list(table.shape), license="GPL-3.0"
            ),
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
