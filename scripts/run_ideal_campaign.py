"""Repeat the ROS-free ideal tracker on the four published avoid configurations."""

import argparse
import json
from pathlib import Path

from evaluate_ideal import run
from run_reference import CASES

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=10)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
records = []
for case, kind, stride, distance in CASES:
    if kind != "avoid":
        continue
    for repeat in range(a.repeats):
        folder = a.output / f"{case}-{repeat}"
        run(stride, distance, folder)
        records.append({"case": case, "repeat": repeat, "metrics": json.loads((folder / "metrics.json").read_text())})
        (a.output / "records.json").write_text(json.dumps(records, indent=2))
