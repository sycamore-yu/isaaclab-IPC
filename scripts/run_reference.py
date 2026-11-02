"""Run unchanged original IPC in isolated Docker; all output stays in this project."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

CASES = [
    ("slow-close", "avoid", 10, 1.0),
    ("medium-close", "avoid", 20, 1.0),
    ("fast-far", "avoid", 42, 2.0),
    ("fast-near", "avoid", 42, 0.5),
    ("forest", "forest", 6, 0.0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ref = args.reference.resolve()
    (out / "runs").mkdir()
    records = []
    for name, kind, stride, distance in CASES:
        for repeat in range(args.repeats):
            private = out / f"config-{name}-{repeat}"
            shutil.copytree(ref / "IPC/config", private)
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                str(__import__("os").getuid()) + ":" + str(__import__("os").getgid()),
                "-e",
                "HOME=/tmp",
                "-e",
                "IPC_ROS_PORT=11366",
                "-e",
                "IPC_TIMEOUT=35",
                "-e",
                f"IPC_REF_DIS={stride}",
                "-e",
                f"IPC_OBS_DIS={distance}",
                "-v",
                f"{ref}:/work:ro",
                "-v",
                f"{out}/runs:/work/repro/results",
                "-v",
                f"{private}:/work/IPC/config",
                "-w",
                "/work",
                "ipc-hku-mars:noetic",
                "bash",
                f"repro/run_{kind}.sh",
            ]
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=100)
            (out / f"{name}-{repeat}.log").write_text(result.stdout)
            match = re.search(r"Evidence: repro/results/(\S+)", result.stdout)
            folder = out / "runs" / match[1] if match else None
            metric = (
                json.loads((folder / "metrics.json").read_text())
                if folder and (folder / "metrics.json").exists()
                else {}
            )
            record = {
                "case": name,
                "repeat": repeat,
                "returncode": result.returncode,
                "evidence": str(folder),
                "metrics": metric,
            }
            records.append(record)
            (out / "records.json").write_text(json.dumps(records, indent=2))
            print(name, repeat, result.returncode, metric.get("passed"), flush=True)
    (out / "complete.json").write_text(
        json.dumps({"complete": True, "expected": len(CASES) * args.repeats, "completed": len(records)}, indent=2)
    )


if __name__ == "__main__":
    main()
