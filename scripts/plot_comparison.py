"""Plot recorded trajectories; geometry thinning is display-only."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results")
records = json.loads((root / "reference-20261102/records.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 2.8), layout="constrained")
for ax, case, folder, title in zip(
    axes,
    ["medium-close", "forest"],
    ["native-medium-final-02", "native-forest-final"],
    ["Sudden obstacle: medium-close", "Original rc_forest map"],
    strict=True,
):
    source = next(r for r in records if r["case"] == case and r["repeat"] == 0)
    original = np.loadtxt(Path(source["evidence"]) / "trajectory.csv", delimiter=",", skiprows=1)[:, 1:4]
    native = np.load(root / folder / "trace-0.npz")["trace"][:, 1:4]
    if case == "forest":
        points = np.load("build/forest/points.npy")
        points = points[
            (points[:, 0] > -1)
            & (points[:, 0] < 9)
            & (abs(points[:, 1]) < 2)
            & (points[:, 2] > 0.2)
            & (points[:, 2] < 2.0)
        ]
        ax.scatter(points[::5, 0], points[::5, 1], s=1, color="#b6c2ad", label="PCD points (display subset)")
    else:
        ax.add_patch(plt.Rectangle((0, 0.8), 10, 0.2, color="#808080", alpha=0.4, label="Wall"))
        ax.add_patch(plt.Rectangle((6.9, -0.1), 0.2, 0.2, color="#303030", label="Final obstacle"))
        ax.add_patch(plt.Circle((7, 0), 0.35, color="#dd9944", alpha=0.16))
    ax.plot(original[:, 0], original[:, 1], color="#277d63", label="Original Docker")
    ax.plot(native[:, 0], native[:, 1], color="#d35f3c", label="Newton / MJWarp")
    ax.scatter(native[-1, 0], native[-1, 1], marker="x", color="#d35f3c", s=70)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.2, 10.2 if case != "forest" else 8.5)
    ax.set_ylim(-1.1, 1.2)
    ax.set_aspect("equal", adjustable="box")
fig.savefig(root / "trajectory-comparison.png", dpi=180, bbox_inches="tight")
