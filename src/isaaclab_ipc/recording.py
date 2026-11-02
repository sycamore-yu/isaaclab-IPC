"""Rerun forest recordings of the actual MPC context and Newton motion."""

from itertools import combinations
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def corridor_edges(planes):
    """Intersect the logged half-spaces without expanding or clipping the SFC."""
    planes = np.asarray(planes, dtype=float).reshape(-1, 4)
    if not np.isfinite(planes).all():
        raise ValueError("Cannot render nonfinite SFC planes")
    if len(planes) < 4:
        return np.empty((0, 2, 3))
    normals, offsets = planes[:, :3], planes[:, 3]
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm == 0):
        raise ValueError("Cannot render a zero-normal SFC plane")
    normals, offsets = normals / norm[:, None], offsets / norm
    triples = np.array(list(combinations(range(len(planes)), 3)))
    matrices = normals[triples]
    nonsingular = np.abs(np.linalg.det(matrices)) > 1e-9
    vertices = np.linalg.solve(matrices[nonsingular], -offsets[triples[nonsingular], None])[..., 0]
    vertices = vertices[(vertices @ normals.T + offsets <= 1e-6).all(1)]
    vertices = np.unique(np.round(vertices, 9), axis=0)
    edges = set()
    for normal, offset in zip(normals, offsets, strict=True):
        ids = np.flatnonzero(np.abs(vertices @ normal + offset) < 1e-6)
        if len(ids) < 3:
            continue
        centered = vertices[ids] - vertices[ids].mean(0)
        axis = centered[0] / np.linalg.norm(centered[0])
        angles = np.arctan2(centered @ np.cross(normal, axis), centered @ axis)
        ids = ids[np.argsort(angles)]
        edges.update(tuple(sorted((a, b))) for a, b in zip(ids, np.roll(ids, -1), strict=True))
    return vertices[np.array(sorted(edges), dtype=int)].reshape(-1, 2, 3)


def forest_blueprint(goal_x):
    overview = rrb.Spatial3DView(
        origin="/world",
        name="Whole route",
        contents=["/world/**", "-/world/sfc/**", "-/world/reference/**", "-/world/prediction/**"],
        eye_controls=rrb.EyeControls3D(
            position=(goal_x / 2, -12, 14), look_target=(goal_x / 2, 0, 1), eye_up=(0, 0, 1)
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="SFC · follow drone",
                eye_controls=rrb.EyeControls3D(
                    position=(3, -8, 6), look_target=(0, 0, 1), eye_up=(0, 0, 1), tracking_entity="/world/drone/body"
                ),
            ),
            rrb.Vertical(
                overview,
                rrb.TimeSeriesView(origin="/metrics", name="Speed / clearance / braking"),
                rrb.TextDocumentView(origin="/status", name="Controller and legend"),
                row_shares=[3, 2, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="sim_time", expanded=True),
        collapse_panels=True,
    )


class ForestRecording:
    def __init__(self, path, env, policy):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if env.num_envs != 1 or env.cfg.scenario != "forest":
            raise ValueError("Forest recording requires one forest environment")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env, self.policy = env, policy
        self.trail = []
        self.mode = None
        self.stream = rr.RecordingStream(f"IPC forest {env.cfg.goal_x:g}m — {policy}")
        self.stream.save(path, default_blueprint=forest_blueprint(env.cfg.goal_x))
        self.stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        points = np.load(Path(env.cfg.scene.forest.spawn.usd_path).parent / "points.npy", mmap_mode="r")
        crop = (points[:, 0] > -2) & (points[:, 0] < env.cfg.goal_x + 3) & (abs(points[:, 1]) < 5)
        self.stream.log("world/map", rr.Points3D(points[crop], radii=0.015, colors=[95, 118, 105]), static=True)
        self.stream.log(
            "world/start_goal",
            rr.Points3D(
                [[0, 0, 1], [env.cfg.goal_x, 0, 1]], radii=0.16, colors=[255, 230, 90], labels=["Start", "Goal"]
            ),
            static=True,
        )
        self.stream.log(
            "world/drone/body", rr.Boxes3D(half_sizes=[0.18, 0.18, 0.045], colors=[60, 160, 255]), static=True
        )
        self.stream.log(
            "world/drone/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3, vectors=np.eye(3) * 0.4, colors=[[255, 70, 70], [70, 255, 70], [70, 70, 255]]
            ),
            static=True,
        )
        self.state()

    def plan(self):
        env, r = self.env, self.stream
        obs, result = env.current_context, env.last_mpc_result
        mode = bool(result["sfc_valid"][0]), bool(result["solved"][0]), bool(result["braking"][0])
        tick = round(float(env.elapsed[0]) * 100)
        if tick % 10 and mode == self.mode:
            return
        self.mode = mode
        r.set_time("sim_time", duration=tick * 0.01)
        r.log("world/sfc", rr.Clear(recursive=True))
        r.log("world/prediction", rr.Clear(recursive=True))
        reference = obs["reference"][0].cpu().numpy()[:, :3]
        if mode[0]:
            r.log("world/reference", rr.LineStrips3D([reference], radii=0.018, colors=[230, 230, 230]))
            planes = obs["planes"][0].cpu().numpy()
            masks = obs["plane_mask"][0].cpu().numpy()
            unique = []
            for k in range(15):
                faces = planes[k, masks[k]]
                if not len(faces) or any(np.array_equal(faces, old) for old in unique):
                    continue
                unique.append(faces)
                steps = [j for j in range(15) if np.array_equal(faces, planes[j, masks[j]])]
                color = [30, 220, 220] if len(unique) == 1 else [255, 170, 50]
                name = f"world/sfc/{len(unique)}"
                r.log(name, rr.LineStrips3D(corridor_edges(faces), colors=color, radii=0.012))
                label = "SFC steps " + ",".join(str(j + 1) for j in steps)
                r.log(name + "/label", rr.Points3D([reference[k]], radii=0.03, colors=color, labels=[label]))
        else:
            r.log("world/reference", rr.Clear(recursive=True))
        if mode[1] and not mode[2]:
            mpc = env.planners[0].mpc
            prediction = mpc.M @ result["state"][0].cpu().numpy() + mpc.C @ result["solution"][0].cpu().numpy()
            positions = prediction.reshape(15, 9)[:, :3]
            r.log("world/prediction/qp", rr.LineStrips3D([positions], radii=0.025, colors=[240, 110, 240]))
        if mode[2]:
            r.log(
                "world/prediction/brake",
                rr.Points3D(
                    result["predicted"][0, :3].cpu().numpy()[None],
                    radii=0.08,
                    colors=[255, 90, 70],
                    labels=["Brake +0.1 s"],
                ),
            )
        planner = env.planners[0]
        r.log(
            "status/current",
            rr.TextDocument(
                f"{self.policy} | Newton real | {env.cfg.goal_x:g} m\n"
                f"SFC valid: {mode[0]} | QP solved: {mode[1]} | braking: {mode[2]}\n"
                f"Frontend failures: {planner.frontend_failures} | recoveries: {planner.recovery_events}\n\n"
                "Blue: actual flight\nWhite: reference\nMagenta: accepted QP prediction\n"
                "Cyan / amber: SFC, labels list prediction steps (0.1 s each)\n"
                "Red point: braking prediction\nPlans sampled at 10 Hz and every mode change."
            ),
        )

    def state(self):
        env, r = self.env, self.stream
        t = float(env.elapsed[0])
        r.set_time("sim_time", duration=t)
        pose, twist = env.body_state()
        position = (pose[0, :3] - env.scene.env_origins[0]).cpu().numpy()
        self.trail.append(position)
        r.log(
            "world/drone", rr.Transform3D(translation=position, rotation=rr.Quaternion(xyzw=pose[0, 3:].cpu().numpy()))
        )
        if round(t * 100) % 10 == 0 or env.reported[0]:
            r.log("world/trajectory", rr.LineStrips3D([np.asarray(self.trail)], radii=0.025, colors=[60, 160, 255]))
        r.log("metrics/speed_mps", rr.Scalars(float(twist[0, :3].norm())))
        clearance = float(env.min_clearance[0])
        if np.isfinite(clearance):
            r.log("metrics/min_clearance_m", rr.Scalars(clearance))
        r.log("metrics/braking", rr.Scalars(float(env.planners[0].braking)))
        if env.reported[0]:
            row = env.records[-1]
            r.log(
                "status/outcome",
                rr.TextDocument(
                    f"Reached: {row['success']}\nCollision: {row['collision']}\nTime: {t:.2f} s\nMinimum clearance: {clearance:.3f} m"
                ),
            )

    def close(self):
        self.stream.flush()
        self.stream.disconnect()
