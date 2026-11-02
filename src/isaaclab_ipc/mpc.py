"""IPC condensed QP; preserve OSQP-Eigen 0.7 upper-triangle semantics."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osqp
import yaml
from scipy import sparse


@dataclass
class Config:
    params: dict
    profile: str = "real"

    @classmethod
    def load(cls, name="forest", ref_dis=None, profile="real"):
        if profile not in ("real", "ideal"):
            raise ValueError("Unknown execution profile: " + profile)
        p = yaml.safe_load((Path(__file__).parent / "config" / f"{name}.yaml").read_text())
        if ref_dis is not None:
            p["fsm"]["ref_dis"] = int(ref_dis)
        p["execution_profile"] = profile
        p["recovery"] = "brake_replan" if profile == "real" else "previous_sequence"
        if profile == "real":
            p["perfect_simu"] = False
        elif profile == "ideal":
            p["perfect_simu"] = True
        return cls(p, profile)


def dynamics(dt, drag=(0.0, 0.0, 0.0)):
    a = np.eye(9)
    b = np.zeros((9, 3))
    eye = np.eye(3)
    a[:3, 3:6] = dt * eye
    a[:3, 6:] = dt**2 / 2 * eye
    a[3:6, 3:6] -= np.diag(drag) * dt
    a[3:6, 6:] = dt * eye
    b[:3] = dt**3 / 6 * eye
    b[3:6] = dt**2 / 2 * eye
    b[6:] = dt * eye
    return a, b


class MPC:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg.params["mpc"]
        self.n = int(c["horizon"])
        n = self.n
        self.dt = float(c["step"])
        self.a, self.b = dynamics(self.dt, [c["D_" + k] for k in "xyz"])
        self.M = np.concatenate([np.linalg.matrix_power(self.a, i + 1) for i in range(n)])
        self.C = np.zeros((9 * n, 3 * n))
        for i in range(n):
            for j in range(i + 1):
                self.C[i * 9 : (i + 1) * 9, j * 3 : (j + 1) * 3] = np.linalg.matrix_power(self.a, i - j) @ self.b
        q = np.tile(np.repeat([c["R_p"], c["R_v"], c["R_a"]], 3), n).astype(float)
        q[-9:] = np.repeat([c["R_pN"], c["R_vN"], c["R_aN"]], 3)
        self.q = q
        # OSQP-Eigen 0.7 discards IPC's lower-only -2*R_u_con blocks.
        continuity = np.full(n, 2.0)
        continuity[[0, -1]] = 1.0
        self.H = self.C.T @ (q[:, None] * self.C) + np.diag(np.repeat(c["R_u"] + continuity * c["R_u_con"], 3))
        self.pi = np.arange(9 * n).reshape(n, 9)[:, :3].ravel()
        self.vi = self.pi + 3
        self.ai = self.pi + 6
        self.A0 = np.vstack((np.eye(3 * n), self.C[self.ai], self.C[self.vi]))
        self.lower = np.array([c[k + "_min"] for k in ["vx", "vy", "vz", "ax", "ay", "az"]])
        self.upper = np.array([c[k + "_max"] for k in ["vx", "vy", "vz", "ax", "ay", "az"]])
        self.jlow = np.tile([c[k + "_min"] for k in ["ux", "uy", "uz"]], n)
        self.jhigh = np.tile([c[k + "_max"] for k in ["ux", "uy", "uz"]], n)

    def problem(self, state, reference, corridors):
        state = np.asarray(state, dtype=float).copy()
        reference = np.asarray(reference, dtype=float)
        if state.shape != (9,) or reference.shape != (self.n, 9) or len(corridors) != self.n:
            raise ValueError("Expected state [9], reference [horizon, 9] and one corridor per step")
        if not np.isfinite(state).all() or not np.isfinite(reference).all():
            raise ValueError("Nonfinite MPC input")
        state[3:] = np.clip(state[3:], self.lower, self.upper)
        offset = self.M @ state
        g = self.C.T @ (self.q * (offset - reference.ravel()))
        lo = np.r_[
            self.jlow,
            np.tile(self.lower[3:], self.n) - offset[self.ai],
            np.tile(self.lower[:3], self.n) - offset[self.vi],
        ]
        hi = np.r_[
            self.jhigh,
            np.tile(self.upper[3:], self.n) - offset[self.ai],
            np.tile(self.upper[:3], self.n) - offset[self.vi],
        ]
        rows = [self.A0]
        highs = [hi]
        lows = [lo]
        for i, planes in enumerate(corridors):
            if len(planes):
                planes = np.asarray(planes, dtype=float)
                if planes.ndim != 2 or planes.shape[1] != 4 or not np.isfinite(planes).all():
                    raise ValueError("Corridor planes must be finite [planes, 4] arrays")
                planes = planes[::-1]  # SetFSC appends each block in reverse row order.
                rows.append(planes[:, :3] @ self.C[9 * i : 9 * i + 3])
                highs.append(-planes[:, 3] - planes[:, :3] @ offset[9 * i : 9 * i + 3])
                lows.append(np.full(len(planes), -np.inf))
        return self.H, g, np.vstack(rows), np.concatenate(lows), np.concatenate(highs), state

    def solve(self, problem):
        h, g, a, lower, u, _ = problem
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(h)),
            q=g,
            A=sparse.csc_matrix(a),
            l=lower,
            u=u,
            verbose=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
            polishing=False,
            warm_starting=False,
        )
        r = solver.solve(raise_error=False)
        return r.x, r.info.status, r.info.prim_res, r.info.dual_res
