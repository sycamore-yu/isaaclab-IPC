"""ROS-free IPC scheduling with the pinned A* and PathCover corridor kernel."""

from collections import deque
from time import perf_counter

import numpy as np

from . import _geometry
from .execution import RecoveryPolicy, brake_jerk, next_braking
from .mpc import MPC


def voxel_centroids(points, leaf=0.2):
    if not len(points):
        return np.empty((0, 3))
    p = np.asarray(points, dtype=np.float32)
    indices = np.floor(p / np.float32(leaf)).astype(np.int64)
    indices -= indices.min(0)
    sizes = indices.max(0) + 1
    keys = indices[:, 0] + sizes[0] * (indices[:, 1] + sizes[1] * indices[:, 2])
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(sums, inverse, p)
    return (sums / count[:, None]).astype(np.float32).astype(float)


def contains(planes, p):
    return bool(len(planes) and np.all(planes[:, :3] @ p + planes[:, 3] <= 0))


class InvalidCorridorError(ValueError):
    """The geometry frontend did not produce finite SFC planes."""


class Planner:
    def __init__(self, cfg, goal):
        self.cfg = cfg
        self.mpc = MPC(cfg)
        self.grid = _geometry.Grid()
        self.goal = np.asarray(goal, dtype=float)
        self.goal[2] = np.clip(self.goal[2], 0.5, 2.5)
        self.frames = deque(maxlen=10)
        self.cloud = np.empty((0, 3))
        self.path = np.empty((0, 3))
        self.nearest = 0
        self.replan = True
        self.previous = None
        self.last_time = 0.0
        self.index = 0
        self.failures = 0
        self.frontend_failures = 0
        self.braking = False
        self.brake_steps = 0
        self.recovery_events = 0

    def update_cloud(self, cloud, position):
        cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
        if not np.isfinite(cloud).all():
            raise ValueError("Point cloud must be finite")
        self.frames.append(cloud.copy())
        points = np.concatenate(self.frames)
        valid = (np.abs(points[:, :2] - position[:2]) <= 9.5).all(1) & (points[:, 2] >= 0.2) & (points[:, 2] <= 3.0)
        self.cloud = voxel_centroids(points[valid])
        self.grid.center([*position[:2], 0.0])
        self.grid.cloud(self.cloud, self.cfg.params["astar"]["expand_fix"])
        if len(self.path) and not self.grid.path_free(self.path[self.nearest :]):
            self.replan = True

    def _move_endpoint(self, p):
        increment = self.cfg.params["astar"]["expand_fix"] / 2
        if self.grid.endpoint_free(p):
            return p
        for radius in np.arange(increment, 20, increment):
            found = False
            for i in range(8):
                candidate = p + radius * np.array([np.sin(2 * np.pi * i / 8), np.cos(2 * np.pi * i / 8), 0.0])
                if self.grid.endpoint_free(candidate):
                    p = candidate
                    found = True
            if found:
                return p
        raise RuntimeError("No free A* endpoint in the local map")

    def _replan(self, position):
        self.grid.cloud(self.cloud, self.cfg.params["astar"]["expand_dyn"])
        start = position.copy()
        goal = self.goal.copy()
        delta = goal - position
        extended = np.max(np.abs(delta[:2])) > 10
        if extended:
            goal[:2] = position[:2] + delta[:2] * 9.9 / np.max(np.abs(delta[:2]))
        start = self._move_endpoint(start)
        goal = self._move_endpoint(goal)
        path = np.asarray(self.grid.search(start, goal))
        if not len(path):
            self.frontend_failures += 1
            self.path = position[None].copy()
            self.replan = False
            return
        if extended and self.grid.free(self.goal):
            path = np.vstack((path, self.goal))
        spacing = self.cfg.params["fsm"]["path_dis"]
        points = []
        for a, b in zip(path[:-1], path[1:], strict=True):
            count = int(np.linalg.norm(b - a) / spacing)
            if count:
                points.append(a + (b - a) * np.arange(count)[:, None] / count)
        points.append(path[-1:])
        self.path = np.concatenate(points)
        self.replan = False

    def _corridor(self, a, b):
        try:
            planes = np.asarray(_geometry.corridor(a, b, self.cloud, self.cfg.params["sfc_dis"])).reshape(-1, 4)
        except _geometry.CorridorError as exc:
            raise InvalidCorridorError(str(exc)) from exc
        if not np.isfinite(planes).all():
            raise InvalidCorridorError("PathCover produced a nonfinite SFC plane")
        return planes

    def context(self, state):
        """Return state, reference and scheduled SFC; callers own their snapshots."""
        state = self.cfg.profile.planner_state(state)
        position = np.asarray(state[:3])
        if self.replan or not len(self.path):
            self._replan(position)
        path = self.path
        self.nearest = near = int(np.argmin(np.linalg.norm(path - position, axis=1)))
        stride = self.cfg.params["fsm"]["ref_dis"]
        spacing = self.cfg.params["fsm"]["path_dis"]
        n = self.mpc.n
        schedule = [np.empty((0, 4)) for _ in range(n)]
        goal_index = near
        if len(path) - near <= stride:
            planes = self._corridor(position, position)
            schedule = [planes] * n
            goal_index = len(path)
        else:
            planes = self._corridor(path[near], path[near])
            first = near
            initial = 0 if contains(planes, position) else int(np.linalg.norm(position - path[near]) / spacing / stride)
            for i in range(near + 1, len(path)):
                if contains(planes, path[i]):
                    first = i
                    goal_index = i
                else:
                    first = max(near, int(first - 0.2 / spacing))
                    break
            mpc_goal = (first - near) // stride + initial + 1
            for i in range(initial, min(mpc_goal + 1, n)):
                schedule[i] = planes
            end = min(first + (n - mpc_goal) * stride, len(path) - 1)
            i = end
            while i >= first and mpc_goal < n - 1:
                if self.grid.free(path[i]) and self.grid.line_free(path[first], path[i]):
                    i = int(i - 0.2 / spacing)
                    if i <= first:
                        break
                    long = self._corridor(path[first], path[i])
                    if len(long):
                        for j in range(mpc_goal + 1, n):
                            schedule[j] = long
                        mpc_goal = n
                        for k in range(first, len(path)):
                            if contains(long, path[k]):
                                goal_index = k
                            else:
                                break
                        break
                i -= 1
            for i in range(mpc_goal + 1, n):
                schedule[i] = planes
        indices = np.minimum(near + np.arange(n) * stride, min(goal_index, len(path) - 1))
        references = np.zeros((n, 9))
        references[:, :3] = path[indices]
        references[:, 3:6] = np.diff(np.vstack((position, references[:, :3])), axis=0) / self.mpc.dt
        return state, references, schedule

    def prepare(self, state):
        start = perf_counter()
        state, references, schedule = self.context(state)
        return self.mpc.problem(state, references, schedule), (perf_counter() - start) * 1000

    def _brake(self, state):
        # Track a velocity-opposing acceleration, with the original jerk bounds.
        jerk = brake_jerk(state, self.mpc.lower, self.mpc.upper, self.mpc.jlow, self.mpc.jhigh, self.mpc.dt)
        self.brake_steps += 1
        return self.mpc.a @ state + self.mpc.b @ jerk, jerk, False

    def consume(self, problem, solution, status, time):
        return self.consume_state(problem[-1], solution, status, time)

    def consume_state(self, state, solution, status, time):
        success = status in ("solved", "solved inaccurate")
        if self.cfg.profile.recovery is RecoveryPolicy.BRAKE_REPLAN:
            if not success:
                self.failures += 1
                if not self.braking:
                    self.recovery_events += 1
                    self.replan = True
            braking = bool(
                next_braking(
                    np.bool_(self.braking), np.bool_(success), np.linalg.norm(state[3:6]), np.linalg.norm(state[6:])
                )
            )
            if self.braking and not braking:
                self.replan = True
            self.braking = braking
            if braking:
                return self._brake(state)
        if success:
            self.previous = np.asarray(solution).reshape(self.mpc.n, 3).copy()
            self.last_time = time
            self.index = int(self.cfg.params["ctrl_delay"] / self.mpc.dt)
            x = state.copy()
            for i in range(self.index + 1):
                jerk = self.previous[min(i, self.mpc.n - 1)]
                x = self.mpc.a @ x + self.mpc.b @ jerk
        else:
            self.failures += 1
            if self.previous is None:
                raise RuntimeError("Initial MPC solve failed: " + status)
            elapsed = time - self.last_time
            if elapsed >= self.mpc.dt:
                self.index += int(elapsed / self.mpc.dt)
                self.last_time = time
            jerk = self.previous[min(self.index, self.mpc.n - 1)]
            x = self.mpc.a @ state + self.mpc.b @ jerk
        return x, jerk, success

    def step(self, state, time):
        problem, frontend_ms = self.prepare(state)
        start = perf_counter()
        solution, status, primal, dual = self.mpc.solve(problem)
        out = self.consume(problem, solution, status, time)
        return *out, dict(
            status=status,
            frontend_ms=frontend_ms,
            solve_ms=(perf_counter() - start) * 1000,
            primal_residual=primal,
            dual_residual=dual,
        )


def flatness(acceleration, jerk, rotation, yaw_target=0.0, yaw_gain=0.5):
    acc = np.asarray(acceleration) + [0, 0, 9.81]
    magnitude = np.linalg.norm(acc)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    z = acc / magnitude
    y = np.cross(z, [np.cos(yaw), np.sin(yaw), 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    hw = (jerk - z.dot(jerk) * z) / magnitude
    error = (yaw_target - yaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([acc @ rotation[:, 2], -hw @ y, hw @ x, error * yaw_gain * z[2]])
