"""ROS-free IPC scheduling with the pinned A* and PathCover corridor kernel."""

from collections import OrderedDict, deque
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
    def __init__(self, cfg, goal, *, cache_corridors=True):
        self.cfg = cfg
        self.mpc = MPC(cfg)
        frontend = cfg.params.get("frontend")
        if frontend:
            half = np.asarray(frontend["size"]) / 2
            self.grid = _geometry.Grid(frontend["resolution"], -half, half, heuristic=0)
            self.grid.center(frontend["center"])
            self.grid.height_limits(frontend["workspace"][0][2], frontend["workspace"][1][2])
        else:
            self.grid = _geometry.Grid()
        self.goal = np.asarray(goal, dtype=float)
        self.goal[2] = np.clip(self.goal[2], 0.5, 2.5)
        self.frames = deque(maxlen=10)
        self.cloud = np.empty((0, 3))
        self.cache_corridors = cache_corridors
        self._corridor_cache = OrderedDict()
        self._corridor_cloud = self.cloud
        self._grid_cloud = None
        self._grid_radius = None
        self.corridor_evaluations = 0
        self.corridor_cache_hits = 0
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
        frontend = self.cfg.params.get("frontend")
        if frontend:
            # Fixed SUPER maps retain observed occupancy; never preload the full collision PCD.
            points = np.vstack((self.cloud, cloud))
            low, high = np.asarray(frontend["workspace"])
            valid = ((points >= low) & (points <= high)).all(1)
            self.cloud = voxel_centroids(points[valid], frontend["cloud_leaf"])
        else:
            self.frames.append(cloud.copy())
            points = np.concatenate(self.frames)
            valid = (np.abs(points[:, :2] - position[:2]) <= 9.5).all(1) & (points[:, 2] >= 0.2) & (points[:, 2] <= 3.0)
            self.cloud = voxel_centroids(points[valid])
        self._corridor_cache.clear()
        self._corridor_cloud = self.cloud
        if not frontend:
            self.grid.center([*position[:2], 0.0])
        self._refresh_grid(self.cfg.params["astar"]["expand_fix"])
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

    def _refresh_grid(self, radius):
        if self.cfg.params.get("frontend"):
            if self.cloud is self._grid_cloud and radius == self._grid_radius:
                return
            self.grid.cloud_fast(self.cloud, radius)
            self._grid_cloud, self._grid_radius = self.cloud, radius
        else:
            self.grid.cloud(self.cloud, radius)

    def _replan(self, position):
        self._refresh_grid(self.cfg.params["astar"]["expand_dyn"])
        start = position.copy()
        goal = self.goal.copy()
        delta = goal - position
        frontend = self.cfg.params.get("frontend")
        extent = frontend["planning_horizon"] if frontend else 9.9
        distance = np.linalg.norm(delta) if frontend else np.max(np.abs(delta[:2]))
        extended = distance > (extent if frontend else 10)
        if extended:
            if frontend:
                goal = position + delta * extent / distance
            else:
                goal[:2] = position[:2] + delta[:2] * extent / distance
        start = self._move_endpoint(start)
        goal = self._move_endpoint(goal)
        path = np.asarray(self.grid.search(start, goal))
        if not len(path):
            self.frontend_failures += 1
            self.path = position[None].copy()
            self.replan = False
            return
        if extended and not frontend and self.grid.free(self.goal):
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
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        radius = float(self.cfg.params["sfc_dis"])
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid corridor endpoints")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Invalid corridor radius")
        if not self.cache_corridors:
            return self._generate_corridor(a, b, radius)
        if self.cloud is not self._corridor_cloud:
            self._corridor_cache.clear()
            self._corridor_cloud = self.cloud
        key = (a.tobytes(), radius)
        if key in self._corridor_cache:
            self.corridor_cache_hits += 1
            self._corridor_cache.move_to_end(key)
        else:
            # RISP depends on the seed/map/radius. The native end argument only checks
            # containment after generating the same polytope; reuse it for candidate ends.
            try:
                self._corridor_cache[key] = self._generate_corridor(a, a, radius)
            except InvalidCorridorError as exc:
                self._corridor_cache[key] = str(exc)
            if len(self._corridor_cache) > 32:
                self._corridor_cache.popitem(last=False)
        planes = self._corridor_cache[key]
        if isinstance(planes, str):
            raise InvalidCorridorError(planes)
        # Keep the native binding's exact end-containment tolerance.
        if (planes[:, :3] @ b + planes[:, 3] > 1e-8).any():
            return np.empty((0, 4))
        return planes.copy()

    def _generate_corridor(self, a, b, radius):
        self.corridor_evaluations += 1
        try:
            frontend = self.cfg.params.get("frontend")
            if frontend:
                planes = _geometry.corridor_numpy(a, b, self.cloud, radius, frontend["workspace"])
            else:
                planes = _geometry.corridor(a, b, self.cloud, radius)
            planes = np.asarray(planes).reshape(-1, 4)
        except _geometry.CorridorError as exc:
            raise InvalidCorridorError(str(exc)) from exc
        if not np.isfinite(planes).all():
            raise InvalidCorridorError("PathCover produced a nonfinite SFC plane")
        return planes

    def context(self, state):
        """Return state, reference and scheduled SFC; callers own their snapshots."""
        state = self.cfg.profile.planner_state(state)
        position = np.asarray(state[:3])
        frontend = self.cfg.params.get("frontend")
        advance = (
            frontend and len(self.path) and np.linalg.norm(self.path[-1] - self.goal) > 0.1
            and np.linalg.norm(position - self.path[-1]) < frontend["receding_distance"]
        )
        if self.replan or not len(self.path) or advance:
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
