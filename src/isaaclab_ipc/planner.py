"""ROS-free IPC scheduling with the pinned A* and PathCover corridor kernel."""

from collections import OrderedDict, deque
from time import perf_counter

import numpy as np

from . import _geometry
from .execution import RecoveryPolicy, brake_jerk, next_braking, safe_climb_jerk
from .mpc import MPC
from .safe_recovery import vertical_segment_clear


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


class AStarEndpointUnavailable(InvalidCorridorError):
    """The native A* grid contains no free endpoint at the searched altitude."""


class SafeClimbPending(InvalidCorridorError):
    """The real controller must regain a collision-checked planning altitude."""


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
        self.incremental_voxels = None
        self.warp_grid = None
        if frontend and frontend.get("gpu_incremental", False):
            from .incremental_voxels import IncrementalVoxelCentroids

            self.incremental_voxels = IncrementalVoxelCentroids(frontend["cloud_leaf"])
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
        self.recovery_target = None
        self.safe_climb_events = 0
        self.safe_climb_steps = 0
        self.last_frontend_error = ""
        self._climb_cloud = None
        self._safe_prediction = None
        self._safe_prediction_goal = None
        self._latest_schedule = []
        self.reused_prediction_count = 0

    def update_cloud(self, cloud, position):
        cloud = np.asarray(cloud, dtype=float).reshape(-1, 3)
        if not np.isfinite(cloud).all():
            raise ValueError("Point cloud must be finite")
        frontend = self.cfg.params.get("frontend")
        if frontend:
            # Fixed SUPER maps accumulate observed points only, never the full collision PCD.
            low, high = np.asarray(frontend["workspace"])
            if self.incremental_voxels is not None:
                valid = ((cloud >= low) & (cloud <= high)).all(1)
                removed, added, rebuilt = self.incremental_voxels.update(cloud[valid])
                if not len(removed) and not len(added):
                    return
                self.cloud = self.incremental_voxels.points
                if self.warp_grid is None:
                    from .warp_inflation import WarpInflatedGrid

                    self.warp_grid = WarpInflatedGrid(
                        frontend["resolution"], frontend["size"], frontend["center"],
                        self.cfg.params["astar"]["expand_fix"],
                    )
                self.warp_grid.update(removed, added, self.grid, reset=rebuilt)
                self._grid_cloud = self.cloud
                self._grid_radius = self.cfg.params["astar"]["expand_fix"]
            else:
                points = np.vstack((self.cloud, cloud))
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
        raise AStarEndpointUnavailable("No free A* endpoint in the local map")

    def _safe_vertical_target(self, position):
        frontend = self.cfg.params.get("frontend")
        if not frontend or not frontend.get("safe_climb", False) or not len(self.cloud):
            return None
        resolution = float(frontend["resolution"])
        radius = float(self.cfg.params["sfc_dis"])
        clearance = radius + float(frontend["cloud_leaf"]) / 2
        lower, upper = np.asarray(frontend["workspace"], dtype=float)
        if position[2] <= lower[2] + clearance or position[2] >= upper[2] - clearance:
            return None
        ceiling = min(position[2] + 1.8, upper[2] - clearance)
        for candidate_z in np.arange(position[2] + resolution, ceiling + 1e-9, resolution):
            target = position.copy()
            target[2] = min(candidate_z + resolution / 2, upper[2] - clearance)
            if not self.grid.endpoint_free(target):
                continue
            if vertical_segment_clear(self.cloud, position, target, clearance):
                return target
        return None

    def _continue_safe_climb(self, state):
        if self.recovery_target is None:
            return
        position = state[:3]
        target = self.recovery_target
        if self._climb_cloud is not self.cloud or np.linalg.norm(position[:2] - target[:2]) > 0.15:
            frontend = self.cfg.params["frontend"]
            clearance = float(self.cfg.params["sfc_dis"]) + float(frontend["cloud_leaf"]) / 2
            if not vertical_segment_clear(self.cloud, position, target, clearance):
                self.recovery_target = None
                self.last_frontend_error = "recovery_climb_blocked"
                raise InvalidCorridorError("Observed obstacle blocks the vertical recovery path")
            self._climb_cloud = self.cloud
        if (
            self.grid.endpoint_free(position)
            and abs(position[2] - target[2]) < 0.1
            and np.linalg.norm(state[3:6]) < 0.2
            and np.linalg.norm(state[6:]) < 0.5
        ):
            self.recovery_target = None
            self._climb_cloud = None
            self.replan = True
            return
        raise SafeClimbPending("Safe vertical recovery is in progress")

    def _grid_segment_free(self, start, end):
        frontend = self.cfg.params.get("frontend")
        resolution = float(frontend["resolution"]) if frontend else float(self.grid.resolution_)
        steps = max(1, int(np.ceil(np.linalg.norm(end - start) / (0.5 * resolution))))
        for alpha in np.linspace(0, 1, steps + 1):
            point = start + (end - start) * alpha
            if frontend:
                low, high = np.asarray(frontend["workspace"])
                if np.any(point < low) or np.any(point >= high):
                    return False
            if not self.grid.free(point):
                return False
        return True

    def _reuse_safe_prefix(self, position):
        frontend = self.cfg.params.get("frontend")
        if not frontend or not frontend.get("reuse_prediction", False):
            return None
        prefix = self._safe_prediction
        if prefix is None or not np.array_equal(self._safe_prediction_goal, self.goal):
            return None
        if not self.grid.endpoint_free(position) or np.linalg.norm(prefix[0] - position) > 2.5:
            return None
        points = np.vstack((position, prefix))
        if any(not self._grid_segment_free(a, b) for a, b in zip(points[:-1], points[1:], strict=True)):
            return None
        return prefix.copy()

    def _remember_safe_prediction(self, state, solution):
        frontend = self.cfg.params.get("frontend")
        self._safe_prediction = None
        if not frontend or not frontend.get("reuse_prediction", False) or not self._latest_schedule:
            return
        count = min(3, self.mpc.n)
        positions = (self.mpc.M @ state + self.mpc.C @ np.asarray(solution)).reshape(self.mpc.n, 9)[:count, :3]
        if any(
            not len(faces) or np.any(faces[:, :3] @ pos + faces[:, 3] > 1e-4)
            for pos, faces in zip(positions, self._latest_schedule[:count], strict=True)
        ):
            return
        self._safe_prediction = positions.copy()
        self._safe_prediction_goal = self.goal.copy()

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
        try:
            start = self._move_endpoint(start)
        except AStarEndpointUnavailable:
            target = self._safe_vertical_target(position)
            if target is not None:
                self.recovery_target = target
                self._climb_cloud = self.cloud
                self.safe_climb_events += 1
                self._safe_prediction = None
                self.last_frontend_error = "astar_start_occupied_safe_climb"
                raise SafeClimbPending("A* start occupied; use verified vertical recovery") from None
            self.last_frontend_error = "astar_start_occupied_no_safe_climb"
            raise
        goal = self._move_endpoint(goal)
        reusable = self._reuse_safe_prefix(position) if np.array_equal(start, position) else None
        if reusable is not None:
            start = reusable[-1]
        path = np.asarray(self.grid.search(start, goal))
        if not len(path):
            self.last_frontend_error = "astar_path_not_found"
            self.frontend_failures += 1
            self.path = position[None].copy()
            self.replan = False
            return
        if reusable is not None:
            path = np.vstack((position, reusable, path[1:]))
            self.reused_prediction_count += 1
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
        self._continue_safe_climb(state)
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
            if frontend and initial == 0 and mpc_goal < n - 1:
                # The next SFC must not become mandatory before the drone can
                # reach the end of the current corridor under acceleration bounds.
                displacement = path[first] - position
                distance = np.linalg.norm(displacement)
                if distance > 0:
                    direction = displacement / distance
                    limits = self.cfg.params["mpc"]
                    max_acc = np.array([
                        limits[f"a{axis}_{'max' if direction[i] >= 0 else 'min'}"]
                        for i, axis in enumerate("xyz")
                    ])
                    along_acc = max(0.0, float(direction @ max_acc))
                    along_speed = float(direction @ state[3:6])
                    for stage in range(mpc_goal + 1, n):
                        t = (stage + 1) * self.mpc.dt
                        if along_speed * t + 0.5 * along_acc * t * t >= distance:
                            mpc_goal = stage - 1
                            break
                    else:
                        mpc_goal = n - 1
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
        self._latest_schedule = [faces.copy() for faces in schedule]
        return state, references, schedule

    def prepare(self, state):
        start = perf_counter()
        state, references, schedule = self.context(state)
        return self.mpc.problem(state, references, schedule), (perf_counter() - start) * 1000

    def _brake(self, state):
        # Track a velocity-opposing acceleration, with the original jerk bounds.
        if self.recovery_target is not None:
            jerk = safe_climb_jerk(
                state, self.recovery_target[2], self.mpc.lower, self.mpc.upper,
                self.mpc.jlow, self.mpc.jhigh, self.mpc.dt,
            )
            self.safe_climb_steps += 1
        else:
            jerk = brake_jerk(state, self.mpc.lower, self.mpc.upper, self.mpc.jlow, self.mpc.jhigh, self.mpc.dt)
        return self.mpc.a @ state + self.mpc.b @ jerk, jerk, False

    def consume(self, problem, solution, status, time):
        return self.consume_state(problem[-1], solution, status, time)

    def consume_recovery(self, state, success):
        """Record one executed control result without interpreting its input vector.

        Both jerk and CTBR controllers use the measured navigation [p,v,a] state
        for the existing real-flight recovery policy. Call only during env.step.
        """
        if self.cfg.profile.recovery is not RecoveryPolicy.BRAKE_REPLAN:
            return False
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
            self._safe_prediction = None
        self.brake_steps += int(braking)
        return braking

    def consume_state(self, state, solution, status, time):
        success = status in ("solved", "solved inaccurate")
        if self.consume_recovery(state, success):
            return self._brake(state)
        if success:
            self.previous = np.asarray(solution).reshape(self.mpc.n, 3).copy()
            self.last_time = time
            self.index = int(self.cfg.params["ctrl_delay"] / self.mpc.dt)
            x = state.copy()
            for i in range(self.index + 1):
                jerk = self.previous[min(i, self.mpc.n - 1)]
                x = self.mpc.a @ x + self.mpc.b @ jerk
            self._remember_safe_prediction(state, solution)
        else:
            self._safe_prediction = None
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
