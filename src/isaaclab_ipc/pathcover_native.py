"""ROS-free NanoVoxMap → JPS/DMP → PathCover/RISP → GCOPTER/MINCO.

The extension is loaded only when a front end is constructed. Each instance owns
its observed map; plans are immutable snapshots suitable for PPO storage.
"""

import math
from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np


class PathCoverPlanningError(RuntimeError):
    """A geometric or trajectory planning failure, distinct from a software error."""


@dataclass(frozen=True)
class PathCoverConfig:
    # Physical workspace bounds; native search/SFC subtract body clearance once.
    bounds_lower: tuple[float, float, float] = (-5.0, -20.0, 0.0)
    bounds_upper: tuple[float, float, float] = (60.0, 20.0, 10.0)
    resolution: float = 0.25
    # Optional coarser JPS/DMP and RISP voxel geometry; mapping keeps resolution.
    search_resolution: float | None = None
    robot_radius: float = 0.25
    robot_radius_z: float | None = None
    max_velocity: float = 6.0
    mass: float = 0.75
    gravity: float = 9.81
    thrust_min: float = 0.0
    thrust_max: float | None = None
    body_rate_max: float = 3.0
    tilt_max: float = 1.2
    max_corridors: int = 6
    pieces_per_polytope: int = 1
    integral_resolution: int = 64
    max_iterations: int = 1000
    relative_tolerance: float = 1e-4
    time_weight: float = 100.0
    safety_margin: float = 0.0
    penalty_weights: tuple[float, ...] = (1e4, 1e3, 1e4, 1e4, 1e4)
    # Width in metres of the smoothed L1 position penalty. Dynamic smoothing
    # retains the upstream 0.01 in each corresponding constraint's units.
    position_penalty_smoothing_m: float = 1e-6
    potential_radius: float = 1.0
    search_radius: float = 2.0
    # Weighted A*: >1 trades shortest-path optimality for fewer expansions.
    dmp_heuristic_weight: float = 1.5
    risp_alpha: float = 0.01
    random_seed: int = 0
    num_threads: int = 1
    use_cuda: bool = True
    cuda_min_grid_voxels: int = 1 << 20
    # Bound on body z angular velocity, not the world heading derivative.
    body_yaw_rate_max: float = np.pi / 2
    yaw_sample_dt: float = 0.02
    yaw_low_speed: float = 0.1

    def native_options(self):
        values = asdict(self)
        if self.search_resolution is None:
            values["search_resolution"] = self.resolution
        if self.robot_radius_z is None:
            values["robot_radius_z"] = self.robot_radius
        if self.thrust_max is None:
            values["thrust_max"] = 3 * self.mass * self.gravity
        return values


@dataclass(frozen=True)
class PathCoverState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float = 0.0
    time: float = 0.0


@dataclass(frozen=True)
class PathCoverSample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    yaw: np.ndarray
    yaw_rate: np.ndarray  # World heading derivative; differs from body z angular velocity.

    @property
    def pva(self):
        return np.concatenate((self.position, self.velocity, self.acceleration), axis=-1)


@dataclass(frozen=True)
class PathCoverSegment:
    corridor_index: int
    start: float
    end: float
    duration: float


def _bounded_body_yaw_profile(trajectory):
    """Project heading rates onto body-r intervals at start/mid/end of each step.

    For c=z.h and s²=1-c², body r=(z_z*yaw_rate+c*p*s)/s².
    Two midpoint/end projections constrain the piecewise-linear heading profile.
    These finite samples are not a continuous-time certificate.
    """
    count = max(1, int(np.ceil(trajectory.duration / trajectory.yaw_sample_dt)))
    times = np.linspace(0.0, trajectory.duration, count + 1)
    half_times = np.linspace(0.0, trajectory.duration, 2 * count + 1)
    _, velocity, acceleration, jerk = trajectory._derivatives(half_times)
    force = acceleration + np.array([0.0, 0.0, trajectory.gravity])
    magnitude = np.linalg.norm(force, axis=-1)
    if not np.isfinite(magnitude).all() or np.any(magnitude <= 0.0) or np.any(force[:, 2] <= 0.0):
        raise ValueError("PathCover heading requires a finite upright thrust direction")
    z = force / magnitude[:, None]
    zdot = (jerk - z * np.sum(z * jerk, axis=-1)[:, None]) / magnitude[:, None]
    # z.(zdot x h) = sin(yaw)*t1 + cos(yaw)*t2 = p*s.
    t1 = z[:, 2] * zdot[:, 0] - z[:, 0] * zdot[:, 2]
    t2 = z[:, 1] * zdot[:, 2] - z[:, 2] * zdot[:, 1]
    target = np.arctan2(velocity[:, 1], velocity[:, 0])
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    yaw = np.empty(count + 1)
    yaw[0] = trajectory.initial_yaw

    def bounds(index, angle):
        cosine, sine = math.cos(angle), math.sin(angle)
        c = z[index, 0] * cosine + z[index, 1] * sine
        s2 = 1.0 - c * c
        offset = c * (t1[index] * sine + t2[index] * cosine)
        return ((-trajectory.body_yaw_rate_max * s2 - offset) / z[index, 2],
                (trajectory.body_yaw_rate_max * s2 - offset) / z[index, 2])

    for i in range(count):
        step = times[i + 1] - times[i]
        angle = yaw[i]
        error = math.atan2(math.sin(target[2 * i + 2] - angle), math.cos(target[2 * i + 2] - angle))
        nominal = error / step if speed[2 * i + 2] > trajectory.yaw_low_speed else 0.0
        first_low, first_high = bounds(2 * i, angle)
        rate = min(max(nominal, first_low), first_high)
        for _ in range(2):
            middle_low, middle_high = bounds(2 * i + 1, angle + 0.5 * step * rate)
            last_low, last_high = bounds(2 * i + 2, angle + step * rate)
            low, high = max(first_low, middle_low, last_low), min(first_high, middle_high, last_high)
            if low > high:
                raise ValueError("PathCover heading step has no common body-r interval")
            rate = min(max(nominal, low), high)
        yaw[i + 1] = angle + step * rate
    return times, yaw


@dataclass(frozen=True)
class PathCoverTrajectory:
    # The upstream coefficient layout is [piece, axis, descending power].
    coefficients: np.ndarray
    durations: np.ndarray
    initial_yaw: float = 0.0
    body_yaw_rate_max: float = np.pi / 2
    yaw_sample_dt: float = 0.02
    yaw_low_speed: float = 0.1
    gravity: float = 9.81
    _boundaries: np.ndarray = field(init=False, repr=False)
    _yaw_times: np.ndarray = field(init=False, repr=False)
    _yaw_values: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if (np.any(np.asarray(self.durations) <= 0) or not np.isfinite(self.durations).all()
                or not np.isfinite(self.coefficients).all()):
            raise ValueError("MINCO coefficients must be finite and durations must be positive")
        object.__setattr__(self, "_boundaries", np.concatenate(([0.0], np.cumsum(self.durations))))
        times, yaw = _bounded_body_yaw_profile(self)
        object.__setattr__(self, "_yaw_times", times)
        object.__setattr__(self, "_yaw_values", yaw)

    @property
    def duration(self):
        return float(self._boundaries[-1])

    def _derivatives(self, times):
        times = np.asarray(times, dtype=np.float64)
        bounded = np.clip(times, 0.0, self.duration)
        indices = np.minimum(np.searchsorted(self._boundaries[1:], bounded, side="left"), len(self.durations) - 1)
        local = bounded - self._boundaries[indices]
        coefficients = self.coefficients[indices]
        degree = coefficients.shape[-1] - 1
        output = []
        for order in range(4):
            value = np.zeros(times.shape + (3,))
            for column in range(degree + 1 - order):
                factor = 1
                for derivative in range(order):
                    factor *= degree - column - derivative
                value = value * local[..., None] + coefficients[..., column] * factor
            if order:
                value = np.where(((times < 0) | (times > self.duration))[..., None], 0.0, value)
            output.append(value)
        return output

    def sample(self, relative_times):
        """Evaluate scalar/vector relative times; hold the terminal pose after expiry."""
        times = np.asarray(relative_times, dtype=np.float64)
        position, velocity, acceleration, jerk = self._derivatives(times)
        yaw = np.interp(times, self._yaw_times, self._yaw_values)
        index = np.clip(np.searchsorted(self._yaw_times[1:], times, side="left"), 0, len(self._yaw_times) - 2)
        yaw_rate = np.diff(self._yaw_values)[index] / np.diff(self._yaw_times)[index]
        yaw_rate = np.where((times < 0) | (times >= self.duration), 0.0, yaw_rate)
        return PathCoverSample(position, velocity, acceleration, jerk, yaw, yaw_rate)


@dataclass(frozen=True)
class PathCoverPlan:
    trajectory: PathCoverTrajectory
    corridors: tuple[np.ndarray, ...]
    segments: tuple[PathCoverSegment, ...]
    start_time: float
    valid_until: float
    path: np.ndarray
    local_goal: np.ndarray
    metrics: dict

    def sample(self, relative_times):
        return self.trajectory.sample(relative_times)


class NativePathCoverFrontend:
    """One observed map and native planning workspace per environment."""

    def __init__(self, config: PathCoverConfig | None = None):
        from . import _pathcover_native

        self.config = config or PathCoverConfig()
        self._native = _pathcover_native.NativeFrontend(self.config.native_options())

    def reset(self):
        self._native.reset()

    def update_cloud(self, points_world, sensor_origin):
        return self._native.update_cloud(
            np.asarray(points_world, dtype=np.float64).reshape(-1, 3),
            np.asarray(sensor_origin, dtype=np.float64).reshape(3),
        )

    def inflated_cloud(self):
        """Copy the occupied map for validation/recording, outside the control path."""
        return self._native.inflated_cloud()

    def obstacle_boxes(self):
        """Copy search/SFC AABBs as (centres[M,3], scalar half extent), for offline checks."""
        resolution = self.config.native_options()["search_resolution"]
        return self._native.obstacle_centers(), 0.5 * resolution

    def failed_problem(self):
        """Retrieve a GCOPTER failure snapshot; successful plans do not copy this data."""
        problem = self._native.failed_problem()
        if problem is not None:
            problem["config"] = self.config.native_options()
        return problem

    def plan(self, points_world, current_state: PathCoverState, goal, *, sensor_origin=None, initial_motion_seed=None):
        """Plan with an optional three-point world-space convex seed for the first SFC.

        CTBR supplies its exact first Euler positions in one batch. Omitting the
        seed retains point RISP; the front end does not infer a control period.
        """
        begin = perf_counter()
        update = None
        if points_world is not None:
            origin = current_state.position if sensor_origin is None else sensor_origin
            update = self.update_cloud(points_world, origin)
        result = self._native.plan(
            np.asarray(current_state.position, dtype=np.float64).reshape(3),
            np.asarray(current_state.velocity, dtype=np.float64).reshape(3),
            np.asarray(current_state.acceleration, dtype=np.float64).reshape(3),
            np.asarray(goal, dtype=np.float64).reshape(3),
            np.empty((0, 3)) if initial_motion_seed is None else np.asarray(initial_motion_seed, dtype=np.float64),
        )
        yaw_begin = perf_counter()
        trajectory = PathCoverTrajectory(
            coefficients=np.stack(result.coefficients), durations=np.asarray(result.durations),
            initial_yaw=current_state.yaw, body_yaw_rate_max=self.config.body_yaw_rate_max,
            yaw_sample_dt=self.config.yaw_sample_dt, yaw_low_speed=self.config.yaw_low_speed, gravity=self.config.gravity,
        )
        yaw_ms = (perf_counter() - yaw_begin) * 1000
        boundaries = np.concatenate(([0.0], np.cumsum(trajectory.durations)))
        segments = tuple(
            PathCoverSegment(int(index), float(boundaries[i]), float(boundaries[i + 1]), float(duration))
            for i, (index, duration) in enumerate(zip(result.corridor_indices, trajectory.durations, strict=True))
        )
        last_map = self._native.last_update
        metrics = {
            "mapping_ms": update.mapping_ms if update is not None else 0.0,
            "inflation_ms": update.inflation_ms if update is not None else 0.0,
            "grid_ms": update.grid_ms if update is not None else 0.0,
            "search_ms": result.search_ms, "corridor_ms": result.corridor_ms,
            "jps_ms": result.jps_ms, "distance_map_ms": result.distance_map_ms,
            "dmp_ms": result.dmp_ms, "dmp_cost": result.dmp_cost,
            "trajectory_ms": result.trajectory_ms, "yaw_ms": yaw_ms, "total_ms": (perf_counter() - begin) * 1000,
            "optimizer_status": result.optimizer_status, "objective": result.cost,
            "occupied_voxels": last_map.occupied_voxels, "inflated_voxels": last_map.inflated_voxels,
            "gpu_mapping": last_map.gpu_mapping, "gpu_inflation": last_map.gpu_inflation,
            "gpu_distance_map": result.gpu_distance_map,
            "speed_initial_m_s": result.initial_speed,
            "speed_nominal_limit_m_s": self.config.max_velocity,
            "speed_effective_limit_m_s": result.effective_speed_limit,
            "speed_nominal_excess_m_s": max(float(result.constraint_metrics[1]) - self.config.max_velocity, 0.0),
        }
        for name, value in zip(("sfc_max_m", "speed_max_m_s", "thrust_min_n", "thrust_max_n",
                                "tilt_max_rad", "zero_yaw_body_rate_max_rad_s"), result.constraint_metrics, strict=True):
            metrics[f"optimizer_{name}"] = float(value)
        return PathCoverPlan(
            trajectory, tuple(result.corridors), segments, float(current_state.time),
            float(current_state.time + trajectory.duration), np.asarray(result.path),
            np.asarray(result.local_goal), metrics,
        )


def _polynomial_maximum(coefficients, duration):
    """Exact scalar polynomial extrema up to numerical root accuracy."""
    coefficients = np.trim_zeros(np.asarray(coefficients), "f")
    if coefficients.size == 0:
        return 0.0
    derivative = np.polyder(coefficients)
    roots = np.roots(derivative) if derivative.size > 1 else []
    times = [0.0, duration]
    times.extend(float(root.real) for root in roots if abs(root.imag) < 1e-7 and 0 < root.real < duration)
    return float(np.max(np.polyval(coefficients, times)))


def validate_pathcover_plan(plan: PathCoverPlan, config: PathCoverConfig, *, sample_dt=0.005):
    """Offline evidence only: exact polynomial SFC/speed extrema and sampled flatness limits."""
    sfc_violation = -np.inf
    max_speed_squared = 0.0
    for coefficients, segment in zip(plan.trajectory.coefficients, plan.segments, strict=True):
        for face in plan.corridors[segment.corridor_index]:
            polynomial = face[:3] @ coefficients
            polynomial[-1] += face[3]
            sfc_violation = max(sfc_violation, _polynomial_maximum(polynomial, segment.duration))
        velocity_coefficients = coefficients[:, :-1] * np.arange(coefficients.shape[1] - 1, 0, -1)
        speed_squared = sum(np.convolve(axis, axis) for axis in velocity_coefficients)
        max_speed_squared = max(max_speed_squared, _polynomial_maximum(speed_squared, segment.duration))
    times = np.linspace(0.0, plan.trajectory.duration, max(2, int(np.ceil(plan.trajectory.duration / sample_dt)) + 1))
    sample = plan.sample(times)
    force = sample.acceleration + np.array([0.0, 0.0, config.gravity])
    norm = np.linalg.norm(force, axis=-1)
    z = force / norm[:, None]
    z_dot = (sample.jerk - z * np.sum(sample.jerk * z, axis=-1)[:, None]) / norm[:, None]
    heading = np.stack((np.cos(sample.yaw), np.sin(sample.yaw), np.zeros_like(sample.yaw)), axis=-1)
    heading_dot = sample.yaw_rate[:, None] * np.stack(
        (-np.sin(sample.yaw), np.cos(sample.yaw), np.zeros_like(sample.yaw)), axis=-1,
    )
    cross = np.cross(z, heading)
    cross_norm = np.linalg.norm(cross, axis=-1)
    y = cross / cross_norm[:, None]
    cross_dot = np.cross(z_dot, heading) + np.cross(z, heading_dot)
    y_dot = (cross_dot - y * np.sum(y * cross_dot, axis=-1)[:, None]) / cross_norm[:, None]
    x = np.cross(y, z)
    body_rates = np.stack((-np.sum(y * z_dot, axis=-1), np.sum(x * z_dot, axis=-1),
                           -np.sum(x * y_dot, axis=-1)), axis=-1)
    return {
        "sfc_max_violation_m": float(sfc_violation),
        "max_speed_m_s": float(np.sqrt(max_speed_squared)),
        "max_acceleration_m_s2": float(np.linalg.norm(sample.acceleration, axis=-1).max()),
        "min_thrust_n": float((config.mass * norm).min()),
        "max_thrust_n": float((config.mass * norm).max()),
        "max_tilt_rad": float(np.arccos(np.clip(z[:, 2], -1, 1)).max()),
        "max_body_rate_rad_s": float(np.linalg.norm(body_rates, axis=-1).max()),
        "max_abs_body_rate_components_rad_s": np.abs(body_rates).max(axis=0).tolist(),
        "max_world_yaw_rate_rad_s": float(np.abs(sample.yaw_rate).max()),
        "finite": bool(np.isfinite(sample.pva).all() and np.isfinite(body_rates).all()),
        "optimizer_status": int(plan.metrics["optimizer_status"]),
        "dynamic_sampling_dt_s": float(times[1] - times[0]),
    }
