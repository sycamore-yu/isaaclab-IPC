"""ROS-free NanoVoxMap → global DMP → PathCover/RISP → GCOPTER/MINCO.

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
    # Physical reference speed diagnostic. The cruise-speed penalty can use a
    # lower nominal value; finite soft violations remain visible in plan metrics.
    max_velocity: float = 6.0
    nominal_velocity: float | None = None  # Defaults to min(3 m/s, physical bound).
    mass: float = 0.75
    gravity: float = 9.81
    # Official reference limits scale with mass; controller factories intersect
    # them with their physical force limits. Explicit historical limits persist.
    thrust_min: float | None = None
    thrust_max: float | None = None
    body_rate_max: float = 3.0
    tilt_max: float = 0.7854
    max_corridors: int = 6
    pieces_per_polytope: int = 1
    integral_resolution: int = 64
    max_iterations: int = 1000
    relative_tolerance: float = 1e-4
    time_weight: float = 100.0
    # Metre-valued inward offset of the reference position-penalty copy only.
    safety_margin: float = 0.15
    penalty_weights: tuple[float, ...] = (1e4, 100.0, 1e4, 1e4, 1e4)
    # Width in metres of the smoothed L1 position penalty. Dynamic smoothing
    # retains the upstream 0.01 in each corresponding constraint's units.
    position_penalty_smoothing_m: float = 1e-6
    potential_radius: float = 1.0
    # Used by the upstream JPS-guide comparison; native global DMP has no guide region.
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

    def __post_init__(self):
        nominal = min(3.0, self.max_velocity) if self.nominal_velocity is None else self.nominal_velocity
        if (not math.isfinite(self.max_velocity) or not math.isfinite(nominal)
                or not 0 < nominal <= self.max_velocity):
            raise ValueError("PathCover speeds require 0 < nominal_velocity <= max_velocity")
        # Resolve defaults in saved snapshots as concrete values, so later
        # default changes cannot reinterpret a recorded planning problem.
        object.__setattr__(self, "nominal_velocity", nominal)
        if self.thrust_min is None:
            object.__setattr__(self, "thrust_min", 2.0 / 0.716 * self.mass)
        if self.thrust_max is None:
            object.__setattr__(self, "thrust_max", 10.0 / 0.716 * self.mass)
        if (not math.isfinite(self.thrust_min) or not math.isfinite(self.thrust_max)
                or not 0 <= self.thrust_min < self.thrust_max):
            raise ValueError("PathCover reference thrust limits require 0 <= thrust_min < thrust_max")

    def native_options(self):
        values = asdict(self)
        if self.search_resolution is None:
            values["search_resolution"] = self.resolution
        if self.robot_radius_z is None:
            values["robot_radius_z"] = self.robot_radius
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
    """Integrate pointwise bounded body-r with native RK4 and Hermite interpolation.

    Heading feedback uses sin(heading - yaw) / yaw_sample_dt and fades by
    min(horizontal_speed / yaw_low_speed, 1). Its antipodal equilibrium is unstable;
    it does not claim global convergence.
    Native local tolerances are 1e-7 rad for RK4 step doubling and 1e-4 rad/s
    for the Hermite body-r defect; these are not a global trajectory certificate.
    """
    from . import _pathcover_native

    profile = _pathcover_native.bounded_body_yaw_profile(
        trajectory.coefficients, trajectory.durations, trajectory.initial_yaw,
        trajectory.body_yaw_rate_max, trajectory.yaw_sample_dt,
        trajectory.yaw_low_speed, trajectory.gravity,
    )
    return profile[:, 0], profile[:, 1], profile[:, 2]


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
    # Saved pre-ODE plans have no endpoint rates; retain their recorded linear
    # profile when reading those historical diagnostic snapshots.
    _yaw_rates: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if (np.any(np.asarray(self.durations) <= 0) or not np.isfinite(self.durations).all()
                or not np.isfinite(self.coefficients).all()):
            raise ValueError("MINCO coefficients must be finite and durations must be positive")
        object.__setattr__(self, "_boundaries", np.concatenate(([0.0], np.cumsum(self.durations))))
        times, yaw, rates = _bounded_body_yaw_profile(self)
        object.__setattr__(self, "_yaw_times", times)
        object.__setattr__(self, "_yaw_values", yaw)
        object.__setattr__(self, "_yaw_rates", rates)

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
        index = np.clip(np.searchsorted(self._yaw_times[1:], times, side="left"), 0, len(self._yaw_times) - 2)
        step = self._yaw_times[index + 1] - self._yaw_times[index]
        if self._yaw_rates is None:
            yaw = np.interp(times, self._yaw_times, self._yaw_values)
            yaw_rate = np.diff(self._yaw_values)[index] / step
        else:
            fraction = (np.clip(times, 0.0, self.duration) - self._yaw_times[index]) / step
            s2, s3 = fraction**2, fraction**3
            y0, y1 = self._yaw_values[index], self._yaw_values[index + 1]
            d0, d1 = self._yaw_rates[index], self._yaw_rates[index + 1]
            yaw = ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + fraction) * step * d0
                   + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * step * d1)
            yaw_rate = (6 * fraction * (1 - fraction) * (y1 - y0) / step
                        + (3 * s2 - 4 * fraction + 1) * d0 + (3 * s2 - 2 * fraction) * d1)
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
        self._yaw_failure = None

    def reset(self):
        self._yaw_failure = None
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
        """Copy failure diagnostics on request; successful plans do not copy this data."""
        if self._yaw_failure is not None:
            result, state, error = self._yaw_failure
            constraint_metrics = np.asarray(result.constraint_metrics).copy()
            return {
                "stage": "yaw", "error": error, "config": self.config.native_options(),
                "initial_pva": np.column_stack((state.position, state.velocity, state.acceleration)),
                "terminal_pva": np.column_stack((result.local_goal, np.zeros(3), np.zeros(3))),
                "initial_yaw": float(state.yaw), "start_time": float(state.time),
                "corridors": tuple(result.corridors), "optimizer_status": result.optimizer_status,
                "trial_durations": np.asarray(result.durations).copy(),
                "constraint_metrics": constraint_metrics,
                "constraint_metrics_order": ("sfc_max_m", "speed_max_m_s", "thrust_min_n",
                                             "thrust_max_n", "tilt_max_rad", "rate_max_rad_s"),
                "constraint_finite": bool(np.isfinite(constraint_metrics).all()),
                "time_initialization": result.time_initialization,
                "candidate": {
                    "coefficients": np.stack(result.coefficients),
                    "durations": np.asarray(result.durations).copy(),
                    "corridor_indices": np.asarray(result.corridor_indices).copy(),
                    "objective": float(result.cost),
                },
            }
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
        self._yaw_failure = None
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
        try:
            trajectory = PathCoverTrajectory(
                coefficients=np.stack(result.coefficients), durations=np.asarray(result.durations),
                initial_yaw=current_state.yaw, body_yaw_rate_max=self.config.body_yaw_rate_max,
                yaw_sample_dt=self.config.yaw_sample_dt, yaw_low_speed=self.config.yaw_low_speed, gravity=self.config.gravity,
            )
        except PathCoverPlanningError as error:
            # Retain the existing native result only. Extract arrays if a caller
            # explicitly requests failed_problem(), after the plan has failed.
            self._yaw_failure = (result, current_state, str(error))
            raise
        yaw_ms = (perf_counter() - yaw_begin) * 1000
        boundaries = np.concatenate(([0.0], np.cumsum(trajectory.durations)))
        segments = tuple(
            PathCoverSegment(int(index), float(boundaries[i]), float(boundaries[i + 1]), float(duration))
            for i, (index, duration) in enumerate(zip(result.corridor_indices, trajectory.durations, strict=True))
        )
        last_map = self._native.last_update
        nominal_velocity = self.config.max_velocity if self.config.nominal_velocity is None else self.config.nominal_velocity
        initialization = result.time_initialization
        metrics = {
            "mapping_ms": update.mapping_ms if update is not None else 0.0,
            "inflation_ms": update.inflation_ms if update is not None else 0.0,
            "grid_ms": update.grid_ms if update is not None else 0.0,
            "search_ms": result.search_ms, "corridor_ms": result.corridor_ms,
            "jps_ms": result.jps_ms, "distance_map_ms": result.distance_map_ms,
            "dmp_ms": result.dmp_ms, "dmp_cost": result.dmp_cost,
            "search_backend": "global_dmp", "jps_ran": False,
            "trajectory_ms": result.trajectory_ms, "yaw_ms": yaw_ms, "total_ms": (perf_counter() - begin) * 1000,
            "time_initialization_ms": initialization["elapsed_ms"],
            "time_initialization_status": initialization["status"],
            "time_initialization_evaluations": initialization["evaluations"],
            "time_initialization_scale": initialization["scale"],
            "optimizer_status": result.optimizer_status, "objective": result.cost,
            "optimizer_quadrature_feasible": result.optimizer_quadrature_feasible,
            "constraint_metrics_semantics": "optimizer quadrature diagnostics; not continuous-time feasibility",
            "reference_margin_requested_m": self.config.safety_margin,
            "reference_margin_applied_per_corridor_m": np.asarray(result.applied_reference_margins).tolist(),
            "occupied_voxels": last_map.occupied_voxels, "inflated_voxels": last_map.inflated_voxels,
            "gpu_mapping": last_map.gpu_mapping, "gpu_inflation": last_map.gpu_inflation,
            "gpu_distance_map": result.gpu_distance_map,
            "speed_initial_m_s": result.initial_speed,
            "speed_nominal_limit_m_s": nominal_velocity,
            "speed_physical_limit_m_s": self.config.max_velocity,
            "speed_acceptance_limit_m_s": result.speed_acceptance_limit,
            "speed_nominal_excess_m_s": max(float(result.constraint_metrics[1]) - nominal_velocity, 0.0),
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
