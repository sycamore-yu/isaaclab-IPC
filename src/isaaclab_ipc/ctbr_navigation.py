"""Freeze native PathCover plans into the same CTBR observations for both runtimes.

Only observation acquisition calls the frontend. PPO receives the resulting
tensors, including the warm start, without calling geometry or changing a plan.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch


@dataclass(frozen=True)
class SampledPlan:
    pva: np.ndarray
    yaw: np.ndarray
    planes: np.ndarray
    plane_mask: np.ndarray
    corridor_index: np.ndarray
    reference_violation: float


def position_sfc_violation(plan, position, now):
    """Measure the observed position against the frontend's active corridor."""
    ends = np.asarray([segment.end for segment in plan.segments])
    relative = np.clip(float(now) - plan.start_time, 0.0, ends[-1])
    index = min(np.searchsorted(ends, relative, side="right"), len(ends) - 1)
    corridors = [plan.segments[index].corridor_index]
    if index > 0 and abs(relative - ends[index - 1]) <= 1e-9:
        corridors.append(plan.segments[index - 1].corridor_index)
    return max(0.0, *(float(np.max(plan.corridors[i][:, :3] @ position + plan.corridors[i][:, 3]))
                       for i in corridors))


def sample_plan(plan, now, cost_nodes, dt, max_planes=128):
    """Sample t0+k*dt; keep adjacent corridor faces at an exact junction.

    Polynomials and their corridor ownership use the frontend's own segment
    times. After the final segment the reference holds its endpoint. This does
    not re-integrate or accept/reject the MPC execution trajectory.
    """
    segments = plan.segments
    ends = np.asarray([segment.end for segment in segments], dtype=np.float64)
    relative = float(now) - float(plan.start_time) + np.arange(cost_nodes) * float(dt)
    clipped = np.clip(relative, 0.0, ends[-1])
    values = plan.trajectory.sample(clipped)
    pva = np.array(values.pva, dtype=np.float64, copy=True)
    pva[relative > ends[-1], 3:] = 0.0
    yaw = np.array(values.yaw, dtype=np.float64, copy=True)
    indices = np.minimum(np.searchsorted(ends, clipped, side="right"), len(segments) - 1)
    planes = np.zeros((cost_nodes, max_planes, 4), dtype=np.float64)
    masks = np.zeros((cost_nodes, max_planes), dtype=bool)
    ownership = np.asarray([segments[index].corridor_index for index in indices], dtype=np.int64)
    violation = 0.0
    for k, (time, index) in enumerate(zip(clipped, indices, strict=True)):
        blocks = [np.asarray(plan.corridors[ownership[k]], dtype=np.float64)]
        # The reference is continuous at a segment boundary. Enforcing both
        # neighboring SFCs there encodes that intersection without a switch FSM.
        if index > 0 and abs(time - ends[index - 1]) <= 1e-9:
            previous = segments[index - 1].corridor_index
            if previous != ownership[k]:
                blocks.append(np.asarray(plan.corridors[previous], dtype=np.float64))
        faces = np.concatenate(blocks, axis=0)
        if len(faces) > max_planes:
            raise ValueError(f"Sample requires {len(faces)} SFC faces; configured capacity is {max_planes}")
        planes[k, :len(faces)], masks[k, :len(faces)] = faces, True
        if len(faces):
            violation = max(violation, float(np.max(faces[:, :3] @ pva[k, :3] + faces[:, 3])))
    return SampledPlan(pva, yaw, planes, masks, ownership, violation)


class PathCoverNavigation:
    """One observed map per environment, one packed transfer per policy batch."""

    def __init__(self, num_envs, ctbr_config, frontend_config, *, device, max_planes=128):
        from .pathcover_native import NativePathCoverFrontend
        from .turbompc.quadrotor import CTBRDynamics

        self.config = ctbr_config
        self.device = torch.device(device)
        self.max_planes = int(max_planes)
        self._dynamics = CTBRDynamics(dt=ctbr_config.dt, mass=ctbr_config.mass, gravity=ctbr_config.gravity)
        self._motion_seed = (torch.compile(self._initial_motion_seed, fullgraph=True, dynamic=False)
                             if self.device.type == "cuda" else self._initial_motion_seed)
        self.frontends = [NativePathCoverFrontend(frontend_config) for _ in range(num_envs)]
        self._workers = (ThreadPoolExecutor(max_workers=min(4, num_envs), thread_name_prefix="pathcover")
                         if num_envs > 1 else None)
        self.last_plans = [None] * num_envs
        self.frontend_times_ms = []
        self.frontend_batch_times_ms = []
        self.last_reference_metrics = [None] * num_envs
        if self.device.type == "cuda":
            # Compile both initial/reset and receding-control forms before the
            # first timed plan, using canonical strides identical to observe().
            with torch.no_grad():
                state = torch.zeros((num_envs, 10), dtype=torch.float64, device=self.device)
                state[:, 3] = 1.0
                acceleration = state.new_zeros((num_envs, 3))
                self._motion_seed(state, acceleration, None)
                self._motion_seed(state, acceleration, state.new_full((num_envs,), ctbr_config.mass * ctbr_config.gravity))
                torch.cuda.synchronize(self.device)

    def _initial_motion_seed(self, state, acceleration, first_force):
        """Protect the first two exact Euler positions in the initial corridor.

        Only the first thrust affects p2; body rates first affect position at p3.
        This geometric seed does not certify feasibility of the complete MPC.
        """
        controls = state.new_zeros((len(state), 4))
        if first_force is None:
            controls[:, 0] = self.config.mass
            gravity = state.new_tensor([0.0, 0.0, self.config.gravity])
            body_z = self._dynamics.acceleration(torch.cat((state, controls), -1)) + gravity
            first_force = self.config.mass * ((acceleration + gravity) * body_z).sum(-1) / body_z.square().sum(-1)
        lower, upper = self.config.specific_thrust_limits
        controls[:, 0] = first_force.clamp(self.config.mass * lower, self.config.mass * upper)
        next_velocity = state[:, 7:10] + self.config.dt * self._dynamics.acceleration(torch.cat((state, controls), -1))
        next_position = state[:, :3] + self.config.dt * state[:, 7:10]
        return torch.stack((state[:, :3], next_position, next_position + self.config.dt * next_velocity), 1)

    def reset(self, env_ids=None):
        ids = range(len(self.frontends)) if env_ids is None else env_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        for index in ids:
            self.frontends[int(index)].reset()
            self.last_plans[int(index)] = None
            self.last_reference_metrics[int(index)] = None

    @torch.no_grad()
    def observe(self, ctbr_state, body_rates, acceleration, goals, clouds, time, *,
                warm_start_controls=None, env_ids=None, allow_failures=False):
        """Plan selected environments and return snapshots in selection order.

        Inputs retain global environment indexing. With ``allow_failures``, return
        ``(snapshots, errors_by_env_id)``. Failed rows are explicit invalid
        placeholders for the environment's termination/reset path, never actions.
        Only expected planning failures are collected; programming errors propagate.
        """
        from .pathcover_native import PathCoverPlanningError, PathCoverState
        from .rl.ctbr_observation import make_ctbr_observation

        ids = list(range(len(self.frontends))) if env_ids is None else env_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        ids = [int(index) for index in ids]
        if not ids:
            raise ValueError("A planning batch needs at least one environment")
        device_ids = torch.as_tensor(ids, device=ctbr_state.device)
        # Keep the GPU kernel's batch shape fixed during partial episode resets.
        # The seed shares the existing packed transfer with measured context.
        force = None if warm_start_controls is None else warm_start_controls[:, 0, 0]
        # RSL uses inference tensors during rollout. Canonical normal tensors
        # match startup warmup and avoid compiling a second set of dispatch keys.
        with torch.inference_mode(False), torch.no_grad():
            motion_seed = self._motion_seed(
                ctbr_state.clone(memory_format=torch.contiguous_format),
                acceleration.to(ctbr_state).clone(memory_format=torch.contiguous_format),
                None if force is None else force.clone(memory_format=torch.contiguous_format),
            )
        # Transfer measured context once; no per-knot device/host round trips.
        physical = torch.cat((ctbr_state[device_ids], acceleration[device_ids], goals[device_ids],
                              motion_seed[device_ids].flatten(1)),
                             dim=-1).detach().cpu().numpy()
        if isinstance(time, torch.Tensor):
            time = time.detach().cpu().numpy()
        times = np.broadcast_to(np.asarray(time, dtype=float), (len(self.frontends),))
        batch_started = perf_counter()

        def plan_one(row):
            index = ids[row]
            frontend, values, now = self.frontends[index], physical[row], times[index]
            qw, qx, qy, qz = values[3:7]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            state = PathCoverState(position=values[:3], velocity=values[7:10],
                                   acceleration=values[10:13], yaw=float(yaw), time=float(now))
            points = clouds.get(index) if isinstance(clouds, dict) else clouds[index]
            started = perf_counter()
            try:
                plan = frontend.plan(points, state, values[13:16], initial_motion_seed=values[16:].reshape(3, 3))
            except PathCoverPlanningError as error:
                return None, error, 1000 * (perf_counter() - started), None
            elapsed = 1000 * (perf_counter() - started)
            snapshot = sample_plan(plan, now, self.config.cost_nodes, self.config.dt, self.max_planes)
            metrics = {
                **plan.metrics,
                "reference_sfc_violation_m": snapshot.reference_violation,
                "corridor_count": len(plan.corridors),
                "trajectory_duration_s": float(plan.valid_until - plan.start_time),
            }
            return plan, snapshot, elapsed, metrics

        if self._workers is None:
            results = [plan_one(0)]
        else:
            futures = [self._workers.submit(plan_one, row) for row in range(len(ids))]
            # Finish the whole batch before surfacing a failure, so reset/close
            # cannot mutate a map that another worker is still using.
            wait(futures)
            results = [future.result() for future in futures]
        self.frontend_times_ms.extend(result[2] for result in results)
        self.frontend_batch_times_ms.append(1000 * (perf_counter() - batch_started))
        errors = {index: result[1] for index, result in zip(ids, results, strict=True) if result[0] is None}
        if errors and not allow_failures:
            raise next(iter(errors.values()))
        sampled = []
        for index, (plan, snapshot, _, metrics) in zip(ids, results, strict=True):
            if plan is not None:
                self.last_plans[index] = plan
                self.last_reference_metrics[index] = metrics
            else:
                nodes = self.config.cost_nodes
                snapshot = SampledPlan(np.zeros((nodes, 9)), np.zeros(nodes),
                                       np.zeros((nodes, self.max_planes, 4)),
                                       np.zeros((nodes, self.max_planes), dtype=bool),
                                       np.full(nodes, -1), 0.0)
                self.last_reference_metrics[index] = {"planning_failure": str(errors[index])}
            sampled.append(snapshot)

        def tensor(name, dtype=torch.float64):
            return torch.as_tensor(np.stack([getattr(value, name) for value in sampled]),
                                   dtype=dtype, device=self.device)

        observations = make_ctbr_observation(
            ctbr_state[device_ids], body_rates[device_ids], goals[device_ids], tensor("pva"), tensor("yaw"),
            tensor("planes"), tensor("plane_mask", torch.bool), ctbr_config=self.config,
            reference_valid=torch.tensor([index not in errors for index in ids], device=self.device),
            warm_start_controls=None if warm_start_controls is None else warm_start_controls[device_ids],
        )
        return (observations, errors) if allow_failures else observations

    def close(self):
        if self._workers is not None:
            self._workers.shutdown(wait=True)
        self.frontends.clear()
        self.last_plans.clear()
