# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, The Isaac Lab Diff Flight Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton port of Isaac Lab's four-thruster Multirotor asset.

Source: Isaac Lab v3.0.0, commit e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4,
``source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py``.

The upstream asset is bound to PhysX. This port retains Newton's initialized
articulation data and reuses the upstream ``MultirotorCfg``, ``Thruster`` and
``MultiRotorActions``. One group owns four independently commanded motors.

The environment's Newton state-force callback calls ``advance_motors`` once per
solver step, then applies ``body_wrench`` to that callback's current state.
``write_data_to_sim`` does not advance the motors or submit another force.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab.utils import instantiate
from isaaclab.utils.string import resolve_matching_names_values
from isaaclab_contrib.actuators import Thruster
from isaaclab_contrib.utils.types import MultiRotorActions
from isaaclab_newton.assets.articulation import Articulation

if TYPE_CHECKING:
    from isaaclab_contrib.assets.multirotor.multirotor_cfg import MultirotorCfg

logger = logging.getLogger(__name__)


class _ThrusterCollection(dict[str, Thruster]):
    """Provide the reset interface used by Newton's articulation lifecycle."""

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        for actuator in self.values():
            actuator.reset_idx(env_ids)


class NewtonMultirotor(Articulation):
    """Four-motor Multirotor API on Newton's concrete articulation backend.

    Commands and motor states use newtons. The official ``Thruster`` integrates
    equivalent rotations per second internally. ``body_wrench`` is ordered
    ``[Fx, Fy, Fz, Tx, Ty, Tz]`` in the root body frame.

    The Torch tensors and their Warp aliases remain at fixed addresses after
    initialization. The step callback must use ``advance_motors`` exactly once
    for each ``cfg.actuators[group].dt`` of simulated time.
    """

    cfg: MultirotorCfg

    @property
    def thruster_names(self) -> list[str]:
        """Names in command/allocation-column order."""
        return self._data.thruster_names

    @property
    def num_thrusters(self) -> int:
        """Number of independently commanded motors."""
        return 4

    @property
    def allocation_matrix(self) -> torch.Tensor:
        """Persistent 6-by-4 body-wrench allocation matrix."""
        return self._allocation_matrix

    @property
    def motor_thrust(self) -> torch.Tensor:
        """Persistent upstream motor state, before the final output clamp [N]."""
        return self._thruster.curr_thrust

    @property
    def body_wrench(self) -> torch.Tensor:
        """Persistent applied body-frame wrench, shaped (num_envs, 6)."""
        return self._body_wrench

    def set_thrust_target(
        self,
        target: torch.Tensor,
        thruster_ids: Sequence[int] | slice | torch.Tensor | None = None,
        env_ids: Sequence[int] | slice | torch.Tensor | None = None,
    ) -> None:
        """Set per-motor target thrust [N] without advancing motor state."""
        env_index = slice(None) if env_ids is None else env_ids
        motor_index = slice(None) if thruster_ids is None else thruster_ids
        if not isinstance(env_index, slice) and not isinstance(motor_index, slice):
            env_index = torch.as_tensor(env_index, device=self.device, dtype=torch.long)[:, None]
        self._data.thrust_target[env_index, motor_index] = target

    @torch.no_grad()
    def advance_motors(self) -> None:
        """Advance the official motor model once and update the applied wrench.

        Run inside the Newton state-force callback, on its current Warp stream.
        Newton performs an eager step before capture, so the same upstream
        Torch operations are warmed before CUDA graph recording.
        """
        stream_context = (
            torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.device))) if self._is_cuda else nullcontext()
        )
        with stream_context:
            # Thruster.compute replaces actions.thrusts with its output.
            self._motor_action.thrusts = self._data.thrust_target
            action = self._thruster.compute(self._motor_action)
            self._data.applied_thrust.copy_(action.thrusts)
            torch.mm(self._data.applied_thrust, self._allocation_transpose, out=self._body_wrench)

    def write_data_to_sim(self) -> None:
        """Leave shared targets for the Newton callback to consume at solver rate."""

    @torch.no_grad()
    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Reset selected motors and targets while retaining all buffer addresses."""
        if env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).flatten()
        elif isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids).long()
        super().reset(env_ids, env_mask)
        index = slice(None) if env_ids is None else env_ids
        # Newton's parent reset has already called the official Thruster reset.
        self._data.thrust_target[index] = self.motor_thrust[index]
        self._data.applied_thrust[index] = self.motor_thrust[index].clamp(
            self._thruster.min_thrust, self._thruster.max_thrust
        )
        # Recomputing this small product preserves the untouched rows exactly.
        torch.mm(self._data.applied_thrust, self._allocation_transpose, out=self._body_wrench)

    @torch.no_grad()
    def reset_motor_thrust(
        self,
        thrust: torch.Tensor,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Initialize selected motor states and targets to physical thrust [N].

        This sets an episode's electromechanical initial condition; it is not
        a feedback controller. For level airborne resets, equal per-motor
        thrust mass*gravity/4 is consistent with zero vertical acceleration.
        A fixed initial RPS is not: independently randomized thrust constants
        otherwise produce unequal forces and a spurious initial torque.

        The official Thruster's persistent dynamic state is curr_thrust.
        Its thrust constants, time constants and other randomized parameters
        are retained. Buffers keep their addresses for captured solver graphs.
        Input may be [selected_envs,4] or a broadcastable [4] thrust vector.
        """
        if isinstance(env_ids, wp.array):
            env_ids = wp.to_torch(env_ids).long()
        index = slice(None) if env_ids is None else env_ids
        current = self._thruster.curr_thrust
        values = torch.as_tensor(thrust, dtype=current.dtype, device=current.device)
        try:
            values = torch.broadcast_to(values, current[index].shape)
        except RuntimeError as error:
            raise ValueError(
                "Motor initial thrust must broadcast to the selected environments and four motors"
            ) from error
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Motor initial thrust must be finite")
        if not bool(((values >= self._thruster.min_thrust) & (values <= self._thruster.max_thrust)).all()):
            raise ValueError("Motor initial thrust must lie within the physical actuator bounds")
        # Index assignment updates the original tensors even for noncontiguous
        # environment IDs. Do not copy_ an advanced-index temporary.
        current[index] = values
        self._data.thrust_target[index] = values
        self._data.applied_thrust[index] = values
        self._body_wrench[index] = values @ self._allocation_transpose

    def _configure_joint_target_modes(self, _event) -> None:
        """Thrusters use body wrenches; there are no joint-actuator target modes."""

    def _process_cfg(self) -> None:
        """Initialize the free-root state without resolving joint target regexes."""
        if self.num_joints != 0:
            raise ValueError("NewtonMultirotor requires a free root and fixed rotor links, without actuated joints.")
        # Newton excludes FREE/FIXED joints from its DOF view. The inherited
        # default joint regex therefore has no target; only the root state is
        # configurable for this asset. Follow the parent root-state layout.
        root_pose = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
        root_velocity = tuple(self.cfg.init_state.lin_vel) + tuple(self.cfg.init_state.ang_vel)
        self._data.default_root_pose = wp.array(
            np.tile(np.asarray(root_pose, dtype=np.float32), (self.num_instances, 1)),
            dtype=wp.transformf,
            device=self.device,
        )
        self._data.default_root_vel = wp.array(
            np.tile(np.asarray(root_velocity, dtype=np.float32), (self.num_instances, 1)),
            dtype=wp.spatial_vectorf,
            device=self.device,
        )

    def _process_actuators_cfg(self) -> None:
        """Add thruster buffers to the data already initialized by Newton."""
        if len(self.cfg.actuators) != 1:
            raise ValueError("NewtonMultirotor uses one Thruster group containing four motors.")

        group_name, actuator_cfg = next(iter(self.cfg.actuators.items()))
        body_ids, names = self.find_bodies(actuator_cfg.thruster_names_expr, preserve_order=True)
        if len(names) != 4:
            raise ValueError(f"Expected four rotor bodies, resolved {names}.")
        self._thruster_body_indices = tuple(body_ids)
        self._data.thruster_names = names
        self._is_cuda = wp.get_device(self.device).is_cuda

        self._allocation_matrix = torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)
        if self._allocation_matrix.shape != (6, 4):
            raise ValueError("The allocation matrix must have shape (6, 4), ordered Fx,Fy,Fz,Tx,Ty,Tz.")
        self._allocation_transpose = self._allocation_matrix.T.contiguous()

        # Resolve RPS before constructing Thruster. It keeps this initial state
        # and uses it again during each partial reset.
        initial_rps = torch.zeros((self.num_instances, 4), device=self.device)
        indices, _, values = resolve_matching_names_values(self.cfg.init_state.rps, names)
        initial_rps[:, indices] = torch.tensor(values, device=self.device, dtype=initial_rps.dtype)
        self._data.default_thruster_rps = initial_rps
        self._thruster = instantiate(
            actuator_cfg,
            thruster_names=names,
            thruster_ids=slice(None),
            num_envs=self.num_instances,
            device=self.device,
            init_thruster_rps=initial_rps,
        )
        if not isinstance(self._thruster, Thruster):
            raise TypeError("The actuator must be the official Isaac Lab Thruster model.")
        self.actuators = _ThrusterCollection({group_name: self._thruster})
        self._has_implicit_actuators = False
        self._has_newton_actuators = False

        # Upstream's persisted motor state is curr_thrust [N]. Keep it as the
        # computed-thrust field rather than creating a second state variable.
        self._data.computed_thrust = self._thruster.curr_thrust
        self._data.thrust_target = self._thruster.curr_thrust.clone()
        self._data.applied_thrust = self._thruster.curr_thrust.clamp(
            self._thruster.min_thrust, self._thruster.max_thrust
        )
        self._body_wrench = torch.empty((self.num_instances, 6), device=self.device)
        torch.mm(self._data.applied_thrust, self._allocation_transpose, out=self._body_wrench)
        self._motor_action = MultiRotorActions(thrusts=self._data.thrust_target, thruster_indices=slice(None))

        # Persistent zero-copy views for the environment's Warp control/force
        # kernel. Motor advancement is owned by advance_motors, not that kernel.
        self.thrust_target_wp = wp.from_torch(self._data.thrust_target, dtype=wp.float32)
        self.motor_thrust_wp = wp.from_torch(self._thruster.curr_thrust, dtype=wp.float32)
        self.applied_thrust_wp = wp.from_torch(self._data.applied_thrust, dtype=wp.float32)
        self.body_wrench_wp = wp.from_torch(self._body_wrench, dtype=wp.float32)

    def _log_articulation_info(self) -> None:
        """Report the actuator layout without querying unused joint gains."""
        logger.info("NewtonMultirotor %s: four motors in order %s", self.cfg.prim_path, self.thruster_names)
