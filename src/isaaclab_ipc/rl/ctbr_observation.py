"""Frozen CTBR policy/MPC observations shared by Newton and MARSIM."""

import numpy as np
import torch

from isaaclab_ipc.turbompc.quadrotor import reference_from_pva


def ctbr_feature_size(cost_nodes):
    return 14 + 11 * cost_nodes


def _reference_controls(reference, pva, mass, gravity, dt):
    thrust = pva[..., 6:9].clone()
    thrust[..., 2] += gravity
    force = mass * thrust.norm(dim=-1, keepdim=True)
    q0, q1 = reference[:, :-1, 3:7], reference[:, 1:, 3:7]
    scalar = (q0 * q1).sum(-1, keepdim=True)
    vector = q0[..., :1] * q1[..., 1:] - q1[..., :1] * q0[..., 1:] - torch.linalg.cross(q0[..., 1:], q1[..., 1:])
    sign = torch.where(scalar < 0, -1.0, 1.0)
    scalar, vector = scalar * sign, vector * sign
    norm = vector.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > 1e-8, 2 * torch.atan2(norm, scalar) / norm.clamp_min(1e-8), 2.0)
    rates = vector * (scale / dt)
    return torch.cat((force, torch.cat((rates, rates[:, -1:]), 1)), -1)


def _policy_features(state, rates, goal, reference, planes, mask, valid, config):
    normals, offsets = planes[..., :3], planes[..., 3]
    clearance = -(normals * reference[..., None, :3]).sum(-1) - offsets
    clearance = clearance / normals.norm(dim=-1).clamp_min(1e-12)
    if planes.shape[-2]:
        clearance = clearance.masked_fill(~mask, torch.inf).amin(-1).clamp(-10, 10)
    else:
        clearance = state.new_full(reference.shape[:2], 10.0)
    return torch.cat(
        (
            (goal.double() - state[:, :3]) / 10,
            state[:, 3:7],
            state[:, 7:10] / 10,
            rates / state.new_tensor(config.body_rate_limits),
            ((reference[..., :3] - state[:, None, :3]) / 10).flatten(1),
            reference[..., 3:7].flatten(1),
            (reference[..., 7:10] / 10).flatten(1),
            clearance / 10,
            valid,
        ),
        -1,
    ).float()


@torch.no_grad()
def make_ctbr_observation(
    ctbr_state, body_rates, goal, reference_pva, yaw, planes, plane_mask, *,
    ctbr_config, reference_valid=None, warm_start_controls=None,
):
    """Copy one rollout context; PPO may replay it without calling a frontend.

    All physical inputs use the same world axes and translation origin. SFC
    faces are [A,-b], in metres after unit-normal normalization. A node k is
    sampled at t_now+k*dt. No corridor is shrunk again here.
    """
    state, rates, reference_pva = ctbr_state.double(), body_rates.double(), reference_pva.double()
    batch, nodes = len(state), ctbr_config.cost_nodes
    if state.shape != (batch, 10) or rates.shape != (batch, 3):
        raise ValueError("Expected CTBR state[B,10] and body_rates[B,3]")
    if goal.shape == (3,):
        goal = goal.expand(batch, -1)
    if goal.shape != (batch, 3) or reference_pva.shape != (batch, nodes, 9):
        raise ValueError("Expected goal[B,3] and PVA[B,cost_nodes,9]")
    if planes.ndim != 4 or planes.shape[:2] != (batch, nodes) or planes.shape[-1] != 4:
        raise ValueError("Expected planes[B,cost_nodes,faces,4]")
    if plane_mask.shape != planes.shape[:-1] or plane_mask.dtype != torch.bool:
        raise ValueError("plane_mask must match planes")
    reference = reference_from_pva(reference_pva, yaw, ctbr_config.gravity)
    sign = torch.where((reference[..., 3:7] * state[:, None, 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    reference[..., 3:7] *= sign
    control_reference = _reference_controls(reference, reference_pva, ctbr_config.mass, ctbr_config.gravity, ctbr_config.dt)
    if reference_valid is None:
        reference_valid = torch.ones((batch, 1), dtype=torch.bool, device=state.device)
    reference_valid = reference_valid.reshape(batch, 1).bool()
    if warm_start_controls is None:
        warm_start_controls = control_reference[:, :-1]
    if warm_start_controls.shape != (batch, nodes - 1, 4):
        raise ValueError("Expected warm_start_controls[B,cost_nodes-1,4]")
    active = planes.double().masked_fill(~plane_mask[..., None], 0)
    features = _policy_features(state, rates, goal, reference, active, plane_mask, reference_valid, ctbr_config)
    yaw = yaw.reshape(batch, -1)
    values = {
        "policy": features,
        "ctbr_state": state,
        "body_rates": rates,
        "goal": goal.double(),
        "ctbr_reference": reference,
        "ctbr_control_reference": control_reference,
        "ctbr_warm_start": warm_start_controls.double(),
        "reference": reference_pva,
        "planes": active,
        "plane_mask": plane_mask,
        "sfc_valid": reference_valid,
        "yaw_target": yaw[:, :1].double(),
    }
    # A B=1 slice may be "contiguous" while retaining an arbitrary leading
    # stride. Canonicalize the owning snapshot to avoid recompiling MPC for
    # physical-state slices or the first (unshifted) control reference.
    return {name: value.clone(memory_format=torch.contiguous_format) for name, value in values.items()}


@torch.no_grad()
def terminal_ctbr_observation(previous, ctbr_state, body_rates, *, ctbr_config):
    """Record terminal measurement with the last executed planning context.

    This performs no planning or trajectory sampling. Completed episodes retain
    their recorded outcome while other environments finish evaluation.
    """
    values = dict(previous)
    state, rates = ctbr_state.double(), body_rates.double()
    values["ctbr_state"], values["body_rates"] = state, rates
    values["policy"] = _policy_features(
        state, rates, previous["goal"], previous["ctbr_reference"], previous["planes"],
        previous["plane_mask"], previous["sfc_valid"], ctbr_config,
    )
    return {name: value.clone(memory_format=torch.contiguous_format) for name, value in values.items()}


class PackedCTBRObservation:
    """One fixed-shape tensor derivation and one packed plan host transfer."""

    def __init__(self, batch, config, max_planes, device, *, compile_kernel=True, backend="inductor"):
        self.batch, self.config, self.faces = batch, config, max_planes
        self.device = torch.device(device)
        self.nodes = config.cost_nodes
        # PVA, heading, planes, mask (0/1), reference validity, warm availability.
        self.plan_width = self.nodes * (10 + 5 * self.faces) + 2
        self.host = np.zeros((batch, self.plan_width), dtype=np.float64)
        with torch.inference_mode(False), torch.no_grad():
            self.empty_warm = torch.zeros((batch, config.horizon, 4), dtype=torch.float64, device=self.device)
        self.derive = (torch.compile(self._derive, fullgraph=True, dynamic=False, backend=backend)
                       if compile_kernel else self._derive)

    def _derive(self, physical, planned):
        batch, nodes, faces = self.batch, self.nodes, self.faces
        pva_end = 9 * nodes
        yaw_end = pva_end + nodes
        planes_end = yaw_end + 4 * nodes * faces
        mask_end = planes_end + nodes * faces
        pva = planned[:, :pva_end].reshape(batch, nodes, 9)
        yaw = planned[:, pva_end:yaw_end]
        planes = planned[:, yaw_end:planes_end].reshape(batch, nodes, faces, 4)
        mask = planned[:, planes_end:mask_end].reshape(batch, nodes, faces).bool()
        valid = planned[:, mask_end:mask_end + 1].bool()
        has_warm = planned[:, mask_end + 1:].bool().reshape(batch, 1, 1)
        # Reuse every current quaternion/control/feature equation verbatim.
        # The default warm reference is computed once with those same controls.
        values = make_ctbr_observation(
            physical[:, :10], physical[:, 10:13], physical[:, 13:16], pva, yaw, planes, mask,
            ctbr_config=self.config, reference_valid=valid,
        )
        warm = physical[:, 16:].reshape(batch, nodes - 1, 4)
        values["ctbr_warm_start"] = torch.where(has_warm, warm, values["ctbr_warm_start"]).clone(
            memory_format=torch.contiguous_format,
        )
        return values

    def pack_plan(self, samples, ids, valid, has_warm):
        """Only selected rows carry plans; unused rows are never sent to the actor."""
        self.host.fill(0)
        nodes, faces = self.nodes, self.faces
        for row, index in enumerate(ids):
            sample = samples[row]
            position = 0
            for array in (sample.pva, sample.yaw, sample.planes, sample.plane_mask):
                values = np.asarray(array).reshape(-1)
                self.host[index, position:position + values.size] = values
                position += values.size
            assert position == nodes * (10 + 5 * faces)
            self.host[index, -2] = valid[row]
            self.host[index, -1] = has_warm
        return self.host

    @torch.no_grad()
    def build(self, state, rates, goal, samples, *, ids=None, valid=None, warm=None, device_ids=None):
        ids = list(range(self.batch)) if ids is None else list(ids)
        valid = [True] * len(ids) if valid is None else valid
        host = self.pack_plan(samples, ids, valid, warm is not None)
        # Both buffers have stable leading strides, including B=1 slices. They
        # are normal tensors under rollout inference mode as well as no_grad.
        with torch.inference_mode(False), torch.no_grad():
            if goal.shape == (3,):
                goal = goal.expand(self.batch, -1)
            controls = self.empty_warm if warm is None else warm
            physical = torch.cat((state, rates, goal, controls.reshape(self.batch, -1)), -1).double()
            planned = torch.as_tensor(host, dtype=torch.float64, device=self.device)
            values = self.derive(physical, planned)
            if ids != list(range(self.batch)):
                indices = (torch.as_tensor(ids, device=self.device) if device_ids is None else device_ids)
                values = {key: value.index_select(0, indices) for key, value in values.items()}
        return values

    @torch.no_grad()
    def prewarm(self):
        with torch.inference_mode(False), torch.no_grad():
            physical = torch.zeros((self.batch, 16 + 4 * self.config.horizon), dtype=torch.float64, device=self.device)
            physical[:, 3] = 1
            planned = torch.as_tensor(self.host, dtype=torch.float64, device=self.device)
            return self.derive(physical, planned)
