"""Indexed Mid-360 rays with official Newton MultiMesh target ownership.

Only read-only geometry and scan directions are shared. Sensor poses, target transforms,
window indices, cached hits and reset masks remain per-environment.
"""

import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.utils.warp import ProxyArray
from isaaclab_newton.sensors import LegacyMultiMeshRayCaster


@wp.kernel(enable_backward=False)
def indexed_raycast(
    env_mask: wp.array(dtype=wp.bool),
    poses: wp.array(dtype=wp.transform),
    directions: wp.array(dtype=wp.vec3),
    window_start: wp.array(dtype=wp.int32),
    decimation: int,
    offset_position: wp.vec3,
    offset_rotation: wp.quat,
    drift: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array2d(dtype=wp.uint64),
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
    max_distance: float,
    sensor_positions: wp.array(dtype=wp.vec3),
    sensor_rotations: wp.array(dtype=wp.quat),
    hits: wp.array2d(dtype=wp.vec3),
):
    env, ray = wp.tid()
    if env_mask[env]:
        pose = poses[env]
        rotation = wp.transform_get_rotation(pose)
        position = wp.transform_get_translation(pose) + wp.quat_rotate(rotation, offset_position) + drift[env]
        local_direction = wp.quat_rotate(offset_rotation, directions[window_start[env] + ray * decimation])
        direction = wp.quat_rotate(rotation, local_direction)
        if ray == 0:
            # Match the official data interface: parent/site orientation, offset in the rays.
            sensor_positions[env] = wp.transform_get_translation(pose) + drift[env]
            sensor_rotations[env] = rotation
        closest = max_distance
        hit = wp.vec3(wp.inf)
        for target in range(mesh_ids.shape[1]):
            mesh_pose = wp.transform(mesh_positions[env, target], mesh_rotations[env, target])
            inverse_pose = wp.transform_inverse(mesh_pose)
            ray_start = wp.transform_point(inverse_pose, position)
            ray_direction = wp.transform_vector(inverse_pose, direction)
            query = wp.mesh_query_ray(mesh_ids[env, target], ray_start, ray_direction, max_distance)
            if query.result and query.t <= closest:
                closest = query.t
                # Match the official local->world reconstruction order. Algebraically
                # equivalent world-origin + t*direction changes voxel bins at boundaries.
                hit = wp.transform_point(mesh_pose, ray_start + query.t * ray_direction)
        hits[env, ray] = hit


class SharedMid360RayCaster(LegacyMultiMeshRayCaster):
    """Keep MultiMesh lifecycle/target tables; eliminate E×R ray-transform buffers."""

    def _initialize_warp_meshes(self):
        self._shared_target_ids = {}
        super()._initialize_warp_meshes()

    def _load_target_prim_warp_mesh(self, target_prim, target_cfg, reference_prim=None):
        if not (self.cfg.reference_meshes and target_cfg.is_shared):
            return super()._load_target_prim_warp_mesh(target_prim, target_cfg, reference_prim)
        # is_shared explicitly asserts identical local geometry, not identical world poses.
        # Scale is baked into mesh vertices by the parent loader, so it belongs in this key.
        key = (target_cfg.prim_expr, tuple(sim_utils.resolve_prim_scale(target_prim)), target_cfg.merge_prim_meshes)
        if key not in self._shared_target_ids:
            self._shared_target_ids[key] = super()._load_target_prim_warp_mesh(target_prim, target_cfg, reference_prim)
        return self._shared_target_ids[key]

    def _initialize_rays_impl(self):
        if self.cfg.ray_alignment != "base" or self.cfg.update_mesh_ids:
            raise ValueError("Compact Mid-360 requires body-aligned rays and position-only output")
        if any(value for limits in self.cfg.ray_cast_drift_range.values() for value in limits):
            raise ValueError("Compact Mid-360 does not add post-query hit drift")
        pattern = self.cfg.pattern_cfg
        self._table_tensor = pattern.load_table(self._device)
        self._direction_table = wp.from_torch(self._table_tensor, dtype=wp.vec3)
        self.num_rays = len(range(0, pattern.rays_per_scan, pattern.ray_decimation))
        self.window_start = ProxyArray(wp.zeros(self._view_count, dtype=wp.int32, device=self._device))
        self.drift = ProxyArray(wp.zeros(self._view_count, dtype=wp.vec3, device=self._device))
        self.ray_cast_drift = ProxyArray(wp.zeros(self._view_count, dtype=wp.vec3, device=self._device))
        self._data.create_buffers(self._view_count, self.num_rays, self._device)
        self._mount_position = wp.vec3(*self.cfg.offset.pos)
        self._mount_rotation = wp.quat(*self.cfg.offset.rot)

    def _update_buffers_impl(self, env_mask):
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        self._update_mesh_transforms()
        wp.launch(
            indexed_raycast,
            dim=(self._view_count, self.num_rays),
            inputs=[
                env_mask,
                self._newton_pose_w,
                self._direction_table,
                self.window_start.warp,
                self.cfg.pattern_cfg.ray_decimation,
                self._mount_position,
                self._mount_rotation,
                self.drift.warp,
                self._mesh_ids_wp,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(self.cfg.max_distance),
            ],
            outputs=[self._data._pos_w, self._data._quat_w, self._data._ray_hits_w],
            device=self._device,
        )
