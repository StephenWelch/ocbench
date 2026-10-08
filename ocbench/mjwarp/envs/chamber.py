from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ocbench.envs.chamber_env import ChamberEnv
from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv


class ChamberMjWarpEnv(ManipulationMjWarpEnv):
    """Batched ChamberEnv backed by MuJoCo Warp physics."""

    KERNEL_MODULE = 'ocbench.mjwarp.envs.chamber_kernels'

    def __init__(
        self,
        nworld: int = 1024,
        nconmax: int = 64,
        njmax: int = 512,
        nccdmax: Optional[int] = None,
        use_cuda_graph: bool = True,
        env_type='easy',
        task_id=1,
        terminate_at_success=True,
        **kwargs,
    ):
        cpu_env = ChamberEnv(
            env_type=env_type,
            task_id=task_id,
            terminate_at_success=terminate_at_success,
            **kwargs,
        )
        super().__init__(
            cpu_env=cpu_env,
            nworld=nworld,
            nconmax=nconmax,
            njmax=njmax,
            nccdmax=nccdmax,
            use_cuda_graph=use_cuda_graph,
            terminate_at_success=terminate_at_success,
        )

        self._gpu_object_qpos_addrs = None
        self._gpu_button_qpos_addrs = None
        self._gpu_button_dof_addrs = None
        self._gpu_button_base_mocap_ids = None
        self._gpu_button_states = None
        self._gpu_target_button_states = None
        self._gpu_mirrored = None
        self._gpu_prev_button_qpos = None
        self._gpu_target_drawer_pos = None
        self._gpu_target_window_pos = None

    def _cache_task_ids(self, model):
        self._object_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'object_joint_{i}').id] for i in range(self._cpu_env._num_cubes)],
            dtype=np.int32,
        )
        self._button_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'buttonbox_joint_{i}').id] for i in range(self._cpu_env._num_buttons)],
            dtype=np.int32,
        )
        self._button_dof_addrs = np.asarray(
            [model.jnt_dofadr[model.joint(f'buttonbox_joint_{i}').id] for i in range(self._cpu_env._num_buttons)],
            dtype=np.int32,
        )
        self._button_base_mocap_ids = np.asarray(
            [model.body_mocapid[body_id] for body_id in self._cpu_env._button_base_body_ids],
            dtype=np.int32,
        )
        self._drawer_base_mocap_id = int(model.body_mocapid[self._cpu_env._drawer_base_body_id])
        self._window_base_mocap_id = int(model.body_mocapid[self._cpu_env._window_base_body_id])
        self._drawer_lock_anchor_mocap_id = int(self._cpu_env._drawer_lock_anchor_mocap_id)
        self._window_lock_anchor_mocap_id = int(self._cpu_env._window_lock_anchor_mocap_id)
        self._drawer_lock_eq_id = int(self._cpu_env._drawer_lock_eq_id)
        self._window_lock_eq_id = int(self._cpu_env._window_lock_eq_id)
        self._drawer_link_body_id = int(self._cpu_env._drawer_link_body_id)
        self._window_link_body_id = int(self._cpu_env._window_link_body_id)
        self._drawer_qpos_addr = int(model.jnt_qposadr[model.joint('drawer_slide').id])
        self._window_qpos_addr = int(model.jnt_qposadr[model.joint('window_slide').id])
        self._drawer_dof_addr = int(model.jnt_dofadr[model.joint('drawer_slide').id])
        self._window_dof_addr = int(model.jnt_dofadr[model.joint('window_slide').id])
        self._drawer_site_id = int(self._cpu_env._drawer_site_id)
        self._window_site_id = int(self._cpu_env._window_site_id)

    def _ensure_task_gpu_step_buffers(self, device):
        wp = self._wp
        self._gpu_object_qpos_addrs = wp.array(self._object_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_qpos_addrs = wp.array(self._button_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_dof_addrs = wp.array(self._button_dof_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_base_mocap_ids = wp.array(self._button_base_mocap_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_states = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.int32, device=device)
        self._gpu_target_button_states = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.int32, device=device)
        self._gpu_mirrored = wp.zeros(self._nworld, dtype=wp.int32, device=device)
        self._gpu_prev_button_qpos = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.float32, device=device)
        self._gpu_target_drawer_pos = wp.zeros(self._nworld, dtype=wp.float32, device=device)
        self._gpu_target_window_pos = wp.zeros(self._nworld, dtype=wp.float32, device=device)

    def _task_mode(self):
        task_name = self._cpu_env.cur_task_info['task_name']
        if task_name in {'put_in', 'put_all_in'}:
            return 0
        if task_name == 'open_all':
            return 2
        return 1

    def _cube_success_type(self):
        success_type = self._cpu_env.cur_task_info['cube_success_type']
        if success_type == 'in_drawer':
            return 0
        if success_type == 'stack_anywhere':
            return 1
        if success_type == 'on_floor':
            return 2
        raise ValueError(f'Unsupported chamber cube success type: {success_type}')

    def _reset_worlds_gpu(self, world_ids, seeds, full_reset=False):
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()
        wp = self._wp
        device = self._data.qpos.device
        world_ids = np.asarray(world_ids, dtype=np.int32)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if full_reset:
            self._mjw.reset_data(self._model, self._data)
        else:
            self._mjw.kinematics(self._model, self._data)
            self._mjw.com_pos(self._model, self._data)
            wp.launch(
                self._ik_kernels.hold_current_attach_targets,
                dim=self._nworld,
                inputs=[
                    self._data.qpos,
                    self._data.site_xpos,
                    self._data.site_xmat,
                    self._ik_target_pos,
                    self._ik_target_xmat,
                    self._ik_target_gripper,
                    self._attach_site_id,
                    self._gripper_opening_joint_id,
                ],
                device=device,
            )

        task_mode = self._task_mode()
        wp.launch(
            self._kernels.reset_chamber_worlds,
            dim=len(world_ids),
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                self._data.time,
                self._data.mocap_pos,
                self._data.mocap_quat,
                self._data.eq_active,
                wp.array(world_ids, dtype=wp.int32, device=device),
                wp.array(seeds.astype(np.int32), dtype=wp.int32, device=device),
                self._ik_target_pos,
                self._ik_target_xmat,
                self._ik_target_gripper,
                self._ik_arm_qpos_ids,
                self._gpu_object_qpos_addrs,
                self._gpu_button_qpos_addrs,
                self._drawer_qpos_addr,
                self._window_qpos_addr,
                self._gpu_button_base_mocap_ids,
                self._drawer_base_mocap_id,
                self._window_base_mocap_id,
                self._drawer_lock_eq_id,
                self._window_lock_eq_id,
                wp.array(self._cpu_env._home_qpos.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(self._cpu_env._nominal_button_base_poss.astype(np.float32), dtype=wp.vec3f, device=device),
                wp.array(self._cpu_env._nominal_button_base_quats.astype(np.float32), dtype=wp.quatf, device=device),
                wp.vec3(*self._cpu_env._nominal_drawer_base_pos.astype(np.float32)),
                wp.vec3(*self._cpu_env._nominal_window_base_pos.astype(np.float32)),
                wp.quat(*self._cpu_env._drawer_base_quat.astype(np.float32)),
                wp.quat(*self._cpu_env._window_base_quat.astype(np.float32)),
                wp.quat(*self._cpu_env._mirrored_drawer_base_quat.astype(np.float32)),
                wp.quat(*self._cpu_env._mirrored_window_base_quat.astype(np.float32)),
                wp.array(self._cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device),
                self._gpu_button_states,
                self._gpu_target_button_states,
                self._gpu_mirrored,
                self._gpu_target_drawer_pos,
                self._gpu_target_window_pos,
                self._cpu_env.model.nq,
                self._cpu_env.model.nv,
                self._cpu_env.model.nu,
                self._cpu_env._num_cubes,
                task_mode,
                float(self._cpu_env._cube_size),
                float(self._cpu_env._min_object_init_dist),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[0, 0]),
                    float(self._cpu_env._object_sampling_bounds[0, 1]),
                ),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[1, 0]),
                    float(self._cpu_env._object_sampling_bounds[1, 1]),
                ),
                wp.vec3(
                    float(self._cpu_env._arm_sampling_bounds[0, 0]),
                    float(self._cpu_env._arm_sampling_bounds[0, 1]),
                    float(self._cpu_env._arm_sampling_bounds[0, 2]),
                ),
                wp.vec3(
                    float(self._cpu_env._arm_sampling_bounds[1, 0]),
                    float(self._cpu_env._arm_sampling_bounds[1, 1]),
                    float(self._cpu_env._arm_sampling_bounds[1, 2]),
                ),
            ],
            device=device,
        )
        self._mjw.forward(self._model, self._data)
        wp.launch(
            self._kernels.place_chamber_objects,
            dim=len(world_ids),
            inputs=[
                self._data.qpos,
                self._data.site_xpos,
                self._data.mocap_pos,
                wp.array(world_ids, dtype=wp.int32, device=device),
                wp.array(seeds.astype(np.int32), dtype=wp.int32, device=device),
                self._gpu_object_qpos_addrs,
                self._drawer_site_id,
                self._drawer_base_mocap_id,
                self._gpu_mirrored,
                self._cpu_env._num_cubes,
                task_mode,
                float(self._cpu_env._cube_size),
                float(self._cpu_env._min_object_init_dist),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[0, 0]),
                    float(self._cpu_env._object_sampling_bounds[0, 1]),
                ),
                wp.vec2(
                    float(self._cpu_env._object_sampling_bounds[1, 0]),
                    float(self._cpu_env._object_sampling_bounds[1, 1]),
                ),
            ],
            device=device,
        )
        self._mjw.forward(self._model, self._data)
        wp.launch(
            self._kernels.capture_chamber_locks,
            dim=self._nworld,
            inputs=[
                self._gpu_button_states,
                self._data.eq_active,
                self._data.mocap_pos,
                self._data.mocap_quat,
                self._data.xpos,
                self._data.xquat,
                self._drawer_lock_eq_id,
                self._window_lock_eq_id,
                self._drawer_lock_anchor_mocap_id,
                self._window_lock_anchor_mocap_id,
                self._drawer_link_body_id,
                self._window_link_body_id,
                1,
            ],
            device=device,
        )
        self._mjw.forward(self._model, self._data)
        self._solve_reset_arm_ik_gpu()

    def _launch_mjwarp_control_step_sequence(self):
        device = self._data.qpos.device
        self._wp.launch(
            self._kernels.store_button_qpos,
            dim=self._nworld,
            inputs=[self._data.qpos, self._gpu_button_qpos_addrs, self._gpu_prev_button_qpos],
            device=device,
        )
        self._advance_control_substeps()
        self._wp.launch(
            self._kernels.update_button_states,
            dim=self._nworld,
            inputs=[self._data.qpos, self._gpu_button_qpos_addrs, self._gpu_prev_button_qpos, self._gpu_button_states],
            device=device,
        )
        self._mjw.forward(self._model, self._data)
        self._wp.launch(
            self._kernels.capture_chamber_locks,
            dim=self._nworld,
            inputs=[
                self._gpu_button_states,
                self._data.eq_active,
                self._data.mocap_pos,
                self._data.mocap_quat,
                self._data.xpos,
                self._data.xquat,
                self._drawer_lock_eq_id,
                self._window_lock_eq_id,
                self._drawer_lock_anchor_mocap_id,
                self._window_lock_anchor_mocap_id,
                self._drawer_link_body_id,
                self._window_link_body_id,
                0,
            ],
            device=device,
        )
        self._mjw.forward(self._model, self._data)

    def _get_task_arrays(self):
        return dict(
            eq_active=self._data.eq_active.numpy(),
            button_states=self._gpu_button_states.numpy(),
            target_button_states=self._gpu_target_button_states.numpy(),
            mirrored=self._gpu_mirrored.numpy(),
            target_drawer_pos=self._gpu_target_drawer_pos.numpy(),
            target_window_pos=self._gpu_target_window_pos.numpy(),
        )

    def compute_success_gpu(self):
        if self._data is None:
            raise ValueError('Call `reset` before computing success.')
        self._ensure_gpu_step_buffers()
        lower = (self._cpu_env._workspace_bounds[0] - 0.2).astype(np.float32)
        upper = (self._cpu_env._workspace_bounds[1] + 0.2).astype(np.float32)
        cube_success_type = self._cube_success_type()
        self._wp.launch(
            self._kernels.chamber_success,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.site_xpos,
                self._data.mocap_pos,
                self._gpu_object_qpos_addrs,
                self._gpu_button_states,
                self._gpu_target_button_states,
                self._gpu_target_drawer_pos,
                self._gpu_target_window_pos,
                self._gpu_mirrored,
                self._gpu_success,
                self._gpu_healthy,
                self._cpu_env._num_cubes,
                cube_success_type,
                float(self._cpu_env._cube_size),
                self._wp.vec3(float(lower[0]), float(lower[1]), float(lower[2])),
                self._wp.vec3(float(upper[0]), float(upper[1]), float(upper[2])),
                self._drawer_qpos_addr,
                self._window_qpos_addr,
                self._drawer_site_id,
                self._drawer_base_mocap_id,
            ],
            device=self._data.qpos.device,
        )
        return self._gpu_success

    def park_done_worlds_gpu(self, done):
        if self._data is None:
            raise ValueError('Call `reset` before parking worlds.')
        self._ensure_ik_buffers()
        self._ensure_gpu_step_buffers()
        self._wp.launch(
            self._kernels.park_done_chamber_worlds,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                done,
                self._gpu_object_qpos_addrs,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gripper_opening_joint_id,
                self._cpu_env._num_cubes,
                self._cpu_env.model.nv,
                len(self._gripper_actuator_ids),
            ],
            device=self._data.qpos.device,
        )

    def _in_drawer(self, obj_pos, drawer_base_pos, drawer_handle_pos, mirrored):
        if mirrored:
            drawer_low_y, drawer_high_y = drawer_handle_pos[1] + 0.11, drawer_handle_pos[1] + 0.31
        else:
            drawer_low_y, drawer_high_y = drawer_handle_pos[1] - 0.31, drawer_handle_pos[1] - 0.11
        drawer_low = np.array([drawer_base_pos[0] - 0.15, drawer_low_y, drawer_base_pos[2] - 0.044])
        drawer_high = np.array([drawer_base_pos[0] + 0.15, drawer_high_y, drawer_base_pos[2] + 0.046])
        return np.all(drawer_low <= obj_pos) and np.all(obj_pos <= drawer_high)

    def _compute_stack_anywhere_successes(self, block_xyzs, arrays):
        order = np.argsort(block_xyzs[:, :, 2], axis=1)
        sorted_xyzs = np.take_along_axis(block_xyzs, order[:, :, None], axis=1)
        target_zs = self._cpu_env._cube_size * (2 * np.arange(self._cpu_env._num_cubes) + 1)

        xy_success = np.all(np.linalg.norm(block_xyzs[:, :, :2] - sorted_xyzs[:, :1, :2], axis=2) <= 0.04, axis=1)
        z_success = np.all(np.abs(sorted_xyzs[:, :, 2] - target_zs[None, :]) <= 0.03, axis=1)
        outside_drawer = np.ones(self._nworld, dtype=bool)
        drawer_base_pos = arrays['mocap_pos'][:, self._drawer_base_mocap_id]
        drawer_handle_pos = arrays['site_xpos'][:, self._drawer_site_id]
        mirrored = arrays['mirrored'].astype(bool)
        for world_id in range(self._nworld):
            outside_drawer[world_id] = all(
                not self._in_drawer(pos, drawer_base_pos[world_id], drawer_handle_pos[world_id], mirrored[world_id])
                for pos in block_xyzs[world_id]
            )
        return xy_success & z_success & outside_drawer

    def _compute_on_floor_successes(self, block_xyzs, arrays):
        floor_success = np.all(np.abs(block_xyzs[:, :, 2] - self._cpu_env._cube_size) <= 0.03, axis=1)
        outside_drawer = np.ones(self._nworld, dtype=bool)
        drawer_base_pos = arrays['mocap_pos'][:, self._drawer_base_mocap_id]
        drawer_handle_pos = arrays['site_xpos'][:, self._drawer_site_id]
        mirrored = arrays['mirrored'].astype(bool)
        for world_id in range(self._nworld):
            outside_drawer[world_id] = all(
                not self._in_drawer(pos, drawer_base_pos[world_id], drawer_handle_pos[world_id], mirrored[world_id])
                for pos in block_xyzs[world_id]
            )
        return floor_success & outside_drawer

    def _post_step(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        qpos = arrays['qpos']
        block_xyzs = np.stack([qpos[:, addr : addr + 3] for addr in self._object_qpos_addrs], axis=1)

        lower = self._cpu_env._workspace_bounds[0] - 0.2
        upper = self._cpu_env._workspace_bounds[1] + 0.2
        self._healthy = np.all((block_xyzs > lower) & (block_xyzs < upper), axis=(1, 2))

        cube_success_type = self._cpu_env.cur_task_info['cube_success_type']
        if cube_success_type == 'stack_anywhere':
            cube_successes = self._compute_stack_anywhere_successes(block_xyzs, arrays)
        elif cube_success_type == 'on_floor':
            cube_successes = self._compute_on_floor_successes(block_xyzs, arrays)
        else:
            drawer_base_pos = arrays['mocap_pos'][:, self._drawer_base_mocap_id]
            drawer_handle_pos = arrays['site_xpos'][:, self._drawer_site_id]
            mirrored = arrays['mirrored'].astype(bool)
            cube_successes = np.array(
                [
                    all(self._in_drawer(pos, drawer_base_pos[world_id], drawer_handle_pos[world_id], mirrored[world_id]) for pos in block_xyzs[world_id])
                    for world_id in range(self._nworld)
                ],
                dtype=bool,
            )
        buttons_ok = np.all(arrays['button_states'] == arrays['target_button_states'], axis=1)
        drawer_ok = np.abs(qpos[:, self._drawer_qpos_addr] - arrays['target_drawer_pos']) <= 0.04
        window_ok = np.abs(qpos[:, self._window_qpos_addr] - arrays['target_window_pos']) <= 0.04
        self._success = self._healthy & cube_successes & buttons_ok & drawer_ok & window_ok

    def compute_ob_info(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        qpos = arrays['qpos']
        qvel = arrays['qvel']
        site_yaw = self._site_yaw(arrays['site_xmat'])

        ob_info: dict[str, Any] = {}
        ob_info['proprio/joint_pos'] = qpos[:, self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = qvel[:, self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = arrays['site_xpos'][:, self._pinch_site_id].copy()
        ob_info['proprio/effector_xmat'] = arrays['site_xmat'][:, self._pinch_site_id].reshape(self._nworld, 3, 3).copy()
        ob_info['proprio/effector_yaw'] = site_yaw
        ob_info['proprio/gripper_opening'] = np.clip(qpos[:, [self._gripper_opening_joint_id]] / 0.8, 0, 1)
        ob_info['proprio/gripper_vel'] = qvel[:, [self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.clip(
            np.linalg.norm(arrays['cfrc_ext'][:, self._right_pad_body_id], axis=1, keepdims=True) / 50,
            0,
            1,
        )

        for i, addr in enumerate(self._object_qpos_addrs):
            quat = qpos[:, addr + 3 : addr + 7]
            ob_info[f'privileged/block_{i}_pos'] = qpos[:, addr : addr + 3].copy()
            ob_info[f'privileged/block_{i}_quat'] = quat.copy()
            ob_info[f'privileged/block_{i}_yaw'] = self._quat_yaw(quat)
        for i, qpos_addr in enumerate(self._button_qpos_addrs):
            ob_info[f'privileged/button_{i}_state'] = arrays['button_states'][:, i].copy()
            ob_info[f'privileged/button_{i}_base_pos'] = arrays['mocap_pos'][:, self._button_base_mocap_ids[i]].copy()
            ob_info[f'privileged/button_{i}_pos'] = qpos[:, [qpos_addr]].copy()
            ob_info[f'privileged/button_{i}_vel'] = qvel[:, [self._cpu_env.model.jnt_dofadr[self._cpu_env.model.joint(f"buttonbox_joint_{i}").id]]].copy()
        ob_info['privileged/mirrored'] = arrays['mirrored'][:, None].astype(np.float32)
        ob_info['privileged/drawer_base_pos'] = arrays['mocap_pos'][:, self._drawer_base_mocap_id].copy()
        ob_info['privileged/drawer_pos'] = qpos[:, [self._drawer_qpos_addr]].copy()
        ob_info['privileged/drawer_vel'] = qvel[:, [self._cpu_env.model.jnt_dofadr[self._cpu_env.model.joint('drawer_slide').id]]].copy()
        ob_info['privileged/drawer_handle_pos'] = arrays['site_xpos'][:, self._drawer_site_id].copy()
        ob_info['privileged/drawer_handle_yaw'] = np.arctan2(
            arrays['site_xmat'][:, self._drawer_site_id, 1, 0],
            arrays['site_xmat'][:, self._drawer_site_id, 0, 0],
        )[:, None]
        ob_info['privileged/window_base_pos'] = arrays['mocap_pos'][:, self._window_base_mocap_id].copy()
        ob_info['privileged/window_pos'] = qpos[:, [self._window_qpos_addr]].copy()
        ob_info['privileged/window_vel'] = qvel[:, [self._cpu_env.model.jnt_dofadr[self._cpu_env.model.joint('window_slide').id]]].copy()
        ob_info['privileged/window_handle_pos'] = arrays['site_xpos'][:, self._window_site_id].copy()
        ob_info['privileged/window_handle_yaw'] = np.arctan2(
            arrays['site_xmat'][:, self._window_site_id, 1, 0],
            arrays['site_xmat'][:, self._window_site_id, 0, 0],
        )[:, None]

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = qpos.copy()
        ob_info['qvel'] = qvel.copy()
        ob_info['control'] = arrays['ctrl'].copy()
        ob_info['mocap_pos'] = arrays['mocap_pos'].copy()
        ob_info['mocap_quat'] = arrays['mocap_quat'].copy()
        ob_info['button_states'] = arrays['button_states'].copy()
        ob_info['dynamics_info'] = np.concatenate(
            [
                arrays['button_states'].astype(np.float64),
                arrays['mirrored'][:, None].astype(np.float64),
                arrays['mocap_pos'][:, self._button_base_mocap_ids].reshape(self._nworld, -1),
                arrays['mocap_pos'][:, self._drawer_base_mocap_id],
                arrays['mocap_pos'][:, self._window_base_mocap_id],
            ],
            axis=1,
        )
        ob_info['time'] = arrays['time'][:, None].copy()
        return ob_info

    def compute_state_observation(self, ob_info=None):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0
        button_scaler = 120.0
        drawer_scaler = 18.0
        window_scaler = 15.0

        if ob_info is None:
            ob_info = self.compute_ob_info()
        ob = [
            ob_info['proprio/joint_pos'],
            ob_info['proprio/joint_vel'],
            (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
            np.cos(ob_info['proprio/effector_yaw']),
            np.sin(ob_info['proprio/effector_yaw']),
            ob_info['proprio/gripper_opening'] * gripper_scaler,
            ob_info['proprio/gripper_contact'],
        ]
        for i in range(self._cpu_env._num_cubes):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/block_{i}_quat'],
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                ]
            )
        for i in range(self._cpu_env._num_buttons):
            button_states = np.eye(self._cpu_env._num_button_states, dtype=np.float32)[
                ob_info[f'privileged/button_{i}_state'].astype(np.int64)
            ]
            ob.extend(
                [
                    button_states,
                    (ob_info[f'privileged/button_{i}_base_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                    ob_info[f'privileged/button_{i}_vel'],
                ]
            )
        ob.extend(
            [
                ob_info['privileged/mirrored'],
                ob_info['privileged/drawer_pos'] * drawer_scaler,
                ob_info['privileged/drawer_vel'],
                (ob_info['privileged/drawer_base_pos'] - xyz_center) * xyz_scaler,
                ob_info['privileged/window_pos'] * window_scaler,
                ob_info['privileged/window_vel'],
                (ob_info['privileged/window_base_pos'] - xyz_center) * xyz_scaler,
            ]
        )
        return np.concatenate(ob, axis=1)

    def copy_world_to_cpu_env(self, world_id: int, *, arrays=None):
        if self._data is None:
            raise ValueError('Call `reset` before copying a world.')
        import mujoco

        if arrays is None:
            arrays = self._get_arrays()
        world_id = int(world_id)
        self._cpu_env.data.qpos[:] = arrays['qpos'][world_id]
        self._cpu_env.data.qvel[:] = arrays['qvel'][world_id]
        self._cpu_env.data.ctrl[:] = arrays['ctrl'][world_id]
        self._cpu_env.data.mocap_pos[:] = arrays['mocap_pos'][world_id]
        self._cpu_env.data.mocap_quat[:] = arrays['mocap_quat'][world_id]
        self._cpu_env.data.eq_active[:] = arrays['eq_active'][world_id]
        self._cpu_env._cur_button_states = arrays['button_states'][world_id].copy()
        self._cpu_env._target_button_states = arrays['target_button_states'][world_id].copy()
        self._cpu_env._target_drawer_pos = float(arrays['target_drawer_pos'][world_id])
        self._cpu_env._target_window_pos = float(arrays['target_window_pos'][world_id])
        self._cpu_env._mirrored = bool(arrays['mirrored'][world_id])
        self._cpu_env._apply_button_states()
        mujoco.mj_forward(self._cpu_env.model, self._cpu_env.data)
