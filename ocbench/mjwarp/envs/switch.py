from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ocbench.envs.switch_env import SwitchEnv
from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv


class SwitchMjWarpEnv(ManipulationMjWarpEnv):
    """Batched SwitchEnv backed by MuJoCo Warp physics."""

    KERNEL_MODULE = 'ocbench.mjwarp.envs.switch_kernels'

    def __init__(
        self,
        nworld: int = 1024,
        nconmax: int = 64,
        njmax: int = 512,
        nccdmax: Optional[int] = None,
        use_cuda_graph: bool = True,
        env_type='5x5',
        task_id=1,
        terminate_at_success=True,
        **kwargs,
    ):
        cpu_env = SwitchEnv(
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

        self._gpu_button_qpos_addrs = None
        self._gpu_button_dof_addrs = None
        self._gpu_button_base_mocap_ids = None
        self._gpu_button_states = None
        self._gpu_prev_button_states = None
        self._gpu_prev_button_qpos = None

    def _restore_cpu_button_model(self):
        self._cpu_env._cur_button_states = np.zeros(self._cpu_env._num_buttons, dtype=np.int64)
        self._cpu_env._apply_button_states()

    def _after_cpu_reset_model(self):
        self._restore_cpu_button_model()

    def _cache_task_ids(self, model):
        self._button_qpos_addrs = np.asarray(
            [model.jnt_qposadr[model.joint(f'buttonbox_joint_{i}').id] for i in range(self._cpu_env._num_buttons)],
            dtype=np.int32,
        )
        self._button_dof_addrs = np.asarray(
            [model.jnt_dofadr[model.joint(f'buttonbox_joint_{i}').id] for i in range(self._cpu_env._num_buttons)],
            dtype=np.int32,
        )
        self._button_base_mocap_ids = np.asarray(self._cpu_env._button_base_mocap_ids, dtype=np.int32)
        self._button_site_ids = np.asarray(self._cpu_env._button_site_ids, dtype=np.int32)

    def _ensure_task_gpu_step_buffers(self, device):
        wp = self._wp
        self._gpu_button_qpos_addrs = wp.array(self._button_qpos_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_dof_addrs = wp.array(self._button_dof_addrs.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_base_mocap_ids = wp.array(self._button_base_mocap_ids.astype(np.int32), dtype=wp.int32, device=device)
        self._gpu_button_states = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.int32, device=device)
        self._gpu_prev_button_states = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.int32, device=device)
        self._gpu_prev_button_qpos = wp.zeros((self._nworld, self._cpu_env._num_buttons), dtype=wp.float32, device=device)

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

        wp.launch(
            self._kernels.reset_switch_worlds,
            dim=len(world_ids),
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                self._data.time,
                self._data.mocap_pos,
                self._data.mocap_quat,
                wp.array(world_ids, dtype=wp.int32, device=device),
                wp.array(seeds.astype(np.int32), dtype=wp.int32, device=device),
                self._ik_target_pos,
                self._ik_target_xmat,
                self._ik_target_gripper,
                self._ik_arm_qpos_ids,
                self._gpu_button_qpos_addrs,
                self._gpu_button_base_mocap_ids,
                wp.array(self._cpu_env._home_qpos.astype(np.float32), dtype=wp.float32, device=device),
                wp.array(self._cpu_env._nominal_button_base_poss.astype(np.float32), dtype=wp.vec3f, device=device),
                wp.array(self._cpu_env._nominal_button_base_quats.astype(np.float32), dtype=wp.quatf, device=device),
                wp.array(self._cpu_env._effector_down_rotation.as_matrix()[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_rotation[None].astype(np.float32), dtype=wp.mat33f, device=device),
                wp.array(self._T_pa_translation[None].astype(np.float32), dtype=wp.vec3f, device=device),
                self._gpu_button_states,
                self._gpu_prev_button_states,
                self._gpu_prev_button_qpos,
                self._cpu_env.model.nq,
                self._cpu_env.model.nv,
                self._cpu_env.model.nu,
                self._cpu_env._num_rows,
                self._cpu_env._num_cols,
                self._cpu_env._num_buttons,
                self._cpu_env.cur_task_info['max_dropped_buttons'],
                self._cpu_env._dropped_button_state,
                self._cpu_env._num_button_colors,
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
        self._solve_reset_arm_ik_gpu()

    def _launch_mjwarp_control_step_sequence(self):
        device = self._data.qpos.device
        self._wp.launch(
            self._kernels.store_button_qpos_and_states,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._gpu_button_qpos_addrs,
                self._gpu_button_states,
                self._gpu_prev_button_states,
                self._gpu_prev_button_qpos,
                self._cpu_env._num_buttons,
            ],
            device=device,
        )
        self._advance_control_substeps()
        self._wp.launch(
            self._kernels.update_switch_button_states,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._gpu_button_qpos_addrs,
                self._gpu_prev_button_qpos,
                self._gpu_button_states,
                self._cpu_env._num_rows,
                self._cpu_env._num_cols,
                self._cpu_env._num_buttons,
                self._cpu_env._dropped_button_state,
                self._cpu_env._num_button_colors,
            ],
            device=device,
        )

    def _get_task_arrays(self):
        return dict(
            button_states=self._gpu_button_states.numpy(),
            prev_button_states=self._gpu_prev_button_states.numpy(),
        )

    def compute_success_gpu(self):
        if self._data is None:
            raise ValueError('Call `reset` before computing success.')
        self._ensure_gpu_step_buffers()
        self._wp.launch(
            self._kernels.switch_success,
            dim=self._nworld,
            inputs=[
                self._gpu_button_states,
                self._gpu_success,
                self._gpu_healthy,
                self._cpu_env._num_buttons,
                self._cpu_env._dropped_button_state,
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
            self._kernels.park_done_switch_worlds,
            dim=self._nworld,
            inputs=[
                self._data.qpos,
                self._data.qvel,
                self._data.ctrl,
                done,
                self._ik_arm_qpos_ids,
                self._gpu_arm_actuator_ids,
                self._gpu_gripper_actuator_ids,
                self._gripper_opening_joint_id,
                self._cpu_env.model.nv,
                len(self._gripper_actuator_ids),
            ],
            device=self._data.qpos.device,
        )

    def _post_step(self, arrays=None):
        if arrays is None:
            arrays = self._get_arrays()
        self._healthy = np.ones(self._nworld, dtype=bool)
        states = arrays['button_states']
        active = states != self._cpu_env._dropped_button_state
        all_zero = np.all((states == 0) | ~active, axis=1)
        all_one = np.all((states == 1) | ~active, axis=1)
        self._success = self._healthy & (all_zero | all_one)

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

        for i, qpos_addr in enumerate(self._button_qpos_addrs):
            state = arrays['button_states'][:, i].copy()
            button_pos = qpos[:, [qpos_addr]].copy()
            button_vel = qvel[:, [self._button_dof_addrs[i]]].copy()
            dropped = state == self._cpu_env._dropped_button_state
            button_pos[dropped] = 0.0
            button_vel[dropped] = 0.0
            ob_info[f'privileged/button_{i}_state'] = state
            ob_info[f'privileged/button_{i}_pos'] = button_pos
            ob_info[f'privileged/button_{i}_vel'] = button_vel

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = qpos.copy()
        ob_info['qvel'] = qvel.copy()
        ob_info['control'] = arrays['ctrl'].copy()
        ob_info['mocap_pos'] = arrays['mocap_pos'].copy()
        ob_info['mocap_quat'] = arrays['mocap_quat'].copy()
        ob_info['prev_button_states'] = arrays['prev_button_states'].copy()
        ob_info['button_states'] = arrays['button_states'].copy()
        ob_info['time'] = arrays['time'][:, None].copy()
        return ob_info

    def compute_state_observation(self, ob_info=None):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0
        button_scaler = 120.0

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
        for i in range(self._cpu_env._num_buttons):
            button_state = np.eye(self._cpu_env._num_button_states, dtype=np.float32)[
                ob_info[f'privileged/button_{i}_state'].astype(np.int64)
            ]
            ob.extend(
                [
                    button_state,
                    ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                    ob_info[f'privileged/button_{i}_vel'],
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
        self._cpu_env._cur_button_states = arrays['button_states'][world_id].copy()
        self._cpu_env._prev_button_states = arrays['prev_button_states'][world_id].copy()
        self._cpu_env._apply_button_states()
        mujoco.mj_forward(self._cpu_env.model, self._cpu_env.data)
