"""GPU simulator extension contracts; run with MUJOCO_GL=egl pytest tests."""

import numpy as np
import pytest
import warp as wp

import ocbench


@wp.kernel
def increment(counter: wp.array(dtype=int), amount: int):
    counter[0] += amount


@pytest.mark.parametrize('task', ['block-double-task2-v0', 'chamber-easy-task1-v0', 'switch-3x3-task1-v0'])
def test_control_callbacks_survive_capture_replacement_and_reset(task, monkeypatch):
    env = ocbench.make(task, nworld=1)
    try:
        env.reset(seeds=np.array([83002], np.uint32))
        counter = wp.zeros(1, dtype=int, device=env.data.qpos.device)
        action = wp.zeros((1, 7), dtype=float, device=env.data.qpos.device)
        launches = []
        launch = wp.launch

        def record_launch(kernel, *args, **kwargs):
            launches.append(kernel.__name__)
            return launch(kernel, *args, **kwargs)

        monkeypatch.setattr(wp, 'launch', record_launch)
        for amount in (1, 2):
            env.set_control_step_callbacks(
                before=lambda env: counter.zero_(),
                after_substep=lambda env, amount=amount: wp.launch(increment, dim=1, inputs=[counter, amount]),
            )
            assert env._step_graph is None
            for _ in range(3):
                env.step_joint_actions_gpu(action)
                np.testing.assert_array_equal(counter.numpy(), [amount * env.cpu_env._n_steps])
            assert env._step_graph is not None
            env.reset(seeds=np.array([83003], np.uint32))
            env.step_joint_actions_gpu(action)
            np.testing.assert_array_equal(counter.numpy(), [amount * env.cpu_env._n_steps])
        if task.startswith('chamber'):
            assert {'store_button_qpos', 'update_button_states', 'capture_chamber_locks'} <= set(launches)
        if task.startswith('switch'):
            assert {'store_button_qpos_and_states', 'update_switch_button_states'} <= set(launches)
        env.set_control_step_callbacks()
        counter.zero_()
        env.step_joint_actions_gpu(action)
        np.testing.assert_array_equal(counter.numpy(), [0])
        with pytest.raises(TypeError):
            env.set_control_step_callbacks(before=1)
    finally:
        env.close()


@pytest.mark.parametrize('arm,gripper', [('delta', 'delta'), ('delta', 'absolute'), ('absolute', 'absolute')])
def test_action_modes_match_actuator_contract(arm, gripper):
    env = ocbench.make('block-double-task2-v0', nworld=4)
    try:
        env.reset(seeds=np.array([83000, 83002, 83003, 83004], np.uint32))
        qpos = env.data.qpos.numpy()
        action = np.array([[0.5] * 7, [20.0] * 7, [-20.0] * 7, [1.0] * 7], np.float32)
        done = np.array([0, 0, 0, 1], np.int32)
        env.set_joint_action_control_gpu(
            wp.array(action, dtype=float, device=env.data.qpos.device),
            wp.array(done, dtype=int, device=env.data.qpos.device),
            arm_mode=arm,
            gripper_mode=gripper,
        )
        ctrl = env.data.ctrl.numpy()
        delta = np.clip(action, -1, 1) * env.cpu_env._joint_action_delta.astype(np.float32)
        delta[done != 0] = 0
        targets = qpos[:, env._arm_qpos_ids] + delta[:, :6]
        if arm == 'absolute':
            targets[done == 0] = action[done == 0, :6]
        bounds = env.cpu_env.model.actuator_ctrlrange[env._arm_actuator_ids].astype(np.float32)
        targets = np.clip(targets, bounds[:, 0], bounds[:, 1])
        opening = qpos[:, env._gripper_opening_joint_id] / np.float32(0.8) + delta[:, 6]
        if gripper == 'absolute':
            opening[done == 0] = action[done == 0, 6]
        opening = np.clip(opening, 0, 1) * 255
        np.testing.assert_allclose(ctrl[:, env._arm_actuator_ids], targets, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            ctrl[:, env._gripper_actuator_ids],
            np.repeat(opening[:, None], len(env._gripper_actuator_ids), axis=1),
            atol=3e-5,
            rtol=0,
        )
        with pytest.raises(ValueError):
            env.set_joint_action_control_gpu(None, arm_mode='invalid')
    finally:
        env.close()


def test_packed_readback_and_native_finalization(monkeypatch):
    env = ocbench.make('block-double-task2-v0', nworld=2)
    try:
        env.reset(seeds=np.array([83002, 83003], np.uint32))
        extra = np.ones((2, 1), np.float32)
        monkeypatch.setattr(env, '_get_task_arrays', lambda: {'extra': extra})
        expected = env.get_arrays()

        def no_global_sync():
            pytest.fail('Packed readback used a device-wide Warp synchronization')

        with monkeypatch.context() as scoped:
            scoped.setattr(wp, 'synchronize', no_global_sync)
            arrays = env.get_arrays(packed=True)
        assert arrays.keys() == expected.keys()
        for key, value in arrays.items():
            assert value.dtype == expected[key].dtype
            np.testing.assert_array_equal(value, expected[key])
        # Both native finalization paths retain task-specific nonfinite checks.
        extra[0] = np.nan
        action = np.zeros((2, 7), np.float32)
        for packed in (False, True):
            _, reward, terminated, _, info = env.finalize_transition(action, env.get_arrays(packed=packed))
            assert info['numerical_failure'][0] and not info['healthy'][0]
            assert terminated[0] and reward[0] == 0
    finally:
        env.close()
