"""Absolute actuator targets for the native batched joint-action controller."""

import warp as wp


@wp.kernel
def set_absolute_gripper(
    action: wp.array2d(dtype=float),
    done: wp.array(dtype=wp.int32),
    actuator_ids: wp.array(dtype=int),
    ctrl: wp.array2d(dtype=float),
    count: int,
):
    world = wp.tid()
    if done[world] == 0:
        for i in range(count):
            ctrl[world, actuator_ids[i]] = 255.0 * wp.clamp(action[world, 6], 0.0, 1.0)


@wp.kernel
def set_absolute_arm(
    action: wp.array2d(dtype=float),
    done: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    low: wp.array(dtype=float),
    high: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    world = wp.tid()
    if done[world] == 0:
        for joint in range(6):
            ctrl[world, ids[joint]] = wp.clamp(action[world, joint], low[joint], high[joint])
