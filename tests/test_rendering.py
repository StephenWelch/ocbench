"""Explicit rendering APIs: stream handoff, temporal ordering and ownership."""

import numpy as np
import pytest
import torch
import warp as wp

import ocbench
from ocbench.mjwarp.rendering import STATE_FIELDS, BatchRenderer, TemporalRenderer, TrajectoryRenderer


@pytest.mark.parametrize('size', [(240, 320), (480, 640)])
@pytest.mark.parametrize('batch', [1, 2, 4])
def test_temporal_pixels_colors_padding_and_owned_output(size, batch):
    wp.init()
    stream = wp.Stream('cuda:0')
    with wp.ScopedStream(stream):
        env = ocbench.make('block-double-task2-v0', nworld=2, height=size[0], width=size[1])
        env.reset(seeds=np.array([83000, 83002], np.uint32))
        renderer = BatchRenderer(env, stream=stream)
        temporal = TemporalRenderer(env, 'block-double-task2-v0', batch, renderer=renderer, stream=stream)
        try:
            slot = temporal.make_slot()
            expected = []
            count = max(1, batch - 1)
            for tick in range(count):
                with torch.cuda.stream(renderer.torch_stream):
                    renderer.geom[0, :, :3].fill_(0.2 + tick * 0.2)
                    renderer.material[1, :, :3].fill_(0.8 - tick * 0.2)
                expected.append(renderer.render())
                temporal.capture(slot, tick)
            pixels = temporal.render(slot, count)
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                consumer.wait_event(slot['ready'])
                retained = pixels.clone()
            consumer.synchronize()
            rgb = retained.cpu().numpy()[..., [2, 1, 0]]
            for tick, images in enumerate(expected):
                for camera, key in enumerate(('front', 'wrist')):
                    np.testing.assert_array_equal(rgb[tick, :, camera], images[key])
            # A subsequent borrowed render must not overwrite the owned slot.
            renderer.render(device=True)
            renderer.torch_stream.synchronize()
            np.testing.assert_array_equal(pixels.cpu(), retained.cpu())
            if count < batch:
                np.testing.assert_array_equal(slot['pixels'][-1].cpu(), retained[-1].cpu())
            temporal.close()
            temporal.close()
            renderer.reset_colors()
            assert renderer.render()['front'].shape == (2, *size, 3)
        finally:
            temporal.close()
            env.close()


@pytest.mark.parametrize('batch,graph', [(1, False), (1, True), (4, False)])
def test_trajectory_plain_fields_padding_and_cpu_output(batch, graph):
    import mujoco_warp as mjw

    env = ocbench.make('block-double-task2-v0', nworld=2 * batch, height=240, width=320)
    env.reset(seeds=np.array([83000, 83002] * batch, np.uint32))
    stream = wp.Stream(env.data.qpos.device)
    renderer = BatchRenderer(env, stream=stream)
    initial = {k: getattr(env.data, k).numpy() for k in STATE_FIELDS}
    lengths = [3, 1]
    arrays = [{k: np.repeat(v[i : i + 1], n, axis=0) for k, v in initial.items()} for i, n in enumerate(lengths)]
    arrays[0]['qpos'][:, 0] += np.arange(3) * 0.1
    replay = TrajectoryRenderer(
        env, arrays, lengths, graph=graph, render_batch_frames=batch, renderer=renderer, stream=stream
    )
    try:
        for tick in (2, 0):
            with wp.ScopedStream(stream):
                for k in STATE_FIELDS:
                    getattr(env.data, k).assign(
                        np.stack([a[k][min(tick + t, n - 1)] for t in range(batch) for a, n in zip(arrays, lengths)])
                    )
                mjw.kinematics(env.model, env.data)
            expected = renderer.render()
            actual = replay.render(tick, device=False)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
        with pytest.raises(ValueError, match='out of range'):
            replay.render(3)
        if batch > 1:
            with pytest.raises(ValueError, match='CUDA graph'):
                TrajectoryRenderer(env, arrays, lengths, graph=True, render_batch_frames=batch)
    finally:
        replay.close()
        env.close()
