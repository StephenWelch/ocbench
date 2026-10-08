"""OCBench MJWarp cameras and a strict image provenance contract."""

import copy
import hashlib
import importlib.metadata
import weakref
from typing import Any

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


def image_shape(size):
    shape = (size, size) if isinstance(size, int) else tuple(size)
    if len(shape) != 2 or min(shape) < 16:
        raise ValueError('Image dimensions must be at least 16')
    return shape


def rendering_profile(model, size):
    # Geometry, assets, and optics are static; colors and damping are task state.
    digest = hashlib.sha256()
    for name in (
        'names',
        'body_parentid',
        'body_pos',
        'body_quat',
        'jnt_type',
        'jnt_pos',
        'jnt_axis',
        'geom_type',
        'geom_group',
        'geom_bodyid',
        'geom_size',
        'geom_pos',
        'geom_quat',
        'geom_matid',
        'geom_dataid',
        'mesh_vert',
        'mesh_face',
        'mesh_texcoord',
        'tex_data',
        'mat_texid',
        'mat_texrepeat',
        'cam_bodyid',
        'cam_pos',
        'cam_quat',
        'cam_fovy',
        'light_pos',
        'light_dir',
    ):
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            # Wrist optics are derived by frame transforms; ignore roundoff.
            value = np.round(value, decimals=10)
            value[value == 0] = 0  # Canonicalize signed zero.
        digest.update(name.encode())
        digest.update(value if isinstance(value, bytes) else np.asarray(value).tobytes())
    identity = digest.hexdigest()
    rgba = model.geom_rgba.copy()
    default = np.all(rgba == [0.5, 0.5, 0.5, 1.0], axis=1)
    material = (model.geom_matid >= 0) & default
    rgba[material] = model.mat_rgba[model.geom_matid[material]]
    return {
        'backend': 'mujoco-warp',
        'revision': 2,
        'visibility': 'groups-0-1-2-nonzero-alpha',
        'hidden_geoms': np.flatnonzero(rgba[:, 3] == 0).tolist(),
        'clip_meters': [
            float(model.vis.map.znear * model.stat.extent),
            float(model.vis.map.zfar * model.stat.extent),
        ],
        'cameras': ['front', 'ur5e/wrist'],
        'resolution': list(image_shape(size)),
        'model_sha256': identity,
        'settings': {
            'use_shadows': False,
            'use_textures': True,
            'use_ambient_lighting': True,
            'samples_per_pixel': 1,
        },
        'versions': {name: importlib.metadata.version(name) for name in ('mujoco', 'mujoco-warp', 'warp-lang')},
    }


class BatchRenderer:
    def __init__(self, env, *, stream=None):
        self.env = weakref.proxy(env)
        model = env.cpu_env.model
        self.device = env.data.qpos.device
        self.stream = stream if stream is not None else wp.get_stream(self.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.worlds = env.data.qpos.shape[0]
        self.height, self.width = (
            int(env.cpu_env._render_height),
            int(env.cpu_env._render_width),
        )
        self.profile = rendering_profile(model, (self.height, self.width))
        with torch.cuda.stream(self.torch_stream):
            self.geom = torch.as_tensor(np.repeat(model.geom_rgba[None], self.worlds, axis=0), device=str(self.device))
            self.material = torch.as_tensor(
                np.repeat(model.mat_rgba[None], self.worlds, axis=0), device=str(self.device)
            )
        env.model.geom_rgba = wp.from_torch(self.geom, dtype=wp.vec4)
        env.model.mat_rgba = wp.from_torch(self.material, dtype=wp.vec4)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            # MJWarp ray tracing ignores alpha: exclude invisible goal markers
            # using a render-only copy, leaving the physics model untouched.
            render_model = copy.copy(model)
            render_model.geom_group[self.profile['hidden_geoms']] = 5
            self.context = mjw.create_render_context(
                render_model,
                nworld=self.worlds,
                cam_res=(self.width, self.height),
                cam_active=self.profile['cameras'],
                render_rgb=True,
                **self.profile['settings'],
            )

    def make_slot(self, temporal_batch=1, *, timing=False):
        """Allocate owned BGRA output and a producer event; callers control reuse."""
        if temporal_batch < 1 or self.worlds % temporal_batch:
            raise ValueError('Temporal batch must divide rendering world count')
        with torch.cuda.stream(self.torch_stream):
            return {
                'pixels': torch.empty(
                    (temporal_batch, self.worlds // temporal_batch, 2, self.height, self.width, 4),
                    dtype=torch.uint8,
                    device=str(self.device),
                ),
                'ready': torch.cuda.Event(enable_timing=timing),
            }

    def reset_colors(self):
        """Refresh reset-dependent appearance without rebuilding the render context."""
        with torch.cuda.stream(self.torch_stream):
            self.geom.copy_(torch.as_tensor(self.env.cpu_env.model.geom_rgba, device=str(self.device)))
            self.material.copy_(torch.as_tensor(self.env.cpu_env.model.mat_rgba, device=str(self.device)))

    def render(self, *, device=False, out=None, ready=None):
        """Render cached live transforms; GPU output is borrowed until the next render.

        GPU BGRA layout is (world, camera, height, width, 4). Supplying ``out``
        copies into caller-owned storage on the render stream; ``ready`` marks
        completion of that copy. CPU output is RGB keyed by front/wrist.
        """
        if out is not None and not device:
            raise ValueError('Output buffers require device=True')
        env = self.env
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            mjw.camlight(env.model, env.data)
            # render() traces against cached bounds; it does not update them.
            mjw.refit_bvh(env.model, env.data, self.context)
            mjw.render(env.model, env.data, self.context)
        if device:
            # Borrowed GPU storage; consumers must finish before the next render.
            pixels = (
                wp.to_torch(self.context.rgb_data)
                .view(-1)
                .view(dtype=torch.uint8)
                .reshape(self.worlds, 2, self.height, self.width, 4)
            )
            return self.copy_output(pixels, out=out, ready=ready)
        # One transfer for both cameras; packed pixels are B,G,R,A bytes.
        self.torch_stream.synchronize()
        packed = self.context.rgb_data.numpy().reshape(self.worlds, 2, self.height, self.width)
        rgb = np.stack([(packed >> shift).astype(np.uint8) for shift in (16, 8, 0)], -1)
        return {view: rgb[:, i] for i, view in enumerate(('front', 'wrist'))}

    def copy_output(self, pixels, *, out=None, ready=None):
        """Hand off borrowed BGRA pixels on this renderer's producer stream."""
        if out is not None:
            if out.shape != pixels.shape or out.dtype != pixels.dtype or out.device != pixels.device:
                raise ValueError('Output buffer shape, dtype and device must match BGRA output')
            with torch.cuda.stream(self.torch_stream):
                out.copy_(pixels)
            pixels = out
        if ready is not None:
            ready.record(self.torch_stream)
        return pixels


# Live snapshots must retain the physics forward pass's transforms. Replay of
# recorded qpos is a different path and intentionally runs kinematics.
RENDER_FIELDS = ('xpos', 'xquat', 'subtree_com', 'geom_xpos', 'geom_xmat')
STATE_FIELDS = ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat', 'time')


class TemporalRenderer:
    """Render live snapshots in timestamp-major/world-minor order.

    Batch 1 borrows the live renderer and must flush before the next physics step.
    Larger batches own a separate rendering environment. No consumer scheduling
    lives here: wait for all slot readers before capture/reuse or close.
    """

    def __init__(self, env, task, batch, *, renderer=None, stream=None):
        from ocbench import make

        if batch < 1:
            raise ValueError('Positive temporal batch required')
        self.env, self.batch = env, batch
        self.worlds = env.data.qpos.shape[0]
        self.stream = stream if stream is not None else wp.get_stream(env.data.qpos.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.render_env = env
        self.closed = False
        try:
            if batch != 1:
                self.render_env = make(
                    task, nworld=batch * self.worlds, width=env.cpu_env._render_width, height=env.cpu_env._render_height
                )
                self.render_env.reset(seeds=np.zeros(batch * self.worlds, np.uint32))
            self.renderer = (
                renderer if batch == 1 and renderer is not None else BatchRenderer(self.render_env, stream=self.stream)
            )
        except BaseException:
            self.close()
            raise

    def make_slot(self):
        slot = self.renderer.make_slot(self.batch)
        with torch.cuda.stream(self.torch_stream):
            slot['states'] = {
                k: torch.empty(
                    (self.batch, *wp.to_torch(getattr(self.env.data, k)).shape),
                    dtype=wp.to_torch(getattr(self.env.data, k)).dtype,
                    device=str(self.env.data.qpos.device),
                )
                for k in RENDER_FIELDS
            }
            slot['colors'] = {
                k: torch.empty(
                    (self.batch, self.worlds, *getattr(self.renderer, k).shape[1:]),
                    device=str(self.env.data.qpos.device),
                )
                for k in ('geom', 'material')
            }
        return slot

    def capture(self, slot, index):
        if self.closed or not 0 <= index < self.batch:
            raise ValueError('Invalid capture slot index or closed renderer')
        with torch.cuda.stream(self.torch_stream):
            for k in RENDER_FIELDS:
                slot['states'][k][index].copy_(wp.to_torch(getattr(self.env.data, k)))
            for k, field in (('geom', 'geom_rgba'), ('material', 'mat_rgba')):
                values = wp.to_torch(getattr(self.env.model, field))
                slot['colors'][k][index].copy_(values if values.ndim == 3 else values[None])

    def render(self, slot, count):
        if self.closed or not 1 <= count <= self.batch:
            raise ValueError('Invalid render count or closed renderer')
        with torch.cuda.stream(self.torch_stream):
            if self.batch != 1:
                for key, states in slot['states'].items():
                    states[count:].copy_(states[count - 1 : count])
                    dest = wp.to_torch(getattr(self.render_env.data, key))
                    dest.copy_(states.reshape_as(dest))
                for key, values in slot['colors'].items():
                    values[count:].copy_(values[count - 1 : count])
                    dest = getattr(self.renderer, key)
                    dest.copy_(values.reshape_as(dest))
            pixels = slot['pixels'].reshape(self.batch * self.worlds, 2, self.renderer.height, self.renderer.width, 4)
            self.renderer.render(device=True, out=pixels, ready=slot['ready'])
        return slot['pixels'][:count]

    def close(self):
        if not self.closed:
            self.closed = True
            self.torch_stream.synchronize()
            if self.render_env is not self.env:
                self.render_env.close()
            self.render_env = None
            self.renderer = None


@wp.kernel
def restore_frame(
    states: wp.array2d(dtype=Any),
    dest: wp.array(dtype=Any),
    tick: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    dest[i] = states[tick[0], i]


class TrajectoryRenderer:
    """Upload trajectories once; map consecutive timestamps to rendering worlds.

    Worlds are ordered timestamp-major, then episode. End states are padded;
    the async consumer exports only each episode's actual frames.
    """

    def __init__(self, env, arrays, lengths, *, graph=False, render_batch_frames=1, renderer=None, stream=None):
        self.env = env
        self.stream = stream if stream is not None else wp.get_stream(env.data.qpos.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.device = env.data.qpos.device
        self.worlds = env.data.qpos.shape[0]
        if not arrays or len(arrays) != len(lengths) or min(lengths) < 1:
            raise ValueError('One positive frame count required per trajectory')
        if any(any(len(a[k]) < n for k in STATE_FIELDS) for a, n in zip(arrays, lengths)):
            raise ValueError('Trajectory fields shorter than requested frame count')
        self.render_batch_frames = render_batch_frames
        if self.worlds != len(arrays) * render_batch_frames:
            raise ValueError('Simulation worlds must match episodes times temporal batch')
        if graph and render_batch_frames != 1:
            raise ValueError('Temporal batching does not support CUDA graph replay')
        self.length = max(lengths)
        self.dest = {k: wp.to_torch(getattr(env.data, k)) for k in STATE_FIELDS}
        with torch.cuda.stream(self.torch_stream):
            self.states = {
                k: torch.as_tensor(
                    np.stack(
                        [
                            a[k][
                                np.minimum(
                                    np.arange(self.length + render_batch_frames - 1),
                                    length - 1,
                                )
                            ]
                            for a, length in zip(arrays, lengths)
                        ],
                        axis=1,
                    ),
                    dtype=self.dest[k].dtype,
                    device=self.dest[k].device,
                )
                for k in STATE_FIELDS
            }
        self.renderer = renderer if renderer is not None else BatchRenderer(env, stream=self.stream)
        self.graph = None
        if graph:
            with wp.ScopedStream(self.stream):
                self.tick = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.inputs = {k: wp.from_torch(v.reshape(self.length, -1)) for k, v in self.states.items()}
            self.outputs = {k: wp.from_torch(v.reshape(-1)) for k, v in self.dest.items()}
            self.launch()
            wp.synchronize_stream(self.stream)
            with wp.ScopedStream(self.stream), wp.ScopedCapture(stream=self.stream) as capture:
                self.launch()
            self.graph = capture.graph

    def launch(self):
        with wp.ScopedStream(self.stream):
            for k in STATE_FIELDS:
                wp.launch(
                    restore_frame,
                    dim=self.outputs[k].size,
                    inputs=[self.inputs[k], self.outputs[k], self.tick],
                    device=self.device,
                )
            mjw.kinematics(self.env.model, self.env.data)
            self.pixels = self.renderer.render(device=True)

    def render(self, tick, *, device=True, out=None, ready=None):
        if out is not None and not device:
            raise ValueError('Output buffers require device=True')
        if not 0 <= tick < self.length:
            raise ValueError('Replay frame index out of range')
        if self.graph is not None:
            with wp.ScopedStream(self.stream):
                self.tick.fill_(tick)
                wp.capture_launch(self.graph, stream=self.stream)
            pixels = self.pixels
        else:
            pixels = self._render(tick)
        pixels = self.renderer.copy_output(pixels, out=out, ready=ready)
        if not device:
            self.torch_stream.synchronize()
            rgb = pixels.cpu().numpy()[..., [2, 1, 0]]
            return {view: rgb[:, i] for i, view in enumerate(('front', 'wrist'))}
        return pixels

    def _render(self, tick):
        env = self.env
        with torch.cuda.stream(self.torch_stream):
            for k in STATE_FIELDS:
                self.dest[k].copy_(self.states[k][tick : tick + self.render_batch_frames].reshape_as(self.dest[k]))
        with wp.ScopedStream(self.stream):
            mjw.kinematics(env.model, env.data)
        return self.renderer.render(device=True)

    def close(self):
        """Release replay buffers after consumers drain; the environment is borrowed."""
        self.torch_stream.synchronize()
        self.graph = None
        self.states.clear()
        self.dest.clear()
        self.inputs = self.outputs = None
        self.pixels = None
        self.renderer = None
