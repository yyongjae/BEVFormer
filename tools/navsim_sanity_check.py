#!/usr/bin/env python
"""Pre-training sanity checks for the BEVFormer NAVSIM port.

Every way this port can be wrong is *silent*.  A 90-degree-rotated BEV, a
swapped ego-motion shift, a ``can_bus`` whose velocity and rotation-rate slots
are the other way round, an ``in_split_tokens`` filter that never fired -- none
of them raise, and all of them show up only as a disappointing mAP after hours
of training.  ``tools/analysis_tools/visual.py`` cannot help either: it is
nuScenes-devkit-only, and the devkit is deliberately never imported on the
NAVSIM path.

So this script re-derives the load-bearing geometry from the info pkl alone,
with numpy, and compares it against the arithmetic the model will actually run.
It needs no GPU, no checkpoint and no model: the config is read with
``mmcv.Config`` purely to pick up the numbers (grid, ranges, camera list, image
scale), and the info pkl with ``pickle``.

The five checks and what each one proves::

  1  info pkl stats       The pkl has the shape the dataset expects: frames,
                          in-split samples (the official navtrain/navtest token
                          subset -- see ``docs/NAVSIM.md`` section 3,
                          "Building the info files"), logs, the prev-chain
                          coverage at each queue depth, and a ``can_bus`` slot
                          audit that catches a converter which forgot to swap
                          NAVSIM's velocity/rotation-rate slots into nuScenes
                          order.
  2  calibration          ``lidar2img`` really maps the ego/LiDAR frame into
                          the images: GT centres and GT corners inside the BEV
                          ROI project in front of a camera and land on the
                          sensor.  A transposed or inverted extrinsic makes
                          this collapse towards 0%.
  3  BEV camera coverage  How many BEV cells no camera can see, under
                          ``encoder.py``'s own predicate (pillar sample points
                          through ``lidar2img``, positive depth, pixel inside
                          the padded image).  3.82-3.88% is right for the
                          64 m x 64 m front-only grid with three front
                          cameras: measured 3.8226% over the 50 frames this
                          tool samples on navtest and 3.8628% on navtrain
                          (3.8349% / 3.8811% over 544 / 4,728 frames, i.e. up
                          to 4 per log).  Near 100% means the projection is
                          broken,
                          and near 0% means the grid is not the front-only one.
  4  ego-motion shift     BEVFormer's ``get_bev_features`` splits the ego
                          translation across the two BEV axes.  That split is
                          frame-dependent (``docs/NAVSIM.md`` section 7.1,
                          "The ego-motion shift axes"), so both
                          ``ego_axes`` settings are recomputed here on real
                          consecutive frame pairs and compared against an
                          independent ground truth,
                          ``-R(yaw_cur)^T (t_prev - t_cur)`` in grid cells.
                          For NAVSIM ``x_forward_y_left`` must be the exact
                          match; ``y_forward_x_right`` (the nuScenes default)
                          is wrong by metres on every moving frame.
  5  BEV PNG per token    A picture of one frame: the camera-coverage mask, the
                          GT boxes with their heading arrows, the ego, and the
                          same frame again in raw (h, w) grid indices with the
                          configured ``rotate_center``.  A 90-degree-wrong BEV
                          or an off-grid rotation centre is visible here
                          instead of being inferred from a bad mAP.

Usage::

    python tools/navsim_sanity_check.py \
        --info   data/infos/navsim_infos_val.pkl \
        --config projects/configs/bevformer/bevformer_navsim_base.py \
        --out-dir work_dirs/navsim_sanity

Exit code is 0 when every check passes, 1 when any check FAILs (warnings do
not fail the run), so this is usable as a pre-flight gate in a job script.
"""

import argparse
import os
import os.path as osp
import pickle
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Defaults.  They are ``docs/NAVSIM.md``'s decisions, not guesses, so the tool
# still reports the right thing when run without a config (e.g. against a
# converter smoke pkl before the configs exist).
# ---------------------------------------------------------------------------
DEFAULT_INFO = 'data/infos/navsim_infos_val.pkl'
DEFAULT_CONFIG = 'projects/configs/bevformer/bevformer_navsim_base.py'
DEFAULT_OUT_DIR = 'work_dirs/navsim_sanity'

# NAVSIM ego/LiDAR frame: +x forward, +y left, +z up, origin at the rear axle
# on the ground.  pc_range is front-only; see ``docs/NAVSIM.md`` section 5,
# "The BEV grid: front only, 64 m, square cells" (and section 4, "Coordinate
# conventions", for the frame itself).
DEFAULT_PC_RANGE = [0.0, -32.0, -3.0, 64.0, 32.0, 5.0]
DEFAULT_BEV_H = 100          # rows, physical Y (lateral)
DEFAULT_BEV_W = 100          # cols, physical X (forward)
# docs/NAVSIM.md section 5, "The BEV grid: front only, 64 m, square cells".
DEFAULT_PILLAR_POINTS = 8
DEFAULT_CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0')
DEFAULT_IMAGE_WH = (1920, 1080)   # every NAVSIM blob, both splits
DEFAULT_QUEUE_LENGTH = 4

# The eight NAVSIM cameras in `navsim.common.dataclasses` order.
NAVSIM_CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2',
                  'CAM_R2', 'CAM_B0')

# NAVSIM is 2 Hz.  Measured over every raw log of both splits, the jitter on a
# genuine 0.5 s neighbour peaks at 1.724 ms (136 navtest logs) / 8.186 ms
# (1,192 navtrain logs), and a bigger gap is a hole in the log, not a
# neighbouring frame: the only other deltas that occur are the 1,000,000 us
# one-frame hole (26 of 71,324 navtest deltas, 656 of 668,396 navtrain) and
# three 1,500,000 us two-frame holes in navtrain.
# Same tolerance as the converter and the dataset.
FRAME_DT_US = 500000
FRAME_DT_TOL_US = 250000

EPS = 1e-5                   # same epsilon as encoder.point_sampling

# Thresholds for the PASS/WARN/FAIL verdicts.  Generous on purpose: this tool
# is here to catch order-of-magnitude mistakes, not to police the dataset.
# ObjectFOVFilter removes 2.5452% of in-ROI boxes on navtest and 2.8279% on
# navtrain over the shipped 64 m ROI, so expect ~0.974 / ~0.972 here.
MIN_CORNER_VISIBLE_FRAC = 0.80
MIN_CENTRE_VISIBLE_FRAC = 0.70
# Measured over the 50 frames this tool samples: 3.8226% blind on navtest,
# 3.8628% on navtrain (3.8349% / 3.8811% over 544 / 4,728 frames).
MAX_BLIND_FRAC = 0.25               # expect ~0.038
MIN_BLIND_FRAC = 1e-4               # exactly 0 == the predicate never fired
MAX_SHIFT_ERR_CELLS = 1e-3          # the match is exact to float eps
# docs/NAVSIM.md section 3, "Why --history-frames, and not --keyframes-only".
# The shipped --history-frames 4 pkls reach a full 3-frame real history on
# 99.8765% of navtest samples and 99.7638% of navtrain; --keyframes-only is
# what leaves ~56.4% (navtest) / ~54.4% (navtrain).  0.40 sits below both, so
# only a badly pruned history pool trips it.
MIN_FULL_HISTORY_FRAC = 0.40


# ===========================================================================
# Small numpy geometry helpers.
#
# Deliberately self-contained: this script must run when the plugin cannot be
# imported (no torch, a half-written sibling module, ...).  Each helper names
# the production code it mirrors so the two can be kept in step.
# ===========================================================================
def quaternion_to_rotation_matrix(q):
    """(w, x, y, z) -> (3, 3) rotation matrix.  Mirrors pyquaternion."""
    w, x, y, z = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_yaw(q):
    """Planar yaw of an ego2global quaternion, in radians.

    Identical to ``navsim_eval_utils.quaternion_yaw`` (and the nuScenes devkit
    function the nuScenes dataset uses): rotate the frame's +x axis into the
    global frame and take its azimuth.  Roll and pitch are discarded, which is
    the same planar assumption BEVFormer's shift arithmetic makes.
    """
    v = quaternion_to_rotation_matrix(q) @ np.array([1.0, 0.0, 0.0])
    return float(np.arctan2(v[1], v[0]))


def build_lidar2img(cams, camera_names, scale=1.0):
    """``(C, 4, 4)`` lidar -> image matrices, optionally image-scaled.

    Same construction as ``navsim_fov_utils.build_lidar2image`` and as the
    NAVSIM dataset's ``get_data_info``: ``K4 @ inv(camera2lidar)``, with
    ``camera2lidar`` assembled from the info pkl's ``sensor2lidar_*`` (which is
    camera -> LiDAR).  ``lidar2ego`` is the identity in every NAVSIM frame, so
    the LiDAR frame *is* the ego frame and no further transform appears here.

    ``scale`` pre-multiplies the pixel axes, exactly as
    ``RandomScaleImageMultiViewImage`` does to ``results['lidar2img']``, so the
    matrices match the resized images the model is fed.
    """
    mats = []
    for name in camera_names:
        cam = cams[name]
        camera2lidar = np.eye(4, dtype=np.float64)
        camera2lidar[:3, :3] = np.asarray(cam['sensor2lidar_rotation'],
                                          dtype=np.float64)
        camera2lidar[:3, 3] = np.asarray(cam['sensor2lidar_translation'],
                                         dtype=np.float64)
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = np.asarray(cam['cam_intrinsic'], dtype=np.float64)
        pix_scale = np.eye(4, dtype=np.float64)
        pix_scale[0, 0] = pix_scale[1, 1] = float(scale)
        mats.append(pix_scale @ viewpad @ np.linalg.inv(camera2lidar))
    return np.stack(mats) if mats else np.zeros((0, 4, 4))


def project_points(points, lidar2img):
    """Project ``(N, 3)`` ego-frame points with ``(C, 4, 4)``.

    Returns ``(uv, depth)`` with shapes ``(C, 2, N)`` and ``(C, N)``.  Depth is
    the camera-frame z before the division, i.e. the sign test for "in front
    of the camera".
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    projected = lidar2img @ homogeneous.T          # (C, 4, N)
    depth = projected[:, 2, :]
    uv = projected[:, :2, :] / np.maximum(depth[:, None, :], EPS)
    return uv, depth


def points_in_view(points, lidar2img, image_wh, margin=0.0):
    """``(C, N)`` per-camera visibility mask.

    This is ``navsim_fov_utils.points_in_view``'s predicate (the one the
    evaluator's camera-FOV filter and ``ObjectFOVFilter`` use), kept per-camera
    here so the report can say *which* camera saw a box.
    """
    uv, depth = project_points(points, lidar2img)
    width, height = image_wh
    visible = depth > EPS
    visible &= (uv[:, 0, :] >= -margin) & (uv[:, 0, :] < width + margin)
    visible &= (uv[:, 1, :] >= -margin) & (uv[:, 1, :] < height + margin)
    return visible


def box_corners(boxes):
    """``(N, 7) -> (N, 8, 3)`` corners of mmdet3d LiDAR boxes.

    ``boxes`` are the info pkl's ``gt_boxes``:
    ``[x, y, z_centre, dx, dy, dz, yaw]`` with ``dx, dy, dz = width, length,
    height`` and ``yaw = -heading - pi/2`` (see the converter).

    The rotation applied here is ``R(-yaw)``, which is *verified* to reproduce
    ``mmdet3d 0.17.1``'s ``LiDARInstance3DBoxes.corners`` -- in this mmdet3d
    version ``rotation_3d_in_axis(..., axis=2)`` right-multiplies by the
    rotation matrix, so the stored yaw turns the box clockwise.  Getting this
    sign wrong would make every box in the PNG 2*heading off, so it is pinned
    to the library rather than to convention.

    Corner order is the ``(i, j, k)`` product over ``(-dx/2, +dx/2) x
    (-dy/2, +dy/2) x (-dz/2, +dz/2)`` with k (z) fastest.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))
    offsets = np.stack(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5],
                                   indexing='ij'), axis=-1).reshape(8, 3)
    local = offsets[None, :, :] * boxes[:, None, 3:6]       # (N, 8, 3)
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    zeros, ones = np.zeros(len(boxes)), np.ones(len(boxes))
    # points @ R, with R = [[c, -s, 0], [s, c, 0], [0, 0, 1]] == R(-yaw) @ p.
    rot = np.stack([np.stack([cos, -sin, zeros], axis=-1),
                    np.stack([sin, cos, zeros], axis=-1),
                    np.stack([zeros, zeros, ones], axis=-1)], axis=1)
    rotated = np.einsum('nij,njk->nik', local, rot)
    return rotated + boxes[:, None, 0:3]


def bottom_face(corners):
    """``(N, 8, 3) -> (N, 4, 2)`` the z-min face as a drawable polygon."""
    # Corner order above has z fastest, so the z-min corners are 0, 2, 4, 6
    # at (i, j) = (0,0), (0,1), (1,0), (1,1); walk 00 -> 01 -> 11 -> 10.
    return corners[:, [0, 2, 6, 4], :2]


def grid_pillar_points(pc_range, bev_h, bev_w, num_points_in_pillar):
    """``(D, H*W, 3)`` the SCA pillar sample points, in metres.

    Reproduces ``BEVFormerEncoder.get_reference_points(dim='3d')`` followed by
    ``point_sampling``'s affine mapping back to metres: cell centres in x and
    y, and ``num_points_in_pillar`` heights at
    ``np.linspace(0.5, Z - 0.5, D) + z_min``, ``Z = z_max - z_min``.  That is
    *not* ``z_min + (0.5 + i) / D * Z`` -- the two agree only when ``D == Z``,
    which the shipped ``D=8`` over a 8 m z range happens to satisfy (both give
    -2.5 .. 4.5 m in 1.0 m steps), and disagree as soon as ``D`` changes:
    ``D=4`` is -2.500, -0.167, 2.167, 4.500, not -2, 0, 2, 4.  The query order
    is ``q = h * bev_w + w``, with ``w`` indexing physical X and ``h``
    physical Y (``docs/NAVSIM.md`` section 5, "The BEV grid: front only,
    64 m, square cells").
    """
    xs = (np.arange(bev_w) + 0.5) / bev_w * (pc_range[3] - pc_range[0]) \
        + pc_range[0]
    ys = (np.arange(bev_h) + 0.5) / bev_h * (pc_range[4] - pc_range[1]) \
        + pc_range[1]
    # encoder.get_reference_points(dim='3d') builds the pillar heights as
    #     zs = linspace(0.5, Z - 0.5, D) / Z, Z = pc_range[5] - pc_range[2]
    # and then maps them back through the same Z. That is NOT
    # (i + 0.5) / D * Z: the two coincide only when D == Z, which happens to
    # be true for this grid (D = 8, Z = 8 m) and would silently stop being
    # true at the documented D = 4 cost-saving setting.
    z_extent = pc_range[5] - pc_range[2]
    zs = np.linspace(0.5, z_extent - 0.5, num_points_in_pillar) + pc_range[2]
    yy, xx = np.meshgrid(ys, xs, indexing='ij')         # (H, W), row = y
    flat_x = np.repeat(xx.reshape(1, -1), len(zs), axis=0)
    flat_y = np.repeat(yy.reshape(1, -1), len(zs), axis=0)
    flat_z = np.repeat(zs.reshape(-1, 1), bev_h * bev_w, axis=1)
    return np.stack([flat_x, flat_y, flat_z], axis=-1)


def encoder_bev_mask(cams, camera_names, pc_range, bev_h, bev_w,
                     num_points_in_pillar, padded_wh, scale):
    """``(bev_h, bev_w)`` bool mask of cells at least one camera can see.

    This is ``encoder.point_sampling``'s own predicate, not an approximation:
    the pillar points go through the scaled ``lidar2img``, depth must exceed
    ``eps``, and the pixel is normalised by the **padded** image shape (what
    ``point_sampling`` divides by is ``img_metas[0]['img_shape']``, which
    ``PadMultiViewImage`` sets to the padded size) and must fall strictly
    inside ``(0, 1)``.  A cell counts as covered if any pillar point in it is
    seen by any camera -- the same ``bev_mask.any()`` the SCA applies.
    """
    lidar2img = build_lidar2img(cams, camera_names, scale=scale)
    points = grid_pillar_points(pc_range, bev_h, bev_w, num_points_in_pillar)
    uv, depth = project_points(points.reshape(-1, 3), lidar2img)
    pad_w, pad_h = padded_wh
    mask = depth > EPS
    mask &= (uv[:, 0, :] / pad_w > 0.0) & (uv[:, 0, :] / pad_w < 1.0)
    mask &= (uv[:, 1, :] / pad_h > 0.0) & (uv[:, 1, :] / pad_h < 1.0)
    # (C, D * H * W) -> (H, W)
    per_point = mask.any(axis=0).reshape(num_points_in_pillar, bev_h * bev_w)
    return per_point.any(axis=0).reshape(bev_h, bev_w)


# ===========================================================================
# Config reading.  Every value falls back to the ``docs/NAVSIM.md`` default so
# the tool runs without a config at all, and reports where each number came
# from.
# ===========================================================================
def _dig(node, *keys):
    """Nested ``get`` over dicts / mmcv Configs; ``None`` if a hop is absent."""
    for key in keys:
        if node is None:
            return None
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def _find_attn(encoder_cfg, type_name):
    """The attention cfg of the given ``type`` inside the encoder layer cfg."""
    attn_cfgs = _dig(encoder_cfg, 'transformerlayers', 'attn_cfgs')
    if attn_cfgs is None:
        return None
    if isinstance(attn_cfgs, dict):
        attn_cfgs = [attn_cfgs]
    for cfg in attn_cfgs:
        if isinstance(cfg, dict) and cfg.get('type') == type_name:
            return cfg
    return None


def _walk_pipeline(pipeline):
    """Flatten a pipeline, descending into ``MultiScaleFlipAug3D``."""
    out = []
    for step in pipeline or []:
        if not isinstance(step, dict):
            continue
        out.append(step)
        if 'transforms' in step:
            out.extend(_walk_pipeline(step['transforms']))
    return out


def _pipeline_geometry(pipeline, stored_wh):
    """``(scale, padded_wh, notes)`` implied by a pipeline's image stages.

    ``RandomScaleImageMultiViewImage`` resizes with ``int()`` truncation and
    pre-multiplies ``lidar2img``; ``PadMultiViewImage`` then pads to a multiple
    of ``size_divisor`` (or to a fixed ``size``).  Both matter to check 3,
    because ``point_sampling`` normalises by the padded shape.
    """
    steps = _walk_pipeline(pipeline)
    scale, notes = 1.0, []
    pad_divisor, pad_size = None, None
    scale_at, pad_at = None, None
    for idx, step in enumerate(steps):
        if step.get('type') == 'RandomScaleImageMultiViewImage':
            scales = step.get('scales') or [1.0]
            scale = float(scales[0])
            scale_at = idx
        elif step.get('type') == 'PadMultiViewImage':
            pad_divisor = step.get('size_divisor')
            pad_size = step.get('size')
            pad_at = idx
    width = int(stored_wh[0] * scale)
    height = int(stored_wh[1] * scale)
    if pad_size is not None:
        # mmcv.impad takes `shape` as (h, w).
        padded = (int(pad_size[1]), int(pad_size[0]))
    elif pad_divisor:
        div = int(pad_divisor)
        padded = (int(np.ceil(width / div) * div),
                  int(np.ceil(height / div) * div))
    else:
        padded = (width, height)
        notes.append('no PadMultiViewImage found; using the unpadded size')
    if scale_at is not None and pad_at is not None and pad_at < scale_at:
        notes.append('PadMultiViewImage runs BEFORE the resize, which pads to '
                     'the wrong canvas (docs/NAVSIM.md section 5, "Images", '
                     'requires the resize first)')
    return scale, padded, notes


def read_config(path, info_path=None):
    """Pull the knobs this tool needs out of a BEVFormer config.

    Returns a dict of values plus ``sources``, mapping each key to
    ``'config'`` or ``'default'`` so the printed report never pretends a
    fallback was read from disk.

    ``info_path`` is the info pkl this run is checking.  It only picks which
    ``data.<split>`` sub-dict the dataset-side knobs (``load_interval``,
    ``filter_by_split``) are read from: the split whose ``ann_file`` is that
    pkl, so the sample count reported here is the count that split will
    actually build.
    """
    knobs = dict(
        config_path=path,
        config_read=False,
        pc_range=list(DEFAULT_PC_RANGE),
        bev_h=DEFAULT_BEV_H,
        bev_w=DEFAULT_BEV_W,
        num_points_in_pillar=DEFAULT_PILLAR_POINTS,
        camera_names=list(DEFAULT_CAMERAS),
        ego_axes='y_forward_x_right',
        rotate_center=[100.0, 100.0],
        rotate_prev_bev=True,
        transformer_num_cams=6,
        sca_num_cams=6,
        head_num_classes=None,
        coder_num_classes=None,
        class_names=None,
        dataset_type=None,
        post_center_range=None,
        queue_length=DEFAULT_QUEUE_LENGTH,
        load_interval=1,
        filter_by_split=True,
        dataset_split=None,
        stored_wh=list(DEFAULT_IMAGE_WH),
        train_scale=1.0,
        train_padded_wh=None,
        test_scale=1.0,
        test_padded_wh=None,
        pipeline_notes=[],
        sources={},
    )
    knobs['sources'] = dict((k, 'default') for k in knobs
                            if k not in ('sources', 'config_path',
                                         'config_read'))

    if not path:
        return knobs
    if not osp.isfile(path):
        print('  [warn] config not found: {} -- using the docs/NAVSIM.md '
              'defaults'.format(path))
        return knobs
    try:
        from mmcv import Config
    except Exception as exc:                        # pragma: no cover
        print('  [warn] mmcv.Config unavailable ({}) -- using the '
              'docs/NAVSIM.md defaults'.format(exc))
        return knobs
    try:
        cfg = Config.fromfile(path)
    except Exception as exc:
        print('  [warn] could not read {}: {} -- using the docs/NAVSIM.md '
              'defaults'.format(path, exc))
        return knobs
    knobs['config_read'] = True

    def take(key, value):
        if value is not None:
            knobs[key] = value
            knobs['sources'][key] = 'config'

    head = _dig(cfg, 'model', 'pts_bbox_head')
    transformer = _dig(head, 'transformer')
    encoder = _dig(transformer, 'encoder')
    coder = _dig(head, 'bbox_coder')
    sca = _find_attn(encoder, 'SpatialCrossAttention')

    pc_range = cfg.get('point_cloud_range', None) or _dig(encoder, 'pc_range')
    take('pc_range', list(pc_range) if pc_range else None)
    take('bev_h', _dig(head, 'bev_h') or cfg.get('bev_h_', None))
    take('bev_w', _dig(head, 'bev_w') or cfg.get('bev_w_', None))
    take('num_points_in_pillar', _dig(encoder, 'num_points_in_pillar'))
    take('ego_axes', _dig(transformer, 'ego_axes'))
    rotate_center = _dig(transformer, 'rotate_center')
    take('rotate_center', list(rotate_center) if rotate_center else None)
    take('rotate_prev_bev', _dig(transformer, 'rotate_prev_bev'))
    take('transformer_num_cams', _dig(transformer, 'num_cams'))
    take('sca_num_cams', _dig(sca, 'num_cams'))
    take('head_num_classes', _dig(head, 'num_classes'))
    take('coder_num_classes', _dig(coder, 'num_classes'))
    post_center_range = _dig(coder, 'post_center_range')
    take('post_center_range',
         list(post_center_range) if post_center_range else None)
    class_names = cfg.get('class_names', None)
    take('class_names', list(class_names) if class_names else None)
    take('dataset_type', cfg.get('dataset_type', None)
         or _dig(cfg, 'data', 'train', 'type'))

    # The dataset-side knobs live on whichever split is defined.
    for split in ('test', 'val', 'train'):
        split_cfg = _dig(cfg, 'data', split)
        if split_cfg is None:
            continue
        cams = split_cfg.get('camera_names', None)
        if cams:
            take('camera_names', [str(c) for c in cams])
            break
    queue_length = _dig(cfg, 'data', 'train', 'queue_length') \
        or cfg.get('queue_length', None)
    take('queue_length', queue_length)

    # ``load_interval`` and ``filter_by_split`` decide how long the dataset
    # actually is, so reporting a sample count without them is reporting a
    # number training will not use: the smoke run in docs/NAVSIM.md section 6
    # sets ``data.train.load_interval=2000`` and nothing else, which takes
    # navtrain from 103,288 samples to 52.  They are per-split, so prefer the
    # split whose ``ann_file`` is the pkl being checked.
    split_order = ['test', 'val', 'train']
    if info_path:
        wanted = osp.basename(str(info_path))
        matched = [s for s in split_order
                   if osp.basename(str(_dig(cfg, 'data', s, 'ann_file') or ''))
                   == wanted]
        split_order = matched + [s for s in split_order if s not in matched]
    for split in split_order:
        split_cfg = _dig(cfg, 'data', split)
        if split_cfg is None:
            continue
        take('dataset_split', split)
        take('load_interval', split_cfg.get('load_interval', None))
        take('filter_by_split', split_cfg.get('filter_by_split', None))
        break
    fov = ('eval_cfg', 'camera_fov', 'image_wh')
    image_wh = _dig(cfg, 'data', 'test', *fov) \
        or _dig(cfg, 'data', 'val', *fov) \
        or _dig(cfg, *fov)
    take('stored_wh', [int(v) for v in image_wh] if image_wh else None)

    notes = []
    for split, scale_key, pad_key in (('train', 'train_scale',
                                       'train_padded_wh'),
                                      ('test', 'test_scale',
                                       'test_padded_wh')):
        pipeline = _dig(cfg, 'data', split, 'pipeline') \
            or cfg.get(split + '_pipeline', None)
        if pipeline is None:
            continue
        scale, padded, step_notes = _pipeline_geometry(pipeline,
                                                       knobs['stored_wh'])
        take(scale_key, scale)
        take(pad_key, list(padded))
        notes.extend(['{} pipeline: {}'.format(split, n) for n in step_notes])
    knobs['pipeline_notes'] = notes
    return knobs


# ===========================================================================
# Info pkl loading / indexing.  Mirrors CustomNavsimDataset.load_annotations
# closely enough that the numbers reported here are the numbers the dataset
# will see, but never imports it (it must work on a bare converter smoke pkl).
# ===========================================================================
def load_infos(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        metadata = data.get('metadata', {})
        infos = data['infos']
    else:
        metadata = {}
        infos = data
    infos = sorted(infos, key=lambda e: (e.get('log_name', ''),
                                         e.get('timestamp', 0)))
    return infos, metadata


def build_prev_index(infos):
    """``(N,)`` int array of each frame's previous frame, ``-1`` where absent.

    The pkl's ``prev`` token is re-validated rather than trusted (same
    log_name, ``|dt - 0.5 s| < 0.25 s``), and when the field is missing
    altogether the chain is rebuilt from the ``(log_name, timestamp)`` order so
    this tool still works on a bevfusion-schema pkl.
    """
    token2idx = {}
    for idx, info in enumerate(infos):
        token2idx[info['token']] = idx
    has_prev_field = any('prev' in info for info in infos)
    prev_index = np.full(len(infos), -1, dtype=np.int64)
    for idx, info in enumerate(infos):
        if has_prev_field:
            cand = token2idx.get(info.get('prev'), -1)
        else:
            cand = idx - 1 if idx > 0 else -1
        if cand < 0:
            continue
        prev_info = infos[cand]
        if prev_info.get('log_name') != info.get('log_name'):
            continue
        # Signed, exactly as CustomNavsimDataset does: abs() here would accept
        # a candidate that is 0.5 s in the FUTURE, reporting history the
        # dataset will refuse to deliver.
        dt = int(info['timestamp']) - int(prev_info['timestamp'])
        if abs(dt - FRAME_DT_US) >= FRAME_DT_TOL_US:
            continue
        prev_index[idx] = cand
    return prev_index, token2idx, has_prev_field


def build_sample_indices(infos, load_interval=1, filter_by_split=True):
    """Positions of the in-split frames (the dataset's ``sample_indices``).

    Mirrors ``CustomNavsimDataset.load_annotations`` exactly, ``load_interval``
    and ``filter_by_split`` included -- both are read off the config by
    :func:`read_config`.  Without them the count printed by check 1 is not the
    count the dataset will build, which is the one thing check 1 exists to
    confirm.  Like the dataset, the pool is never subsampled (that would delete
    the history the surviving samples walk back into) and the stride is applied
    after the fallback, so ``load_interval`` thins whichever list was chosen.
    """
    load_interval = max(1, int(load_interval))
    has_flag = any('in_split_tokens' in info for info in infos)
    if not filter_by_split or not has_flag:
        return np.arange(len(infos))[::load_interval], False
    flagged = np.array([bool(info.get('in_split_tokens', True))
                        for info in infos])
    if not flagged.any():
        return np.arange(len(infos))[::load_interval], False
    return np.flatnonzero(flagged)[::load_interval], True


# ===========================================================================
# Reporting helpers
# ===========================================================================
class Report(object):
    """Collects per-check verdicts so the tool can exit non-zero."""

    def __init__(self):
        self.results = []

    def _add(self, level, name, message):
        self.results.append((level, name, message))
        print('  [{}] {}'.format(level, message))

    def ok(self, name, message):
        self._add('PASS', name, message)

    def warn(self, name, message):
        self._add('WARN', name, message)

    def fail(self, name, message):
        self._add('FAIL', name, message)

    @property
    def failed(self):
        return any(level == 'FAIL' for level, _, _ in self.results)

    def summary(self):
        print()
        print('=' * 78)
        print('SUMMARY')
        print('=' * 78)
        counts = dict(PASS=0, WARN=0, FAIL=0)
        for level, name, message in self.results:
            counts[level] += 1
            if level != 'PASS':
                print('  {:<4}  {:<22}  {}'.format(level, name, message))
        print('  {} pass, {} warn, {} fail'
              .format(counts['PASS'], counts['WARN'], counts['FAIL']))
        return 1 if counts['FAIL'] else 0


def section(title):
    print()
    print('-' * 78)
    print(title)
    print('-' * 78)


def sample_positions(count, wanted):
    """``wanted`` indices spread evenly over ``range(count)``."""
    if count <= 0 or wanted <= 0:
        return np.zeros(0, dtype=np.int64)
    if wanted >= count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, wanted).astype(np.int64))


# ===========================================================================
# Check 0 (preamble): the config's own numbers, and the ones that must agree
# ===========================================================================
def audit_config(knobs, report, metadata=None):
    section('configuration')
    pc_range = knobs['pc_range']
    bev_h, bev_w = int(knobs['bev_h']), int(knobs['bev_w'])
    real_w = pc_range[3] - pc_range[0]      # physical X span, the bev_w axis
    real_h = pc_range[4] - pc_range[1]      # physical Y span, the bev_h axis
    cell_x, cell_y = real_w / bev_w, real_h / bev_h
    src = knobs['sources']
    # Without a config there is nothing to blame for a wrong value, so the
    # config-level verdicts drop to warnings: the tool is then reporting the
    # library defaults, not what training would run.
    blame = report.fail if knobs['config_read'] else report.warn

    print('  config               {} ({})'.format(
        knobs['config_path'] or '<none>',
        'read' if knobs['config_read']
        else 'NOT read -- docs/NAVSIM.md defaults below'))
    print('  point_cloud_range    {}   [{}]'.format(pc_range, src['pc_range']))
    print('  BEV grid             bev_h={} (y / lateral)  bev_w={} '
          '(x / forward)   [{}]'.format(bev_h, bev_w, src['bev_h']))
    print('  cell size            {:.4f} m along x  x  {:.4f} m along y'
          .format(cell_x, cell_y))
    print('  dataset_type         {}   [{}]'.format(
        knobs['dataset_type'], src['dataset_type']))
    print('  camera_names         {}   [{}]'.format(
        knobs['camera_names'], src['camera_names']))
    print('  ego_axes             {}   [{}]'.format(knobs['ego_axes'],
                                                    src['ego_axes']))
    print('  rotate_center        {}   [{}]'.format(knobs['rotate_center'],
                                                    src['rotate_center']))
    # The encoder's own heights: linspace(0.5, Z - 0.5, D) + z_min, exactly as
    # grid_pillar_points() builds them.  Printing the (0.5 + i) / D * Z form
    # here instead would agree only while D == Z (the shipped D=8 / Z=8 m).
    print('  num_points_in_pillar {}   [{}]  -> z = {}'.format(
        knobs['num_points_in_pillar'], src['num_points_in_pillar'],
        np.round(np.linspace(0.5, (pc_range[5] - pc_range[2]) - 0.5,
                             knobs['num_points_in_pillar']) + pc_range[2],
                 3).tolist()))
    print('  load_interval        {}   [{}]'.format(knobs['load_interval'],
                                                    src['load_interval']))
    print('  filter_by_split      {}   [{}]   (data.{} supplied the two)'
          .format(knobs['filter_by_split'], src['filter_by_split'],
                  knobs['dataset_split'] or '<none>'))
    print('  stored image size    {}   [{}]   '
          '(eval_cfg.camera_fov.image_wh)'.format(knobs['stored_wh'],
                                                  src['stored_wh']))

    # The converter scales every ``cam_intrinsic`` by ``image_width / 1920``
    # and records the width it used in ``metadata['image_width']``.
    # ``eval_cfg.camera_fov.image_wh`` is the canvas ObjectFOVFilter, the
    # evaluator and this tool's check 2 test those intrinsics against, so the
    # two must name the same image.  A stale ``image_wh`` raises nothing: it
    # just resizes the FOV predicate, silently keeping or dropping the boxes
    # near the image border.  Both numbers are data -- one from the pkl, one
    # from the config (or, with no config, the converter's own 1920 default) --
    # so a disagreement is a FAIL either way, not a config-only verdict.
    pkl_width = metadata.get('image_width') if metadata else None
    print('  pkl image_width      {}   [info pkl metadata]'
          .format(pkl_width if pkl_width is not None else '<absent>'))
    cfg_width = int(knobs['stored_wh'][0])
    if pkl_width is None:
        report.warn('config', "the info pkl carries no "
                    "metadata['image_width'], so the stored cam_intrinsic "
                    'scale cannot be matched against '
                    'eval_cfg.camera_fov.image_wh {}'
                    .format(list(knobs['stored_wh'])))
    elif int(pkl_width) != cfg_width:
        report.fail('config', "the pkl was converted at image_width={} but "
                    'eval_cfg.camera_fov.image_wh[0]={}; the converter scaled '
                    'every cam_intrinsic by {}/{}, so ObjectFOVFilter and the '
                    'evaluator would test those intrinsics against the wrong '
                    'canvas. Re-convert with --image-width {} or set image_wh '
                    'to [{}, {}].'
                    .format(int(pkl_width), cfg_width, int(pkl_width),
                            DEFAULT_IMAGE_WH[0], cfg_width, int(pkl_width),
                            int(round(int(pkl_width)
                                      * float(DEFAULT_IMAGE_WH[1])
                                      / DEFAULT_IMAGE_WH[0]))))
    else:
        report.ok('config', "the pkl's metadata image_width ({}) matches "
                  'eval_cfg.camera_fov.image_wh[0], so the stored '
                  'cam_intrinsic and the FOV canvas agree'.format(cfg_width))
    for split in ('train', 'test'):
        scale = knobs[split + '_scale']
        padded = knobs[split + '_padded_wh']
        print('  {:<5} images         scale {} -> {} -> padded {}   [{}]'
              .format(split, scale,
                      [int(knobs['stored_wh'][0] * scale),
                       int(knobs['stored_wh'][1] * scale)],
                      padded, src[split + '_scale']))
    for note in knobs['pipeline_notes']:
        report.warn('config', note)

    # The rotation centre is the ego's own cell; `rotate_center` is (col, row)
    # from the top-left because that is what torchvision's rotate() takes.
    ego_col = (0.0 - pc_range[0]) / real_w * bev_w
    ego_row = (0.0 - pc_range[1]) / real_h * bev_h
    print('  ego cell (col, row)  [{:.1f}, {:.1f}]'.format(ego_col, ego_row))
    if knobs['rotate_prev_bev']:
        got = [float(v) for v in knobs['rotate_center']]
        if abs(got[0] - ego_col) > 0.5 or abs(got[1] - ego_row) > 0.5:
            blame('config', 'rotate_center {} is not the ego cell '
                  '[{:.1f}, {:.1f}]; rotate_prev_bev would spin the previous '
                  'BEV about the wrong point (docs/NAVSIM.md section 7.2, '
                  '"rotate_center -- config")'
                  .format(got, ego_col, ego_row))
        if abs(cell_x - cell_y) > 1e-6:
            report.warn('config', 'cells are not square ({:.4f} x {:.4f} m) '
                        'but rotate_prev_bev=True; the rotation is only '
                        'geometrically valid on square cells'
                        .format(cell_x, cell_y))

    n_cams = len(knobs['camera_names'])
    for key in ('transformer_num_cams', 'sca_num_cams'):
        if int(knobs[key]) != n_cams:
            blame('config', '{}={} but {} cameras are configured ({} '
                  'camera_names); both PerceptionTransformer.num_cams and '
                  'SpatialCrossAttention.num_cams must match '
                  '(docs/NAVSIM.md section 9, "Pre-flight check"; the '
                  'NAVSIM rig is the three front cameras of section 5)'
                  .format(key, knobs[key], n_cams, src['camera_names']))
    if knobs['class_names'] is not None:
        n_classes = len(knobs['class_names'])
        for key in ('head_num_classes', 'coder_num_classes'):
            if knobs[key] is not None and int(knobs[key]) != n_classes:
                blame('config', '{}={} but {} class_names; both the head '
                      'and bbox_coder num_classes must match the seven '
                      'nuPlan classes (docs/NAVSIM.md section 4, "Coordinate '
                      'conventions")'
                      .format(key, knobs[key], n_classes))
    if knobs['post_center_range'] is not None:
        pcr_min, pcr_max = pc_range[:3], pc_range[3:]
        post = knobs['post_center_range']
        tight = [i for i in range(3)
                 if post[i] > pcr_min[i] + 1e-6
                 or post[3 + i] < pcr_max[i] - 1e-6]
        if tight:
            report.warn('config', 'post_center_range {} is tighter than '
                        'point_cloud_range on axes {}; the head already clamps '
                        'centres inside pc_range, so this can only delete '
                        'valid boxes (docs/NAVSIM.md section 5, "The BEV '
                        'grid: front only, 64 m, square cells")'
                        .format(post, tight))

    unknown = [c for c in knobs['camera_names'] if c not in NAVSIM_CAMERAS]
    if unknown:
        blame('config', 'unknown NAVSIM cameras: {}'.format(unknown))
    return dict(pc_range=pc_range, bev_h=bev_h, bev_w=bev_w,
                cell_x=cell_x, cell_y=cell_y, real_w=real_w, real_h=real_h,
                ego_col=ego_col, ego_row=ego_row)


# ===========================================================================
# Check 1 -- info pkl stats
# ===========================================================================
def check_info_stats(infos, metadata, prev_index, sample_indices, used_flag,
                     has_prev_field, knobs, report):
    section('check 1/5 -- info pkl stats '
            '(does the pkl have the shape the dataset expects?)')
    n_frames = len(infos)
    logs = sorted(set(info.get('log_name', '') for info in infos))
    scenes = set(info.get('scene_token', '') for info in infos)
    print('  frames (history pool)   {}'.format(n_frames))
    if used_flag:
        why = ''
    elif not knobs['filter_by_split']:
        why = '  [filter_by_split=False -> every frame is a sample]'
    else:
        why = '  [no usable in_split_tokens -> all frames]'
    print('  in-split samples        {} ({:.1f}% of frames){}'.format(
        len(sample_indices), 100.0 * len(sample_indices) / max(n_frames, 1),
        why))
    if int(knobs['load_interval']) != 1:
        print('  load_interval           {}  (already applied above: the '
              'dataset keeps every {}-th sample and never thins the pool)'
              .format(knobs['load_interval'], knobs['load_interval']))
    print('  logs                    {}'.format(len(logs)))
    print('  scene_tokens            {}'.format(len(scenes)))
    if metadata:
        print('  metadata                {}'.format(
            dict((k, v) for k, v in sorted(metadata.items()))))
    if not has_prev_field:
        report.warn('info stats', "no 'prev' field in the pkl; the prev chain "
                    'was rebuilt from (log_name, timestamp) order, so this '
                    'measures the data, not the converter')

    if n_frames == 0:
        report.fail('info stats', 'the info pkl has no frames')
        return
    if len(sample_indices) == 0:
        report.fail('info stats',
                    'no in-split samples; the dataset would have length 0')
        return

    # Prev-chain coverage: how deep a real history the queue can reach.  With
    # queue_length=4 the model wants 3 history frames; the shortfall is padded
    # by repeating a frame, which costs the temporal attention its motion cue.
    queue_length = int(knobs['queue_length'])
    depth = np.zeros(len(sample_indices), dtype=np.int64)
    for i, frame_idx in enumerate(sample_indices):
        cursor, found = int(frame_idx), 0
        while found < queue_length - 1:
            cursor = int(prev_index[cursor])
            if cursor < 0:
                break
            found += 1
        depth[i] = found
    print('  prev-chain coverage over {} samples (queue_length={}):'
          .format(len(sample_indices), queue_length))
    for k in range(1, queue_length):
        frac = float((depth >= k).mean())
        print('    >= {} real history frame(s) at k*0.5 s   {:6.2f}%'
              .format(k, 100.0 * frac))
    full = float((depth >= queue_length - 1).mean())
    if full < MIN_FULL_HISTORY_FRAC:
        report.warn('info stats', 'only {:.1f}% of samples have a full '
                    '{}-frame history; the shipped --history-frames 4 pkls '
                    'measure 99.8765% (navtest) / 99.7638% (navtrain), and '
                    'even --keyframes-only leaves ~56.4% / ~54.4% '
                    '(docs/NAVSIM.md section 3, "Why --history-frames, and '
                    'not --keyframes-only"), so a much lower number means the '
                    'history pool was pruned too hard (--history-frames)'
                    .format(100.0 * full, queue_length - 1))
    else:
        report.ok('info stats', '{} frames / {} samples / {} logs, {:.1f}% '
                  'with a full {}-frame real history'
                  .format(n_frames, len(sample_indices), len(logs),
                          100.0 * full, queue_length - 1))

    # Class histogram, over the in-split samples only (what training sees).
    counts = {}
    for frame_idx in sample_indices:
        names = infos[frame_idx].get('gt_names')
        if names is None:
            continue
        for name in np.asarray(names).reshape(-1):
            counts[str(name)] = counts.get(str(name), 0) + 1
    if counts:
        total = sum(counts.values())
        # UNFILTERED, and labelled so: this counts every gt_name in the
        # frame, with no range filter.  docs/NAVSIM.md has mistaken the 54.8
        # it comes to on navtest for the 64 m ROI count before; the shipped
        # ObjectRangeFilter leaves 20.8129 boxes per navtest in-split sample
        # and 30.7781 per navtrain one.
        print('  GT boxes in samples     {} ({:.1f} per frame, UNFILTERED -- '
              'the whole frame, not the BEV ROI; the shipped front-only 64 m '
              'ROI leaves 20.8129/sample on navtest, 30.7781 on navtrain)'
              .format(total, total / float(len(sample_indices))))
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print('    {:<16} {:8d}  {:5.2f}%'
                  .format(name, count, 100.0 * count / total))
        if knobs['class_names']:
            unknown = sorted(set(counts) - set(knobs['class_names']))
            if unknown:
                report.warn('info stats', 'GT names absent from class_names '
                            '(they become label -1 and are dropped): {}'
                            .format(unknown))
    else:
        report.warn('info stats', 'no gt_names in the pkl; a test-mode pkl is '
                    'fine, a train pkl is not')

    _audit_can_bus(infos, sample_indices, report)


def _audit_can_bus(infos, sample_indices, report):
    """Verify the can_bus slot layout the whole shift arithmetic depends on.

    ``docs/NAVSIM.md`` section 4, "Coordinate conventions" / "can_bus":
    NAVSIM ships ``[10:13] = velocity, [13:16] = rotation
    rate`` while every BEVFormer comment (and any pretrained ``can_bus_mlp``)
    assumes nuScenes' ``[10:13] = rotation rate, [13:16] = velocity``.  The
    converter swaps them; if it did not, this is where it shows.
    """
    probe = [infos[i] for i in sample_indices[:200]
             if infos[i].get('can_bus') is not None]
    if not probe:
        report.warn('can_bus', "no 'can_bus' field in the pkl; "
                    'PerceptionTransformer.get_bev_features needs it')
        return
    # Capture the STORED dtype before the cast below; checking `can.dtype`
    # afterwards can only ever say float64 and the guard would never fire.
    stored_dtypes = set(np.asarray(info['can_bus']).dtype for info in probe)
    can = np.stack([np.asarray(info['can_bus'], dtype=np.float64).reshape(18)
                    for info in probe])
    trans = np.stack([np.asarray(info['ego2global_translation'],
                                 dtype=np.float64).reshape(3) for info in probe])
    rot = np.stack([np.asarray(info['ego2global_rotation'],
                               dtype=np.float64).reshape(4) for info in probe])
    pos_err = float(np.abs(can[:, 0:3] - trans).max())
    rot_err = float(np.abs(can[:, 3:7] - rot).max())
    print('  can_bus[0:3] vs ego2global_translation  max |err| {:.3e}'
          .format(pos_err))
    print('  can_bus[3:7] vs ego2global_rotation     max |err| {:.3e}'
          .format(rot_err))
    if pos_err > 1e-6 or rot_err > 1e-9:
        report.warn('can_bus', 'can_bus[0:3]/[3:7] do not hold the ego pose; '
                    'get_data_info overwrites them anyway, but the pkl is not '
                    'what docs/NAVSIM.md section 4, "can_bus", describes')
    if any(dt != np.float64 for dt in stored_dtypes):
        report.warn('can_bus', 'can_bus is stored as {} rather than float64; '
                    'ego2global_translation is UTM-like (measured mean '
                    '(5.0e5, 4.0e6, 2.6e2) m, up to (6.6e5, 4.7e6, 6.3e2)) '
                    'where the float32 ULP is 0.25 m, so differencing two '
                    'poses can lose a quarter metre of a mean 2.536 m '
                    '(navtest) / 2.321 m (navtrain) ego step -- ~10% of it'
                    .format(sorted(str(d) for d in stored_dtypes)))

    state = [info.get('ego_dynamic_state') for info in probe]
    if any(s is None for s in state):
        return
    state = np.stack([np.asarray(s, dtype=np.float64).reshape(4) for s in state])
    vel = state[:, 0:2]                     # [vel_x, vel_y] in the ego frame
    as_nuscenes = float(np.abs(can[:, 13:15] - vel).max())
    as_navsim = float(np.abs(can[:, 10:12] - vel).max())
    print('  can_bus[13:15] vs ego velocity (nuScenes slots)  max |err| {:.3e}'
          .format(as_nuscenes))
    print('  can_bus[10:12] vs ego velocity (raw NAVSIM)      max |err| {:.3e}'
          .format(as_navsim))
    if as_nuscenes <= 1e-6:
        report.ok('can_bus', 'velocity sits in the nuScenes slots [13:16]; the '
                  'converter applied the [10:13]<->[13:16] swap')
    elif as_navsim <= 1e-6:
        report.fail('can_bus', 'velocity is still in NAVSIM slots [10:13]; the '
                    'converter did NOT swap [10:13]<->[13:16], so can_bus_mlp '
                    'reads rotation rate as velocity (docs/NAVSIM.md '
                    'section 4, "can_bus")')
    else:
        report.warn('can_bus', 'ego velocity matches neither slot layout '
                    '(nuScenes err {:.3e}, NAVSIM err {:.3e})'
                    .format(as_nuscenes, as_navsim))


# ===========================================================================
# Check 2 -- calibration: GT boxes project into the images
# ===========================================================================
def check_calibration(infos, sample_indices, knobs, grid, num_frames, report):
    section('check 2/5 -- calibration '
            '(do GT boxes inside the ROI project into the images?)')
    camera_names = knobs['camera_names']
    stored_wh = tuple(knobs['stored_wh'])
    pc_range = grid['pc_range']
    picks = sample_positions(len(sample_indices), num_frames)
    print('  {} frames sampled, cameras {}, stored image size {}'
          .format(len(picks), list(camera_names), list(stored_wh)))
    print('  predicate: navsim_fov_utils.points_in_view (positive depth and '
          'the pixel inside the stored image)')

    n_total = n_in_roi = n_centre = n_corner = n_behind = 0
    per_cam = np.zeros(len(camera_names), dtype=np.int64)
    depths = []
    used = 0
    for pos in picks:
        info = infos[sample_indices[pos]]
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))),
                           dtype=np.float64).reshape(-1, 7)
        cams = info.get('cams')
        if cams is None or any(name not in cams for name in camera_names):
            continue
        used += 1
        n_total += len(boxes)
        if len(boxes) == 0:
            continue
        centres = boxes[:, 0:3]
        in_roi = ((centres[:, 0] >= pc_range[0]) & (centres[:, 0] <= pc_range[3])
                  & (centres[:, 1] >= pc_range[1]) & (centres[:, 1] <= pc_range[4])
                  & (centres[:, 2] >= pc_range[2]) & (centres[:, 2] <= pc_range[5]))
        n_behind += int((centres[:, 0] < 0.0).sum())
        if not in_roi.any():
            continue
        boxes = boxes[in_roi]
        n_in_roi += len(boxes)
        lidar2img = build_lidar2img(cams, camera_names, scale=1.0)
        centre_vis = points_in_view(boxes[:, 0:3], lidar2img, stored_wh)
        n_centre += int(centre_vis.any(axis=0).sum())
        per_cam += centre_vis.sum(axis=1)
        corners = box_corners(boxes).reshape(-1, 3)
        corner_vis = points_in_view(corners, lidar2img, stored_wh)
        corner_hits = corner_vis.any(axis=0).reshape(len(boxes), 8).sum(axis=1)
        n_corner += int((corner_hits >= 1).sum())
        _, depth = project_points(boxes[:, 0:3], lidar2img)
        depths.append(depth[centre_vis].reshape(-1))

    print('  GT boxes                {} ({} inside pc_range, {} behind the '
          'ego)'.format(n_total, n_in_roi, n_behind))
    if used == 0 or n_in_roi == 0:
        report.fail('calibration', 'no in-ROI GT boxes in {} sampled frames; '
                    'nothing to check'.format(len(picks)))
        return
    centre_frac = n_centre / float(n_in_roi)
    corner_frac = n_corner / float(n_in_roi)
    print('  centre projects in-image  {:6.2f}%  ({}/{})'
          .format(100.0 * centre_frac, n_centre, n_in_roi))
    print('  >=1 corner in-image       {:6.2f}%  ({}/{})   '
          '[this is ObjectFOVFilter / the evaluator camera_fov predicate]'
          .format(100.0 * corner_frac, n_corner, n_in_roi))
    for name, hits in zip(camera_names, per_cam):
        print('    {:<8} sees {:7d} centres ({:5.2f}% of in-ROI)'
              .format(name, int(hits), 100.0 * hits / n_in_roi))
    if depths:
        depth = np.concatenate(depths)
        print('  visible-centre camera depth  min {:.2f} m  median {:.2f} m  '
              'max {:.2f} m'.format(depth.min(), np.median(depth), depth.max()))
        if depth.min() <= 0.0:
            report.warn('calibration', 'a "visible" centre has non-positive '
                        'depth, which should be impossible')

    if corner_frac < MIN_CORNER_VISIBLE_FRAC:
        report.fail('calibration', 'only {:.1f}% of in-ROI boxes have a corner '
                    'in an image (expected ~97.5% on navtest / ~97.2% on '
                    'navtrain over the shipped 64 m ROI); lidar2img is '
                    'probably transposed or inverted'
                    .format(100.0 * corner_frac))
    elif centre_frac < MIN_CENTRE_VISIBLE_FRAC:
        report.warn('calibration', 'only {:.1f}% of in-ROI box centres project '
                    'in-image'.format(100.0 * centre_frac))
    else:
        report.ok('calibration', '{:.1f}% of in-ROI boxes have >=1 corner '
                  'in-image, {:.1f}% have their centre in-image'
                  .format(100.0 * corner_frac, 100.0 * centre_frac))


# ===========================================================================
# Check 3 -- BEV cells no camera can see
# ===========================================================================
def check_bev_coverage(infos, sample_indices, knobs, grid, num_frames, report,
                       max_frames=50):
    section('check 3/5 -- BEV camera coverage '
            "(how many cells are blind under encoder.py's own predicate?)")
    camera_names = knobs['camera_names']
    pc_range = grid['pc_range']
    bev_h, bev_w = grid['bev_h'], grid['bev_w']
    padded_wh = tuple(knobs['test_padded_wh']
                      or [int(knobs['stored_wh'][0] * knobs['test_scale']),
                          int(knobs['stored_wh'][1] * knobs['test_scale'])])
    scale = float(knobs['test_scale'])
    picks = sample_positions(len(sample_indices), min(num_frames, max_frames))
    print('  {} frames sampled (the mask is a property of the rig '
          'calibration, which varies a little per vehicle)'.format(len(picks)))
    print('  predicate: encoder.point_sampling -- {} pillar points per cell, '
          'depth > {:g}, pixel strictly inside the padded {}x{} canvas '
          '(lidar2img pre-scaled by {})'
          .format(knobs['num_points_in_pillar'], EPS, padded_wh[0],
                  padded_wh[1], scale))

    blind_fracs, first_mask = [], None
    per_log = {}
    for pos in picks:
        info = infos[sample_indices[pos]]
        cams = info.get('cams')
        if cams is None or any(name not in cams for name in camera_names):
            continue
        mask = encoder_bev_mask(cams, camera_names, pc_range, bev_h, bev_w,
                                int(knobs['num_points_in_pillar']), padded_wh,
                                scale)
        if first_mask is None:
            first_mask = mask
        frac = float((~mask).mean())
        blind_fracs.append(frac)
        per_log.setdefault(info.get('log_name', '?'), []).append(frac)
    if not blind_fracs:
        report.fail('bev coverage', 'no frame had all {} cameras; cannot build '
                    'the coverage mask'.format(len(camera_names)))
        return None
    blind = np.array(blind_fracs)
    cells = bev_h * bev_w
    print('  blind cells   mean {:6.3f}%   min {:6.3f}%   max {:6.3f}%   '
          '({:.0f} of {} cells at the mean)'
          .format(100.0 * blind.mean(), 100.0 * blind.min(),
                  100.0 * blind.max(), blind.mean() * cells, cells))
    if len(per_log) > 1:
        print('  per log (the extrinsics are per vehicle, so this is the '
              'honest spread):')
        for log_name in sorted(per_log):
            vals = np.array(per_log[log_name])
            print('    {:<42} {:6.3f}%  (n={})'
                  .format(log_name[:42], 100.0 * vals.mean(), len(vals)))
    # Where the blind cells are: for a front-camera rig they must be the near
    # lateral corners (the cameras cannot look down and sideways at once).
    rows, cols = np.nonzero(~first_mask)
    if len(rows):
        x_m = (cols + 0.5) / bev_w * grid['real_w'] + pc_range[0]
        y_m = (rows + 0.5) / bev_h * grid['real_h'] + pc_range[1]
        print('  blind cells lie at   x in [{:.1f}, {:.1f}] m, |y| in '
              '[{:.1f}, {:.1f}] m   (mean x {:.1f} m)'
              .format(x_m.min(), x_m.max(), np.abs(y_m).min(),
                      np.abs(y_m).max(), x_m.mean()))
    if blind.mean() > MAX_BLIND_FRAC:
        report.fail('bev coverage', '{:.2f}% of cells are blind (expected '
                    '3.82% on navtest / 3.86% on navtrain for this grid and '
                    '3 front cameras); the projection or the grid is wrong'
                    .format(100.0 * blind.mean()))
    elif blind.mean() < MIN_BLIND_FRAC:
        report.warn('bev coverage', 'not a single blind cell, which a '
                    'front-only 64 m grid with 3 cameras should have 3.82% '
                    '(navtest) / 3.86% (navtrain) of; check that the '
                    'predicate is firing at all')
    else:
        report.ok('bev coverage', '{:.2f}% of BEV cells are blind (measured '
                  '3.8226% on navtest / 3.8628% on navtrain over 50 frames of '
                  'this grid and 3 front cameras, 3.8349% / 3.8811% over '
                  '544 / 4,728 frames; 7.63% / 7.72% if the ROI is cut to '
                  '32 m -- docs/NAVSIM.md section 5, "The BEV grid: front '
                  'only, 64 m, square cells")'.format(100.0 * blind.mean()))
    return first_mask


# ===========================================================================
# Check 4 -- the ego-motion shift
# ===========================================================================
def bevformer_shift_cells(delta_global_xy, yaw_cur_rad, cell_x, cell_y,
                          ego_axes):
    """``get_bev_features``' shift, in grid cells, for one ``ego_axes``.

    Line-for-line the arithmetic in
    ``projects/mmdet3d_plugin/bevformer/modules/transformer.py``, with the
    ``/ bev_w`` and ``/ bev_h`` normalisation undone so the result is in cells
    (the model keeps it normalised because it is added to ``ref_2d``, which
    lives in [0, 1]).

    ``delta_global_xy`` is ``can_bus[0:2]`` after ``union2one``'s differencing,
    i.e. ``t_cur - t_prev`` in the GLOBAL frame, and ``yaw_cur_rad`` is
    ``can_bus[-2]``, the absolute global yaw (``union2one`` only differences
    ``can_bus[-1]``, the degrees slot).
    """
    delta_x, delta_y = float(delta_global_xy[0]), float(delta_global_xy[1])
    ego_angle = yaw_cur_rad / np.pi * 180.0
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180.0
    bev_angle = ego_angle - translation_angle
    forward = translation_length * np.cos(bev_angle / 180.0 * np.pi)
    rightward = translation_length * np.sin(bev_angle / 180.0 * np.pi)
    if ego_axes == 'y_forward_x_right':
        # nuScenes LIDAR_TOP: +y is ego-forward, +x is ego-right.
        shift_w = rightward / cell_x
        shift_h = forward / cell_y
    else:
        # NAVSIM / mmdet3d LiDAR: +x is forward, +y is LEFT.
        shift_w = forward / cell_x
        shift_h = -rightward / cell_y
    return np.array([shift_w, shift_h])


def check_ego_shift(infos, prev_index, knobs, grid, num_pairs, report):
    section('check 4/5 -- ego-motion shift '
            '(which ego_axes setting matches the real ego motion?)')
    cell_x, cell_y = grid['cell_x'], grid['cell_y']
    pairs = []
    for idx in range(len(infos)):
        prev = int(prev_index[idx])
        if prev < 0:
            continue
        pairs.append((prev, idx))
        if len(pairs) >= num_pairs:
            break
    print('  {} consecutive frame pairs from the prev chain '
          '(same log, |dt - 0.5 s| < 0.25 s)'.format(len(pairs)))
    print('  ground truth: delta_ego = R(yaw_cur)^T (t_prev - t_cur), the '
          'previous origin seen from the current frame;')
    print('                shift_truth = -delta_ego / cell, in cells '
          '(cell = {:.4f} m along x, {:.4f} m along y)'
          .format(cell_x, cell_y))
    if not pairs:
        report.fail('ego shift', 'no consecutive frame pairs in the pkl; '
                    'cannot check the shift arithmetic')
        return

    settings = ('y_forward_x_right', 'x_forward_y_left')
    errors = dict((s, []) for s in settings)
    truths, steps = [], []
    for prev_idx, cur_idx in pairs:
        prev_info, cur_info = infos[prev_idx], infos[cur_idx]
        t_prev = np.asarray(prev_info['ego2global_translation'],
                            dtype=np.float64).reshape(3)
        t_cur = np.asarray(cur_info['ego2global_translation'],
                           dtype=np.float64).reshape(3)
        yaw_cur = quaternion_yaw(cur_info['ego2global_rotation'])

        # --- independent ground truth, straight from the quaternion ---------
        cos, sin = np.cos(yaw_cur), np.sin(yaw_cur)
        rot_z = np.array([[cos, -sin], [sin, cos]])
        delta_ego = rot_z.T @ (t_prev[:2] - t_cur[:2])
        # prev_bev, once de-rotated to the current heading, holds the world as
        # seen from the previous origin.  A point now at p was at p - delta_ego
        # then, so the reference points must move by -delta_ego.
        truth = np.array([-delta_ego[0] / cell_x, -delta_ego[1] / cell_y])
        truths.append(truth)
        steps.append(np.linalg.norm(t_cur[:2] - t_prev[:2]))

        # --- what the model computes, from the can_bus slots ----------------
        # get_data_info writes can_bus[-2] from the yaw wrapped to [0, 360);
        # reproduce that wrap, even though cos/sin do not care.
        patch_angle = yaw_cur / np.pi * 180.0
        if patch_angle < 0:
            patch_angle += 360.0
        yaw_slot = patch_angle / 180.0 * np.pi
        delta_global = t_cur[:2] - t_prev[:2]      # union2one's can_bus[:3] -=
        for setting in settings:
            got = bevformer_shift_cells(delta_global, yaw_slot, cell_x, cell_y,
                                        setting)
            errors[setting].append(np.linalg.norm(got - truth))

    truths = np.asarray(truths)
    steps = np.asarray(steps)
    print('  ego step               mean {:.3f} m   max {:.3f} m   '
          '(= {:.2f} / {:.2f} cells)'
          .format(steps.mean(), steps.max(), steps.mean() / cell_x,
                  steps.max() / cell_x))
    print('  |shift_truth|          mean {:.3f} cells   max {:.3f} cells'
          .format(np.linalg.norm(truths, axis=1).mean(),
                  np.linalg.norm(truths, axis=1).max()))
    print()
    print('  {:<20} {:>14} {:>14}   {}'
          .format('ego_axes', 'mean err', 'max err', 'verdict'))
    print('  {:<20} {:>14} {:>14}   {}'
          .format('', '[cells]', '[cells]', ''))
    stats = {}
    for setting in settings:
        err = np.asarray(errors[setting])
        stats[setting] = (float(err.mean()), float(err.max()))
        print('  {:<20} {:>14.6e} {:>14.6e}   {}'
              .format(setting, err.mean(), err.max(),
                      'MATCHES' if err.max() <= MAX_SHIFT_ERR_CELLS
                      else 'wrong'))
    matching = [s for s in settings if stats[s][1] <= MAX_SHIFT_ERR_CELLS]
    configured = knobs['ego_axes']
    print()
    print('  configured ego_axes    {}   [{}]'
          .format(configured, knobs['sources']['ego_axes']))

    if 'x_forward_y_left' not in matching:
        report.fail('ego shift', "'x_forward_y_left' does NOT match the ego "
                    'motion on this data (mean err {:.4f} cells, max {:.4f}); '
                    'docs/NAVSIM.md section 7.1, "The ego-motion shift axes", '
                    'expects it to be the exact match (measured max error '
                    '2.63e-14 cells over all 12,140 in-split navtest pairs) '
                    '-- do not adjust the test, investigate the convention'
                    .format(*stats['x_forward_y_left']))
    if configured not in matching:
        # No config read means this is the library default, not a choice
        # anybody made, so it is reported but it does not fail the run.
        blame = report.fail if knobs['config_read'] else report.warn
        blame('ego shift', "configured ego_axes='{}' is wrong for this "
              'data: mean error {:.3f} cells ({:.2f} m), max {:.3f} '
              'cells. Use {}.'
              .format(configured, stats[configured][0],
                      stats[configured][0] * cell_x,
                      stats[configured][1],
                      matching[0] if matching else
                      'neither -- both are wrong'))
    elif len(matching) == 1:
        report.ok('ego shift', "ego_axes='{}' reproduces the ground-truth "
                  'shift to {:.2e} cells; the other setting is off by {:.3f} '
                  'cells on average ({:.2f} m)'
                  .format(configured, stats[configured][1],
                          stats[settings[0] if configured == settings[1]
                                else settings[1]][0],
                          stats[settings[0] if configured == settings[1]
                                else settings[1]][0] * cell_x))
    else:
        report.warn('ego shift', 'both ego_axes settings match, which only '
                    'happens when the ego never moves; re-run on moving frames')


# ===========================================================================
# Check 5 -- a BEV picture per token
# ===========================================================================
def check_bev_pngs(infos, token2idx, sample_indices, tokens, knobs, grid,
                   out_dir, report):
    section('check 5/5 -- BEV PNG per token '
            '(is the BEV the right way round?)')
    try:
        import matplotlib
        matplotlib.use('Agg')                 # no display, no GPU, no Qt
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:
        report.warn('bev png', 'matplotlib unavailable ({}); skipping the '
                    'plots'.format(exc))
        return

    if not tokens:
        picks = sample_positions(len(sample_indices), 3)
        tokens = [infos[sample_indices[p]]['token'] for p in picks]
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)

    camera_names = knobs['camera_names']
    pc_range = grid['pc_range']
    bev_h, bev_w = grid['bev_h'], grid['bev_w']
    padded_wh = tuple(knobs['test_padded_wh']
                      or [int(knobs['stored_wh'][0] * knobs['test_scale']),
                          int(knobs['stored_wh'][1] * knobs['test_scale'])])
    written = 0
    for token in tokens:
        if token not in token2idx:
            report.warn('bev png', 'token {} is not in the info pkl'
                        .format(token))
            continue
        info = infos[token2idx[token]]
        cams = info.get('cams')
        if cams is None or any(name not in cams for name in camera_names):
            report.warn('bev png', 'frame {} lacks one of {}'
                        .format(token, list(camera_names)))
            continue
        mask = encoder_bev_mask(cams, camera_names, pc_range, bev_h, bev_w,
                                int(knobs['num_points_in_pillar']), padded_wh,
                                float(knobs['test_scale']))
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))),
                           dtype=np.float64).reshape(-1, 7)
        names = np.asarray(info.get('gt_names', [])).reshape(-1)
        in_roi = np.ones(len(boxes), dtype=bool)
        if len(boxes):
            centres = boxes[:, 0:3]
            in_roi = ((centres[:, 0] >= pc_range[0])
                      & (centres[:, 0] <= pc_range[3])
                      & (centres[:, 1] >= pc_range[1])
                      & (centres[:, 1] <= pc_range[4]))

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # ---- panel 1: metric ego frame (x forward right, y left up) -------
        ax = axes[0]
        ax.imshow(mask.astype(np.float32), origin='lower', cmap='Greys_r',
                  vmin=0.0, vmax=1.5, alpha=0.9,
                  extent=[pc_range[0], pc_range[3], pc_range[1], pc_range[4]],
                  aspect='equal', interpolation='nearest')
        faces = bottom_face(box_corners(boxes)) if len(boxes) else []
        for i in range(len(boxes)):
            colour = 'tab:red' if in_roi[i] else 'tab:gray'
            ax.add_patch(Polygon(faces[i], closed=True, fill=False,
                                 edgecolor=colour, linewidth=1.2))
            # The arrow uses NAVSIM's own heading, recovered by inverting the
            # converter's yaw = wrap(-heading - pi/2).  If the port ever gets
            # that 90-degree offset wrong, the cars point across their lane
            # here instead of along it.
            heading = -boxes[i, 6] - np.pi / 2
            length = max(boxes[i, 4], 1.0)
            ax.arrow(boxes[i, 0], boxes[i, 1],
                     0.5 * length * np.cos(heading),
                     0.5 * length * np.sin(heading),
                     head_width=0.5, color=colour, linewidth=0.8)
        ax.plot([0.0], [0.0], marker='s', color='tab:blue', markersize=8)
        ax.arrow(0.0, 0.0, 4.0, 0.0, head_width=1.0, color='tab:blue')
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_xlabel('x [m]  ego forward  ->')
        ax.set_ylabel('y [m]  ego left  ^')
        ax.set_title('{} boxes ({} in ROI); grey = camera-covered, '
                     'black = blind'.format(len(boxes), int(in_roi.sum())),
                     fontsize=10)
        ax.grid(alpha=0.2, linewidth=0.4)

        # ---- panel 2: the BEV tensor as it is indexed ---------------------
        # q = h * bev_w + w, w <-> physical X, h <-> physical Y.  Plotting the
        # raw indices is what makes a transposed grid or an off-grid
        # rotate_center obvious.
        ax = axes[1]
        ax.imshow(mask.astype(np.float32), origin='lower', cmap='Greys_r',
                  vmin=0.0, vmax=1.5, aspect='equal', interpolation='nearest')
        if len(boxes):
            col = (boxes[:, 0] - pc_range[0]) / grid['real_w'] * bev_w - 0.5
            row = (boxes[:, 1] - pc_range[1]) / grid['real_h'] * bev_h - 0.5
            ax.scatter(col[in_roi], row[in_roi], s=9, c='tab:red',
                       label='GT centres')
        ax.plot([grid['ego_col'] - 0.5], [grid['ego_row'] - 0.5], marker='s',
                color='tab:blue', markersize=7, label='ego cell')
        # rotate_center is (col, row) from the top-left, because that is what
        # torchvision's rotate() takes.  Drawn as a big hollow ring so an
        # off-grid value (upstream's [100, 100] default) is impossible to miss.
        rc = [float(v) for v in knobs['rotate_center']]
        ax.plot([rc[0] - 0.5], [rc[1] - 0.5], marker='o', markerfacecolor='none',
                markeredgecolor='tab:orange', markersize=18, markeredgewidth=2.5,
                label='rotate_center {}'.format(rc))
        # Widen the view if rotate_center falls outside the grid, so an
        # off-grid rotation centre is drawn rather than silently clipped.
        ax.add_patch(Polygon([(-0.5, -0.5), (bev_w - 0.5, -0.5),
                              (bev_w - 0.5, bev_h - 0.5), (-0.5, bev_h - 0.5)],
                             closed=True, fill=False, edgecolor='tab:green',
                             linestyle='--', linewidth=1.0))
        ax.set_xlim(min(-0.5, rc[0] - 0.5) - 2, max(bev_w - 0.5, rc[0] - 0.5) + 2)
        ax.set_ylim(min(-0.5, rc[1] - 0.5) - 2, max(bev_h - 0.5, rc[1] - 0.5) + 2)
        ax.set_xlabel('w  (bev_w = {}, physical X {} .. {} m)'
                      .format(bev_w, pc_range[0], pc_range[3]))
        ax.set_ylabel('h  (bev_h = {}, physical Y {} .. {} m)'
                      .format(bev_h, pc_range[1], pc_range[4]))
        ax.set_title('BEV tensor indices, q = h * bev_w + w', fontsize=10)
        # Legend under the axes: an off-grid rotate_center lands in a corner,
        # which is exactly where an in-axes legend would hide it.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                  fontsize=8, frameon=False)
        ax.grid(alpha=0.2, linewidth=0.4)

        fig.suptitle('{}   log {}   {} blind cells ({:.2f}%)'
                     .format(token, info.get('log_name', '?'),
                             int((~mask).sum()), 100.0 * (~mask).mean()),
                     fontsize=11)
        fig.tight_layout()
        path = osp.join(out_dir, 'bev_{}.png'.format(token))
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written += 1
        counts = {}
        for name in names[in_roi] if len(names) == len(boxes) else []:
            counts[str(name)] = counts.get(str(name), 0) + 1
        print('  wrote {}   ({} boxes in ROI: {})'
              .format(path, int(in_roi.sum()),
                      ', '.join('{} x{}'.format(k, v)
                                for k, v in sorted(counts.items())) or '-'))
    if written:
        report.ok('bev png', '{} BEV PNG(s) written to {}; eyeball them -- the '
                  'cars must drive along the lanes, the blind cells must be '
                  'the near lateral corners, and rotate_center must sit on the '
                  'ego cell'.format(written, out_dir))
    else:
        report.warn('bev png', 'no BEV PNG could be written')


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='navsim_sanity_check.py',
        description='Pre-training sanity checks for the BEVFormer NAVSIM '
                    'port: numpy + matplotlib only, no GPU, no checkpoint, no '
                    'model built.',
        epilog=__doc__.split('The five checks and what each one proves::')[1]
        .split('Usage::')[0].strip('\n'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--info', default=DEFAULT_INFO,
                        help='NAVSIM info pkl from '
                             'tools/data_converter/navsim_converter.py '
                             '(default: %(default)s)')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='BEVFormer config to read the grid, ranges, '
                             'camera list and image scale from. Only read, '
                             'never built. Missing or unreadable falls back to '
                             'the docs/NAVSIM.md defaults '
                             '(default: %(default)s)')
    parser.add_argument('--tokens', default=None,
                        help='comma-separated frame tokens to plot in check 5 '
                             '(default: 3 in-split tokens spread over the pkl)')
    parser.add_argument('--num-frames', type=int, default=300,
                        help='frames to sample for check 2 and frame pairs for '
                             'check 4; check 3 uses at most 50 of them '
                             '(default: %(default)s)')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                        help='where the check-5 PNGs go (default: %(default)s)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print('=' * 78)
    print('BEVFormer NAVSIM sanity check')
    print('=' * 78)
    print('  info     {}'.format(args.info))
    print('  config   {}'.format(args.config))
    print('  out-dir  {}'.format(args.out_dir))

    if not osp.isfile(args.info):
        print('ERROR: info pkl not found: {}'.format(args.info))
        return 2
    infos, metadata = load_infos(args.info)
    prev_index, token2idx, has_prev_field = build_prev_index(infos)

    report = Report()
    knobs = read_config(args.config, args.info)
    # Built from the config's own load_interval / filter_by_split so the count
    # check 1 prints is the count the dataset will build.
    sample_indices, used_flag = build_sample_indices(
        infos, load_interval=knobs['load_interval'],
        filter_by_split=knobs['filter_by_split'])
    grid = audit_config(knobs, report, metadata)

    check_info_stats(infos, metadata, prev_index, sample_indices, used_flag,
                     has_prev_field, knobs, report)
    check_calibration(infos, sample_indices, knobs, grid, args.num_frames,
                      report)
    check_bev_coverage(infos, sample_indices, knobs, grid, args.num_frames,
                       report)
    check_ego_shift(infos, prev_index, knobs, grid, args.num_frames, report)
    tokens = [t for t in (args.tokens or '').split(',') if t]
    check_bev_pngs(infos, token2idx, sample_indices, tokens, knobs, grid,
                   args.out_dir, report)
    return report.summary()


if __name__ == '__main__':
    sys.exit(main())
