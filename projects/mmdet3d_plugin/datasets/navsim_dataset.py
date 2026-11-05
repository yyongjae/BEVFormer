"""NAVSIM (OpenScene / nuPlan) temporal dataset for BEVFormer.

Consumes the info pickles produced by ``tools/data_converter/navsim_converter.py``.
This is the NAVSIM counterpart of :class:`CustomNuScenesDataset`: it keeps
BEVFormer's temporal queue machinery (``prepare_train_data`` / ``union2one``)
and replaces everything nuScenes-specific -- the devkit evaluator, the
``LIDAR_TOP`` calibration chain and the keyframe bookkeeping.  Nothing in this
file imports ``nuscenes`` / ``nuscenes-devkit``; the evaluator is the
nuScenes-*styled* pure-numpy one in :mod:`navsim_eval_utils`, shared
byte-for-byte with the BEVFusion NAVSIM port so the two score identically.

Coordinate conventions
----------------------
NAVSIM stores everything in the *ego* frame, whose origin is the rear axle
projected onto the ground plane (z = 0 is the ground, +x forward, +y left,
+z up).  ``lidar2ego`` is the identity in every verified frame, so the ego
frame *is* the LiDAR frame and no extra transform appears anywhere below.

This is **not** the frame BEVFormer was written for.  nuScenes boxes live in
``LIDAR_TOP``, whose ``lidar2ego`` is a ~-90 degree z-rotation, so there
``+y`` is ego-forward and ``+x`` is ego-right.  Two consequences:

* The BEV grid here is ``pc_range = [0, -32, -3, 64, 32, 5]`` -- forward-only
  on the ``w`` (X) axis, symmetric on the ``h`` (Y) axis -- because the three
  front cameras observe nothing behind the car.  See the configs.
* ``PerceptionTransformer`` must be built with
  ``ego_axes='x_forward_y_left'``, or its ego-motion BEV shift lands on the
  wrong axis with the lateral sign inverted (see the module docstring there).
  That is a model-side setting; this file only supplies ``can_bus``.

``gt_boxes`` in the info pkl are already in the mmdet3d LiDAR box convention
(see the converter for the exact transformation from NAVSIM's
``[x, y, z, length, width, height, heading]``)::

    [x, y, z_center, dx, dy, dz, yaw, vx, vy]
    dx, dy, dz = width, length, height          # length/width SWAP
    yaw        = -heading - pi / 2

The stored z is the **box centre**, hence ``origin=(0.5, 0.5, 0.5)`` when
building :class:`LiDARInstance3DBoxes`.

The frame pool vs. the sample list
----------------------------------
The official ``navtrain`` / ``navtest`` token lists are a *subset* of each
log's 2 Hz frames, so a pkl built from the token lists alone has real history
at all three queue slots for only ~56.4% (navtest) / ~54.4% (navtrain) of its
samples -- see ``docs/NAVSIM.md`` section 3, "Why ``--history-frames``, and not
``--keyframes-only``" -- while a pkl whose every frame counted as a sample
would hand the evaluator one frame per pooled frame and divide the reported
recall by the pool/sample ratio.  Neither is usable on its own, so this dataset
separates the two roles:

* ``self.data_infos``    -- **every** frame in the pkl: the *history pool*.
* ``self.sample_indices`` -- positions of the in-split frames: the *sample list*.

Measured on the shipped ``--history-frames 4`` pkls, the pool is **1.6030x**
the sample list on navtest (19,470 frames / 12,146 samples; 7,324 frames, 37.6%
of the pool, exist only as history) and **1.4917x** on navtrain (154,073 /
103,288; 50,785 extra, 33.0%).  Those extra frames are exactly what buys a full
3-frame real history for **99.8765%** of navtest samples and **99.7638%** of
navtrain ones; counting them as samples instead would score 1.60x / 1.49x too
many frames.

``__len__``, ``_set_group_flag``, ``prepare_test_data``, ``get_cat_ids``,
``load_gt`` and ``_format_bbox`` run over **sample** indices;
``get_data_info`` and ``get_ann_info`` take **frame** indices.  The queue walks
the real ``prev`` token chain through the pool, so history frames are always
real frames and never have to be in the split.
"""

import copy
import json
import random
import tempfile
import warnings
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset

from .navsim_eval_utils import (
    DetectionBox,
    DetectionMetricDataList,
    DetectionMetrics,
    EvalBoxes,
    accumulate,
    calc_ap,
    calc_tp,
    center_distance,
    quaternion_yaw,
)
from .navsim_fov_utils import (
    boxes_in_view,
    build_lidar2image,
    image_size_from_intrinsics,
)

# The eight NAVSIM cameras, in the order used by ``navsim.common.dataclasses``.
NAVSIM_CAMERAS = (
    'CAM_F0',
    'CAM_L0',
    'CAM_R0',
    'CAM_L1',
    'CAM_R1',
    'CAM_L2',
    'CAM_R2',
    'CAM_B0',
)

# nuPlan ``TrackedObjectType`` names that appear in ``anns['gt_names']``.
NAVSIM_CLASSES = (
    'vehicle',
    'pedestrian',
    'bicycle',
    'traffic_cone',
    'barrier',
    'czone_sign',
    'generic_object',
)

# NAVSIM logs are 2 Hz.  Measured over every raw log pkl of both splits, the
# inter-frame delta is 500000 us with at most 1724 us of jitter on the 136
# navtest logs and 8186 us on the 1,192 navtrain logs.  The only other deltas
# that occur are the 1000000 us one-frame hole -- 26 of 71,324 navtest deltas
# in 9.6% of its logs, 656 of 668,396 navtrain deltas in 19.6% of its logs,
# where NAVSIM itself sets sample_prev / sample_next to None -- and three
# 1500000 us two-frame holes in navtrain.  A +-250000 us window therefore
# accepts every genuine neighbour with a 30x margin on the worst jitter and
# rejects every hole by 249000 us, without a single borderline case.
NAVSIM_FRAME_DT_US = 500000
NAVSIM_FRAME_DT_TOL_US = 250000


@DATASETS.register_module()
class CustomNavsimDataset(Custom3DDataset):
    """NAVSIM dataset with BEVFormer's temporal queue.

    Args:
        ann_file (str): Path of the info pickle written by
            ``tools/data_converter/navsim_converter.py``.
        pipeline (list[dict], optional): Processing pipeline.
        data_root (str, optional): Root that relative ``data_path`` /
            ``lidar_path`` entries in the info pkl are resolved against.  This
            is mmdet3d's ``Custom3DDataset`` parameter name (BEVFusion calls
            the same thing ``dataset_root``).
        classes (tuple[str], optional): Classes used by the detector.  Defaults
            to the seven :data:`NAVSIM_CLASSES`.
        camera_names (tuple[str]): Subset of :data:`NAVSIM_CAMERAS` to feed to
            the model, **in the given order** -- that order is the camera axis
            of the image tensor and of ``lidar2img``, so it must match
            ``num_cams`` in both ``transformer`` and ``SpatialCrossAttention``.
            Defaults to the three front cameras, the standard NAVSIM rig.
        queue_length (int): Number of frames per training sample, history
            included (so ``queue_length - 1`` history frames + the current one).
        bev_size (tuple[int, int]): ``(bev_h, bev_w)``, kept only so the
            dataset can be introspected; the model reads its own config.
        load_interval (int): Take every ``load_interval``-th *sample* (the
            frame pool is never subsampled -- that would punch holes in the
            history chain).
        with_velocity (bool): Append ``(vx, vy)`` to the boxes.
        modality (dict, optional): ``use_camera`` / ``use_lidar`` switches.
            Defaults to camera-only.
        box_type_3d (str): Must stay ``'LiDAR'``; NAVSIM's ego frame *is* the
            LiDAR frame.
        filter_empty_gt (bool): Drop training samples whose *current* frame has
            no labelled object.  History frames are never filtered -- see
            :meth:`prepare_train_data`.
        test_mode (bool): No ``ann_info``, no queue, no group flag.
        eval_cfg (dict, optional): nuScenes-styled evaluation configuration.
            Defaults to :attr:`DefaultEvalCfg`.
        use_valid_flag (bool): Filter GT with the ``valid_flag`` field.  Always
            leave this True for NAVSIM: ``num_lidar_pts`` is ``-1`` everywhere
            (NAVSIM ships no per-box point count), so the nuScenes-style
            ``num_lidar_pts > 0`` fallback would empty every single frame.
        filter_by_split (bool): Build the sample list from the
            ``in_split_tokens`` flag, i.e. the official navtrain / navtest
            token set.  Ignored when the field is absent or selects nothing.
        temporal_key (str): Which field defines a temporal unit, i.e. where
            ``prev_bev`` is reset.  ``'log_name'`` (default) gives maximum
            continuity -- a NAVSIM log runs for minutes, and BEV warping only
            needs a valid relative ego pose, which holds across the whole log.
            ``'scene_token'`` restricts continuity to NAVSIM's 20 s / 40-frame
            scenes, the closest analogue of a nuScenes scene; use it to
            reproduce nuScenes-like sequence lengths.  Either way, in-log gaps
            also break the segment (see ``temporal_segment``).
        max_dt_us (int): Reserved knob for the maximum tolerated age of a
            history frame, in microseconds.  Defaults to 750000, i.e. one
            dropped frame; the ``prev``-chain validation below is already
            stricter than this (it requires a direct 0.5 s neighbour), so this
            value is informational unless that validation is relaxed.
        retry_limit (int): How many times ``__getitem__`` may resample before
            giving up.  Upstream's ``while True`` loops forever on an
            all-empty dataset, which is exactly what NAVSIM produces if
            ``use_valid_flag`` is ever turned off.
    """

    #: ``classes=None`` falls back to this, so a config may omit ``classes``.
    CLASSES = NAVSIM_CLASSES

    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    #: Used when the config passes no ``eval_cfg``.  ``class_range`` mirrors the
    #: default BEV grid (``pc_range = [0, -32, -3, 64, 32, 5]``) for every class:
    #: the head's sigmoid already hard-clamps decoded centres inside
    #: ``pc_range``, so a *smaller* eval range can only delete valid boxes, and
    #: a per-class taper (BEVFusion's stale 50/40/30) would silently drop the
    #: far field of every class but ``vehicle``.
    DefaultEvalCfg = {
        'point_cloud_range': [0.0, -32.0, -3.0, 64.0, 32.0, 5.0],
        'dist_ths': [0.5, 1.0, 2.0, 4.0],
        'dist_th_tp': 2.0,
        'min_recall': 0.1,
        'min_precision': 0.1,
        'mean_ap_weight': 5,
        'class_names': list(NAVSIM_CLASSES),
        # NAVSIM has no box attributes, so there is no 'attr_err' here.  That
        # makes NDS a weighted mean over 9 terms (5 * mAP + 4 TP scores) rather
        # than nuScenes' 10, so **NAVSIM NDS is not comparable to nuScenes
        # NDS** -- it reads systematically higher for the same mAP/TP errors.
        'tp_metrics': ['trans_err', 'scale_err', 'orient_err', 'vel_err'],
        'err_name_maping': ErrNameMapping,
        'class_range': {name: (64, 32) for name in NAVSIM_CLASSES},
        'camera_fov': {
            'enabled': False,
            'min_visible_corners': 1,
            'margin': 0.0,
            'apply_within': None,
            # Stored image size.  NAVSIM's cy is 560 on 1080-row images, so
            # never let this fall through to the principal point.
            'image_wh': [1920, 1080],
        },
    }

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 camera_names=('CAM_F0', 'CAM_L0', 'CAM_R0'),
                 queue_length=4,
                 bev_size=(100, 100),
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_cfg=None,
                 use_valid_flag=True,
                 filter_by_split=True,
                 temporal_key='log_name',
                 max_dt_us=750000,
                 retry_limit=50,
                 **kwargs):
        # Everything ``load_annotations`` reads must exist before
        # ``Custom3DDataset.__init__`` runs it, and so must everything
        # ``__len__`` reads, because ``__init__`` then calls
        # ``_set_group_flag`` -> ``len(self)``.
        assert temporal_key in ('log_name', 'scene_token'), (
            "temporal_key must be 'log_name' or 'scene_token', got "
            '{!r}'.format(temporal_key))
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.filter_by_split = filter_by_split
        self.temporal_key = temporal_key
        self.max_dt_us = int(max_dt_us)
        self.queue_length = queue_length
        self.bev_size = tuple(bev_size)
        self.retry_limit = int(retry_limit)
        self.with_velocity = with_velocity

        self.camera_names = list(camera_names) if camera_names else list(
            NAVSIM_CAMERAS)
        unknown = [c for c in self.camera_names if c not in NAVSIM_CAMERAS]
        assert not unknown, 'Unknown NAVSIM cameras: {}'.format(sorted(unknown))

        # Configs for this dataset are routinely copied from the nuScenes ones,
        # which carry fields this class has no use for (``overlap_test``,
        # ``load_augmented``, ...).  Accept and ignore them so a copy-paste does
        # not crash, but say so once -- a silently swallowed typo is worse.
        if kwargs:
            warnings.warn(
                'CustomNavsimDataset ignores unknown arguments: {}'.format(
                    sorted(kwargs.keys())))
        self.extra_kwargs = dict(kwargs)

        super(CustomNavsimDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.eval_cfg = eval_cfg if eval_cfg is not None else copy.deepcopy(
            self.DefaultEvalCfg)

        if self.modality is None:
            # BEVFormer on NAVSIM is camera-only.  ``use_external`` is False
            # unlike the nuScenes configs: there it flags the separately
            # downloaded CAN-bus expansion, whereas NAVSIM's ``can_bus`` ships
            # inside the info pkl.
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_annotations(self, ann_file):
        """Load the info pkl and build the frame pool + index maps.

        Returns the **full** frame list, so ``self.data_infos`` is the history
        pool.  The sample list lands in ``self.sample_indices``.
        """
        data = mmcv.load(ann_file)
        if isinstance(data, dict):
            self.metadata = data.get('metadata', {})
            data_infos = list(data['infos'])
        else:
            # A bare list is accepted so a hand-built pkl works for debugging.
            self.metadata = {}
            data_infos = list(data)

        # Deterministic, time-ordered frame layout within each log.  Everything
        # below (the prev chain, the temporal segments, the sample list) is
        # positional, so this sort has to happen before any of it.  The
        # converter sorts the same way; repeating it here makes the dataset
        # independent of how the pkl was assembled.
        data_infos = sorted(
            data_infos, key=lambda e: (e['log_name'], e['timestamp']))

        self.token2idx = {
            info['token']: i
            for i, info in enumerate(data_infos)
        }

        # --- prev chain -------------------------------------------------- #
        # The converter already nulls ``prev`` across pruning holes, log
        # boundaries and dt outliers, but re-validate here: an info pkl may
        # have been built by an older converter, concatenated from several
        # runs, or sliced by hand, and a wrong ``prev`` is silent (it would
        # warp BEV features with a bogus ego delta).  A valid neighbour must be
        # in the same log and exactly one 0.5 s step back.
        num_frames = len(data_infos)
        prev_index = np.full(num_frames, -1, dtype=np.int64)
        for i, info in enumerate(data_infos):
            prev_token = info.get('prev')
            if not prev_token:
                continue
            j = self.token2idx.get(prev_token)
            if j is None:
                continue
            prev_info = data_infos[j]
            if prev_info['log_name'] != info['log_name']:
                continue
            dt = int(info['timestamp']) - int(prev_info['timestamp'])
            # dt > 0 is implied by the window, which (with the sort above)
            # also guarantees j < i, i.e. the chain only ever walks backwards.
            if abs(dt - NAVSIM_FRAME_DT_US) >= NAVSIM_FRAME_DT_TOL_US:
                continue
            prev_index[i] = j
        self.prev_index = prev_index

        # --- temporal segments ------------------------------------------- #
        # ``scene_token`` in ``img_metas`` is what both ``union2one`` and
        # ``BEVFormer.forward_test`` compare to decide whether ``prev_bev`` is
        # reusable, so it must change at *every* discontinuity, not just at log
        # boundaries.  A string of ``<temporal_key value>#<break counter>``
        # does that with no detector change: the counter advances at each
        # missing ``prev`` (log start or in-log hole), and with
        # ``temporal_key='scene_token'`` the key itself also changes at scene
        # boundaries even though the prev chain runs straight through them.
        segments = []
        segment_id = -1
        for i, info in enumerate(data_infos):
            if prev_index[i] < 0:
                segment_id += 1
            key = info.get(self.temporal_key)
            if key is None:
                key = info['log_name']
            segments.append('{}#{}'.format(key, segment_id))
        self.temporal_segment = segments

        # --- sample list ------------------------------------------------- #
        # ``in_split_tokens`` marks the official navtrain / navtest frames.
        # Everything else in the pkl exists only to give those frames real
        # history.  Fall back to "every frame is a sample" when the flag is
        # absent (older pkl) or selects nothing (mis-specified scene filter) --
        # a zero-length dataset is a much more confusing failure.
        has_flag = any('in_split_tokens' in info for info in data_infos)
        if self.filter_by_split and has_flag:
            sample_indices = [
                i for i, info in enumerate(data_infos)
                if bool(info.get('in_split_tokens', True))
            ]
            if not sample_indices:
                warnings.warn(
                    'in_split_tokens selected 0 of {} frames in {}; falling '
                    'back to all frames'.format(num_frames, ann_file))
                sample_indices = list(range(num_frames))
        else:
            sample_indices = list(range(num_frames))

        # Subsample *samples*, never the pool: thinning the pool would delete
        # the history the remaining samples walk back into.  This is what the
        # two index spaces buy, measured on the shipped pkls: the pool is
        # 1.6030x the sample list on navtest (19,470 frames / 12,146 samples)
        # and 1.4917x on navtrain (154,073 / 103,288), so 37.6% / 33.0% of the
        # frames here exist only as history -- and with them 99.8765% /
        # 99.7638% of samples reach a full 3-frame real history.  Index into
        # ``data_infos`` with a sample index and the evaluator scores 1.60x /
        # 1.49x too many frames.
        self.sample_indices = np.asarray(
            sample_indices[::self.load_interval], dtype=np.int64)

        # --- sample-level segments (test mode only) ------------------------ #
        # ``temporal_segment`` above is a property of the FRAME POOL, which is
        # the right thing for training: ``prepare_train_data`` walks the pool's
        # prev chain, so every queue hop really is one 0.5 s frame.
        #
        # Test mode does not.  ``BEVFormer.forward_test`` is handed the in-split
        # samples one after another and carries ``prev_bev`` whenever
        # ``scene_token`` is unchanged -- but consecutive in-split frames are
        # often NOT adjacent pool frames, because the official token list skips
        # frames inside a log.  Measured on navtest: 591 of 10,631 continued
        # pairs (5.6%) are 2-5 pool frames apart, up to 2.5 s and 27.5 m, and
        # ``load_interval > 1`` makes that every pair.  The pool-level string
        # cannot see those jumps, so the model would warp a multi-second-old BEV
        # by a correspondingly large can_bus delta -- an input training never
        # produces, since the training queue only ever hops 0.5 s (1.0 s at most
        # through the random drop in ``_queue_frame_indices``).
        #
        # So test mode gets its own counter, advanced whenever the previous
        # SAMPLE is not the immediate prev-chain predecessor of this one.
        # ``prepare_test_data`` substitutes it; training is untouched, and no
        # detector change is needed.
        sample_segments = []
        seg = -1
        prev_frame = -1
        for frame_index in self.sample_indices.tolist():
            if prev_frame < 0 or int(self.prev_index[frame_index]) != prev_frame:
                seg += 1
            sample_segments.append(
                '{}#s{}'.format(self.temporal_segment[frame_index].rsplit('#', 1)[0], seg))
            prev_frame = frame_index
        self.sample_segment = sample_segments

        return data_infos

    def __len__(self):
        """Number of **samples** (in-split frames), not of pooled frames."""
        return len(self.sample_indices)

    def _abs_path(self, path):
        """Join a pkl-relative sensor path onto ``data_root``."""
        if path is None or self.data_root is None or osp.isabs(path):
            return path
        return osp.join(self.data_root, path)

    # ------------------------------------------------------------------ #
    # Sample construction -- these two take FRAME indices
    # ------------------------------------------------------------------ #
    def get_data_info(self, index):
        """Build the pipeline input dict for a **frame** index.

        Args:
            index (int): Index into ``self.data_infos`` (the frame pool), *not*
                into the sample list.  ``prepare_train_data`` /
                ``prepare_test_data`` do the sample -> frame mapping.

        Returns:
            dict: The keys BEVFormer's pipeline and
                :class:`CustomCollect3D` expect, i.e. ``sample_idx``,
                ``pts_filename``, ``sweeps``, ``timestamp``, ``img_filename``,
                ``lidar2img``, ``cam_intrinsic``, ``lidar2cam``,
                ``prev_idx``/``next_idx``, ``scene_token``, ``can_bus``,
                ``frame_idx``, plus ``navsim_token`` / ``log_name`` / ``cams``
                for the evaluator and the FOV filter.
        """
        index = int(index)
        info = self.data_infos[index]

        # BUG FIX (spec 2.4): ``can_bus`` must be a fresh array.  Both
        # ``union2one`` and ``BEVFormer.forward_test`` do
        # ``can_bus[:3] -= prev_pos`` *in place*, and neither ``get_data_info``
        # nor ``CustomCollect3D`` copies, so aliasing ``info['can_bus']``
        # mutates the info pkl in memory.  That is self-healing across calls
        # (the fields are rewritten below every time) except when one queue
        # contains the same frame twice -- which the history padding does at
        # every segment start.  Both occurrences are then the *same object*,
        # so union2one differences a slot against itself-already-zeroed.
        # Measured on this grid with the aliasing code restored, queue
        # ``[p, p, p, i]``: the three history elements read 4,052,984.9 m of
        # fake ego motion and the CURRENT element -- the one that actually
        # drives the head's BEV shift -- reads 8,105,969.8 m
        # (= |T_i| + |T_p|, the UTM pose magnitudes).  With the copy, all four
        # read 0.0001 m, the true 0.5 s step of a near-stationary frame.
        #
        # float64 is load-bearing, not tidiness: NAVSIM's
        # ``ego2global_translation`` is UTM-like (measured mean
        # (5.0e5, 4.0e6, 2.6e2) m over the navtest logs, up to
        # (6.6e5, 4.7e6, 6.3e2)), where the float32 ULP is 0.25 m.  The ego
        # step this differencing has to recover averages 2.536 m on navtest and
        # 2.321 m on navtrain, so float32 would throw away up to a quarter
        # metre of it -- ~10%, not a rounding detail.
        raw_can_bus = info.get('can_bus')
        if raw_can_bus is None:
            can_bus = np.zeros(18, dtype=np.float64)
        else:
            can_bus = np.array(raw_can_bus, dtype=np.float64,
                               copy=True).reshape(-1)
            assert can_bus.shape[0] == 18, (
                'can_bus must have 18 slots (nuScenes layout), got {} for '
                'token {}'.format(can_bus.shape[0], info['token']))

        input_dict = dict(
            sample_idx=info['token'],
            # Kept under a second name because ``sample_idx`` is overwritten by
            # some pipelines and the evaluator needs the raw NAVSIM token.
            navsim_token=info['token'],
            log_name=info['log_name'],
            pts_filename=self._abs_path(info.get('lidar_path')),
            # BEVFormer is camera-only (``max_sweeps`` defaults to 0 in the
            # converter), so this is normally empty; resolve the paths anyway
            # so a LiDAR-side experiment works unchanged.
            sweeps=[
                dict(s, data_path=self._abs_path(s.get('data_path')))
                for s in info.get('sweeps', [])
            ],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info.get('prev'),
            next_idx=info.get('next'),
            # NOT the raw NAVSIM ``scene_token``: the temporal segment, so
            # ``prev_bev`` is reset at log starts *and* at in-log frame drops.
            scene_token=self.temporal_segment[index],
            can_bus=can_bus,
            frame_idx=int(info.get('frame_idx', 0)),
            timestamp=info['timestamp'] / 1e6,
            # Raw calibration, for ``_camera_fov_mask`` and the FOV filter.
            cams=info.get('cams'),
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type in self.camera_names:
                cam_info = info['cams'][cam_type]
                image_paths.append(self._abs_path(cam_info['data_path']))

                # NAVSIM gives camera -> LiDAR directly, so the chain is just
                # an inverse and a padded K.  This is algebraically the same
                # matrix ``CustomNuScenesDataset`` builds with its
                # transpose-shuffle (``viewpad @ lidar2cam_rt.T``), written the
                # short way.  float64 throughout, matching the nuScenes path:
                # ``RandomScaleImageMultiViewImage`` left-multiplies by a
                # float64 ``np.eye(4)`` and the model casts to float32 itself.
                camera2lidar = np.eye(4, dtype=np.float64)
                camera2lidar[:3, :3] = cam_info['sensor2lidar_rotation']
                camera2lidar[:3, 3] = cam_info['sensor2lidar_translation']
                lidar2camera = np.linalg.inv(camera2lidar)

                intrinsic = np.asarray(
                    cam_info['cam_intrinsic'], dtype=np.float64)
                viewpad = np.eye(4, dtype=np.float64)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rts.append(viewpad @ lidar2camera)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2camera)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            input_dict['ann_info'] = self.get_ann_info(index)

        # Written exactly as ``CustomNuScenesDataset`` does, so the 18 slots mean
        # the same thing on both datasets.  The converter has already swapped
        # NAVSIM's [10:13]=velocity / [13:16]=rotation-rate into the nuScenes
        # order, so only the pose slots are refreshed here.
        #
        # TRAP, deliberately preserved: ``can_bus[3:7] = rotation`` does NOT
        # write (w, x, y, z).  :class:`pyquaternion.Quaternion` defines
        # ``__float__`` but neither ``__iter__`` nor ``__len__``, so numpy
        # treats it as a scalar and broadcasts ``float(q) == q.w`` into all
        # four slots (verified on this env, numpy 1.19.5: the slots become
        # [w, w, w, w]).  Upstream BEVFormer has always behaved this way on
        # nuScenes.
        #
        # It is preserved for upstream parity, NOT for weight transfer: nothing
        # here loads a pretrained ``can_bus_mlp`` (``bevformer_navsim_base``
        # loads an FCOS3D *2D backbone*, and the BEV head trains from scratch),
        # so this line is free to change.  The reason not to is that it would
        # make NAVSIM and nuScenes runs differ in an input channel that has
        # nothing to do with the port, and the information is not lost anyway:
        # slot [16] already carries the absolute global yaw.  Writing
        # ``can_bus[3:7] = rotation.elements`` is the one-line fix if someone
        # wants the real quaternion, and it should be a deliberate, measured
        # change.  Code that needs the true rotation reads ``info['can_bus']``
        # or ``ego2global_rotation``, both of which are correct.
        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        # [16] = absolute global yaw in radians, [17] = the same in degrees.
        # ``union2one`` differences [-1] (degrees) and the transformer reads
        # [-2] (radians); both must be the *absolute* heading, hence the
        # [0, 360) wrap rather than [-180, 180).
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def get_ann_info(self, index):
        """Annotations for a **frame** index.

        Args:
            index (int): Index into ``self.data_infos``.
        """
        info = self.data_infos[int(index)]

        gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32).reshape(-1, 7)
        gt_names = np.asarray(info['gt_names']).reshape(-1)

        # ``valid_flag`` only -- never the nuScenes ``num_lidar_pts > 0`` rule.
        # NAVSIM has no per-box point count and the converter writes -1
        # everywhere, so that rule would empty every frame.
        if self.use_valid_flag and 'valid_flag' in info:
            mask = np.asarray(info['valid_flag'], dtype=bool).reshape(-1)
        else:
            mask = np.ones(len(gt_boxes), dtype=bool)

        gt_boxes = gt_boxes[mask]
        gt_names = gt_names[mask]

        if self.with_velocity:
            gt_velocity = np.asarray(
                info['gt_velocity'], dtype=np.float32).reshape(-1, 2)[mask]
            # NAVSIM's ``gt_velocity_3d`` legitimately contains NaN (objects
            # with no track history).  NaN in a regression target poisons the
            # whole loss, and the velocity code weights are only 0.2, so zero
            # is the right stand-in.
            gt_velocity = np.nan_to_num(gt_velocity, nan=0.0)
            gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)

        # -1 for names outside ``CLASSES``; ``ObjectNameFilter`` /
        # ``filter_empty_gt`` treat -1 as "ignore".
        gt_labels = np.array(
            [
                self.CLASSES.index(n) if n in self.CLASSES else -1
                for n in gt_names
            ],
            dtype=np.int64)

        # origin=(0.5, 0.5, 0.5): the stored z is the box centre, not its base.
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
            gt_names=gt_names,
        )

    # ------------------------------------------------------------------ #
    # Queue assembly -- these take SAMPLE indices
    # ------------------------------------------------------------------ #
    def _queue_frame_indices(self, frame_index):
        """Frame indices of one training queue, oldest first.

        Rebuilds upstream's ``range(index - queue_length, index)`` on the real
        ``prev`` chain instead of on raw positions.  Positional arithmetic is
        wrong here for two reasons: ``data_infos`` is a concatenation of logs,
        so ``index - 1`` can belong to a different log, and an in-log frame
        drop makes ``index - 1`` 1.0 s old rather than 0.5 s.

        Returns:
            list[int]: ``queue_length`` frame indices ending with
                ``frame_index``.  Entries may repeat when there is not enough
                real history (the front is padded by repetition, exactly as
                upstream's ``max(0, i)`` clamp does at the dataset start).
        """
        frame_index = int(frame_index)

        history = []
        cursor = frame_index
        for _ in range(self.queue_length):
            prev = int(self.prev_index[cursor])
            if prev < 0:
                break
            history.append(prev)
            cursor = prev
        history.reverse()  # oldest -> newest

        if len(history) == self.queue_length:
            # Upstream's ``random.shuffle(index_list); sorted(index_list[1:])``
            # = drop one of the ``queue_length`` candidates uniformly at random
            # and keep the rest in time order.  It is a temporal-stride
            # augmentation, so reproduce it rather than always taking the
            # newest.
            drop = random.randrange(len(history))
            history = history[:drop] + history[drop + 1:]

        # Pad the front by repeating the oldest frame we do have (or the
        # current frame when there is no history at all).  A repeated frame
        # contributes a zero ego delta, which is the honest answer at a
        # segment start; the detector additionally drops ``prev_bev`` there via
        # ``prev_bev_exists``.
        while len(history) < self.queue_length - 1:
            history.insert(0, history[0] if history else frame_index)

        history.append(frame_index)
        return history

    def prepare_train_data(self, index):
        """Training data preparation for a **sample** index.

        Args:
            index (int): Index into ``self.sample_indices``.

        Returns:
            dict | None: One ``union2one``-merged queue, or None when the
                sample must be resampled.
        """
        frame_index = int(self.sample_indices[index])
        index_list = self._queue_frame_indices(frame_index)
        last = len(index_list) - 1

        queue = []
        for pos, i in enumerate(index_list):
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if example is None:
                return None
            # ``filter_empty_gt`` applies to the CURRENT frame only.  History
            # frames contribute nothing but ``prev_bev`` -- their GT is never
            # read by ``obtain_history_bev`` -- so rejecting a sample because
            # some history frame is empty is pure sample loss.  Measured over
            # this front-only 64 m ROI: the empty-current-frame rate is 3.0304%
            # of navtrain in-split samples (4.7835% of navtest), and it is
            # independent of ``queue_length`` because only the current frame is
            # tested.  Testing every queue frame as upstream
            # ``CustomNuScenesDataset`` does would take navtrain to 4.3853% at
            # ``queue_length=4`` -- +1.355 pp, a 44.7% relative increase in
            # resampling, for no benefit.
            if pos == last and self.filter_empty_gt:
                labels = example.get('gt_labels_3d')
                if labels is None:
                    return None
                labels = labels._data if hasattr(labels, '_data') else labels
                if ~(labels != -1).any():
                    return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        """Merge a queue into one sample.

        Verbatim from :class:`CustomNuScenesDataset`:
        ``BEVFormer.forward_train`` slices ``img[:, :-1]`` / ``img[:, -1]`` and
        indexes ``img_metas[i][len_queue - 1]``, so the stacked-image /
        dict-of-metas shape below is a hard contract.

        The ``can_bus`` differencing converts the absolute global pose written
        by ``get_data_info`` into a delta against the previous queue element,
        which is what ``PerceptionTransformer.get_bev_features`` consumes.  It
        mutates ``can_bus`` in place -- see the copy in ``get_data_info``.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                # NOTE: ``can_bus[-1]`` (the yaw delta in degrees) comes out
                # +-360 deg off whenever the heading crosses the 0/360 seam,
                # because ``get_data_info`` wraps the absolute global yaw into
                # [0, 360) and this subtraction is not re-wrapped.  Measured on
                # the shipped pkls: 108 of 17,954 navtest pool pairs (0.60%)
                # and 712 of 140,523 navtrain ones (0.51%) -- 1.58% / 1.35% of
                # ``queue_length=4`` samples contain at least one such hop.
                # The recorded delta is then the true delta -/+ 360 (observed
                # magnitudes 342.3-359.9 deg on navtest, the true turn being
                # 0.05-17.68 deg).  It is harmless in both places that read it
                # geometrically, which is why it is left exactly as upstream
                # BEVFormer wrote it:
                #   * ``get_bev_features``' BEV shift does not use this slot at
                #     all -- it takes the translation from ``can_bus[0:2]`` and
                #     the heading from ``can_bus[-2]``, the *absolute* yaw in
                #     radians, which ``union2one`` never differences;
                #   * ``rotate_prev_bev`` does use it (``transformer.py``:
                #     ``rotation_angle = can_bus[-1]``), but it goes straight
                #     into ``torchvision.transforms.functional.rotate``, whose
                #     affine matrix is built from cos/sin of the angle -- so
                #     theta and theta +- 360 are the same rotation, to float
                #     round-off.
                # The only consumer that sees the spike as a *value* is the raw
                # ``can_bus_mlp`` input.  Do NOT "fix" the arithmetic here: it
                # is upstream behaviour, deliberately preserved, and changing it
                # would make NAVSIM and nuScenes runs differ in an input channel
                # that has nothing to do with this port.
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(
            torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def prepare_test_data(self, index):
        """Test data preparation for a **sample** index.

        No queue: ``BEVFormer.forward_test`` carries ``prev_bev`` across calls
        itself and resets it whenever ``scene_token`` changes, which is why
        ``get_data_info`` puts a temporal segment there.  That also means the
        test loader must not be shuffled.

        The segment substituted here is the **sample-level** one, not the
        pool-level one ``get_data_info`` wrote: at test time the model only sees
        the in-split frames, so a jump over skipped pool frames is a real
        temporal discontinuity even though the pool chain is unbroken.  See the
        note in :meth:`load_annotations`.
        """
        input_dict = self.get_data_info(int(self.sample_indices[index]))
        input_dict['scene_token'] = self.sample_segment[index]
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def get_cat_ids(self, index):
        """Category ids of one **sample**, used by :class:`CBGSDataset`."""
        info = self.data_infos[int(self.sample_indices[index])]
        gt_names = np.asarray(info['gt_names']).reshape(-1)
        if self.use_valid_flag and 'valid_flag' in info:
            gt_names = gt_names[np.asarray(
                info['valid_flag'], dtype=bool).reshape(-1)]
        return [
            self.cat2id[n] for n in set(gt_names.tolist()) if n in self.CLASSES
        ]

    def __getitem__(self, idx):
        """Get one sample by **sample** index."""
        if self.test_mode:
            return self.prepare_test_data(idx)

        # Bounded, unlike upstream's ``while True``.  If every sample is
        # rejected -- which is what ``use_valid_flag=False`` produces on
        # NAVSIM, since ``num_lidar_pts`` is -1 everywhere -- the upstream loop
        # hangs the dataloader worker with no message at all.
        for _ in range(self.retry_limit + 1):
            data = self.prepare_train_data(idx)
            if data is not None:
                return data
            idx = self._rand_another(idx)
        raise RuntimeError(
            'CustomNavsimDataset: {} consecutive samples were rejected '
            'starting from index {}. Every attempt returned an empty or '
            'unloadable frame. Check use_valid_flag (must be True for NAVSIM: '
            'num_lidar_pts is -1 everywhere) and filter_empty_gt, and that '
            'the image files under data_root actually exist.'.format(
                self.retry_limit + 1, idx))

    # ------------------------------------------------------------------ #
    # Formatting & evaluation (nuScenes-styled, shared with the BEVFusion port)
    # ------------------------------------------------------------------ #
    def _in_eval_range(self, x, y, class_name):
        """Whether a box centre lies in both the class and the model range."""
        class_range = self.eval_cfg.get('class_range', {})
        if class_name not in class_range:
            return False
        range_x, range_y = class_range[class_name]
        if abs(x) > range_x or abs(y) > range_y:
            return False

        # The front-only grid is not symmetric, so the class radius above is
        # not sufficient: a box at x = -5 passes ``abs(x) <= 64`` but is behind
        # the ego and outside ``pc_range``, where the head can never place a
        # prediction.  Scoring it would be an unreachable miss.
        point_cloud_range = self.eval_cfg.get('point_cloud_range')
        if point_cloud_range is not None:
            return (point_cloud_range[0] <= x <= point_cloud_range[3]
                    and point_cloud_range[1] <= y <= point_cloud_range[4])
        return True

    def _camera_fov_mask(self, info, boxes):
        """``(N,)`` mask of boxes inside the configured camera evaluation area.

        ``load_gt`` builds the evaluation GT straight from ``data_infos``,
        bypassing the pipeline, so ``ObjectFOVFilter`` never touches it.
        Without this mask on both GT and predictions, detections outside the
        rig's view would still count as false positives even though the
        matching GT was removed.  Controlled by ``eval_cfg['camera_fov']``;
        absent or ``enabled: False`` keeps every box.
        """
        cfg = self.eval_cfg.get('camera_fov')
        if not cfg or not cfg.get('enabled', False) or len(boxes) == 0:
            return np.ones(len(boxes), dtype=bool)

        cams = info.get('cams')
        if not cams or any(name not in cams for name in self.camera_names):
            return np.ones(len(boxes), dtype=bool)

        # Prefer the configured size.  ``image_size_from_intrinsics`` assumes a
        # centred principal point, which NAVSIM does not have vertically
        # (cy = 560 on 1080-row images), so it over-estimates the height by
        # 40 px and makes the test slightly permissive.
        image_wh = cfg.get('image_wh') or image_size_from_intrinsics(
            cams, self.camera_names)

        if isinstance(boxes, LiDARInstance3DBoxes):
            box_corners = boxes.corners.numpy()
            box_centres = boxes.gravity_center.numpy()
        else:
            boxes = np.asarray(boxes)
            lidar_boxes = LiDARInstance3DBoxes(
                boxes, box_dim=boxes.shape[-1], origin=(0.5, 0.5, 0.5))
            box_corners = lidar_boxes.corners.numpy()
            box_centres = lidar_boxes.gravity_center.numpy()

        mask = boxes_in_view(
            box_corners,
            build_lidar2image(cams, self.camera_names),
            tuple(image_wh),
            min_corners=cfg.get('min_visible_corners', 1),
            margin=cfg.get('margin', 0.0))

        apply_within = cfg.get('apply_within')
        if apply_within is not None:
            x_min, y_min, x_max, y_max = apply_within
            considered = ((box_centres[:, 0] >= x_min)
                          & (box_centres[:, 0] <= x_max)
                          & (box_centres[:, 1] >= y_min)
                          & (box_centres[:, 1] <= y_max))
            mask |= ~considered
        return mask

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Serialise predictions to the nuScenes-styled result json.

        ``results`` is produced by iterating this dataset, so ``sample_id`` is
        a **sample** index.  The token therefore has to go through
        ``self.sample_indices``; indexing ``self.data_infos`` directly (as the
        nuScenes code does, where the two coincide) attributes every prediction
        to the wrong frame, and the only symptom is a bad number.
        """
        assert len(results) == len(self.sample_indices), (
            'got {} results for {} samples; _format_bbox maps result index -> '
            'sample index -> frame index and cannot align a partial result '
            'list'.format(len(results), len(self.sample_indices)))

        navsim_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']

            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            # mmdet3d yaw -> NAVSIM heading (the converter's inverse).
            box_yaw = -box3d.yaw.numpy() - np.pi / 2

            info = self.data_infos[int(self.sample_indices[sample_id])]
            sample_token = info['token']
            fov_mask = self._camera_fov_mask(info, box3d)

            for i in range(len(box3d)):
                if not fov_mask[i]:
                    continue
                name = mapped_class_names[labels[i]]
                center_x = box_gravity_center[i, 0].item()
                center_y = box_gravity_center[i, 1].item()
                if not self._in_eval_range(center_x, center_y, name):
                    continue
                annos.append(
                    dict(
                        sample_token=sample_token,
                        translation=box_gravity_center[i].tolist(),
                        size=box_dims[i].tolist(),
                        rotation=list(
                            Quaternion(axis=[0, 0, 1], radians=box_yaw[i])),
                        velocity=[
                            box3d.tensor[i, 7].item(),
                            box3d.tensor[i, 8].item(),
                        ],
                        detection_name=name,
                        # ``DetectionBox`` asserts ``type(...) == float``, so a
                        # numpy/torch scalar must be ``.item()``-ed here.
                        detection_score=float(scores[i].item()),
                        attribute_name=name,
                    ))
            navsim_annos[sample_token] = annos

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_navsim.json')
        print('Results writes to', res_path)
        mmcv.dump({'meta': self.modality, 'results': navsim_annos}, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Write the result json(s) and return their path(s)."""
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            return self._format_bbox(results, jsonfile_prefix), tmp_dir

        result_files = {}
        for name in results[0]:
            print('\nFormating bboxes of {}'.format(name))
            results_ = [out[name] for out in results]
            result_files[name] = self._format_bbox(
                results_, osp.join(jsonfile_prefix, name))
        return result_files, tmp_dir

    def load_gt(self):
        """Build the GT :class:`EvalBoxes` from the **sample** frames.

        Only in-split frames are scored.  Adding the pooled history frames
        here would contribute GT with no matching prediction and divide recall
        by the pool/sample ratio, measured on the shipped pkls as **1.6030x**
        on navtest (19,470 pooled frames against 12,146 samples) and
        **1.4917x** on navtrain (154,073 against 103,288).
        """
        all_annotations = EvalBoxes()
        for frame_index in self.sample_indices:
            info = self.data_infos[int(frame_index)]
            sample_boxes = []
            gt_boxes = np.asarray(
                info['gt_boxes'], dtype=np.float32).reshape(-1, 7)
            gt_names = np.asarray(info['gt_names']).reshape(-1)
            gt_velocity = np.asarray(
                info.get('gt_velocity', np.zeros((len(gt_boxes), 2))),
                dtype=np.float32).reshape(-1, 2)
            num_pts = np.asarray(
                info.get('num_lidar_pts', np.full(len(gt_boxes),
                                                  -1))).reshape(-1)
            if self.use_valid_flag and 'valid_flag' in info:
                valid_mask = np.asarray(
                    info['valid_flag'], dtype=bool).reshape(-1)
            else:
                valid_mask = np.ones(len(gt_boxes), dtype=bool)

            fov_mask = self._camera_fov_mask(info, gt_boxes)

            for j in range(len(gt_boxes)):
                class_name = gt_names[j]
                if not valid_mask[j] or not fov_mask[j]:
                    continue
                if not self._in_eval_range(gt_boxes[j, 0], gt_boxes[j, 1],
                                           class_name):
                    continue

                # Back to the NAVSIM heading, the same convention
                # ``_format_bbox`` uses for predictions, so ``yaw_diff``
                # compares like with like.
                heading = -gt_boxes[j, 6] - np.pi / 2
                sample_boxes.append(
                    DetectionBox(
                        sample_token=info['token'],
                        translation=tuple(gt_boxes[j, 0:3].tolist()),
                        size=tuple(gt_boxes[j, 3:6].tolist()),
                        rotation=list(
                            Quaternion(axis=[0, 0, 1], radians=heading)),
                        velocity=tuple(gt_velocity[j].tolist()),
                        num_pts=int(num_pts[j]),
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name=class_name,
                    ))
            all_annotations.add_boxes(info['token'], sample_boxes)
        return all_annotations

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 **kwargs):
        """nuScenes-styled detection evaluation on NAVSIM.

        Note on NDS: NAVSIM has no box attributes, so ``tp_metrics`` has no
        ``attr_err`` and :meth:`DetectionMetrics.nd_score` normalises by
        ``mean_ap_weight + 4 = 9`` instead of nuScenes' 10.  **NAVSIM NDS is
        therefore not comparable to nuScenes NDS** -- it reads systematically
        higher for the same mAP and TP errors.  Compare mAP and the per-class
        TP errors across datasets, never NDS.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            result_path = result_files.get('pts_bbox',
                                           list(result_files.values())[0])
        else:
            result_path = result_files

        with open(result_path) as f:
            result_data = json.load(f)

        pred_boxes = EvalBoxes.deserialize(result_data['results'],
                                           DetectionBox)
        gt_boxes = self.load_gt()

        metric_data_list = DetectionMetricDataList()
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data_list.set(
                    class_name, dist_th,
                    accumulate(gt_boxes, pred_boxes, class_name,
                               center_distance, dist_th))

        metrics = DetectionMetrics(self.eval_cfg)
        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                metrics.add_label_ap(
                    class_name, dist_th,
                    calc_ap(metric_data_list[(class_name, dist_th)],
                            self.eval_cfg['min_recall'],
                            self.eval_cfg['min_precision']))
            for metric_name in self.eval_cfg['tp_metrics']:
                metrics.add_label_tp(
                    class_name, metric_name,
                    calc_tp(
                        metric_data_list[(class_name,
                                          self.eval_cfg['dist_th_tp'])],
                        self.eval_cfg['min_recall'], metric_name))

        metrics_summary = metrics.serialize()
        err_name_mapping = self.eval_cfg.get('err_name_maping',
                                             self.ErrNameMapping)

        print('mAP: %.4f' % metrics_summary['mean_ap'])
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping.get(tp_name, tp_name), tp_val))
        print('NDS: %.4f' % metrics_summary['nd_score'])

        print('\nPer-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' %
                  (class_name, class_aps[class_name],
                   class_tps[class_name]['trans_err'],
                   class_tps[class_name]['scale_err'],
                   class_tps[class_name]['orient_err'],
                   class_tps[class_name]['vel_err']))

        detail = {}
        prefix = 'bbox_NAVSIM'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                detail['{}/{}_AP_dist_{}'.format(prefix, name,
                                                 k)] = float('%.4f' % v)
            for k, v in metrics_summary['label_tp_errors'][name].items():
                detail['{}/{}_{}'.format(prefix, name,
                                         k)] = float('%.4f' % v)
        for k, v in metrics_summary['tp_errors'].items():
            detail['{}/{}'.format(prefix, err_name_mapping.get(
                k, k))] = float('%.4f' % v)
        detail['{}/NDS'.format(prefix)] = metrics_summary['nd_score']
        detail['{}/mAP'.format(prefix)] = metrics_summary['mean_ap']

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return detail
