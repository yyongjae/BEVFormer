"""NAVSIM-specific pipeline transforms for BEVFormer.

Ported from the BEVFusion NAVSIM port
(``<bevfusion>/mmdet3d/datasets/pipelines/transforms_3d.py``, commits ``cf3e02a``
and ``ee05061``) and adapted to BEVFormer's result-dict conventions.  The
visibility predicate itself is *not* reimplemented here: it lives in
:mod:`projects.mmdet3d_plugin.datasets.navsim_fov_utils`, which is a verbatim
copy of BEVFusion's ``fov_utils.py``, so both ports filter and score NAVSIM with
exactly the same rule.

Differences from the BEVFusion original, all forced by this repository:

* BEVFormer stores the projection matrices as ``results['lidar2img']`` -- a
  *list* of ``(4, 4)`` arrays, one per configured camera -- where BEVFusion uses
  ``results['lidar2image']`` as a stacked ``(C, 4, 4)``.
* ``image_wh`` is a required argument.  BEVFusion could fall back on
  ``results['ori_shape']`` because its multi-view loader stores PIL's
  ``(W, H)``; here ``ori_shape`` is mmdet3d's ``(H, W, 3 * C)`` from
  ``LoadMultiViewImageFromFiles`` and then a *list* of per-camera ``(H, W, C)``
  after ``PadMultiViewImage`` / ``RandomScaleImageMultiViewImage`` -- three
  incompatible layouts for one key.  Inferring from the intrinsics is worse
  still: NAVSIM's principal point is not vertically centred (``cy`` = 560 on
  1080 rows), so ``2 * cy`` overestimates the height by 40 px and quietly makes
  the test permissive.  An explicit size is the only layout-independent answer.
* ``enabled`` defaults to ``False`` (BEVFusion's class default was ``True``,
  with every shipped config setting ``enabled: false``).  The default here
  matches what the configs actually do; see the class docstring for why.

Nothing in this module is reachable from a nuScenes config: it is only
instantiated where a config asks for ``ObjectFOVFilter``.
"""

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

from ..navsim_fov_utils import boxes_in_view, build_lidar2image

__all__ = ['ObjectFOVFilter']

# How many corners a 3D box has.  ``min_visible_corners`` above this would
# silently reject every box, because ``boxes_in_view`` only clamps the lower
# bound (``>= max(1, min_corners)``).
_NUM_BOX_CORNERS = 8


@PIPELINES.register_module()
class ObjectFOVFilter(object):
    """Drop GT boxes that none of the configured cameras has observed.

    ``ObjectRangeFilter`` keeps a BEV *rectangle* and ``ObjectNameFilter`` keeps
    a class list, but NAVSIM's three front cameras (``CAM_F0`` / ``CAM_L0`` /
    ``CAM_R0``) only span about -87.3 to +87.0 degrees, so the front-only
    ``point_cloud_range`` still admits boxes the rig never saw.  This filter
    projects the eight corners of each box through ``lidar2img`` and keeps the
    box only when at least ``min_visible_corners`` of them land inside an image
    with positive depth -- BEVFormer's own ``point_sampling`` predicate, applied
    to labels instead of to BEV queries.  See
    :mod:`projects.mmdet3d_plugin.datasets.navsim_fov_utils` for why the test is
    by projection and not by an azimuth threshold (the cameras sit off the LiDAR
    origin, so the observed region is not a clean wedge from the ego frame).

    **Why it is off by default.**  BEVFormer's reference predicate for "is this
    box a target" is centre-in-rectangle with no visibility test at all -- which
    is precisely what ``point_cloud_range`` + ``ObjectRangeFilter`` already do,
    since ``in_range_bev`` tests the same centre.  Turning this on therefore
    *changes the task*, not just the data loading, and makes the NAVSIM numbers
    incomparable to any run that did not use it.

    The effect is small and lives almost entirely in the near field, where a
    forward-facing camera mounted 1.52 m up cannot look down far enough.
    BEVFusion measured **-5.2% of in-ROI boxes** on 400 navtrain frames over its
    0-32 m ROI (28.8% of boxes at 2-5 m, 16.1% at 5-10 m, 4.1% at 10-20 m, 2.8%
    beyond 20 m).  Re-measured here through this class on 400 navtest frames:

    ============  =====================  ======================
    distance      removed, 0-32 m ROI    removed, 0-64 m ROI
    ============  =====================  ======================
    2-5 m         16.5% (188 boxes)      16.5% (188 boxes)
    5-10 m        14.4% (1029)           14.4% (1029)
    10-20 m       2.5% (3535)            2.6% (3513)
    beyond 20 m   3.5% (4411)            1.1% (13764)
    **total**     **3.9%** (10747)       **2.2%** (19112)
    ============  =====================  ======================

    The per-bin shape matches BEVFusion's; the lower total is the split (navtest
    rather than navtrain) plus, for the right-hand column, this port's 64 m
    range diluting the near field with far-field boxes the rig sees in full.
    **2.2% is the number that matters here** -- the near-field objects it would
    delete are a small minority of a 64 m label set, so deleting them buys
    little and costs comparability.  Keep it as an ablation knob:
    ``enabled=True`` in the config is the only change needed.

    **Where it must go in the pipeline.**  Directly after ``ObjectNameFilter``
    and *before* every image or BEV augmentation.  Two concrete reasons, both
    silent if ignored:

    1. ``RandomScaleImageMultiViewImage`` left-multiplies every ``lidar2img`` by
       the image scale factor (0.8334 in the NAVSIM base config).  Run after it,
       the projected coordinates are in *resized* image pixels while
       ``image_wh`` is still the stored ``(1920, 1080)``, so the acceptance
       window is 1.2x too large and the filter turns permissive.
       ``PadMultiViewImage`` has the same effect in reverse for the padded
       border.
    2. ``GlobalRotScaleTransImage`` rotates / scales / flips the LiDAR frame and
       right-multiplies ``lidar2img`` by the inverse, so projection stays
       self-consistent -- but the sampled transform differs every epoch, which
       would make the *label set* epoch-dependent.  Visibility is a property of
       an object's real position, not of a sampled augmentation, and the
       evaluator (``CustomNavsimDataset.load_gt``, which never runs the
       pipeline) necessarily uses the raw geometry.  Filtering on raw sensor
       geometry keeps training and evaluation targets identical.

    The matching evaluation-side mask lives in ``eval_cfg['camera_fov']`` and
    must be configured with the *same* values: masking only the training targets
    would leave detections outside the camera region counting as false
    positives with no GT left to match.

    Args:
        min_visible_corners (int): how many of the eight box corners must land
            in an image.  Defaults to 1, which keeps boxes clipped by the image
            border -- still legitimate detection targets.  Must be in 1..8.
        margin (float): pixels of slack allowed outside the image border.
        enabled (bool): ``False`` (the default) makes this a pure no-op, so the
            stage can stay in the pipeline and be switched on from a config
            without editing the pipeline list.
        apply_within (sequence[float], optional): ``[x_min, y_min, x_max,
            y_max]`` BEV rectangle, in LiDAR/ego metres (``+x`` forward, ``+y``
            left for NAVSIM), restricting the test to boxes whose gravity centre
            falls inside it; boxes outside are kept untouched.  Useful on a
            360-degree grid to preserve rear supervision while trimming only the
            camera-queried front region.  ``None`` (the default) filters
            everywhere, which is what the front-only NAVSIM grid wants.
        image_wh (sequence[float]): stored image ``(width, height)`` in pixels,
            e.g. ``[1920, 1080]`` for NAVSIM.  **Required when**
            ``enabled=True``; there is no safe fallback (see the module
            docstring).
        camera_names (sequence[str], optional): camera order to use when
            ``use_raw_cams=True``.  Must match the dataset's ``camera_names`` so
            the matrices line up with the images the model is fed.
        use_raw_cams (bool): read the per-frame ``results['cams']`` dict and
            rebuild the matrices with
            :func:`~projects.mmdet3d_plugin.datasets.navsim_fov_utils.build_lidar2image`
            instead of using ``results['lidar2img']``.  ``False`` by default,
            because inside the pipeline ``lidar2img`` is already built by
            ``CustomNavsimDataset.get_data_info`` from the same ``cams`` entry
            and the two agree exactly.  Set it to ``True`` only if this stage
            has to sit after an augmentation that rewrites ``lidar2img``: it
            then reads raw sensor geometry regardless of pipeline order, which
            is also byte-for-byte what the evaluator's mask uses.
    """

    def __init__(self,
                 min_visible_corners=1,
                 margin=0.0,
                 enabled=False,
                 apply_within=None,
                 image_wh=None,
                 camera_names=None,
                 use_raw_cams=False):
        self.min_visible_corners = int(min_visible_corners)
        if not 1 <= self.min_visible_corners <= _NUM_BOX_CORNERS:
            raise ValueError(
                'min_visible_corners must be in 1..{}, got {}'.format(
                    _NUM_BOX_CORNERS, self.min_visible_corners))
        self.margin = float(margin)
        self.enabled = bool(enabled)
        self.use_raw_cams = bool(use_raw_cams)
        self.camera_names = (None if camera_names is None else
                             tuple(str(name) for name in camera_names))

        # Stored image size.  Validated here rather than in ``__call__`` so a
        # misconfiguration fails when the pipeline is built, not silently on the
        # first batch.  Only required when the filter actually runs, so a config
        # may leave it out while ``enabled=False``.
        if image_wh is None:
            self.image_wh = None
        else:
            image_wh = tuple(float(v) for v in image_wh)
            if len(image_wh) != 2 or image_wh[0] <= 0 or image_wh[1] <= 0:
                raise ValueError(
                    'image_wh must be a positive (width, height), got '
                    '{}'.format(image_wh))
            self.image_wh = image_wh
        if self.enabled and self.image_wh is None:
            raise ValueError(
                'ObjectFOVFilter(enabled=True) requires image_wh=(width, '
                'height) of the stored images, e.g. [1920, 1080] for NAVSIM. '
                "It cannot be inferred: results['ori_shape'] changes layout "
                'across this repository\'s loaders, and the intrinsics are '
                'not centred vertically (cy = 560 on 1080 rows).')
        if self.enabled and self.use_raw_cams and not self.camera_names:
            raise ValueError(
                'ObjectFOVFilter(use_raw_cams=True) requires camera_names, in '
                "the same order as the dataset's camera_names.")

        # ``[x_min, y_min, x_max, y_max]`` in LiDAR/ego metres.  float64 to
        # match ``gravity_center``'s promoted dtype and avoid a float32
        # comparison at the rectangle border.
        self.apply_within = (None if apply_within is None else np.asarray(
            apply_within, dtype=np.float64))
        if self.apply_within is not None and self.apply_within.shape != (4, ):
            raise ValueError(
                'apply_within must be [x_min, y_min, x_max, y_max], got shape '
                '{}'.format(tuple(self.apply_within.shape)))

    def _lidar2img(self, results):
        """Return the ``(C, 4, 4)`` lidar -> image matrices, or ``None``.

        ``None`` means "not enough information to filter", which the caller
        turns into a no-op rather than an error: the same pipeline list is used
        by stages that carry no calibration.
        """
        if self.use_raw_cams:
            # Raw sensor geometry straight from the info pkl, immune to any
            # augmentation that has already rewritten ``lidar2img``.
            cams = results.get('cams')
            if not cams or any(name not in cams
                               for name in self.camera_names):
                return None
            return build_lidar2image(cams, self.camera_names)

        # BEVFormer's own key: a *list* of (4, 4) arrays, one per camera, built
        # as ``K4 @ lidar2camera``.  ``reshape`` also accepts an already-stacked
        # (C, 4, 4) array and a lone (4, 4).
        mats = results.get('lidar2img')
        if mats is None or len(mats) == 0:
            return None
        return np.asarray(mats, dtype=np.float64).reshape(-1, 4, 4)

    def __call__(self, results):
        """Filter ``gt_bboxes_3d`` / ``gt_labels_3d`` by camera visibility.

        Args:
            results (dict): result dict from the loading pipeline.

        Returns:
            dict: ``results``, with ``gt_bboxes_3d``, ``gt_labels_3d`` and
                ``gt_names`` / ``gt_names_3d`` (whichever are present) reduced
                by the same mask.  Returned unmodified when the filter is
                disabled, when there are no boxes, or when no projection
                matrices are available.
        """
        if not self.enabled:
            return results

        gt_bboxes_3d = results.get('gt_bboxes_3d')
        if gt_bboxes_3d is None or len(gt_bboxes_3d) == 0:
            return results

        lidar2img = self._lidar2img(results)
        if lidar2img is None or len(lidar2img) == 0:
            return results

        # ``corners`` is (N, 8, 3) in the LiDAR frame, which for NAVSIM is the
        # ego frame (``lidar2ego`` is the identity in every frame): +x forward,
        # +y left, +z up, origin at the rear axle.  ``lidar2img`` consumes
        # exactly that frame, so no conversion is needed here.
        mask = boxes_in_view(
            gt_bboxes_3d.corners.numpy(),
            lidar2img,
            self.image_wh,
            min_corners=self.min_visible_corners,
            margin=self.margin,
        )

        if self.apply_within is not None:
            centre = gt_bboxes_3d.gravity_center.numpy()
            x_min, y_min, x_max, y_max = self.apply_within
            considered = ((centre[:, 0] >= x_min) & (centre[:, 0] <= x_max)
                          & (centre[:, 1] >= y_min) & (centre[:, 1] <= y_max))
            # Keep everything outside the region unconditionally.
            mask |= ~considered

        # ``BaseInstance3DBoxes.__getitem__`` indexes its internal tensor, so it
        # needs a torch bool tensor; the label/name arrays are numpy and take
        # the numpy mask.  Both are the *same* mask, so the two stay aligned --
        # mmdet3d's loss and assigner zip them positionally and a length
        # mismatch there surfaces only as a wrong number.
        results['gt_bboxes_3d'] = gt_bboxes_3d[torch.from_numpy(mask)]
        if results.get('gt_labels_3d') is not None:
            results['gt_labels_3d'] = self._index(results['gt_labels_3d'],
                                                  mask)
        # ``gt_names`` is what mmdet3d's ``get_ann_info`` calls it and what
        # ``CustomNavsimDataset`` passes through; ``gt_names_3d`` is BEVFusion's
        # spelling.  Filter whichever is present so a config that collects names
        # (e.g. for debugging or visualisation) does not end up with names and
        # boxes of different lengths.
        for key in ('gt_names', 'gt_names_3d'):
            if results.get(key) is not None:
                results[key] = np.asarray(results[key])[mask]
        return results

    @staticmethod
    def _index(array, mask):
        """Apply a numpy bool ``mask`` to a numpy array or a torch tensor."""
        if isinstance(array, torch.Tensor):
            return array[torch.from_numpy(mask)]
        return np.asarray(array)[mask]

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(min_visible_corners={}, '.format(
            self.min_visible_corners)
        repr_str += 'margin={}, '.format(self.margin)
        repr_str += 'enabled={}, '.format(self.enabled)
        repr_str += 'apply_within={}, '.format(
            None if self.apply_within is None else self.apply_within.tolist())
        repr_str += 'image_wh={}, '.format(self.image_wh)
        repr_str += 'camera_names={}, '.format(self.camera_names)
        repr_str += 'use_raw_cams={})'.format(self.use_raw_cams)
        return repr_str
