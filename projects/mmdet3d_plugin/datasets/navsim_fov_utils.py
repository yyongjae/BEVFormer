# Copied verbatim from the BEVFusion NAVSIM port:
#     <bevfusion>/mmdet3d/datasets/fov_utils.py
# Kept byte-identical on purpose so the two ports score NAVSIM the same way.
# Depends only on numpy (+ pyquaternion); no mmdet3d / mmcv / nuscenes-devkit.
# Do not "clean up" local duplication here without changing it in BEVFusion too.

"""Camera field-of-view tests for the NAVSIM front-camera rig.

NAVSIM ships a 360-degree merged point cloud, but an agent built on the three
front cameras (``CAM_F0`` / ``CAM_L0`` / ``CAM_R0``) only ever observes a wedge
of roughly +-87 degrees.  The rectangular front-only ``bev_roi`` therefore
still admits objects that no camera in the rig has seen, which a
camera-conditioned teacher cannot learn and should not be scored on.

Visibility is decided by **projection**, mirroring BEVFormer's
``point_sampling``: push the point through the full ``lidar -> image`` matrix,
require positive depth, and require the pixel to land inside the image.  A
fixed azimuth threshold is not equivalent -- the cameras sit off the LiDAR
origin, so the observed region is not a clean wedge from the ego frame. On a
fixed 150-frame navtrain sample, projection drops 5.2% of the boxes inside the
front-only ROI where an equivalent +-87 degree wedge drops only 1.6%.

The test is deliberately run on the **raw sensor geometry**, before
``ImageAug3D`` / ``GlobalRotScaleTrans``.  Those augmentations rotate the LiDAR
frame virtually and the model undoes them via ``lidar_aug_matrix`` when it
projects, so an object's visibility is a property of its original position, not
of the sampled augmentation.  Keeping the test pre-augmentation also makes the
label set deterministic across epochs.
"""

from typing import Sequence, Tuple

import numpy as np

EPS = 1e-5

__all__ = [
    "build_lidar2image",
    "image_size_from_intrinsics",
    "points_in_view",
    "boxes_in_view",
]


def build_lidar2image(cams: dict, camera_names: Sequence[str]) -> np.ndarray:
    """Return the ``(C, 4, 4)`` lidar -> image matrices for ``camera_names``.

    ``cams`` is the per-frame dict from a NAVSIM info pickle, whose
    ``sensor2lidar_*`` entries are camera -> LiDAR.
    """
    mats = []
    for name in camera_names:
        cam = cams[name]
        camera2lidar = np.eye(4, dtype=np.float64)
        camera2lidar[:3, :3] = cam["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = cam["sensor2lidar_translation"]
        intrinsics = np.eye(4, dtype=np.float64)
        intrinsics[:3, :3] = cam["cam_intrinsic"]
        mats.append(intrinsics @ np.linalg.inv(camera2lidar))
    return np.stack(mats) if mats else np.zeros((0, 4, 4))


def image_size_from_intrinsics(
    cams: dict, camera_names: Sequence[str]
) -> Tuple[float, float]:
    """Infer ``(width, height)`` from the principal point.

    Last resort only.  It assumes a centred principal point, which NAVSIM does
    not have vertically: ``cy`` is 560 on 1080-row images, so the height comes
    out 40 px too large and the test turns slightly permissive.  Prefer an
    explicit size (``camera_fov.image_wh``, or ``ori_shape`` when the pipeline
    has loaded an image).
    """
    for name in camera_names:
        K = np.asarray(cams[name]["cam_intrinsic"], dtype=np.float64)
        return float(2.0 * K[0, 2]), float(2.0 * K[1, 2])
    return 0.0, 0.0


def points_in_view(
    points: np.ndarray,
    lidar2image: np.ndarray,
    image_wh: Tuple[float, float],
    margin: float = 0.0,
) -> np.ndarray:
    """``(N,)`` mask of points that project into at least one image.

    Args:
        points: ``(N, 3)`` in the LiDAR frame.
        lidar2image: ``(C, 4, 4)``.
        image_wh: stored image ``(width, height)`` in pixels.
        margin: pixels of slack allowed outside the image border.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(lidar2image) == 0:
        return np.zeros(len(points), dtype=bool)

    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    # (C, 4, 4) @ (4, N) -> (C, 4, N)
    projected = lidar2image @ homogeneous.T
    depth = projected[:, 2, :]
    uv = projected[:, :2, :] / np.maximum(depth[:, None, :], EPS)

    width, height = image_wh
    visible = depth > EPS
    visible &= (uv[:, 0, :] >= -margin) & (uv[:, 0, :] < width + margin)
    visible &= (uv[:, 1, :] >= -margin) & (uv[:, 1, :] < height + margin)
    return visible.any(axis=0)


def boxes_in_view(
    corners: np.ndarray,
    lidar2image: np.ndarray,
    image_wh: Tuple[float, float],
    min_corners: int = 1,
    margin: float = 0.0,
) -> np.ndarray:
    """``(N,)`` mask of boxes with at least ``min_corners`` visible corners.

    Args:
        corners: ``(N, 8, 3)`` box corners in the LiDAR frame.
        min_corners: how many of the eight corners must land in an image.  1
            keeps every box the rig has partially seen, which is the right
            default for detection: a box clipped by the image border is still
            a legitimate target.
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 8, 3)
    if len(corners) == 0:
        return np.zeros(0, dtype=bool)
    flat = points_in_view(corners.reshape(-1, 3), lidar2image, image_wh, margin)
    return flat.reshape(len(corners), 8).sum(axis=1) >= max(1, int(min_corners))
