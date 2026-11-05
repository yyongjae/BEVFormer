"""NAVSIM (OpenScene / nuPlan) -> BEVFormer info converter.

Reads the raw NAVSIM log pickles (``navsim_logs/{split}/<log_name>.pkl``) and
emits a single info pickle consumable by
:class:`projects.mmdet3d_plugin.datasets.CustomNavsimDataset`.

This is a port of ``bevfusion/tools/data_converter/navsim_converter.py``.  The
box conversion, path resolution and scene-filter reading are kept byte-for-byte
so that the two ports see the *same* labels and the *same* images; everything
added here (``prev`` / ``next`` / ``can_bus`` / ``scene_token`` / pruning) exists
only because BEVFormer is a *temporal* detector and BEVFusion is not.

Input (one entry per frame inside each log pkl; a log pkl is a flat, time-ordered
``list`` of frame dicts at 2 Hz)::

    token, frame_idx, timestamp, log_name, log_token, scene_name, scene_token,
    map_location, roadblock_ids, traffic_lights, vehicle_name,
    driving_command (4,), ego_dynamic_state [vx, vy, ax, ay] (EGO frame),
    can_bus (18,) float64,
    ego2global_translation (3,), ego2global_rotation (4, wxyz),
    lidar2ego_translation (3, == 0), lidar2ego_rotation (4, == identity),
    ego2global / lidar2ego / lidar2global (4, 4),
    lidar_path (str, relative to the sensor-blob root),
    sample_prev / sample_next (str token or None),
    cams: {CAM_*: {data_path, sensor2lidar_rotation (3,3),
                   sensor2lidar_translation (3,), cam_intrinsic (3,3),
                   distortion (5,)}},
    anns: {gt_boxes (N,7), gt_names (N,), gt_velocity_3d (N,3),
           instance_tokens, track_tokens},
    occ_gt_final_path, flow_gt_final_path

Output::

    {
      'infos':    [<per-frame dict>, ...],   # sorted by (log_name, timestamp)
      'metadata': {'version': <split>, 'navsim_converter': <version string>,
                   'image_width': int, 'history_frames': int,
                   'keyframes_only': bool, ...},
    }

Per-frame dict (the complete schema; dtypes are what the dataset relies on)::

    token            str          NAVSIM frame token (unique across the dataset)
    prev             str | None   token of the previous kept frame, or None
    next             str | None   token of the next kept frame, or None
    scene_token      str          raw NAVSIM per-scene token (20 s / 40 frames)
    scene_name       str          e.g. 'log-0001-scene-0001'
    log_name         str          e.g. '2021.05.25.14.16.10_veh-35_00083_00485'
    frame_idx        int          the raw NAVSIM 'frame_idx': the index within
                                  the 40-frame SCENE (0..39).  Same semantics as
                                  nuScenes' key, and the value BEVFusion's
                                  converter also writes there.
    log_frame_idx    int          index of this frame within its LOG (0-based)
    timestamp        int          microseconds
    location         str          map location (alias: 'map_location')
    map_location     str          same string, under the raw NAVSIM key
    in_split_tokens  bool         frame is a reference frame of the official
                                  navtrain / navtest token list
    lidar_path       str | None   '<lidar-prefix>/<split>/<log>/MergedPointCloud/
                                  <token><lidar-ext>'
    sweeps           list[dict]   previous LiDAR frames, empty by default
                                  (--max-sweeps 0: BEVFormer is camera-only)
    lidar2ego_translation  list[3]  always [0, 0, 0]  (see below)
    lidar2ego_rotation     list[4]  always [1, 0, 0, 0]
    ego2global_translation list[3]  UTM-like; measured mean
                                  (5.0e5, 4.0e6, 2.6e2) m over the navtest
                                  logs, up to (6.6e5, 4.7e6, 6.3e2)
    ego2global_rotation    list[4]  quaternion, w x y z
    can_bus          float64 (18,)  nuScenes slot order (see below)
    cams             dict         {CAM_*: {data_path str,
                                   sensor2lidar_rotation float32 (3,3),
                                   sensor2lidar_translation float32 (3,),
                                   cam_intrinsic float32 (3,3)}}
    gt_boxes         float32 (N,7)  [x, y, z_center, dx, dy, dz, yaw]
    gt_names         '<U16'  (N,)   (longest class name is 14 chars)
    gt_velocity      float32 (N,2)  (vx, vy) in the ego frame, NaN -> 0
    num_lidar_pts    int32   (N,)   all -1 (see below)
    valid_flag       bool    (N,)   all True
    ego_dynamic_state float32 (4,)  [vx, vy, ax, ay], ego frame
    driving_command   float32 (4,)  one-hot route command

Dropped from the raw frame, deliberately: ``occ_gt_final_path`` and
``flow_gt_final_path`` (the paths do not resolve on any machine we have),
``distortion`` (a real accuracy knob -- k1 = -0.356 on CAM_F0, measured over
the 136 navtest logs: mean -0.35610, range -0.35612 .. -0.35269 -- but both this
port and the BEVFusion port use the pinhole model, so storing it would only
invite half-applied undistortion), ``traffic_lights`` and ``roadblock_ids``
(planning inputs, unused by a detector; ``roadblock_ids`` is read here only by
the scene filter), ``log_token``, ``vehicle_name`` and the redundant 4x4
``ego2global`` / ``lidar2ego`` / ``lidar2global`` matrices (the dataset rebuilds
them from the translation + quaternion, as the nuScenes path does).
``sample_prev`` / ``sample_next`` are also dropped: they index the *raw* log and
are superseded by ``prev`` / ``next``, which index the *kept* frames.

Coordinate conventions
----------------------
``lidar2ego`` is the identity in every NAVSIM frame, so the ego frame **is** the
LiDAR frame: +x forward, +y left, +z up, z = 0 the ground, origin the rear axle.
That differs from nuScenes, whose ``LIDAR_TOP`` frame is rotated ~-90 deg (there
+y_lidar is ego-forward).  Consequences that matter elsewhere in the port:
BEVFormer's ego-motion shift must be told which convention it is in
(``PerceptionTransformer(ego_axes='x_forward_y_left')``), and the BEV grid is
front-only (``point_cloud_range = [0, -32, -3, 64, 32, 5]``).

Box convention
--------------
NAVSIM ``gt_boxes`` are ``[x, y, z_center, length, width, height, heading]`` in
the ego (== LiDAR) frame.  They are rewritten to the mmdet3d LiDAR box
convention, identical to the nuScenes converter and to the BEVFusion port::

    [x, y, z_center, width, length, height, -heading - pi/2]
     ^-- note the length/width swap: mmdet3d dims are (dx, dy, dz)

``heading`` is unwrapped in the raw data (it reaches +-2*pi); the stored ``yaw``
is wrapped to ``[-pi, pi)``.  ``z`` is the box *centre*, hence
``origin=(0.5, 0.5, 0.5)`` when the dataset builds ``LiDARInstance3DBoxes``.

``num_lidar_pts`` is -1 for every box: NAVSIM ships no per-box point count.
A dataset that filters on ``num_lidar_pts > 0`` would therefore empty every
frame -- ``valid_flag`` (all True) is the field to filter on.

can_bus
-------
NAVSIM already ships an 18-element ``can_bus`` per frame, but **two slots are
transposed relative to nuScenes**.  Verified numerically on the raw logs
(``[0:3] == ego2global_translation``, ``[3:7] == ego2global_rotation``,
``[7],[8] == ego_dynamic_state[2],[3]``, ``[10],[11] == ego_dynamic_state[0],[1]``)::

    NAVSIM   [0:3] pos  [3:7] quat(wxyz)  [7:10] accel  [10:13] VELOCITY       [13:16] ROTATION_RATE  [16:18] 0,0
    nuScenes [0:3] pos  [3:7] quat(wxyz)  [7:10] accel  [10:13] ROTATION_RATE  [13:16] VELOCITY       [16:18] 0,0

The swap is done **once, here**, so that every BEVFormer comment, and any
pretrained ``can_bus_mlp``, finds in each slot what it expects.  Nothing
downstream may reorder it again.

The array stays **float64**: slots ``[0:3]`` hold a UTM-like translation (~4e6),
where the float32 ULP is 0.25 m, and the dataset *differences* consecutive
translations to recover the ego step -- which is **2.536 m** on average over the
12,140 linked navtest pairs and **2.321 m** over the 103,206 navtrain ones
(max 7.407 / 8.378 m).  In float32 that differencing can throw away a quarter
metre of it, i.e. **~10%** of a mean step.

Temporal linking
----------------
``prev`` / ``next`` are recomputed **after** pruning and after any ``--check-files``
drop, per log, and are ``None`` across:

  * a pruning hole (the neighbour was not kept),
  * a log boundary (linking never crosses logs),
  * a ``dt`` outside ``500000 +- 250000 us`` (9.6% of the 136 navtest logs and
    19.6% of the 1,192 navtrain logs contain at least one ``dt = 1000000 us``
    hole where NAVSIM itself reports no neighbour; navtrain also has three
    ``dt = 1500000 us`` two-frame holes),
  * a disagreement with NAVSIM's own ``sample_prev`` / ``sample_next``.

The dataset re-validates the same predicates when it builds its ``prev_index``,
so a stale pkl cannot silently produce a bogus history queue.

Which frames are kept
---------------------
The official ``navtrain`` / ``navtest`` token lists are a *subset* of each log's
frames (every 40-frame scene contributes a window of reference frames).  Keeping
only those costs history: with ``queue_length=4`` only ~56.4% of navtest samples
(~54.4% of navtrain) then have three real history frames at k*0.5 s, against
**99.8765% / 99.7638%** with the default ``--history-frames 4``.  Flagging every
frame instead inflates the sample set **5.8834x** on navtest (71,460 raw frames
in the 136 selected logs against 12,146 official tokens) and **6.4827x** on
navtrain (669,588 against 103,288).  So this converter keeps both notions
apart:

  * it emits frames *around* the split as well, each flagged ``in_split_tokens``;
  * ``--history-frames N`` keeps a frame if it is in the split **or** within N
    frames before an in-split frame of the same log (``0`` = in-split only,
    ``-1`` = every frame of the selected logs);
  * the dataset uses the full frame list as its *history pool* and
    ``in_split_tokens`` as its *sample list*.

Usage
-----
    python tools/data_converter/navsim_converter.py \
        --navsim-logs  /data/datasets/navsim/navsim_logs/trainval \
        --sensor-root  /data/datasets/navsim/sensor_blobs \
        --split        trainval \
        --scene-filter .../train_test_split/scene_filter/navtrain.yaml \
        --history-frames 4 --workers 16 \
        --out          data/infos/navsim_infos_train.pkl
"""

import argparse
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os import path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

CONVERTER_VERSION = "bevformer-navsim-1.0"

# The seven nuPlan class names are at most 14 characters ("generic_object"),
# so U16 keeps the same safety margin as a wider dtype at half the bytes.  It
# matters: numpy fixed-width strings are padded to 4 bytes per character, so
# gt_names at U16 is 64 B x 12,962,377 boxes = 830 MB of the 1,888 MB navtrain
# pkl (44%) and 68 MB of the 178 MB navtest pkl (38%).  U32 would double both,
# costing another 830 MB on navtrain for nothing.
GT_NAME_DTYPE = "<U16"

ORIGINAL_IMAGE_WIDTH = 1920
ORIGINAL_IMAGE_HEIGHT = 1080

# NAVSIM runs at 2 Hz.  Measured over every raw log of both splits, the
# inter-frame dt is 500000 us with at most 1724 us of jitter on the 136 navtest
# logs and 8186 us on the 1,192 navtrain logs.  The only other values seen are
# the 1000000 us one-frame hole -- 26 of 71,324 navtest deltas (9.6% of its
# logs), 656 of 668,396 navtrain deltas (19.6% of its logs) -- and three
# 1500000 us two-frame holes in navtrain.  A 250000 us window therefore accepts
# the worst jitter with a 30x margin and rejects the nearest hole (999047 us)
# by 249047 us.  The dataset uses the same two constants -- keep them in sync.
FRAME_INTERVAL_US = 500000
FRAME_INTERVAL_TOL_US = 250000

CAN_BUS_LEN = 18

NAVSIM_CAMERAS = [
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
]

# The official token list, published to the worker processes once at pool
# start-up instead of riding along in every job tuple.  navtrain ships 103,288
# tokens, which pickle to 1.96 MB; one copy per job, over the 1,192 jobs its
# scene filter selects, would move 2.34 GB through the queue for data that
# never changes.  (1,310 is the number of pkls in the raw trainval directory,
# not the number of jobs -- the scene filter selects 1,192 of them.)
_SPLIT_TOKENS = None  # type: Optional[set]


def _init_worker(official_tokens):
    """Pool initializer: publish the official token set to this process."""
    global _SPLIT_TOKENS
    _SPLIT_TOKENS = official_tokens


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _candidate_rel_paths(raw_path: str, split: str) -> List[str]:
    """Candidate paths relative to the sensor-blob root.

    Different NAVSIM releases store ``data_path`` either with or without the
    leading split directory, and sometimes as an absolute path.
    """
    raw = str(raw_path).replace("\\", "/").lstrip("/")
    cands = [raw]
    if not raw.startswith(split + "/"):
        cands.append("{}/{}".format(split, raw))
    # Absolute paths from another machine: keep the tail after 'sensor_blobs/'.
    if "sensor_blobs/" in raw:
        tail = raw.split("sensor_blobs/")[-1]
        cands.append(tail)
        if not tail.startswith(split + "/"):
            cands.append("{}/{}".format(split, tail))
    # De-duplicate, preserving order.
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_rel_path(raw_path: str, sensor_root: str, split: str) -> Optional[str]:
    """Return the sensor-root-relative path that actually exists on disk."""
    for cand in _candidate_rel_paths(raw_path, split):
        if osp.exists(osp.join(sensor_root, cand)):
            return cand
    return None


def _lidar_rel_path(raw_path: str, split: str) -> str:
    """Sensor-root-relative LiDAR path, without touching the disk.

    The raw ``.pcd`` blobs are converted offline to ``.bin``, so this path is
    not expected to exist yet and must not be probed.
    """
    rel = str(raw_path).replace("\\", "/").lstrip("/")
    if "sensor_blobs/" in rel:
        rel = rel.split("sensor_blobs/")[-1]
    if not rel.startswith(split + "/"):
        rel = "{}/{}".format(split, rel)
    return rel


# --------------------------------------------------------------------------- #
# Scene filtering (mirrors navsim.common.dataloader.filter_scenes)
# --------------------------------------------------------------------------- #
def compute_split_tokens(
    frames: List[Dict[str, Any]],
    num_history_frames: int = 4,
    num_future_frames: int = 10,
    frame_interval: int = 1,
    has_route: bool = True,
) -> set:
    """Tokens that are the reference frame of a structurally valid NAVSIM scene.

    FALLBACK ONLY, used when no ``--scene-filter`` is given.  It reproduces
    NAVSIM's *structural* scene enumeration -- enough history, enough future, a
    route -- but **not** the official navtrain / navtest token lists, which are
    a curated subset that cannot be recomputed from the logs.  Measured on the
    136 navtest logs, this returns **69,405** tokens against the official
    **12,146** -- a **5.71x** SUPERSET -- so an info file built this way flags
    ~5.7x too many frames as ``in_split_tokens`` and the evaluator then scores
    ~5.7x too many frames, dividing reported recall accordingly.

    Always pass ``--scene-filter`` with a yaml that carries a ``tokens:`` list.
    """
    num_frames = num_history_frames + num_future_frames
    tokens = set()
    for start in range(0, len(frames), frame_interval):
        chunk = frames[start: start + num_frames]
        if len(chunk) < num_frames:
            continue
        ref = chunk[num_history_frames - 1]
        if has_route and len(ref.get("roadblock_ids", [])) == 0:
            continue
        tokens.add(ref["token"])
    return tokens


def load_scene_filter(scene_filter_yaml: Optional[str]):
    """Return ``(log_names, tokens)`` from a navsim scene_filter yaml.

    The official navtrain / navtest filters ship an explicit ``tokens`` list
    (103,288 and 12,146 entries respectively).  When it is present it defines
    the split exactly, and :func:`compute_split_tokens` is not needed.
    """
    if scene_filter_yaml is None:
        return None, None
    import yaml

    with open(scene_filter_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("log_names")
    tokens = cfg.get("tokens")
    return (set(names) if names else None), (set(tokens) if tokens else None)


# --------------------------------------------------------------------------- #
# Per-frame conversion
# --------------------------------------------------------------------------- #
def convert_boxes(anns: Dict[str, Any]):
    """NAVSIM annotations -> (gt_boxes, gt_names, gt_velocity)."""
    boxes = np.asarray(anns["gt_boxes"], dtype=np.float32).reshape(-1, 7)
    names = np.asarray(anns["gt_names"]).reshape(-1)

    if len(boxes) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            names.astype(GT_NAME_DTYPE),
            np.zeros((0, 2), dtype=np.float32),
        )

    xyz = boxes[:, 0:3]
    length, width, height = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    heading = boxes[:, 6]

    # mmdet3d LiDAR boxes: dims = (dx, dy, dz) = (width, length, height)
    # with yaw = -heading - pi/2  (same convention as the nuScenes converter).
    dims = np.stack([width, length, height], axis=-1)
    # NAVSIM headings are not wrapped (they reach +-2*pi); normalise to
    # [-pi, pi) so the stored boxes stay readable and comparable.
    yaw = -heading - np.pi / 2
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    gt_boxes = np.concatenate([xyz, dims, yaw[:, None]], axis=-1).astype(np.float32)

    velocity = np.asarray(anns.get("gt_velocity_3d"), dtype=np.float32)
    if velocity.size == 0:
        gt_velocity = np.zeros((len(boxes), 2), dtype=np.float32)
    else:
        gt_velocity = np.nan_to_num(velocity.reshape(-1, 3)[:, :2], nan=0.0)

    return gt_boxes, names.astype(GT_NAME_DTYPE), gt_velocity.astype(np.float32)


def build_can_bus(frame: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """NAVSIM ``can_bus`` -> nuScenes slot order.  Returns (array, synthesised).

    The only transformation is the slot swap documented in the module
    docstring::

        out[10:13] = navsim[13:16]   # rotation rate  (nuScenes slot)
        out[13:16] = navsim[10:13]   # velocity       (nuScenes slot)

    Slots ``[0:3]``, ``[3:7]``, ``[16]`` and ``[17]`` are rewritten by the
    dataset's ``get_data_info`` on every access (position, quaternion, yaw in
    radians, yaw in degrees), exactly as ``CustomNuScenesDataset`` does, so the
    values stored here only have to be *well-formed*.  Slots ``[7:16]`` are
    what the ``can_bus_mlp`` actually consumes from the pkl.

    Kept float64: see the module docstring on UTM-scale translations.
    """
    can_bus = np.zeros(CAN_BUS_LEN, dtype=np.float64)
    raw = frame.get("can_bus")
    raw = np.asarray(raw, dtype=np.float64).reshape(-1) if raw is not None \
        else np.zeros(0, dtype=np.float64)

    if raw.size == CAN_BUS_LEN:
        can_bus[:] = raw
        navsim_velocity = can_bus[10:13].copy()
        navsim_rotation_rate = can_bus[13:16].copy()
        can_bus[10:13] = navsim_rotation_rate
        can_bus[13:16] = navsim_velocity
        return can_bus, False

    # No (or a differently sized) can_bus in this release: synthesise the slots
    # BEVFormer reads so the pkl stays usable, and let the caller warn.  The
    # rotation-rate slots stay zero because NAVSIM's ego_dynamic_state has no
    # yaw rate; the ego-motion shift uses [0:3] and [16]/[17], which the dataset
    # fills from the pose, so temporal fusion is unaffected.
    can_bus[: min(raw.size, CAN_BUS_LEN)] = raw[:CAN_BUS_LEN]
    state = np.asarray(frame.get("ego_dynamic_state", [0.0] * 4), dtype=np.float64)
    can_bus[0:3] = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    can_bus[3:7] = np.asarray(frame["ego2global_rotation"], dtype=np.float64)
    if state.size >= 4:
        can_bus[7:9] = state[2:4]     # accel  (x, y), ego frame
        can_bus[13:15] = state[0:2]   # velocity (x, y), nuScenes slot
    can_bus[16:18] = 0.0
    return can_bus, True


def ego_pose(frame: Dict[str, Any]):
    """(R, t) of the frame's ego (== LiDAR) pose in global coordinates."""
    from pyquaternion import Quaternion

    R = Quaternion(np.asarray(frame["ego2global_rotation"], dtype=np.float64)).rotation_matrix
    t = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    return R, t


def build_sweeps(
    frames: List[Dict[str, Any]],
    index: int,
    max_sweeps: int,
    split: str,
    lidar_prefix: str,
    lidar_ext: str,
) -> List[Dict[str, Any]]:
    """Previous LiDAR frames expressed in the current frame, nuScenes-style.

    ``--max-sweeps`` defaults to **0** here, so this normally returns ``[]``:
    BEVFormer is camera-only and its pipeline never loads points.  The code is
    kept so that one info pkl can serve a LiDAR-fusion experiment later.

    ``LoadPointsFromMultiSweeps`` expects each sweep to carry a
    ``sensor2lidar_rotation`` / ``sensor2lidar_translation`` that maps its
    points into the *current* keyframe, plus a timestamp for the 5th point
    channel.  nuScenes fills this from the 20 Hz non-keyframe LiDAR records;
    NAVSIM has no intermediate LiDAR, so a "sweep" is the previous 2 Hz frame
    of the same log.  The transform is built the same way either way::

        p_cur = R_cur^T R_prev p_prev + R_cur^T (t_prev - t_cur)

    NOTE the temporal scale: 10 nuScenes sweeps span 0.5 s, 10 NAVSIM frames
    span 5 s (median 2.6 m of ego motion each). Accumulating that far smears
    moving objects; see docs/NAVSIM.md for the measured coverage per sweep.
    """
    if max_sweeps <= 0:
        return []

    R_cur, t_cur = ego_pose(frames[index])
    R_cur_T = R_cur.T
    sweeps = []
    for k in range(1, max_sweeps + 1):
        j = index - k
        if j < 0:
            break
        prev = frames[j]
        if not prev.get("lidar_path"):
            break
        R_prev, t_prev = ego_pose(prev)

        rel = _lidar_rel_path(prev["lidar_path"], split)
        sweeps.append(dict(
            data_path=osp.join(lidar_prefix, osp.splitext(rel)[0] + lidar_ext),
            timestamp=int(prev["timestamp"]),
            sensor2lidar_rotation=(R_cur_T @ R_prev).astype(np.float32),
            sensor2lidar_translation=(R_cur_T @ (t_prev - t_cur)).astype(np.float32),
        ))
    return sweeps


def convert_frame(
    frame: Dict[str, Any],
    log_name: str,
    log_frame_idx: int,
    split: str,
    sensor_root: str,
    intrinsic_scale: float,
    sensor_prefix: str,
    lidar_prefix: str,
    lidar_ext: str,
    camera_names: List[str],
    check_files: bool,
    in_split: bool,
    sweeps: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """One raw NAVSIM frame -> one info dict.  Returns (info, can_bus_synthesised).

    ``info`` is ``None`` when a requested camera is missing from the frame, or
    when ``check_files`` is set and its image is not on disk.
    """
    cams = {}  # type: Dict[str, Dict[str, Any]]
    for cam_name in camera_names:
        cam = frame["cams"].get(cam_name)
        if cam is None:
            return None, False

        rel = resolve_rel_path(cam["data_path"], sensor_root, split)
        if rel is None:
            if check_files:
                return None, False
            # Fall back to the recorded path so the pkl stays usable.
            rel = _candidate_rel_paths(cam["data_path"], split)[-1]

        # Intrinsics are expressed for the 1920x1080 originals.  Scaling rows
        # 0-1 (fx, skew, cx / fy, cy) by width/1920 retargets them at whatever
        # size the blobs were exported at.  Row 2 must not be touched.
        # Never infer the image *height* from this matrix: NAVSIM does not
        # centre the principal point vertically (cy = 560 on 1080 rows).
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float32).copy()
        K[:2, :] *= intrinsic_scale

        cams[cam_name] = dict(
            data_path=osp.join(sensor_prefix, rel),
            sensor2lidar_rotation=np.asarray(
                cam["sensor2lidar_rotation"], dtype=np.float32
            ),
            sensor2lidar_translation=np.asarray(
                cam["sensor2lidar_translation"], dtype=np.float32
            ),
            cam_intrinsic=K,
        )

    # LiDAR: point at the converted .bin, mirroring the raw pcd layout.
    lidar_path = None
    raw_lidar = frame.get("lidar_path")
    if raw_lidar:
        rel = _lidar_rel_path(raw_lidar, split)
        lidar_path = osp.join(lidar_prefix, osp.splitext(rel)[0] + lidar_ext)

    gt_boxes, gt_names, gt_velocity = convert_boxes(frame["anns"])
    can_bus, can_bus_synthesised = build_can_bus(frame)

    info = dict(
        token=frame["token"],
        # Filled in by link_prev_next() once the kept set of this log is known.
        prev=None,
        next=None,
        # The raw per-scene token.  The dataset does NOT feed this to the
        # detector: it derives its own 'scene_token' from the prev chain so
        # that prev_bev is reset at in-log gaps too (see CustomNavsimDataset).
        scene_token=frame.get("scene_token", ""),
        scene_name=frame.get("scene_name", ""),
        log_name=log_name,
        # 'frame_idx' is the raw NAVSIM value, which restarts at every 40-frame
        # scene boundary (verified: 0..39).  That is deliberately the same
        # semantics nuScenes gives the key -- an index within the 20 s scene --
        # and the same value BEVFusion's NAVSIM converter passes through, so the
        # two info pickles stay comparable.  'log_frame_idx' is the position
        # within the whole log, which is what the prev chain walks.  Neither is
        # load-bearing: BEVFormer's temporal bookkeeping keys off scene_token
        # and the prev chain only.
        frame_idx=int(frame.get("frame_idx", 0)),
        log_frame_idx=int(log_frame_idx),
        timestamp=int(frame["timestamp"]),
        location=frame.get("map_location", frame.get("location", "")),
        map_location=frame.get("map_location", frame.get("location", "")),
        in_split_tokens=bool(in_split),
        lidar_path=lidar_path,
        sweeps=sweeps,
        # NAVSIM annotations live in the ego frame and lidar2ego is the identity.
        lidar2ego_translation=[0.0, 0.0, 0.0],
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        ego2global_translation=np.asarray(
            frame["ego2global_translation"], dtype=np.float64
        ).tolist(),
        ego2global_rotation=np.asarray(
            frame["ego2global_rotation"], dtype=np.float64
        ).tolist(),
        can_bus=can_bus,
        cams=cams,
        gt_boxes=gt_boxes,
        gt_names=gt_names,
        gt_velocity=gt_velocity,
        # NAVSIM ships no per-box point count: -1 means "unknown", and the
        # dataset must filter on valid_flag instead of num_lidar_pts > 0.
        num_lidar_pts=np.full(len(gt_boxes), -1, dtype=np.int32),
        valid_flag=np.ones(len(gt_boxes), dtype=bool),
        ego_dynamic_state=np.asarray(
            frame.get("ego_dynamic_state", [0.0] * 4), dtype=np.float32
        ),
        driving_command=np.asarray(
            frame.get("driving_command", [0] * 4), dtype=np.float32
        ),
    )
    return info, can_bus_synthesised


# --------------------------------------------------------------------------- #
# Pruning and temporal linking (both per-log, hence inside the worker)
# --------------------------------------------------------------------------- #
def select_frame_indices(in_split: List[bool], history_frames: int) -> List[int]:
    """Raw indices of the frames to keep from one log.

    Args:
        in_split (list[bool]): per raw frame, whether it is an official
            navtrain / navtest reference frame.
        history_frames (int): ``-1`` keeps every frame of the log, ``0`` keeps
            only the in-split frames, ``N > 0`` additionally keeps the N frames
            immediately preceding each in-split frame -- the history pool the
            dataset's queue walks back into.

    The window is measured in raw frame positions, not in time; a ``dt`` hole
    inside the window is harmless because :func:`link_prev_next` refuses to
    link across it, so the queue simply stops there.
    """
    n = len(in_split)
    if history_frames < 0:
        return list(range(n))
    if history_frames == 0:
        return [i for i in range(n) if in_split[i]]

    keep = [False] * n
    next_in_split = None
    for i in range(n - 1, -1, -1):
        if in_split[i]:
            next_in_split = i
            keep[i] = True
        elif next_in_split is not None and next_in_split - i <= history_frames:
            keep[i] = True
    return [i for i in range(n) if keep[i]]


def link_prev_next(
    infos: List[Dict[str, Any]],
    raw_indices: List[int],
    frames: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Rebuild ``prev`` / ``next`` on one log's surviving frames.

    Must run *after* pruning and after any ``--check-files`` drop: a link is
    only created between two frames that both made it into the pkl, so the
    dataset can resolve every non-``None`` token.  A link is refused when

      * the two kept frames are not adjacent in the raw log (a pruning hole),
      * ``dt`` is outside ``FRAME_INTERVAL_US +- FRAME_INTERVAL_TOL_US``,
      * NAVSIM's own ``sample_prev`` / ``sample_next`` name a different frame.

    Log boundaries need no test: this function only ever sees one log, so the
    first frame's ``prev`` and the last frame's ``next`` stay ``None``.

    Returns a breakdown so a collapse of the temporal chain is *loud* in the
    converter's summary rather than silent in the training curve.
    """
    stats = Counter()
    for info in infos:
        info["prev"] = None
        info["next"] = None

    for k in range(len(infos) - 1):
        ia, ib = raw_indices[k], raw_indices[k + 1]
        if ib != ia + 1:
            stats["break_hole"] += 1
            continue
        dt = int(frames[ib]["timestamp"]) - int(frames[ia]["timestamp"])
        if abs(dt - FRAME_INTERVAL_US) >= FRAME_INTERVAL_TOL_US:
            stats["break_dt"] += 1
            continue
        # NAVSIM states adjacency itself; a None there means "no neighbour in
        # this log", which the dt test above has already caught in every case
        # measured.  Only an explicit disagreement breaks the link, so a
        # release that drops the fields degrades to the dt test rather than to
        # an empty chain.
        prev_token = frames[ib].get("sample_prev")
        next_token = frames[ia].get("sample_next")
        if (prev_token is not None and prev_token != frames[ia]["token"]) or \
                (next_token is not None and next_token != frames[ib]["token"]):
            stats["break_chain"] += 1
            continue

        infos[k]["next"] = infos[k + 1]["token"]
        infos[k + 1]["prev"] = infos[k]["token"]
        stats["linked"] += 1

    return dict(stats)


def convert_log(job: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one log pkl.  Runs in a worker process.

    ``job`` is a dict rather than a positional tuple: this function takes 12
    parameters and a mis-ordered unpack would be a silent, data-corrupting bug.
    """
    log_path = job["log_path"]
    split = job["split"]
    log_name = osp.basename(log_path).replace(".pkl", "")
    try:
        with open(log_path, "rb") as f:
            frames = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        return dict(log_name=log_name, infos=[], warning="failed to read: {}".format(exc),
                    stats={})

    # The official token list defines the split exactly when the scene-filter
    # yaml ships one; otherwise reproduce it from the log's own structure.
    if _SPLIT_TOKENS is not None:
        split_tokens = _SPLIT_TOKENS
    else:
        split_tokens = compute_split_tokens(frames, **job["filter_kwargs"])

    in_split = [frame["token"] in split_tokens for frame in frames]
    keep_indices = select_frame_indices(in_split, job["history_frames"])

    infos, kept_raw_indices = [], []
    stats = Counter()
    stats["raw_frames"] = len(frames)
    stats["raw_in_split"] = sum(in_split)
    for idx in keep_indices:
        sweeps = build_sweeps(
            frames, idx, job["max_sweeps"], split, job["lidar_prefix"],
            job["lidar_ext"],
        )
        info, can_bus_synthesised = convert_frame(
            frames[idx], log_name, idx, split, job["sensor_root"],
            job["intrinsic_scale"], job["sensor_prefix"], job["lidar_prefix"],
            job["lidar_ext"], job["camera_names"], job["check_files"],
            in_split[idx], sweeps,
        )
        if info is None:
            stats["dropped"] += 1
            continue
        if can_bus_synthesised:
            stats["can_bus_synthesised"] += 1
        infos.append(info)
        kept_raw_indices.append(idx)

    # Pruning and linking are per-log, so they parallelise with the conversion;
    # only the final sort and the global token map have to be serial.
    stats.update(link_prev_next(infos, kept_raw_indices, frames))

    msgs = []
    if stats["dropped"]:
        msgs.append("dropped {} frames (missing images)".format(stats["dropped"]))
    if stats["can_bus_synthesised"]:
        msgs.append("synthesised can_bus for {} frames".format(
            stats["can_bus_synthesised"]))
    return dict(log_name=log_name, infos=infos,
                warning="; ".join(msgs) if msgs else None, stats=dict(stats))


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="NAVSIM -> BEVFormer info converter")
    p.add_argument("--navsim-logs", required=True,
                   help="directory of raw NAVSIM log pickles (<log_name>.pkl)")
    p.add_argument("--sensor-root", required=True,
                   help="root of the sensor blobs (contains trainval/ and test/)")
    p.add_argument("--split", required=True, choices=["trainval", "test", "mini"])
    p.add_argument("--out", required=True, help="output .pkl path")
    p.add_argument("--image-width", type=int, default=1920,
                   help="width of the stored images; intrinsics are scaled by "
                        "image-width / 1920 (default: 1920, the native size)")
    p.add_argument("--scene-filter", default=None,
                   help="navsim scene_filter yaml; its log_names list selects "
                        "the logs and its tokens list defines in_split_tokens")
    p.add_argument("--sensor-prefix", default="sensor_blobs",
                   help="prefix written into data_path, relative to data_root")
    # BEVFormer is camera-only, so lidar_path is informational.  The defaults
    # name the clouds NAVSIM actually ships -- the merged .pcd files inside the
    # sensor blobs -- so `data/navsim/lidar -> <sensor blobs>` makes the path
    # resolve with no conversion step.  Pass `--lidar-prefix lidar_bin
    # --lidar-ext .bin` if you have run BEVFusion's navsim_lidar_to_bin.py.
    p.add_argument("--lidar-prefix", default="lidar",
                   help="prefix written into lidar_path, relative to data_root")
    p.add_argument("--lidar-ext", default=".pcd",
                   help="extension of the point clouds (default: .pcd, as "
                        "NAVSIM ships them)")
    p.add_argument("--cameras", nargs="+", default=NAVSIM_CAMERAS,
                   help="cameras to record in the info file; the dataset picks "
                        "its own subset (camera_names) out of these")
    p.add_argument("--history-frames", type=int, default=4,
                   help="keep frames that are not in the split but sit within N "
                        "frames before an in-split frame of the same log -- the "
                        "history pool BEVFormer's queue walks back into. "
                        "0 = in-split frames only, -1 = every frame "
                        "(default: 4, one more than queue_length-1 = 3)")
    p.add_argument("--keyframes-only", action="store_true",
                   help="shorthand for --history-frames 0 (no history pool); "
                        "halves the pkl but leaves only ~56.4%% of navtest "
                        "samples (~54.4%% of navtrain) with three real "
                        "history frames, against 99.88%% / 99.76%% at the "
                        "default --history-frames 4")
    p.add_argument("--check-files", action="store_true",
                   help="drop frames whose images are missing on disk")
    p.add_argument("--max-sweeps", type=int, default=0,
                   help="previous LiDAR frames recorded per sample. 0 by "
                        "default: BEVFormer is camera-only and never loads "
                        "points (default: 0)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-logs", type=int, default=None,
                   help="debug / staged testing: only convert the first N logs")
    # Scene-filter parameters (defaults match navtrain/navtest).
    p.add_argument("--num-history-frames", type=int, default=4)
    p.add_argument("--num-future-frames", type=int, default=10)
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--no-route-filter", action="store_true")
    return p.parse_args()


def _queue_depth_histogram(infos, token2idx, max_depth=3):
    """How many real history frames each in-split sample can reach.

    This is the number the keyframes-only / all-frames trade-off turns on: a
    sample with fewer than ``queue_length - 1`` reachable predecessors has its
    queue padded by repetition, which contributes no new BEV evidence.
    """
    hist = Counter()
    for info in infos:
        if not info["in_split_tokens"]:
            continue
        depth, token = 0, info["prev"]
        while token is not None and depth < max_depth:
            idx = token2idx.get(token)
            if idx is None:
                break
            depth += 1
            token = infos[idx]["prev"]
        hist[depth] += 1
    return hist


def main():
    args = parse_args()

    history_frames = 0 if args.keyframes_only else args.history_frames

    log_files = sorted(
        osp.join(args.navsim_logs, f)
        for f in os.listdir(args.navsim_logs)
        if f.endswith(".pkl")
    )
    keep, official_tokens = load_scene_filter(args.scene_filter)
    if keep is not None:
        log_files = [f for f in log_files
                     if osp.basename(f).replace(".pkl", "") in keep]
    if official_tokens is not None:
        print("[navsim_converter] using the official token list "
              "({} tokens) from {}".format(len(official_tokens), args.scene_filter))
    else:
        print("[warn] no 'tokens:' list in the scene filter -- in_split_tokens "
              "will be computed structurally by compute_split_tokens(), which "
              "is a ~5.7x SUPERSET of the official navtrain/navtest split "
              "(measured 69,405 tokens against navtest's official 12,146). "
              "The evaluator would then score ~5.7x too many frames and "
              "divide reported recall accordingly. Pass a scene-filter yaml "
              "that carries a tokens: list.")
    if args.max_logs:
        log_files = log_files[: args.max_logs]

    intrinsic_scale = args.image_width / ORIGINAL_IMAGE_WIDTH
    print("[navsim_converter] {} logs, split={}".format(len(log_files), args.split))
    print("[navsim_converter] intrinsic scale = {:.6f} ({} -> {})".format(
        intrinsic_scale, ORIGINAL_IMAGE_WIDTH, args.image_width))
    print("[navsim_converter] history_frames = {} ({})".format(
        history_frames,
        "every frame" if history_frames < 0 else
        "in-split frames only" if history_frames == 0 else
        "in-split frames + {} preceding".format(history_frames)))

    filter_kwargs = dict(
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        frame_interval=args.frame_interval,
        has_route=not args.no_route_filter,
    )
    jobs = [
        dict(log_path=f, split=args.split, sensor_root=args.sensor_root,
             intrinsic_scale=intrinsic_scale, sensor_prefix=args.sensor_prefix,
             lidar_prefix=args.lidar_prefix, lidar_ext=args.lidar_ext,
             camera_names=args.cameras, check_files=args.check_files,
             history_frames=history_frames, filter_kwargs=filter_kwargs,
             max_sweeps=args.max_sweeps)
        for f in log_files
    ]

    all_infos, warnings = [], []
    totals = Counter()

    def _collect(result):
        all_infos.extend(result["infos"])
        totals.update(result["stats"])
        if result["warning"]:
            warnings.append("{}: {}".format(result["log_name"], result["warning"]))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(official_tokens,)) as pool:
            for result in tqdm(pool.map(convert_log, jobs), total=len(jobs),
                               desc="logs"):
                _collect(result)
    else:
        _init_worker(official_tokens)
        for job in tqdm(jobs, desc="logs"):
            _collect(convert_log(job))

    for w in warnings[:20]:
        print("[warn] {}".format(w))
    if len(warnings) > 20:
        print("[warn] ... and {} more".format(len(warnings) - 20))

    # ----- serial from here: the global ordering and the global token map ----
    # Sorting by (log_name, timestamp) gives the dataset a deterministic,
    # temporally monotone frame list, which is what its prev_index / segment
    # construction and the 'frames are contiguous within a log' assumption of
    # every sampler rest on.
    all_infos.sort(key=lambda i: (i["log_name"], i["timestamp"]))
    token2idx = {info["token"]: idx for idx, info in enumerate(all_infos)}
    if len(token2idx) != len(all_infos):
        print("[warn] {} duplicate tokens in the output".format(
            len(all_infos) - len(token2idx)))

    # prev/next are built per log, so they should already point at kept frames;
    # verify globally rather than trust it, and null out anything dangling so
    # the dataset can never KeyError on a stale pkl.
    dangling = 0
    for info in all_infos:
        for key in ("prev", "next"):
            if info[key] is not None and info[key] not in token2idx:
                info[key] = None
                dangling += 1
    if dangling:
        print("[warn] nulled {} dangling prev/next links".format(dangling))

    n_split = sum(1 for i in all_infos if i["in_split_tokens"])
    n_boxes = sum(len(i["gt_boxes"]) for i in all_infos)
    n_prev = sum(1 for i in all_infos if i["prev"] is not None)
    n_next = sum(1 for i in all_infos if i["next"] is not None)
    n_sweeps = float(np.mean([len(i["sweeps"]) for i in all_infos])) if all_infos else 0.0
    n_logs = len(set(i["log_name"] for i in all_infos))

    print("[navsim_converter] {} frames from {} logs "
          "({} flagged in_split_tokens, {} boxes)".format(
              len(all_infos), n_logs, n_split, n_boxes))
    if totals.get("dropped"):
        print("[navsim_converter] dropped {} frames (missing images)".format(
            totals["dropped"]))

    # The load-bearing check.  --check-files removes frames silently, and a
    # removed *official* token is a hole in the evaluation set whose only
    # symptom is a slightly-off number.  `raw_in_split` counts the official
    # tokens the converted logs contained; `n_split` counts the ones that
    # survived.  They must be equal.
    expected_split = totals.get("raw_in_split", n_split)
    if n_split != expected_split:
        present = set(i["token"] for i in all_infos)
        lost = sorted(t for t in (official_tokens or set())
                      if t not in present)
        raise SystemExit(
            "[navsim_converter] FATAL: {} of the {} official tokens in the "
            "converted logs are missing from the output{}. --check-files "
            "dropped frames the evaluator needs: fix the sensor blobs, or "
            "re-run without --check-files and accept missing images."
            .format(expected_split - n_split, expected_split,
                    " (e.g. {})".format(lost[:5]) if lost else ""))
    print("[navsim_converter] all {} official tokens of the converted logs "
          "survived".format(n_split))
    print("[navsim_converter] mean {:.2f} sweeps/frame".format(n_sweeps))
    print("[navsim_converter] prev set on {} frames, next on {} frames; "
          "links refused: {} pruning holes, {} dt gaps, {} chain mismatches"
          .format(n_prev, n_next, totals.get("break_hole", 0),
                  totals.get("break_dt", 0), totals.get("break_chain", 0)))
    if totals.get("can_bus_synthesised", 0):
        print("[navsim_converter] can_bus synthesised for {} frames".format(
            totals["can_bus_synthesised"]))
    hist = _queue_depth_histogram(all_infos, token2idx, max_depth=3)
    if n_split:
        print("[navsim_converter] in-split samples by reachable history depth "
              "(queue_length=4 wants 3): " + ", ".join(
                  "{}: {} ({:.1f}%)".format(d, hist.get(d, 0),
                                            100.0 * hist.get(d, 0) / n_split)
                  for d in range(4)))

    out = dict(
        infos=all_infos,
        metadata=dict(
            version=args.split,
            navsim_converter=CONVERTER_VERSION,
            image_width=args.image_width,
            history_frames=history_frames,
            keyframes_only=bool(args.keyframes_only or history_frames == 0),
            split=args.split,
            cameras=list(args.cameras),
            max_sweeps=args.max_sweeps,
            scene_filter=args.scene_filter,
            num_logs=n_logs,
            num_frames=len(all_infos),
            num_in_split=n_split,
        ),
    )

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("[navsim_converter] wrote {}".format(args.out))


if __name__ == "__main__":
    main()
