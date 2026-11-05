# BEVFormer on NAVSIM

This fork trains BEVFormer (v1, camera-only, temporal) on
[NAVSIM](https://github.com/autonomousvision/navsim) (OpenScene / nuPlan).  It
mirrors the [BEVFusion NAVSIM port](../../bevfusion/docs/NAVSIM.md) — same
classes, same box convention, same front-only BEV box, same nuScenes-styled
evaluator — so the two models' numbers are directly comparable and a BEV
distillation pair is a straight copy.

The port is **additive**: `CustomNuScenesDataset`, every `projects/configs/bevformer*`
nuScenes config and the BEVFormerV2 / DD3D path are untouched.  Everything new
is named `navsim_*`; the four shared files that changed are listed in section 10.

---

## 1. Environment

BEVFormer needs the versions in [`install.md`](install.md): torch 1.9.1+cu111,
mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d 0.17.1, python 3.8.
On this machine the `ssr` conda env already matches exactly, so clone it rather
than build from scratch or mutate it (`/home/yongjae/e2e/SSR/mmdetection3d` is
live behind its egg-link):

```bash
conda create -n bevformer --clone ssr
conda activate bevformer
```

`detectron2` / `fvcore` / `timm` (install.md step f) are **not** required.  They
are only used by BEVFormerV2's auxiliary DD3D head, which has no NAVSIM path
(NAVSIM ships no mono3d annotations and its `num_lidar_pts` is `-1`, so DD3D's
`min_num_lidar_points` filter is unavailable).  Three imports are now guarded so
the plugin loads without them — see section 10.  Install the extras if you want
BEVFormerV2 on nuScenes; nothing else changes.

Pretrained weights:

```bash
mkdir -p ckpts
# base: the DETR3D / BEVFormer R101-DCN 2D pretrain
wget -P ckpts https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
# tiny needs nothing: torchvision://resnet50 is fetched on first use
```

`load_from` transfers more than the backbone.  Compared the way
`mmcv.runner.load_state_dict` does (key present **and** shape equal), it fills
**690 of the model's 1,089 tensors = 47,688,934 of 62,613,657 parameters,
76.16%**:

| prefix | tensors | parameters | loaded |
| --- | --- | --- | --- |
| `img_backbone` | 676 / 676 | 44,410,342 | **100%** |
| `img_neck` | 14 / 14 | 3,278,592 | **100%** |
| `pts_bbox_head` | 0 / 399 | 0 / 14,924,723 | **0%** |

The whole FPN neck transfers, not just `fpn_convs`: the checkpoint's
`lateral_convs` are `(256,512,1,1)`, `(256,1024,1,1)`, `(256,2048,1,1)`, exactly
BEVFormer's `in_channels=[512,1024,2048]`.  Only the 5th extra level
(`fpn_convs.4`, from `num_outs=5` vs this config's 4) is dropped.  All of
`pts_bbox_head.*` (BEV encoder, decoder, heads) trains from scratch.

Two things the filename hides.  **There are zero shape mismatches**: the
checkpoint's head lives under `bbox_head.*` and the model's under
`pts_bbox_head.*`, so the keys never collide and mmcv reports them as
*unexpected keys*, not as size mismatches — expect
`unexpected key in source state_dict: bbox_head...` (103 tensors, plus
`img_neck.fpn_convs.4.{weight,bias}`) and
`missing keys in source state_dict: pts_bbox_head...`.  And **check what the
file actually is**: the `ckpts/r101_dcn_fcos3d_pretrain.pth` present on this
machine was not downloaded from that URL — it was copied from another checkout
— and its `meta.exp_name` reads
`pgd_r101_caffe_fpn_gn-head_dcn_2x16_1x_nus-mono3d_prior_finetune.py`
(mmdet3d 0.12.0, epoch 20, 10 nuScenes classes), i.e. a **PGD** checkpoint, not
FCOS3D.  The 2D backbone and neck are what transfer either way, so nothing is
broken — but "FCOS3D pretrain" is a filename, not a provenance.  After
downloading, confirm with
`torch.load(p, map_location='cpu')['meta']['exp_name']`, and expect the numbers
in the table above to shift if your file differs from the one measured here.  `bevformer_navsim_tiny.py` has no `load_from` at
all (`pretrained=dict(img='torchvision://resnet50')`), so none of this applies
to it.

---

## 2. Data layout

As the ordinary NAVSIM download leaves it:

```
<openscene>/
  navsim_logs/{trainval,test}/<log_name>.pkl        1310 / 147 logs
  sensor_blobs/{trainval,test}/<log_name>/
      CAM_{F0,L0,L1,L2,R0,R1,R2,B0}/<token>.jpg     all 1920x1080
      MergedPointCloud/<frame_token>.pcd            (unused: BEVFormer is camera-only)
```

The `{navtrain,navtest}.yaml` scene filters live in the navsim repo under
`navsim/planning/script/config/common/train_test_split/scene_filter/`.  Their
`log_names` list selects the logs and their `tokens` list *is* the official split
(103,288 navtrain / 12,146 navtest).

Inside this repo:

```
data/navsim/sensor_blobs -> <openscene>/sensor_blobs
data/navsim/lidar        -> <openscene>/sensor_blobs    # only if you want lidar_path to resolve
data/infos/navsim_infos_{train,val}.pkl
ckpts/r101_dcn_fcos3d_pretrain.pth
```

---

## 3. Building the info files

```bash
SF=<navsim>/navsim/planning/script/config/common/train_test_split/scene_filter

python tools/data_converter/navsim_converter.py \
    --navsim-logs  <openscene>/navsim_logs/trainval \
    --sensor-root  <openscene>/sensor_blobs \
    --split trainval --scene-filter $SF/navtrain.yaml \
    --history-frames 4 --check-files --workers 24 \
    --out data/infos/navsim_infos_train.pkl

python tools/data_converter/navsim_converter.py \
    --navsim-logs  <openscene>/navsim_logs/test \
    --sensor-root  <openscene>/sensor_blobs \
    --split test --scene-filter $SF/navtest.yaml \
    --history-frames 4 --check-files --workers 24 \
    --out data/infos/navsim_infos_val.pkl
```

### Why `--history-frames`, and not `--keyframes-only`

This is the one design decision a reader should understand before anything else,
because getting it wrong is silent either way.

BEVFormer is temporal: each training sample is a queue of `queue_length` frames,
and the previous frames have to be *real* previous frames.  The official navtrain
/ navtest token lists are a **subset** of each log's frames, so an info file
containing only those tokens does not have the history.  Measured on this data,
`queue_length=4` (3 history slots):

| info file built with | all 3 history slots real, at k x 0.5 s | frames scored by the evaluator |
| --- | --- | --- |
| `--keyframes-only` (= `--history-frames 0`) | navtrain 54.4%, navtest 56.4% | correct |
| every frame of the selected logs | navtrain 99.76%, navtest 99.88% | **6.48x / 5.88x too many** |
| **`--history-frames 4`** (the default) | **navtrain 99.76%, navtest 99.88%** | **correct** |

The inflation factor is the whole split, not a subset: the 1,192 navtrain logs
hold **669,588** raw frames against **103,288** official tokens (**6.4827x**),
and the 136 navtest logs **71,460** against **12,146** (**5.8834x**).  Rows 2
and 3 have *identical* history quality because `--history-frames 4` already
keeps one more frame than the 3 a `queue_length=4` queue needs; the residual
0.1-0.2% are log starts and `dt` holes, which keeping every frame cannot fix
either.  What separates them is only the sample count — which is why the
inflation is a dataset-indexing problem, not a conversion one.

The converter therefore keeps a frame when it is in the official split **or**
when it sits within `N` frames before one, and flags each with `in_split_tokens`.
`CustomNavsimDataset` then separates *which frames exist* from *which frames are
samples*:

* `self.data_infos` is the **full frame pool** (the history),
* `self.sample_indices` are the positions of the in-split frames (the samples).

`__len__`, `_set_group_flag`, `prepare_test_data`, `get_cat_ids`, `load_gt` and
`_format_bbox` all run over **sample** indices; `get_data_info` and
`get_ann_info` take **frame** indices.  The single riskiest line in the port is
`_format_bbox`'s token lookup, which must go through `sample_indices` — a
one-sample misalignment takes a perfect-prediction mAP from 1.0000 to 0.1412 and
produces no other symptom, so it is now guarded by an explicit length assert.

Measured on the full conversion (24 workers, under two minutes for both splits):

| split | logs | frames kept | official tokens | 3 real history frames | pkl |
| --- | --- | --- | --- | --- | --- |
| navtest | 136 | 19,470 | 12,146 (all) | 99.88% | 170 MB |
| navtrain | 1192 | 154,073 | 103,288 (all) | 99.76% | 1.8 GB |

`--check-files` drops frames whose JPEGs are not on disk.  Measured over the
frames the converter actually considers (navtest 19,470; navtrain 166,322 at
`--history-frames 4`), checking all eight cameras: navtest loses **0**, navtrain
loses **12,249** — of which **12,242 are missing all eight cameras at once** and
**7 are missing only `CAM_B0`**.  So checking three cameras instead of eight
would keep 7 extra navtrain frames and change nothing else; it is near-enough
all-or-nothing, not exactly.  **No official token was lost in either split** (0
of the 12,249 drops is an in-split frame) — the converter asserts that.

**Memory.** The 1.8 GiB train pkl unpickles to **5.1 GiB of python objects**
(2.9x its disk size; the 170 MB navtest pkl to 0.60 GiB, 3.65x) in ~20 s.
`load_annotations` goes through `mmcv.load`, which buffers the file, so a bare
loader process **peaks at 7.9 GiB RSS** against a 1.0 GiB import baseline.
Every rank loads its own copy, so an 8-GPU run settles at **~41 GiB** of info
before any dataloader worker forks; the workers share it copy-on-write at first
and drift upwards as refcounting touches pages.  Fine on a 500 GB box, worth
knowing on a smaller one — `--history-frames 0` cuts the pool by **33%**
(navtrain, 50,785 history-only frames of 154,073) / **38%** (navtest, 7,324 of
19,470) at the cost of the temporal chain, and `data.train.load_interval`
subsamples without re-converting.

### The temporal chain

The queue walks the real `prev` token chain, never `index - k`.  The converter
sets `prev`/`next` only across a verified link: same log, no pruning hole, and
`dt` inside `500000 +/- 250000 us` (NAVSIM is 2 Hz; measured jitter is
+/-2.3 ms, and real holes are exactly 1,000,000 us).  The dataset turns breaks in
that chain into a per-frame `temporal_segment` string, `"<log_name>#<n>"`, and
hands it to the detector as `scene_token`.  That is what makes
`union2one` and `BEVFormer.forward_test` reset `prev_bev` at log starts **and**
at in-log gaps, with no change to the detector.

`temporal_key='scene_token'` switches to NAVSIM's own 20 s / 40-frame scenes,
which is the closest analogue of a nuScenes scene; `'log_name'` (the default)
keeps continuity across the whole log.

---

## 4. Coordinate conventions (verified against the data, not assumed)

* The ego frame **is** the LiDAR frame: `lidar2ego` is the identity in every
  frame.  Origin is the rear axle projected to the ground, **+x forward, +y
  left, +z up**.  This is the inverse of nuScenes' `LIDAR_TOP` frame, where
  `lidar2ego` is a ~-90 degree z-rotation so +y is forward and +x is right.
  That 90 degrees is the whole of bug 7.1.
* The vertical range is therefore `[-3, +5]`, the inverse of nuScenes' `[-5, 3]`.
* Quaternions are `(w, x, y, z)` (pyquaternion order) everywhere.
* `cams[*]['sensor2lidar_*']` is **camera -> LiDAR**, so
  `lidar2img = pad(K, 4x4) @ inv(camera2lidar)`.
* NAVSIM boxes are `[x, y, z_centre, length, width, height, heading]` with
  `heading` **unwrapped** (it reaches +/-2pi).  The converter rewrites them to the
  mmdet3d LiDAR convention used by both ports:
  `[x, y, z_centre, width, length, height, wrap(-heading - pi/2)]` — note the
  length/width swap.  `z` is the **centre**, so boxes are built with
  `origin=(0.5, 0.5, 0.5)`.
* Timestamps are microseconds.
* Classes are the seven nuPlan `TrackedObjectType` names: `vehicle`,
  `pedestrian`, `bicycle`, `traffic_cone`, `barrier`, `czone_sign`,
  `generic_object`.  Frequency is very skewed — `generic_object` outnumbers
  `vehicle`, and `czone_sign` / `bicycle` are ~0.2% each.
* NAVSIM has **no attributes**, so there is no `attr_err` / mAAE and NDS is
  divided by 9, not 10.  **NAVSIM NDS is not comparable to nuScenes NDS.**

### `can_bus`

NAVSIM already ships an 18-slot `can_bus` per frame, but **two of its slots are
ordered differently from nuScenes'**:

```
NAVSIM   [0:3] pos  [3:7] quat  [7:10] accel  [10:13] VELOCITY       [13:16] ROTATION RATE  [16:18] 0,0
nuScenes [0:3] pos  [3:7] quat  [7:10] accel  [10:13] ROTATION RATE  [13:16] VELOCITY       [16:18] 0,0
```

The converter swaps `[10:13]` with `[13:16]` once, so every comment in the
BEVFormer codebase means what it says.  `tools/navsim_sanity_check.py` asserts
the swap was applied.

Of the 18, only four slots are read positionally: `[0]`/`[1]` (the translation
delta) and `[16]`/`[17]` (absolute / differenced global yaw).  The rest is raw
`can_bus_mlp` input.  The array is kept **float64** throughout: NAVSIM's
`ego2global_translation` is UTM-like (~4e6 m), where the float32 spacing is
**0.25 m** — **~10% of the measured mean ego step** (2.536 m over the 12,140
linked navtest pairs, 2.321 m over navtrain's 103,206).

One upstream quirk is preserved and documented in `navsim_dataset.py`:
`can_bus[3:7] = rotation` writes `[w, w, w, w]`, not the quaternion, because
`pyquaternion.Quaternion` has `__float__` but no `__len__` so numpy broadcasts it.
That is how upstream BEVFormer has always behaved on nuScenes; the true
quaternion is still in `info['can_bus']` and `ego2global_rotation`.

---

## 5. The BEV grid: front only, 64 m, square cells

```
point_cloud_range: [0.0, -32.0, -3.0, 64.0, 32.0, 5.0]
bev_w_ = 100   # x, forward,  0 .. 64 m   -> 0.64 m cells
bev_h_ = 100   # y, lateral, -32 .. 32 m  -> 0.64 m cells
```

BEVFormer's BEV axis mapping is `bev_w <-> pc_range x`, `bev_h <-> pc_range y`
(`encoder.py` builds `xs` from `W` and `ys` from `H`; `bevformer_head.py` sets
`real_w = x extent`, `real_h = y extent`).  All three upstream configs are square,
so this had never been exercised; a NAVSIM grid is the first test.

Why this grid, matching `bevfusion/.../swint_convfuser_64m.yaml`:

* **Front only.** The three front cameras (`CAM_F0`, `CAM_L0`, `CAM_R0`) span
  ~+/-87 degrees.  There is no rear observation to encode.
* **64 m rather than 32 m.**  Measured on the shipped front-only ROI
  (`0 < x < 64`, `-32 < y < 32`), over **every** in-split sample — 12,146 on
  navtest, 103,288 on navtrain — with the real `get_ann_info` →
  `ObjectRangeFilter` → `ObjectNameFilter` predicate:

  | measurement | 64 m (shipped) | 32 m |
  | --- | --- | --- |
  | GT boxes / sample, navtest | **20.81** | 12.47 |
  | GT boxes / sample, navtrain | **30.78** | 19.19 |
  | samples with 0 boxes, navtest | **4.78%** | 8.13% |
  | samples with 0 boxes, navtrain | **3.03%** | 5.57% |
  | blind BEV cells, navtest | **3.8349%** | 7.6287% |
  | blind BEV cells, navtrain | **3.8811%** | 7.7212% |

  The empty-sample rate **is** the `filter_empty_gt` resample probability here,
  and it is independent of `queue_length`: `prepare_train_data` applies the
  filter to the current frame only (upstream's nuScenes version tests every
  queue frame, which is what compounds).  Blind cells are the fraction getting
  zero spatial cross-attention under `encoder.point_sampling`'s own predicate,
  over up to 4 in-split frames from every log (544 frames / 136 logs on
  navtest, 4,728 / 1,192 on navtrain).  For scale, the **unfiltered whole frame** — no
  range and no name filter, 360 degrees, every class — holds **54.79** boxes on
  navtest and **83.66** on navtrain, so the 64 m front ROI keeps 38.0% /
  36.8% of the labels; that number is *not* comparable to the ROI rows above.
* **Square cells.** `rotate_prev_bev` rotates the previous BEV as an image, which
  is only a valid yaw rotation when the cells are isotropic.  0.64 x 0.64 m is.
  (The alternative 32 m / 100x100 grid that matches PARA-SSR cell-for-cell has
  0.32 x 0.64 m cells, and an image-space rotation on it maps a point at 10 m to
  20 m — if you need that grid for distillation, set `rotate_prev_bev=False`.)

Blind cells sit where you would expect: on the first navtest frame they lie at
`x in [0.32, 2.88] m`, `|y| in [0.32, 31.68] m`, mean `x` 1.26 m — a strip
hugging the ego along the `x = 0` edge, which forward-facing cameras cannot look
down into.

`num_points_in_pillar` (`D`) is **8**, not the upstream 4.
`BEVFormerEncoder.get_reference_points(dim='3d')` builds
`zs = linspace(0.5, Z - 0.5, D) / Z` and `point_sampling` maps it back through
the same `Z`, so the sample heights are

```
Z  = z_max - z_min
zs = linspace(0.5, Z - 0.5, D) + z_min
```

— **not** `z_min + (0.5 + i)/D * Z`.  The two coincide only when `D == Z`, which
is exactly the shipped case (`D = Z = 8`), so the difference is invisible until
you change `D`:

| z range | D | sample heights | spacing |
| --- | --- | --- | --- |
| nuScenes `[-5, 3]` | 4 | -4.500, -2.167, 0.167, 2.500 | 2.333 |
| NAVSIM `[-3, 5]` | 4 | -2.500, -0.167, 2.167, 4.500 | 2.333 |
| **NAVSIM `[-3, 5]`** | **8** | **-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5** | **1.000** |

What `D=8` buys, measured as the fraction of in-ROI GT box **centre** heights
within +/-0.5 m of some sample height (every box surviving range+name filtering
over every in-split sample: 252,794 boxes / 12,146 samples on navtest,
3,179,006 / 103,288 on navtrain):

| D | navtest | navtrain |
| --- | --- | --- |
| 4 | **35.56%** | **31.51%** |
| **8** (shipped) | **99.9980%** | **99.9980%** |

`D=8`'s +/-0.5 m bands tile `[-3, 5]` contiguously, so every centre inside the z
range is covered by construction; the 5 / 63 misses are boxes whose stored z
falls *outside* `[-3, 5]` (navtrain's extreme is -251.81 m — `ObjectRangeFilter`
tests x and y only, so `point_cloud_range`'s z limits never reach the labels).
`D=4` fails because NAVSIM centre heights are massed just above its second
covered band: median **0.4737 m** (navtest) / **0.5142 m** (navtrain), densest
0.25 m bin `[0.50, 0.75)`, while `D=4`'s nearest sample height is -0.1667 m.  So
`D=4` is not "hits nothing" — it covers about a third of object centres — and
`D=8` covers all of them.

`D=8` is also why **`MSDeformableAttention3D.num_points` is 16**, not upstream's
8.  `num_points` is the *total* sampling offsets per head per level;
`spatial_cross_attention.py` reshapes them as
`(num_points // num_Z_anchors, num_Z_anchors)` with
`num_Z_anchors == num_points_in_pillar`.  Upstream's 8 over `D=4` is 2 offsets
per height anchor, so leaving `num_points=8` with `D=8` would silently halve it
to 1 — trading sampling density for height coverage instead of adding height
coverage.  `num_points` must also be an integer multiple of `D` or that
`.view()` raises.

**SCA cost is not linear in `D`.**  The deformable-attention cost is
`num_levels * num_points` sampling points per head and `D` does not appear in
it at all; `D` only scales `point_sampling`'s projection and the reference-point
tensor.  Setting `D` back to 4 therefore saves very little on its own, and
halving `num_points` with it is the part that would hurt.

`post_center_range` is `[-10, -42, -10, 74, 42, 10]` = `pc_range` + 10 m slack,
BEVFormer's own convention.  The head's sigmoid already clamps every decoded
centre inside `pc_range`, so this filter can only ever *delete* valid boxes — a
symmetric carry-over of nuScenes' `+/-61.2` would silently drop the far field.

### Images

NAVSIM ships 1920x1080.  Upstream `bevformer_base` has no resize stage at all
(it pads only), so the NAVSIM configs add
`RandomScaleImageMultiViewImage(scales=[0.8334])` **before** `PadMultiViewImage`:

| config | resize | result | padded (`size_divisor=32`) |
| --- | --- | --- | --- |
| base | 0.8334 | 1600 x 900 | 1600 x 928 |
| tiny | 0.4167 | 800 x 450 | 800 x 480 |

That is the same geometry as today's nuScenes base/tiny, so the pretrain keeps
its scale statistics.  The literal decimals are **not** needed for the integer
size: `RandomScaleImageMultiViewImage` truncates with `int()`, but IEEE-754
rounds `5/6` *up* to 0.8333333333333334, so `int(1920 * 5/6) == 1600` and
`int(1080 * 5/6) == 900` — exactly what `0.8334` gives (`5/12` and `0.4167`
likewise both give 800x450).  What does truncate is a **truncated** decimal
copy: `0.8333` gives 1599x899, `0.41666` gives 799x449.  Both shipped values are
the *rounded* decimal, written out so the intent survives being copied at finite
precision.

The two spellings *do* differ in `lidar2img`, which the transform
pre-multiplies by `rand_scale`: `0.8334 - 5/6 = 6.67e-05`, i.e. at most
**0.128 px** at `u = 1920` and 0.072 px at `v = 1080`.  Measured consequence at
base 64 m / 100x100 / `D=8`: the blind-cell verdict differs for **0 of 2,000,000
BEV cells** over 200 navtest frames.  The resize left-multiplies `lidar2img`,
which is why `ObjectFOVFilter` must run before it.

---

## 6. Training

```bash
# 8 GPUs (dist_train.sh sets PYTHONPATH for you)
./tools/dist_train.sh projects/configs/bevformer/bevformer_navsim_base.py 8

# single GPU -- note PYTHONPATH: tools/train.py imports the plugin by module
# path, and plain `python tools/train.py` puts tools/ on sys.path, not the repo
# root. This is upstream behaviour, not specific to the NAVSIM port.
PYTHONPATH=. python tools/train.py \
    projects/configs/bevformer/bevformer_navsim_base.py

# smoke / debug: R50, 50x50 BEV, 800x480, 3-frame queue, ~50 samples
PYTHONPATH=. python tools/train.py \
    projects/configs/bevformer/bevformer_navsim_tiny.py \
    --work-dir runs/smoke_tiny --no-validate \
    --cfg-options total_epochs=1 runner.max_epochs=1 data.train.load_interval=2000
```

Measured on one RTX A6000 at `samples_per_gpu=1`: base uses **11.4 GB**
(1600x928, 3 cameras, 4-frame queue) and runs ~1.9 s/iteration; tiny uses
2.4 GB.  There is room to raise `samples_per_gpu` on 48 GB cards.

`total_epochs=24` is carried over from upstream, but navtrain is ~3.7x nuScenes
(103,288 vs 28,130 samples), so 24 epochs is ~3.7x the gradient steps of the
nuScenes baseline — BEVFusion settled on 20 for its 64 m NAVSIM teacher.  Decide
that against your compute budget before a long run.

---

## 7. Four bugs this port had to fix

All four are silent on nuScenes and silent on NAVSIM; none of them raises.

### 7.1 The ego-motion shift axes — `transformer.py`

`get_bev_features` decomposes the ego translation into `shift_y = L cos(theta)`
(along the heading) and `shift_x = L sin(theta)` (to the right), then feeds
`shift = [shift_x, shift_y]` to the temporal self-attention as
`(bev_w axis, bev_h axis)`.  That is correct only when +y is forward and +x is
right, i.e. in the nuScenes `LIDAR_TOP` frame.  In NAVSIM's ego frame forward is
+x and the lateral axis points **left**, so the two components must swap and the
lateral one must flip sign.

`PerceptionTransformer` now takes `ego_axes`:

| value | frame | used by |
| --- | --- | --- |
| `'y_forward_x_right'` (default) | nuScenes `LIDAR_TOP` | every existing nuScenes config, unchanged |
| `'x_forward_y_left'` | NAVSIM / nuPlan ego, mmdet3d LiDAR canonical | the NAVSIM configs |

Measured over **all 12,140** navtest in-split frames whose pool `prev` survives
re-validation — the last hop of every training queue, and the hop `union2one`
differences — against a ground truth computed independently as
`R(yaw_cur)^T (t_prev - t_cur)`.  Both rows are that same population of 12,140
pairs; cells are 0.64 m:

| `ego_axes` | mean error | max error |
| --- | --- | --- |
| `y_forward_x_right` | **5.60 cells (3.58 m)** | **16.37 cells (10.48 m)** |
| `x_forward_y_left` | **1.66e-15 cells** | **2.63e-14 cells** |

The wrong setting's error is *larger than the true shift itself* (mean 3.96
cells over the same pairs), so using it is worse than not shifting at all.  The
right setting is exact to float64 round-off, not approximate.  navtrain agrees:
mean 5.13 / max 18.51 cells over its 103,206 pairs.

### 7.2 `rotate_center` — config

`rotate_center=[100, 100]` is a hard-coded default that no upstream config
overrides, and it is the ego's pixel in a 200x200 nuScenes grid.
`torchvision.rotate` takes `center` as `(col, row)` from the top-left, so the
correct value is

```
col = (0 - pc_range[0]) / real_w * bev_w     # = 0   (the ego sits at x = 0)
row = (0 - pc_range[1]) / real_h * bev_h     # = 50
```

i.e. `[0.0, 50.0]` for base and `[0.0, 25.0]` for tiny.  Both NAVSIM configs set
it explicitly.  (Upstream `bevformer_tiny` is *already* broken by this: its grid
is 50x50, so `[100, 100]` is off-grid and the rotation retains ~0% of the
content.)  The rotation **angle** sign needs no change — in both conventions
`(col, row) = (that frame's x, y)` and `R_z` commutes with the `R_z(90 deg)`
separating the frames.

### 7.3 `can_bus` aliasing — the new dataset only

`get_data_info` returns `info['can_bus']` without copying, and neither
`CustomCollect3D` nor `union2one` copies either, so `can_bus[:3] -= prev_pos`
mutates the cached info in place.  It is self-healing across calls because
`get_data_info` rewrites those slots every time — **except** when the same frame
index appears twice in one queue, which is exactly what happens at a log start
where the history is padded by repetition.  The second occurrence is then the
same array object and the delta comes out as `0 - translation`, i.e. ~4e6 m of
fabricated ego motion.  `CustomNavsimDataset` takes a fresh float64 copy.
`CustomNuScenesDataset` is left alone.

### 7.4 Test-mode `prev_bev` across SAMPLE-level gaps — `navsim_dataset.py`

`temporal_segment` is a property of the **frame pool**, which is right for
training: `prepare_train_data` walks the pool's `prev` chain, so every queue hop
really is one 0.5 s frame.  Test mode does not walk it.
`BEVFormer.forward_test` is handed the in-split **samples** one after another
and carries `prev_bev` whenever `scene_token` is unchanged — but the official
token list *skips* frames inside a log, so consecutive in-split frames are often
not adjacent pool frames.  Measured on navtest: **591 of 10,631 continued pairs
(5.56%)** are 2-5 pool frames apart — gap histogram
`{1: 10040, 2: 211, 3: 153, 4: 115, 5: 112}` — up to **2.50 s** and **27.46 m**
apart; navtrain has 4,562 of 89,778 (5.08%), up to 2.50 s and 31.51 m.  A
pool-level segment string cannot see those jumps, so the model would warp a
multi-second-old BEV by a correspondingly large `can_bus` delta — an input
training never produces, since the training queue only ever hops 0.5 s (1.0 s at
most, through the random drop in `_queue_frame_indices`).

`CustomNavsimDataset.load_annotations` therefore builds a second,
**sample-level** segment counter, advanced whenever the previous *sample* is not
the immediate `prev`-chain predecessor of this one, and `prepare_test_data`
substitutes it into `scene_token`.  That adds exactly those 591 `prev_bev`
resets on navtest; with `load_interval > 1` every pair resets.  Training is
unaffected — it still walks the pool chain — and the detector needs no change.

---

## 8. Evaluation

`CustomNavsimDataset.evaluate` is BEVFusion's nuScenes-styled evaluator, ported
unchanged, so the two models are scored identically:

```
dist_ths   [0.5, 1.0, 2.0, 4.0]      dist_th_tp 2.0
class_range  all 7 classes [64, 32]  (per-axis |x|, |y|, i.e. the whole grid)
tp_metrics [trans_err, scale_err, orient_err, vel_err]    # no attr_err
NDS = (5 * mAP + 4 TP scores) / 9
```

Verified end to end on 395 navtest samples: feeding the GT back as predictions
gives **mAP 1.0000 / NDS 1.0000** with every TP error 0 and AP 1.0 for all seven
classes; jittering the centres degrades it monotonically (0.5 m → 0.705,
1.5 m → 0.325, 6 m → 0.023).

The optional `camera_fov` filter (project the eight box corners through
`lidar2img`, keep a box with at least one corner in an image) is wired on both
sides — `ObjectFOVFilter` in the train pipeline and `eval_cfg.camera_fov` for GT
*and* predictions — and is **off** by default, matching BEVFusion, so the two
ports score the same label set.  On the **shipped 64 m ROI** turning it on
removes **2.5452%** of in-ROI boxes (navtest: 6,434 of all 252,794 in-ROI boxes
over the 12,146 in-split samples) and **2.8279%** on navtrain (89,900 of
3,179,006).  Essentially every removal is near-field in **forward** distance:
**6,434 of 6,434 (100.000%)** navtest removals have `x < 5 m`, **64.67%** of the
boxes at `x < 2 m` go, and **0.000%** beyond `x = 5 m` do.  Radial banding makes
the filter look like it bites out to 32 m; it does not — the radial "2.8%
removed at 20-32 m" is boxes sitting *beside* the ego (small `x`, large `|y|`),
outside a forward rig's view.

For comparison, BEVFusion publishes **5.2%** removed over its **32 m** ROI (400
navtrain frames).  The same predicate over this port's full splits at 32 m gives
**4.5353%** (navtrain) / **4.2484%** (navtest) — same order, ~0.7 pp lower, same
per-band shape.  The shipped 64 m figure is the lower one because the extra
range dilutes the near field with far-field boxes the rig sees in full; quote
2.55% / 2.83% for this port and 5.2% only as BEVFusion's 32 m number.

Flip `eval_cfg.camera_fov.enabled` and both sides follow; masking only one side
would make detections outside the region count as false positives with no GT
left to match.

**`image_wh` is an unchecked cross-file invariant.**  `ObjectFOVFilter` and
`eval_cfg.camera_fov` both test against **stored, unscaled** pixels, so
`eval_cfg.camera_fov.image_wh` must equal the converter's `--image-width` and
its matching height (`--image-width * 1080/1920`).  The converter scales rows
0-1 of every `cam_intrinsic` by `--image-width / 1920`, so a pkl built at
`--image-width 960` needs `image_wh=[960, 540]`; the shipped pair is
`[1920, 1080]` against the default `--image-width 1920`, which the pkl records
as `metadata['image_width']`.  **Nothing compares the two at runtime** — not the
dataset, not the transform, and not even `tools/navsim_sanity_check.py`, which
reads the config's `image_wh` as its canvas (and derives the resize/pad geometry
from it) but never looks at the pkl's `metadata['image_width']`.  Get them out
of step and the FOV test silently accepts or rejects the wrong boxes.

```bash
./tools/dist_test.sh projects/configs/bevformer/bevformer_navsim_base.py \
    work_dirs/bevformer_navsim_base/latest.pth 8
```

Use `dist_test.sh` even on one GPU: upstream `tools/test.py` has `assert False`
on its non-distributed branch, so `python tools/test.py` cannot run at all.
`dist_test.sh` already appends `--eval bbox`.

Evaluation must also keep the sampler it ships with.  `nonshuffler_sampler` is a
plain `DistributedSampler` that hands each rank a **contiguous** block, and
`custom_multi_gpu_test` **concatenates** the per-rank parts, which is what makes
prediction *i* line up with sample *i* in `_format_bbox` — and what keeps the
detector's `prev_bev` chain temporally valid inside a rank.  A striding sampler
would silently chain `prev_bev` across non-adjacent frames and misattribute
every prediction.

---

## 9. Pre-flight check

The port's failure modes are silent and `tools/analysis_tools/visual.py` is
nuScenes-devkit-only, so run this before any training:

```bash
python tools/navsim_sanity_check.py \
    --info data/infos/navsim_infos_val.pkl \
    --config projects/configs/bevformer/bevformer_navsim_base.py \
    --out-dir work_dirs/navsim_sanity
```

It audits the config (rotate_center vs the computed ego cell, square cells under
`rotate_prev_bev`, both `num_cams`, both `num_classes`, `post_center_range` vs
`pc_range`, the resize/pad geometry) and then runs five checks: info-pkl and
prev-chain stats with a `can_bus` slot audit, GT projection into the three
cameras, the BEV camera-coverage fraction under `encoder.point_sampling`'s own
predicate, the ego-motion shift against both `ego_axes` settings, and a BEV PNG
per token.  Exit code 1 on any failure, so it works as a gate.

Reference output on navtest: velocity confirmed in the nuScenes slots, **~97.4%**
of in-ROI boxes with at least one corner in-image (97.4548% over the full split,
the complement of section 8's 2.5452%), **~3.82%** blind BEV cells on the tool's
own 50-frame sample (3.8349% over up to 4 frames from every log — section 5),
and `x_forward_y_left` matching the ground-truth shift to float64 round-off
(2.63e-14 cells at worst over all 12,140 in-split pairs).

---

## 10. Changes to shared code

Four edits outside the `navsim_*` files, all additive and all reversible:

* `projects/mmdet3d_plugin/bevformer/modules/transformer.py` — the `ego_axes`
  keyword of section 7.1.  Default is the existing nuScenes behaviour, so every
  nuScenes config is byte-identical.  `PerceptionTransformerV2` is untouched.
* `projects/mmdet3d_plugin/__init__.py` — `from .dd3d import *` is now a
  `try/except ImportError` with a warning naming the missing extras.
* `projects/mmdet3d_plugin/datasets/__init__.py` — the same guard around
  `CustomNuScenesDatasetV2`, plus `CustomNavsimDataset`.
* `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` — the same guard
  around `DD3DMapper`, plus `ObjectFOVFilter`.

New files:

```
tools/data_converter/navsim_converter.py                 info-pkl writer
tools/navsim_sanity_check.py                             pre-flight checker
projects/mmdet3d_plugin/datasets/navsim_dataset.py       CustomNavsimDataset
projects/mmdet3d_plugin/datasets/navsim_eval_utils.py    BEVFusion copy + 6-line header
projects/mmdet3d_plugin/datasets/navsim_fov_utils.py     BEVFusion copy + 6-line header
projects/mmdet3d_plugin/datasets/pipelines/navsim_transforms.py   ObjectFOVFilter
projects/configs/bevformer/bevformer_navsim_base.py
projects/configs/bevformer/bevformer_navsim_tiny.py
docs/NAVSIM.md
```

`navsim_eval_utils.py` and `navsim_fov_utils.py` are copies of BEVFusion's
`nuscenes_styled_eval_utils.py` and `fov_utils.py`, **verbatim apart from a
six-line provenance header** at the top of each, on purpose — both depend only
on numpy and pyquaternion, and keeping the bodies identical is what makes the
two ports score NAVSIM the same way.  Do not "clean up" one without the other.
