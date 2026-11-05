# ============================================================
# NAVSIM (OpenScene / nuPlan) base configuration for BEVFormer
#
# Key differences from projects/configs/bevformer/bevformer_base.py:
#   - Does NOT inherit ../datasets/custom_nus-3d.py.  That file is nuScenes
#     specific and mmcv merges _base_ recursively, so it would leak
#     nuScenes class_names / dataset_type / data_root / eval_pipeline in.
#     Everything the NAVSIM contract needs is therefore spelled out here, and
#     only ../_base_/default_runtime.py is inherited.
#   - dataset_type: CustomNavsimDataset (no nuscenes-devkit anywhere).
#   - 7 nuPlan object classes instead of 10 nuScenes ones (head AND bbox_coder).
#   - 3 front cameras (CAM_F0 / CAM_L0 / CAM_R0) instead of 6 ring cameras, so
#     num_cams=3 in the transformer AND in the SpatialCrossAttention cfg.
#   - Front-only BEV: x_forward in [0, 64] m, y_left in [-32, 32] m, and the
#     vertical range flips from [-5, 3] to [-3, 5] because NAVSIM's ego origin
#     is the rear axle ON THE GROUND, not a roof LiDAR.
#   - ego_axes='x_forward_y_left': NAVSIM's lidar2ego is the identity, so the
#     ego frame IS the LiDAR frame with +x forward / +y left.  nuScenes'
#     LIDAR_TOP frame is rotated -90 deg from that, which is what the upstream
#     ego-motion shift code assumes.  See the comment at `ego_axes` below.
#   - rotate_center=[0.0, 50.0]: the ego sits on the x=0 EDGE of a front-only
#     grid, not at its centre, so the hard-coded [100, 100] default is wrong.
#   - num_points_in_pillar=8: the z range flipped, and at D=4 the pillar
#     samples straddle the band where nuPlan object centres sit, covering
#     only about a third of them.  See the comment below.
#   - post_center_range follows pc_range + 10 m instead of a symmetric +-61.2,
#     which would delete the whole far field.
#   - A RandomScaleImageMultiViewImage stage is ADDED (base upstream only pads),
#     because NAVSIM ships 1920x1080 blobs and nuScenes ships 1600x900.
#   - input_modality.use_external=False: nuScenes sets it True for its separate
#     CAN-bus expansion; NAVSIM's can_bus vector ships inside the info pkl.
#   - evaluation is nuScenes-STYLED but not nuScenes-comparable; see eval_cfg.
#
# The companion smoke/debug config is bevformer_navsim_tiny.py.
# Data prep and the measured numbers quoted below: docs/NAVSIM.md.
# ============================================================

_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ------------------------------------------------------------------ #
# BEV region of interest
# ------------------------------------------------------------------ #
# [x_min, y_min, z_min, x_max, y_max, z_max] in the NAVSIM ego frame, which is
# also the LiDAR frame (lidar2ego is the identity in every NAVSIM log): +x
# forward, +y LEFT, +z up, z = 0 is the ground, origin = the rear axle.
#
# Front-only in x: the three front cameras have no rear observation at all, so
# encoding the rear would ask the spatial cross-attention to hallucinate half
# the grid.  Matches bevfusion's 64 m NAVSIM teacher one-to-one
# (configs/navsim/det/transfusion/secfpn/camera+lidar/swint_convfuser_64m.yaml),
# so the two models' BEVs and their numbers are directly comparable.
#
# z is [-3, 5], not nuScenes' [-5, 3]: the origin moved from a roof LiDAR
# (~1.84 m up) down to the rear axle on the ground.  The 8 m extent is kept so
# the pillar geometry below stays a pure re-sampling question.
#
# 64 m rather than 32 m forward was chosen on three measured grounds.  Every
# box count below is measured over THIS front-only ROI (0 <= x <= 64,
# -32 <= y <= 32) on the in-split samples of a named split -- not over a
# symmetric +-64 m region, which is what an earlier revision of this comment
# quoted (its "40.5 / 25.0 boxes, 1.24% / 2.78% empty" is the +-64 m / +-32 m
# x-range on navtest, and its resample figures compounded those over the queue):
#   - 20.8129 GT boxes/sample in-ROI vs 12.4686 at 32 m, and 4.7835% vs 8.1344%
#     of samples left with no box at all (navtest, 12,146 in-split samples).
#     navtrain, 103,288 in-split samples: 30.7781 vs 19.1901 boxes, 3.0304% vs
#     5.5737% empty.  That empty rate is exactly what filter_empty_gt resamples
#     -- see the comment at filter_empty_gt below.
#   - 3.8349% of BEV cells get zero spatial-cross-attention contribution (no
#     camera projects into them) vs 7.6287% at 32 m -- navtest, 544 in-split
#     frames drawn from all 136 logs, D=8 pillar samples projected onto the
#     padded 1600x928 canvas.  navtrain (4,728 frames / 1,192 logs) gives
#     3.8811% / 7.7212%.  The blind cells are a strip hugging the ego along the
#     x=0 edge, where a forward rig that cannot look down sees nothing.
#   - with bev_h_ == bev_w_ == 100 the cells come out SQUARE (0.64 m), which
#     rotate_prev_bev requires; the 32 m / 100x100 grid is 0.32 x 0.64 m and an
#     image-space rotate on it shears the previous BEV.
point_cloud_range = [0.0, -32.0, -3.0, 64.0, 32.0, 5.0]

# Dead keys, kept because NMSFreeCoder / train_cfg still accept them (grep
# finds no live reader in projects/mmdet3d_plugin).  Recomputed from the NAVSIM
# range anyway so a reader is not misled: 64 m / 0.2 m = 320 cells, z extent is
# still 8 m.
voxel_size = [0.2, 0.2, 8]

# Backbone-coupled, NOT dataset-coupled: this is the Caffe-style BGR
# normalisation the R101-DCN mono3D pretrain was trained with (that checkpoint
# is a PGD one despite its filename -- see the note at load_from).  Unchanged
# from bevformer_base.py on purpose.
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# nuPlan TrackedObjectType names exactly as they appear in the info pkl's
# gt_names (see tools/data_converter/navsim_converter.py).  The ORDER is the
# label order: CustomNavsimDataset maps a name to its index in this list, and
# the evaluator's class_range below is keyed by the same names.
class_names = [
    'vehicle', 'pedestrian', 'bicycle', 'traffic_cone', 'barrier',
    'czone_sign', 'generic_object'
]

# use_external is False, unlike nuScenes: there it flags the separately
# downloaded CAN-bus expansion, whereas NAVSIM's 18-slot can_bus vector is
# built by the converter and ships inside the info pkl.
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4

# BEV grid.  The axis mapping is NOT symmetric and is worth stating once,
# because every upstream config is square and so never exercises it:
#   bev_w_ (column index w) <-> physical X <-> pc_range[0] .. pc_range[3]
#   bev_h_ (row    index h) <-> physical Y <-> pc_range[1] .. pc_range[4]
#   query index q = h * bev_w + w
# (encoder.py builds xs from W and ys from H; bevformer_head.py sets
# real_w = pc_range[3]-pc_range[0] and real_h = pc_range[4]-pc_range[1].)
#
# 64 m forward / 64 m lateral over 100 x 100 -> SQUARE 0.64 m cells.  Square
# matters: rotate_prev_bev does an image-space rotation of the previous BEV,
# which is only a valid yaw rotation when the cells are isotropic.
bev_h_ = 100
bev_w_ = 100

# 4 frames = 1.5 s of history at NAVSIM's 2 Hz (nuScenes keyframes are also
# 2 Hz, so this is the same temporal span as bevformer_base.py).
queue_length = 4

# The ego's own cell, as torchvision.transforms.functional.rotate wants it:
# center = (x, y) = (col, row) from the top-left.
#   col = (0 - pc_range[0]) / real_w * bev_w = (0 -   0.0) / 64 * 100 =  0
#   row = (0 - pc_range[1]) / real_h * bev_h = (0 - -32.0) / 64 * 100 = 50
# The upstream default is the hard-coded [100, 100], i.e. the ego cell of a
# 200x200 nuScenes grid; no config overrides it, so bevformer_tiny upstream is
# already rotating its 50x50 grid about an off-grid point and retaining ~0% of
# the previous BEV.  With a front-only ROI the ego is on the x=0 edge, so the
# correct centre is the LEFT edge, half way up.
rotate_center_ = [0.0, 50.0]

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        # Deliberately left at the nuScenes value.  The NAVSIM ROI is smaller
        # and has 3 fewer classes, so 900 is generous, but reducing it changes
        # the query_embedding shape and breaks weight reuse against any
        # existing BEVFormer checkpoint.  Tune later, measured.
        num_query=900,
        # 7, not 10.  num_classes appears TWICE (here and in bbox_coder below)
        # and nothing cross-checks them.
        num_classes=7,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            # CAM_F0 / CAM_L0 / CAM_R0.  num_cams is an independent field here
            # and in the SpatialCrossAttention cfg below, both defaulting to 6;
            # a mismatch with the number of loaded images fails loudly
            # (IndexError / RuntimeError), but set both anyway.
            num_cams=3,
            rotate_center=rotate_center_,
            # Which physical axis each BEV axis carries, used ONLY to split the
            # ego translation between the two BEV axes in get_bev_features.
            #   'y_forward_x_right'  nuScenes LIDAR_TOP (the upstream default):
            #                        lidar2ego is a ~-90 deg z-rotation, so
            #                        +y is forward and +x is right.
            #   'x_forward_y_left'   NAVSIM / nuPlan ego, which is the mmdet3d
            #                        LiDAR canonical frame: +x forward, +y left.
            # Measured on real NAVSIM poses with this grid, the nuScenes
            # setting emits shift (0.00007, 0.10313) where the truth is
            # (0.10313, -0.00006): right magnitude, swapped axis, inverted
            # lateral sign.  That one frame is 10.31 cells of error, and it
            # is an ILLUSTRATION, not the average: over all 12,140 in-split
            # navtest pairs the nuScenes setting is off by mean 5.60 / max
            # 16.37 cells (3.58 m / 10.48 m), which is larger than the true
            # shift itself (mean 3.96 cells) -- i.e. worse than not shifting
            # at all.  navtrain, 103,206 pairs: mean 5.13 / max 18.51.  The
            # shipped setting reproduces the truth to max 2.63e-14 cells,
            # which is float64 round-off, not an approximation.  Wrong on
            # every moving frame and completely silent.  The rotate_prev_bev
            # ANGLE sign needs no change -- a z-rotation is invariant to a
            # yaw-only reframing.
            ego_axes='x_forward_y_left',
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                # 8, not 4.  The pillar sample heights are a linspace over
                # the z EXTENT, not fractions of D:
                #     Z  = z_max - z_min
                #     zs = linspace(0.5, Z - 0.5, D) + z_min
                # (get_reference_points builds linspace(0.5, Z-0.5, D)/Z and
                # point_sampling scales it by (z_max - z_min) and adds z_min.)
                # It is NOT z_min + (0.5 + i)/D * Z.  The two forms agree only
                # when D == Z, which is exactly the shipped D=8 / Z=8 case --
                # which is why the wrong form sat here unnoticed, and why it
                # matters that it be stated correctly before anyone changes D.
                # Real heights:
                #   nuScenes [-5, 3], D=4 -> -4.500, -2.167,  0.167,  2.500
                #   NAVSIM   [-3, 5], D=4 -> -2.500, -0.167,  2.167,  4.500
                #   NAVSIM   [-3, 5], D=8 -> -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5
                # Both D=4 rows are spaced 2.333 m apart; D=8 is spaced exactly
                # 1.0 m, so its +-0.5 m bands TILE [-3, 5] contiguously.
                # Measured on in-ROI GT box centre heights (navtest, 252,794
                # boxes over 12,146 in-split samples), the fraction lying within
                # 0.5 m of some pillar sample is 35.5629% at D=4 against
                # 99.9980% at D=8; navtrain (3,179,006 boxes) gives 31.5053%
                # vs 99.9980%.  So D=4 does not "hit nothing", as an earlier
                # revision of this comment claimed -- it misses about two
                # thirds, because the NAVSIM centre-height mode sits just above
                # D=4's -0.167 m sample (median 0.4737 m navtest / 0.5142 m
                # navtrain; densest 0.25 m bin [0.50, 0.75)).
                # SCA cost is LINEAR in D, so reverting to 4 halves the
                # cross-attention cost if memory is tight -- at the price of
                # that coverage, and num_points below then yields 4 offsets per
                # height anchor instead of 2 unless it drops to 8 as well.
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            # The second place num_cams has to be 3 (see above).
                            num_cams=3,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                # 16, not upstream's 8, BECAUSE num_points_in_pillar
                                # went 4 -> 8.  `num_points` is the TOTAL sampling
                                # points per head per level, and
                                # spatial_cross_attention.py:365-369 reshapes them as
                                #     (num_points // num_Z_anchors, num_Z_anchors)
                                # with num_Z_anchors == num_points_in_pillar.  So
                                # upstream's 8 points over 4 heights is 2 offsets per
                                # height anchor; leaving it at 8 with 8 heights would
                                # silently drop that to 1, trading sampling density
                                # for height coverage instead of adding height
                                # coverage.  16 keeps 2 per anchor.
                                # num_points MUST be an integer multiple of
                                # num_points_in_pillar or the .view() above raises
                                # (spatial_cross_attention.py:369 then asserts that
                                # the reshape was exact, i.e. the same divisibility
                                # requirement -- note its `num_points` is a rebound
                                # local meaning offsets-per-anchor, not this field).
                                # The cost of 8->16 has NOT been measured in this
                                # pass -- it needs a GPU run -- so no figure is
                                # quoted here; the earlier "+3.7% memory, +5.5%
                                # step time" was never reproduced.  The shape is
                                # certain though: num_all_points, and so the
                                # deformable-attention sampling work, is linear
                                # in num_points.
                                num_points=16,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            # pc_range widened by BEVFormer's own 10 m slack convention
            # ([-51.2, 51.2] -> [-61.2, 61.2] upstream), applied to a range
            # that is no longer symmetric.  The head sigmoid-maps every decoded
            # centre INSIDE pc_range already, so this box can only ever delete
            # valid detections: carrying the symmetric +-61.2 over, or
            # bevfusion's x_max=37 (sized for its 32 m grid), would silently
            # drop everything decoded beyond those x.
            post_center_range=[-10.0, -42.0, -10.0, 74.0, 42.0, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            # The second place num_classes has to be 7 (see the head above).
            num_classes=7),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            # row <-> bev_h <-> Y, col <-> bev_w <-> X.  Equal here, so a swap
            # would be undetectable; keep the pairing correct for the day a
            # non-square grid is tried.
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        # grid_size / out_size_factor are dead (no reader in the plugin); kept
        # at the NAVSIM-consistent 64 m / 0.2 m = 320 so they do not lie.
        grid_size=[320, 320, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNavsimDataset'
# Sensor blobs: <data_root>/<log_name>/CAM_*/<token>.jpg, i.e. the info pkl's
# relative data_path / lidar_path are joined against this.
data_root = 'data/navsim/'
# Info pickles live apart from the blobs (they are built per split and per
# converter flag set); same layout as bevfusion's info_root.
info_root = 'data/infos/'
file_client_args = dict(backend='disk')

# The three front cameras, in this order.  The order is the camera order the
# model sees, so it must match num_cams above and stay stable between training
# and evaluation.  NAVSIM also ships CAM_L1/R1/L2/R2/B0 for full 360 coverage.
camera_names = ['CAM_F0', 'CAM_L0', 'CAM_R0']

# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #
# nuScenes-STYLED, not nuScenes-comparable.  nuPlan boxes have no attributes,
# so tp_metrics has no 'attr_err' and the NDS weighted sum is divided by
# mean_ap_weight + 4 = 9 instead of 10.  NAVSIM NDS must never be quoted
# against a nuScenes NDS.
eval_cfg = dict(
    # Identical to the model's range: rear and far GT must not count as missed
    # for a model that was never asked to look there.
    point_cloud_range=point_cloud_range,
    # Per-axis [|x| limit, |y| limit] (CustomNavsimDataset._in_eval_range), not
    # nuScenes' single radius.  [64, 32] is exactly the grid, so every cell the
    # model can predict in is scored and nothing outside it is.  (bevfusion's
    # stale 50/40/30 radii were sized for its 32 m grid.)
    class_range=dict(
        vehicle=[64, 32],
        pedestrian=[64, 32],
        bicycle=[64, 32],
        traffic_cone=[64, 32],
        barrier=[64, 32],
        czone_sign=[64, 32],
        generic_object=[64, 32]),
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    min_recall=0.1,
    min_precision=0.1,
    mean_ap_weight=5,
    # No 'attr_err': see the NDS note above.
    tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    class_names=class_names,
    # Key spelling ('maping') matches the evaluator's lookup; do not "fix" it.
    err_name_maping=dict(
        trans_err='mATE',
        scale_err='mASE',
        orient_err='mAOE',
        vel_err='mAVE',
        attr_err='mAAE'),
    # Same projection test the (optional) training-time label filter uses, so
    # the model is not scored on objects its cameras never saw.  OFF by default
    # to match bevfusion's reference runs.  image_wh MUST be given explicitly:
    # NAVSIM does not centre the vertical principal point (cy = 560 on 1080
    # rows), so inferring the image size from the intrinsics is wrong.
    camera_fov=dict(
        enabled=False,
        min_visible_corners=1,   # 1 keeps boxes clipped by the image border
        margin=0.0,              # pixels of slack outside the border
        apply_within=None,       # None = filter everywhere in the grid
        image_wh=[1920, 1080]))

# ------------------------------------------------------------------ #
# Pipelines
# ------------------------------------------------------------------ #
# Image scale.  bevformer_base.py has NO resize stage -- it feeds nuScenes'
# native 1600x900 and only pads to 1600x928 (29 * 32).  NAVSIM ships
# 1920x1080, so a resize stage is ADDED here, sized to land on exactly the same
# model input:
#
#   1920 x 1080  --x0.8334-->  1600 x 900  --pad/32-->  1600 x 928
#
# Both sizes are exactly 16:9, which is why one scalar reproduces the upstream
# geometry byte for byte and the pretrained backbone keeps its scale
# statistics.
# RandomScaleImageMultiViewImage truncates the resulting size with int(), but
# 5/6 is NOT the trap: IEEE-754 rounds it UP to 0.8333333333333334, so
# 1920 * (5/6) == 1600.0 and 1080 * (5/6) == 900.0 exactly -- the same 1600x900
# that 0.8334 gives (1600.128 -> 1600; 900.072 -> 900).  The literal decimal is
# written out so the intent survives a 4-decimal copy: it is 0.8333 that
# truncates, to 1599x899.  The two scales do differ in lidar2img, which is
# pre-multiplied by the scale, but by at most 0.128 px horizontally and 0.072 px
# vertically -- measured over 200 navtest frames (2,000,000 BEV cells) that
# changes the blind-cell verdict for 0 cells.
_img_scale_ = [0.8334]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # Camera-FOV label filter, OFF by default (see eval_cfg.camera_fov).  It
    # must sit HERE: it projects the eight box corners through
    # results['lidar2img'], so it has to run on the raw, unscaled calibration
    # and before any image/BEV augmentation, which also makes the label set
    # identical in every epoch.  image_wh is required -- never inferred from
    # ori_shape or from the intrinsics.
    dict(type='ObjectFOVFilter',
         enabled=eval_cfg['camera_fov']['enabled'],
         min_visible_corners=eval_cfg['camera_fov']['min_visible_corners'],
         margin=eval_cfg['camera_fov']['margin'],
         apply_within=eval_cfg['camera_fov']['apply_within'],
         image_wh=eval_cfg['camera_fov']['image_wh']),
    # Mandatory, not optional: the head sigmoid-maps every prediction inside
    # point_cloud_range, so GT outside it would be an unreachable target.
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=_img_scale_),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        # Bookkeeping only (MultiScaleFlipAug3D never resizes images here); set
        # to the STORED NAVSIM size so the record is honest.  The real resize
        # is the RandomScaleImageMultiViewImage below, which must run before
        # PadMultiViewImage -- hence Pad lives inside this block, unlike
        # bevformer_base.py where it sits at the top level.
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=_img_scale_),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    # 1 sample per GPU, as upstream: one sample is a queue of `queue_length`
    # frames x 3 cameras at 1600x928, so the effective image batch is 12.
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=info_root + 'navsim_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        camera_names=camera_names,
        modality=input_modality,
        eval_cfg=eval_cfg,
        test_mode=False,
        # MUST be True on every split.  NAVSIM has no per-box LiDAR point
        # count, so the converter writes num_lidar_pts = -1 everywhere; with
        # use_valid_flag=False the dataset would fall back to num_lidar_pts > 0
        # and return zero GT for every single frame.
        use_valid_flag=True,
        with_velocity=True,
        # Which frames may share a prev_bev chain.  'log_name' (the default)
        # gives maximum temporal continuity; 'scene_token' restricts it to
        # NAVSIM's 20 s scenes, the closest analogue of a nuScenes scene.
        # Either way CustomNavsimDataset re-validates each hop's dt, so a gap
        # inside a log still starts a fresh segment and resets prev_bev.
        temporal_key='log_name',
        # Applied to the CURRENT frame only: prepare_train_data tests
        # `pos == last and self.filter_empty_gt`, and history frames contribute
        # nothing but prev_bev.  So the resample probability IS the
        # empty-current-frame rate and is INDEPENDENT of queue_length -- there
        # is no 1-(1-p)^n compounding here, unlike upstream
        # CustomNuScenesDataset, which tests every frame in the queue.
        # Measured over this front-only 64 m ROI: 4.78347% of navtest in-split
        # samples (12,146) and 3.03036% of navtrain (103,288) are empty, hence
        # resampled, at ANY queue_length.  The 32 m ROI would make that
        # 8.13437% / 5.57374%.
        filter_empty_gt=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        # NAVSIM boxes are already in the mmdet3d LiDAR convention
        # (dx, dy, dz = width, length, height; yaw = -heading - pi/2).
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=info_root + 'navsim_infos_val.pkl',
             pipeline=test_pipeline,
             bev_size=(bev_h_, bev_w_),
             classes=class_names,
             camera_names=camera_names,
             modality=input_modality,
             eval_cfg=eval_cfg,
             # Set explicitly: bevformer_base.py inherits these two from
             # custom_nus-3d.py, which this config deliberately does not.
             test_mode=True,
             use_valid_flag=True,
             with_velocity=True,
             temporal_key='log_name',
             queue_length=queue_length,
             box_type_3d='LiDAR',
             samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=info_root + 'navsim_infos_val.pkl',
              pipeline=test_pipeline,
              bev_size=(bev_h_, bev_w_),
              classes=class_names,
              camera_names=camera_names,
              modality=input_modality,
              eval_cfg=eval_cfg,
              test_mode=True,
              use_valid_flag=True,
              with_velocity=True,
              temporal_key='log_name',
              queue_length=queue_length,
              box_type_3d='LiDAR'),
    # Group-aware sampler: CustomNavsimDataset sets the group flag over its
    # SAMPLE list (the official navtrain tokens), not over the frame pool.
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# Same checkpoint file as bevformer_base.py, but three things this comment
# used to get wrong, all verified against the file and a built model:
#   - CHECK WHAT YOUR FILE IS.  The ckpts/r101_dcn_fcos3d_pretrain.pth that was
#     measured here did not come from the upstream URL (it was copied from
#     another checkout on this machine), and its meta.exp_name reads
#     pgd_r101_caffe_fpn_gn-head_dcn_2x16_1x_nus-mono3d_prior_finetune.py --
#     mmdet3d 0.12.0, epoch 20, 10 nuScenes classes -- i.e. PGD, not FCOS3D.
#     The 2D backbone + neck are what transfer either way, so nothing is
#     broken, but "FCOS3D pretrain" is a filename here, not a provenance, and
#     the counts below are for THAT file.  Verify yours with
#     torch.load(p, map_location='cpu')['meta']['exp_name'].
#   - The FPN NECK transfers in full too, not just the backbone: all 676
#     img_backbone tensors (44,410,342 params) AND all 14 img_neck tensors
#     (3,278,592 params) load, i.e. 690 tensors / 47,688,934 params = 76.16% of
#     this model's 62,613,657.  Not an accident of shape: PGD's
#     in_channels=[256,512,1024,2048] with start_level=1 builds the same three
#     laterals as in_channels=[512,1024,2048] here, and only its 5th FPN level
#     (num_outs=5 vs 4) is dropped.
#   - There are ZERO shape mismatches.  The checkpoint's head lives under
#     bbox_head.* and this model's under pts_bbox_head.*, so the keys never
#     collide: mmcv reports them as UNEXPECTED keys, not as size mismatches.
#     Expect "unexpected key in source state_dict: bbox_head..." together with
#     "missing keys in source state_dict: pts_bbox_head...".  All 399
#     pts_bbox_head tensors (14,924,723 params) start from init.
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
