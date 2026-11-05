# ============================================================
# NAVSIM (OpenScene / nuPlan) TINY configuration for BEVFormer
#
# THIS IS THE SMOKE / DEBUG CONFIG.  It exists to exercise the whole NAVSIM
# path end to end -- converter output -> CustomNavsimDataset -> queue ->
# PerceptionTransformer -> BEVFormerHead -> the NAVSIM evaluator -- on one GPU
# in minutes, not to produce a number worth reporting.  Everything here is
# chosen to be CHEAP.  Train bevformer_navsim_base.py for real results.
#
# Key differences from projects/configs/bevformer/bevformer_tiny.py are the
# same NAVSIM contract as bevformer_navsim_base.py -- read that file's header
# for the full list and the reasoning.  In one line each:
#   - no ../datasets/custom_nus-3d.py inheritance (nuScenes specific; mmcv
#     merges _base_ recursively and would leak its classes/paths in)
#   - CustomNavsimDataset, 7 nuPlan classes, 3 front cameras (num_cams=3 twice)
#   - front-only BEV [0, 64] x [-32, 32] m, z [-3, 5] (rear-axle origin)
#   - ego_axes='x_forward_y_left', rotate_center=[0.0, 25.0]
#   - num_points_in_pillar=8, post_center_range per the NAVSIM range
#   - a resize stage sized for NAVSIM's 1920x1080 blobs
#   - input_modality.use_external=False (can_bus ships inside the info pkl)
#
# And, exactly as bevformer_tiny.py is to bevformer_base.py, cheaper than
# bevformer_navsim_base.py in five ways:
#   smaller backbone:  R101-DCN      -> R50 (torchvision ImageNet pretrain)
#   smaller BEV:       100x100       -> 50x50     (1.28 m cells, still SQUARE)
#   fewer enc. layers: 6             -> 3
#   smaller input:     1600x928      -> 800x480
#   multi-scale FPN    4 levels (C3-C5) -> 1 level (C5)
#   shorter queue:     4 frames      -> 3 frames
# ============================================================

_base_ = [
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ------------------------------------------------------------------ #
# BEV region of interest -- identical to bevformer_navsim_base.py, so the two
# configs score over the same region and their numbers are comparable.
# ------------------------------------------------------------------ #
# [x_min, y_min, z_min, x_max, y_max, z_max] in the NAVSIM ego frame, which is
# also the LiDAR frame (lidar2ego is the identity in every NAVSIM log): +x
# forward, +y LEFT, +z up, z = 0 is the ground, origin = the rear axle.
# Front-only in x because the three front cameras see nothing behind the car;
# z is [-3, 5] rather than nuScenes' [-5, 3] because the origin moved from a
# roof LiDAR down to the rear axle on the ground.
point_cloud_range = [0.0, -32.0, -3.0, 64.0, 32.0, 5.0]

# Dead keys (no live reader in projects/mmdet3d_plugin), recomputed from the
# NAVSIM range anyway so they do not mislead: 64 m / 0.2 m = 320, z extent 8 m.
voxel_size = [0.2, 0.2, 8]

# Backbone-coupled, NOT dataset-coupled: torchvision RGB statistics, because
# this variant starts from the torchvision ImageNet R50.  Unchanged from
# bevformer_tiny.py on purpose (base uses Caffe BGR for its mono3D-pretrained
# R101-DCN).
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# nuPlan TrackedObjectType names exactly as they appear in the info pkl's
# gt_names.  The ORDER is the label order; eval_cfg.class_range below is keyed
# by the same names.
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
# Single FPN level (C5 only), as bevformer_tiny.py.
_num_levels_ = 1

# BEV grid.  The axis mapping is NOT symmetric:
#   bev_w_ (column index w) <-> physical X <-> pc_range[0] .. pc_range[3]
#   bev_h_ (row    index h) <-> physical Y <-> pc_range[1] .. pc_range[4]
#   query index q = h * bev_w + w
#
# 64 m forward / 64 m lateral over 50 x 50 -> SQUARE 1.28 m cells (base has
# 0.64 m).  Square matters: rotate_prev_bev rotates the previous BEV in image
# space, which is only a valid yaw rotation when the cells are isotropic.
# 2500 queries instead of 10000 is most of why this config is cheap.
bev_h_ = 50
bev_w_ = 50

# 3 frames = 1.0 s of history at NAVSIM's 2 Hz.  One frame shorter than base,
# matching bevformer_tiny.py: it cuts the per-sample image cost by a quarter.
queue_length = 3

# The ego's own cell, as torchvision.transforms.functional.rotate wants it:
# center = (x, y) = (col, row) from the top-left.
#   col = (0 - pc_range[0]) / real_w * bev_w = (0 -   0.0) / 64 * 50 =  0
#   row = (0 - pc_range[1]) / real_h * bev_h = (0 - -32.0) / 64 * 50 = 25
# The upstream default is a hard-coded [100, 100] that no config overrides --
# the ego cell of a 200x200 nuScenes grid.  bevformer_tiny.py is ALREADY broken
# by it: on a 50x50 grid [100, 100] is off-grid and the rotation retains ~0% of
# the previous BEV.  With a front-only ROI the ego sits on the x=0 edge, so the
# correct centre is the LEFT edge, half way up.
rotate_center_ = [0.0, 25.0]

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    # ImageNet R50, no mono3D-detection pretrain and no DCN: this variant is
    # for plumbing, not accuracy, so `load_from` stays at default_runtime.py's
    # None and there is nothing to download first.
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        # Left at the nuScenes value so the head shape matches base; the
        # decoder is not the bottleneck in this config.
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
            # a mismatch fails loudly, but set both anyway.
            num_cams=3,
            rotate_center=rotate_center_,
            # Which physical axis each BEV axis carries, used ONLY to split the
            # ego translation between the two BEV axes in get_bev_features.
            #   'y_forward_x_right'  nuScenes LIDAR_TOP (the upstream default):
            #                        lidar2ego is a ~-90 deg z-rotation, so
            #                        +y is forward and +x is right.
            #   'x_forward_y_left'   NAVSIM / nuPlan ego, which is the mmdet3d
            #                        LiDAR canonical frame: +x forward, +y left.
            # Measured on real NAVSIM poses with the 100x100 grid, the nuScenes
            # setting emits shift (0.00007, 0.10313) where the truth is
            # (0.10313, -0.00006): right magnitude, swapped axis, inverted
            # lateral sign -- wrong on every moving frame and completely
            # silent.  The rotate_prev_bev ANGLE sign needs no change.
            ego_axes='x_forward_y_left',
            encoder=dict(
                type='BEVFormerEncoder',
                # 3 layers, as bevformer_tiny.py.
                num_layers=3,
                pc_range=point_cloud_range,
                # 8, not 4.  The pillar sample heights are a linspace over
                # the z EXTENT, not fractions of D:
                #     Z  = z_max - z_min
                #     zs = linspace(0.5, Z - 0.5, D) + z_min
                # It is NOT z_min + (0.5 + i)/D * Z; the two forms agree only
                # when D == Z, which is exactly this shipped D=8 / Z=8 case.
                # Real heights:
                #   nuScenes [-5, 3], D=4 -> -4.500, -2.167,  0.167,  2.500
                #   NAVSIM   [-3, 5], D=4 -> -2.500, -0.167,  2.167,  4.500
                #   NAVSIM   [-3, 5], D=8 -> -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5 4.5
                # D=8's 1.0 m spacing makes its +-0.5 m bands tile [-3, 5]
                # contiguously: 99.9980% of in-ROI GT box centre heights land
                # within 0.5 m of a sample, against 35.5629% at D=4 (navtest,
                # 252,794 boxes over 12,146 in-split samples; navtrain,
                # 3,179,006 boxes: 99.9980% vs 31.5053%).  See the longer note
                # in bevformer_navsim_base.py.
                # SCA cost is LINEAR in D, so drop back to 4 if even this
                # config is too heavy -- at the price of that coverage, and
                # num_points below then yields 4 offsets per height anchor
                # instead of 2 unless it drops to 8 as well.
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
                            # The second place num_cams has to be 3.
                            num_cams=3,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                # 16 because num_points_in_pillar is 8: offsets per
                                # height anchor = num_points // num_points_in_pillar
                                # (spatial_cross_attention.py:365-369), so 8 here
                                # would give 1 per anchor instead of upstream's 2.
                                # See the longer note in bevformer_navsim_base.py.
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
            # The second place num_classes has to be 7.
            num_classes=7),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            # row <-> bev_h <-> Y, col <-> bev_w <-> X.  Equal here, so a swap
            # would be undetectable; keep the pairing correct anyway.
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
        # grid_size / out_size_factor are dead; kept NAVSIM-consistent.
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
# Info pickles live apart from the blobs; same layout as bevfusion's info_root.
info_root = 'data/infos/'
file_client_args = dict(backend='disk')

# The three front cameras, in this order.  The order is the camera order the
# model sees, so it must match num_cams above and stay stable between training
# and evaluation.  NAVSIM also ships CAM_L1/R1/L2/R2/B0 for full 360 coverage.
camera_names = ['CAM_F0', 'CAM_L0', 'CAM_R0']

# ------------------------------------------------------------------ #
# Evaluation -- identical to bevformer_navsim_base.py
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
# Image scale.  bevformer_tiny.py halves nuScenes' native 1600x900 to 800x450
# and pads to 800x480 (15 * 32).  NAVSIM ships 1920x1080, so the scale is
# re-expressed relative to THAT to land on exactly the same model input:
#
#   1920 x 1080  --x0.4167-->  800 x 450  --pad/32-->  800 x 480
#
# Both sizes are exactly 16:9, which is why one scalar reproduces the upstream
# geometry byte for byte.  RandomScaleImageMultiViewImage truncates the
# resulting size with int(), but 5/12 is NOT the trap: IEEE-754 rounds it UP to
# 0.4166666666666667, so 1920 * (5/12) == 800.0 and 1080 * (5/12) == 450.0
# exactly -- the same 800x450 that 0.4167 gives (800.064 -> 800;
# 450.036 -> 450).  The literal decimal is written out so the intent survives a
# 4-decimal copy: it is the truncation 0.41666 that gives 799x449.
_img_scale_ = [0.4167]

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
        # PadMultiViewImage.
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
    # frames x 3 cameras at 800x480, so the effective image batch is 9.
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
        # empty-current-frame rate and is INDEPENDENT of queue_length -- the
        # "~3.7% at queue_length=3" this comment used to quote was wrong twice
        # over: measured on a symmetric +-64 m region, then compounded across
        # the queue, which is upstream CustomNuScenesDataset's rule, not this
        # port's.  Measured over this front-only 64 m ROI: 4.78347% of navtest
        # in-split samples (12,146) and 3.03036% of navtrain (103,288) are
        # empty, hence resampled, at this or any other queue_length.
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
             # Set explicitly: bevformer_tiny.py inherits these two from
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
# 24 as upstream bevformer_tiny.py.  For an actual smoke run do not edit this:
# pass `--cfg-options total_epochs=1 runner.max_epochs=1 data.train.load_interval=50`
# so the config on disk stays the reference one.
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# No load_from: `pretrained` above pulls the torchvision R50 and
# ../_base_/default_runtime.py leaves load_from = None, so this config needs
# nothing in ckpts/ to start.
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
