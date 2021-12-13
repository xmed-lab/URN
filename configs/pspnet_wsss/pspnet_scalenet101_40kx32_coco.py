_base_ = './pspnet_r50-d8_40kx32_coco.py'
model = dict(
    pretrained='data/models/scalenet/weights/scalenet101.pth',
    backbone=dict(
        type='ScaleNet',
        layers=[3, 4, 23, 3],
        structure='data/models/scalenet/structures/scalenet101.json',
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=False),
    decode_head=dict(
        in_channels=2048,
        loss_decode=dict(pus_type='clamp', pus_beta=0.8, pus_k=0.8)),
    auxiliary_head=dict(
        in_channels=1024,
        loss_decode=dict(pus_type='clamp', pus_beta=0.8, pus_k=0.8))
    )

# 512 out of memory, so reduce to 448
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 448), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(448, 448), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    train=dict(pipeline=train_pipeline)
)
