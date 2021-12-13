_base_ = './pspnet_r50-d8_20k_voc12aug.py'
model = dict(
    pretrained='data/models/resnet101-5d3b4d8f.pth',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        deep_stem=False,
        contract_dilation=True,
        frozen_stages=-1),
    decode_head=dict(
        loss_decode=dict(weight_thresh=0.05)),
    auxiliary_head=dict(
        loss_decode=dict(weight_thresh=0.05))
)
data = dict(train=dict(ann_dir='urn_r2n'))
