_base_ = './pspnet_r50-d8_40kx32_coco.py'
model = dict(
    pretrained='data/models/res2net101_v1b_26w_4s-0812c246.pth',
    backbone=dict(
        type='Res2Net',
        layers=[3, 4, 23, 3],
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=False),
    decode_head=dict(
        loss_decode=dict(weight_thresh=0.05)),
    auxiliary_head=dict(
        loss_decode=dict(weight_thresh=0.05))
)
data = dict(train=dict(ann_dir='voc_format/urn_s101_coco'))

