_base_ = './pspnet_r50-d8_20k_voc12aug.py'
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
        in_channels=2048),
    auxiliary_head=dict(
        in_channels=1024)
    )
data = dict(train=dict(ann_dir='cyclic_mask_s101'))
