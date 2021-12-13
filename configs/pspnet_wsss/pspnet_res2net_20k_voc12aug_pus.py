_base_ = './pspnet_r50-d8_20k_voc12aug.py'
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
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.5)),
    auxiliary_head=dict(
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.5))
)
data = dict(train=dict(ann_dir='ppmg'))
