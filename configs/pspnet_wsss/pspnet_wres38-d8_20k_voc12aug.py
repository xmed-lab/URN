_base_ = './pspnet_r50-d8_20k_voc12aug.py'
model = dict(
    pretrained='data/models/res38d.pth',
    backbone=dict(
        type='WideRes38'),
    decode_head=dict(
        in_channels=4096),
    auxiliary_head=dict(
        in_channels=1024)
    )
data = dict(train=dict(ann_dir='cyclic_mask'))
