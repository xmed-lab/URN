_base_ = './pspnet_r50-d8_20k_voc12aug.py'
model = dict(
    pretrained='data/models/res38d.pth',
    backbone=dict(
        type='WideRes38'),
    decode_head=dict(
        in_channels=4096,
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.5)),
    auxiliary_head=dict(
        in_channels=1024,
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.5))
    )
