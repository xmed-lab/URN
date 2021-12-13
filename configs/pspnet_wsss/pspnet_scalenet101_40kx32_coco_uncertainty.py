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
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.8)),
    auxiliary_head=dict(
        in_channels=1024,
        loss_decode=dict(pus_type='clamp', pus_beta=0.5, pus_k=0.8))
    )
test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480), crf=True,
                pred_output_path='data/coco2014/voc_format/urn_s101_coco',
                scales=[0.4,0.5,0.6,2,3,4])
data = dict(test=dict(split='voc_format/train.txt'))
