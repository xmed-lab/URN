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
test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480), crf=True,
                pred_output_path='data/voc12/VOC2012/urn_wr38',
                scales=[0.15,0.2,0.25,4,5,6])
data = dict(test=dict(split='ImageSets/Segmentation/trainaug.txt'))
