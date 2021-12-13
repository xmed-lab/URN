_base_ = './pspnet_r50-d8_20k_voc12aug.py'
model = dict(
    pretrained='data/models/res2net101_v1b_26w_4s-0812c246.pth',
    backbone=dict(
        type='Res2Net',
        layers=[3, 4, 23, 3],
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_eval=False)
)
test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(480, 480), crf=True,
                pred_output_path='data/voc12/VOC2012/cyclic_mask_r2n')
data = dict(test=dict(split='ImageSets/Segmentation/trainaug.txt'))
