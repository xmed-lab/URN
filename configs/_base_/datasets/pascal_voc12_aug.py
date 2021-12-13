_base_ = './pascal_voc12.py'
# dataset settings
data = dict(
    samples_per_gpu=2,
    train=dict(
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ]))
