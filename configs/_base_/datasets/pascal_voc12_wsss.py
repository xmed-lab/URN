_base_ = './pascal_voc12.py'
data_root = 'data/voc12/VOC2012/'

data = dict(
    samples_per_gpu=2,
    train=dict(
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='ppmg',
        split='ImageSets/Segmentation/trainaug.txt',
        ),
    test=dict(
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        ))
