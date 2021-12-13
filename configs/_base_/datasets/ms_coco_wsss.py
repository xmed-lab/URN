_base_ = './pascal_voc12.py'
dataset_type = 'COCODataset'
data_root = 'data/coco2014/'

data = dict(
    samples_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='voc_format/ppmg',
        split='voc_format/train.txt',
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='voc_format/class_labels',
        #split='voc_format/val_mini.txt', # eval mini, then test val
        split='voc_format/val.txt',
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='voc_format/class_labels',
        split='voc_format/val.txt',
        ))
