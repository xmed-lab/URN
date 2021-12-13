_base_ = [
    '../_base_/models/pspnet_r50_wsss.py',
    '../_base_/datasets/ms_coco_wsss.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=91), auxiliary_head=dict(num_classes=91))
