#!/usr/bin/env bash

PARTITION=$1
DATASET=$2
ARCH=$3

PORT=${PORT:-29500}

if [ ${DATASET} == voc12 ]
then
    if [ ${ARCH} == res38 ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_wres38-d8_20k_voc12aug_pus.py work_dirs/voc12_res38_pus 8
        python tools/pick_best_pth.py work_dirs/voc12_res38_pus
        ckpts=(`find work_dirs/voc12_res38_pus -name "best*"`)
        bash tools/slurm_test.sh ${PARTITION} python configs/pspnet_wsss/pspnet_wres38-d8_20k_voc12aug_cp.py ${ckpts[0]} 8
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_wres38-d8_20k_voc12aug.py work_dirs/voc12_res38_cp 8
    elif [ ${ARCH} == r2n ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_res2net_20k_voc12aug_pus.py work_dirs/voc12_r2n_pus 8
        python tools/pick_best_pth.py work_dirs/voc12_r2n_pus
        ckpts=(`find work_dirs/voc12_r2n_pus -name "best*"`)
        bash tools/slurm_test.sh ${PARTITION} python configs/pspnet_wsss/pspnet_res2net_20k_voc12aug_cp.py ${ckpts[0]} 8
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_res2net_20k_voc12aug.py work_dirs/voc12_r2n_cp 8
    elif [ ${ARCH} == s101 ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_scalenet101_20k_voc12aug_pus.py work_dirs/voc12_s101_pus 8
        python tools/pick_best_pth.py work_dirs/voc12_s101_pus
        ckpts=(`find work_dirs/voc12_s101_pus -name "best*"`)
        bash tools/slurm_test.sh ${PARTITION} python configs/pspnet_wsss/pspnet_scalenet101_20k_voc12aug_cp.py ${ckpts[0]} 8
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_scalenet101_20k_voc12aug.py work_dirs/voc12_s101_cp 8
    else
        echo architecture false
    fi
elif [ ${DATASET} == coco14 ]
then
    if [ ${ARCH} == res38 ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_wres38-d8_40kx32_coco.py work_dirs/coco14_res38_pus 16
    elif [ ${ARCH} == r2n ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_res2net-d8_40kx32_coco.py work_dirs/coco14_r2n_pus 16
    elif [ ${ARCH} == s101 ]
    then
        bash tools/slurm_train.sh ${PARTITION} python configs/pspnet_wsss/pspnet_scalenet101_40kx32_coco.py work_dirs/coco14_s101_pus 16
    else
        echo architecture false
    fi
else
    echo false dataset
fi
