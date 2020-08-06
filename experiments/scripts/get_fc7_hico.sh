#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 python tools/get_fc7_nms_vcoco.py \
--saved_model_path "pth/res152_faster_rcnn_iter_1190000.pth" \
--subset   "train" \
--type  "human" \
--file_dir "your_path/hico_152" \
--image_path "your_path/hico_20160224_det/images/"

# subset: train, test
# type: human, object, union
# file_dir: the dir of hoi_candidates_{subset}.hdf5