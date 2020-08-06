#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 python tools/extract_boxes_scores.py \
--im_in_out_json your "your path" \
--saved_model_path "pth/res50_faster_rcnn_iter_1190000.pth"