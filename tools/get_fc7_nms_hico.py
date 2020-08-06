#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import h5py

from model.nms_wrapper import nms
from model.test import _get_blobs
import argparse
from tqdm import tqdm
import numpy as np
import os, cv2

from nets.resnet_v1 import resnetv1
from torch.autograd import Variable
import torch


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.array(list(range(len(y2))))
    keep = []
    all_map = - np.ones(len(y2))
    while len(order) > 0:
        i = order[0]
        # print(i)
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr < thresh)[0]
        inds_1 = np.where(ovr >= thresh)[0]
        # print(len(inds), len(inds_1), len(ovr), len(y2))

        assert len(inds) + len(inds_1) == len(ovr)
        # print(order[inds_1])
        all_map[order[0]] = order[0]
        all_map[order[inds_1 + 1]] = order[0]

        order = order[inds + 1]

    # print(all_map)
    assert (all_map >= 0).all()

    return keep, all_map


def im_detect_union(net, im, person_boxes, object_boxes):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    person_boxes = np.array(person_boxes)
    object_boxes = np.array(object_boxes)
    union_boxes = []
    for i in range(len(person_boxes)):
        list = [0.0, min(person_boxes[i, 0], object_boxes[i, 0]),
                min(person_boxes[i, 1], object_boxes[i, 1]),
                max(person_boxes[i, 2], object_boxes[i, 2]),
                max(person_boxes[i, 3], object_boxes[i, 3])]
        union_boxes.append(list)

    fc7_U = torch.empty(0, 2048).cuda()
    union_boxes = np.array(union_boxes)
    union_boxes = union_boxes * im_scales[0]

    keep, all_map = py_cpu_nms(union_boxes, 1)
    union_boxes = union_boxes[keep]

    time = len(union_boxes) // 512
    for i in range(time):
        if i == time - 1:
            union_boxes_ = Variable(torch.FloatTensor(union_boxes[i * 512:, :])).contiguous().cuda().to(
                net._device)
            fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
            fc7_U = torch.cat((fc7_U, fc7_U_))
        else:
            union_boxes_ = Variable(torch.FloatTensor(union_boxes[i * 512:(i + 1) * 512, :])).contiguous().cuda().to(
                net._device)
            fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
            fc7_U = torch.cat((fc7_U, fc7_U_))

    if time == 0:
        union_boxes_ = Variable(torch.FloatTensor(union_boxes)).contiguous().cuda().to(
            net._device)
        fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
        fc7_U = torch.cat((fc7_U, fc7_U_))

    print(fc7_U.shape, len(keep))
    if fc7_U.shape[0] > 1:
        if (fc7_U[0].data.cpu().numpy() == fc7_U[1].data.cpu().numpy()).all():
            print("==")
        else:
            print("!=")
    else:
        print("fc7_U.shape[0] == 1")
    return fc7_U, keep, all_map


def im_detect(net, im, person_boxes):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    person_boxes = np.array(person_boxes)
    union_boxes = []
    for i in range(len(person_boxes)):
        list = [0, person_boxes[i, 0],
                person_boxes[i, 1],
                person_boxes[i, 2],
                person_boxes[i, 3]]
        union_boxes.append(list)

    fc7_U = torch.empty(0, 2048).cuda()
    union_boxes = np.array(union_boxes)
    union_boxes = union_boxes * im_scales[0]

    keep, all_map = py_cpu_nms(union_boxes, 1)
    union_boxes = union_boxes[keep]
    print(len(keep))

    time = len(union_boxes) // 1024
    for i in range(time):
        if i == time - 1:
            union_boxes_ = Variable(torch.FloatTensor(union_boxes[i * 1024:, :])).contiguous().cuda().to(
                net._device)
            fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
            fc7_U = torch.cat((fc7_U, fc7_U_))
        else:
            union_boxes_ = Variable(torch.FloatTensor(union_boxes[i * 1024:(i + 1) * 1024, :])).contiguous().cuda().to(
                net._device)
            fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
            fc7_U = torch.cat((fc7_U, fc7_U_))

    if time == 0:
        union_boxes_ = Variable(torch.FloatTensor(union_boxes)).contiguous().cuda().to(
            net._device)
        fc7_U_ = net.get_fc7(blobs['data'], blobs['im_info'], union_boxes_, avg_pooling=False)
        fc7_U = torch.cat((fc7_U, fc7_U_))

    # if len(keep) == 1:
    #     print(union_boxes)
    #     for i in range(len(person_boxes)):
    #         list = [0, min(person_boxes[i, 0], object_boxes[i, 0]),
    #                 min(person_boxes[i, 1], object_boxes[i, 1]),
    #                 max(person_boxes[i, 2], object_boxes[i, 2]),
    #                 max(person_boxes[i, 3], object_boxes[i, 3])]
    #         print(list)
    #     assert False

    if fc7_U.shape[0] > 1:
        if (fc7_U[0].data.cpu().numpy() == fc7_U[1].data.cpu().numpy()).all():
            print("==")
        else:
            print("!=")
    else:
        print("fc7_U.shape[0] == 1")
    return fc7_U, keep, all_map


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('--saved_model_path')
    parser.add_argument('--subset')
    parser.add_argument('--type')
    parser.add_argument('--file_dir')
    parser.add_argument('--image_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    saved_model_path = args.saved_model_path
    subset = args.subset
    type = args.type
    file_path = os.path.join(args.file_path, f'hoi_candidates_{subset}.hdf5')
    write_file_path = os.path.join(args.file_path, f'hoi_candidates_{type}_feats_{subset}.hdf5')
    ###########################################################################
    assert_err = 'Saved model file not found'
    assert (os.path.isfile(saved_model_path)), assert_err

    # Load network
    num = 152 if "152" in saved_model_path else 50
    net = resnetv1(num_layers=num)
    net.create_architecture(
        81,
        tag='default',
        anchor_scales=[4, 8, 16, 32])
    net.load_state_dict(torch.load(saved_model_path))
    print("loaded")
    net.eval()
    net.cuda()

    file = h5py.File(file_path, "r")
    print(len(file))

    union_file = h5py.File(write_file_path, "w")

    for global_id in tqdm(file.keys()):
        if global_id in union_file:
            continue
        data = file[global_id]["boxes_scores_rpn_ids_hoi_idx"][()]

        person_boxes = data[:, 0:4].tolist()
        object_boxes = data[:, 4:8].tolist()

        if "test" in global_id:
            image_path = args.image_path + "test2015/" + global_id + ".jpg"
        else:
            image_path = args.image_path +"train2015/" + global_id + ".jpg"
        assert os.path.exists(image_path)

        im = cv2.imread(image_path)
        if type == "human":
            fc7_U, keep, all_map = im_detect(net, im, person_boxes)
        elif type == "object":
            fc7_U, keep, all_map = im_detect(net, im, object_boxes)
        elif type == "union":
            fc7_U, keep, all_map = im_detect_union(net, im, person_boxes, object_boxes)
        fc = fc7_U.data.cpu().numpy()
        assert fc.shape[1] == 2048
        assert len(all_map) == len(person_boxes)

        assert fc.shape[0] == len(keep)
        union_file.create_group(global_id)
        union_file[global_id].create_dataset(
            'features',
            data=fc)

        keep_id_to_feature_idx = {item: index for index, item in enumerate(keep)}
        features_indexs = - np.ones(len(all_map))
        for i in range(len(all_map)):
            features_index = keep_id_to_feature_idx[all_map[i]]
            features_indexs[i] = features_index
        assert len(features_indexs) == len(all_map)
        assert (features_indexs >= 0).all()
        union_file[global_id].create_dataset(
            'indexs',
            data=features_indexs)

    union_file.close()
    file.close()
