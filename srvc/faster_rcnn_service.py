#! /usr/bin/env python2.7
# -*- coding: utf-8
# file: faster_rcnn_service.py
# author: Thomas Wood
# email: thomas@synpon.com
# description: A service to run faster-rcnn.pytorch on an input image.

from __future__ import print_function

# Import _init_paths to set up sys.path to import modules from faster-rcnn.
import _init_paths
import os
import os.path as osp
import json
import time
from pprint import pprint

# Import numerical libraries.
import numpy as np
import cv2
import torch
from scipy.misc import imread

# Import modules from faster-rcnn.pytorch.
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms
from model.utils.net_utils import vis_detections


def _save_array(filename, arr):
    """
    """
    with open(filename,'wb') as f:
        np.save(f, arr)

def _load_array(filename):
    """
    """
    with open(filename, 'rb') as f:
        return np.load(f)

def _get_image_blob(im, cfg):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    # print(cfg.PIXEL_MEANS)
    im_orig -= cfg.PIXEL_MEANS[::-1]
    im_orig /= 255.
    # print(im_orig)

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
          im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("image", im)
        # time.sleep(4)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _set_dirnames():
    """Gets directory names for the project.
    Arguments:
      None
    Returns:
      srvc_dir (string): name of the current directory containing this file.
      fstrrcnn_dir (string): name of the parent directory of faster-rcnn.pytorch
    """
    srvc_dir = os.getcwd()
    fstrrcnn_dir = osp.dirname(srvc_dir)
    return srvc_dir, fstrrcnn_dir

def _get_cfg():
    """Gets the configuration for the resnet101 yaml file.
    Arguments:
      None
    Returns:
      cfg (config object): a configuration for the resnet
    """
    _, fstrrcnn_dir = _set_dirnames()
    cfg_file = osp.join(fstrrcnn_dir, "cfgs", "res101.yml")
    cfg_from_file(cfg_file)

def _cuda_available():
    """Wrapper for torch.cuda.is_available
    Arguments:
      None
    Returns:
      torch.cuda.is_available (bool): a boolean value showing availability of cuda.
    """
    return torch.cuda.is_available


def _get_classes(class_name="pascal"):
    """Returns output classes of the object classifier.
    Arguments:
      class_name (string): name of the dataset to use classes for.
    Returns:
      classes (np.array): array of strings with class labels.
    """
    pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    if class_name == "pascal":
        return pascal_classes


def _get_model(classes):
    """Returns output classes of the object classifier.
    Arguments:
      class_name (string): name of the dataset to use classes for.
    Returns:
      classes (np.array): array of strings with class labels.
    """
    fasterRCNN = resnet(classes, 101, pretrained=True,class_agnostic=None)
    srvc_dir, fstrrcnn_dir = _set_dirnames()
    # Change to faster-rcnn.pytorch directory to load fasterRCNN.
    os.chdir(fstrrcnn_dir)
    fasterRCNN.create_architecture()
    os.chdir(srvc_dir)
    # if _cuda_available():
    #     fasterRCNN.cuda()
    fasterRCNN.eval()
    return fasterRCNN

def _init_tensors():
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # if _cuda_available():
    #     im_data = im_data.cuda()
    #     im_info = im_info.cuda()
    #     num_boxes = num_boxes.cuda()
    #     gt_boxes = gt_boxes.cuda()

    return im_data, im_info, num_boxes, gt_boxes

def _get_images_from_dir():
    """Collect images from input directory.
    Arguments:
      None
    Returns:
      image_dir (string): String of directory containing input images.
      imagelist (list): List of strings with input image file names.
    """
    srvc_dir, fstrrcnn_dir = _set_dirnames()
    image_dir = osp.join(fstrrcnn_dir, "new_images")
    return image_dir, os.listdir(image_dir)

def _process_image(im_file, cfg):
    """Collect images from input directory and insert data into torch tensors.
    Arguments:
      im_file (string): Location of input image.
    Returns:
      Returns:
        im_data (torch.FloatTensor): image data
        im_info (torch.FloatTensor): image information
        num_boxes (torch.LongTensor): number of boxes
        gt_boxes (torch.FloatTensor): ground truth boxes
      """
    im_in = np.array(imread(im_file))
    # print(im_file)
    # cv2.imshow("image", im_in)
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :,::-1]
    # im = im_in
    blobs, im_scales = _get_image_blob(im, cfg)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data, im_info, num_boxes, gt_boxes = _init_tensors()
    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()
    return im_data, im_info, gt_boxes, num_boxes, im_scales, im

def _collect_regions(model, im_data, gt_boxes, mode="align"):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    base_feats = model.RCNN_base(im_data)
    if mode == "align":
        pooled_feat = model.RCNN_roi_align(base_feat, gt_boxes.view(-1, 5))
    elif mode == "pooling":
        pooled_feat = model.RCN_roi_pool(base_feat, gt_boxes.view(-1, 5))
    return pooled_feat.numpy()

def _run_model_infer(model, im_data, im_info, gt_boxes, num_boxes):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    with torch.no_grad():
        return model(im_data, im_info, gt_boxes, num_boxes)

def _run_model_gt(model, im_data, gt_boxes, mode="align"):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    with torch.no_grad():
        return _collect_regions(model, im_data, mode=mode)

def _process_images_infer(image_dir, imagelist, model, cfg):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    res = []
    for image in imagelist:
        im_file = osp.join(image_dir, image)
        im_data, im_info, gt_boxes, num_boxes, im_scales, im = \
            _process_image(im_file, cfg)
        # print(im.shape)
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label  \
        = _run_model_infer(
            model, im_data, im_info, gt_boxes, num_boxes
        )
        r = {
                "image_file": im_file,
                "im_info": im_info,
                "im": im,
                "bbox_pred": bbox_pred,
                "rois": rois,
                "cls_prob": cls_prob,
                "rpn_loss_cls": rpn_loss_cls,
                "rpn_loss_box": rpn_loss_box,
                "RCNN_loss_cls": RCNN_loss_cls,
                "RCNN_loss_bbox": RCNN_loss_bbox,
                "rois_label": rois_label,
                "im_scales": im_scales
            }
        res.append(r)
    return res


def _process_images_gt(image_dir, imagelist, model, cfg):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    res = []
    for image in imagelist:
        im_file = osp.join(image_dir, image)
        im_data, im_info, gt_boxes, num_boxes = _process_image(im_file, cfg)


def _nms(cls_prob, rois, bbox_pred, im, im_info, im_scales, classes, vis, thresh, cfg):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      cls_prob (torch.FloatTensor): class probabilities
      rois
      bbox_pred
      im_scales
      classes
      vis, thresh
      cfg
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    print(boxes.shape)
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # if args.cuda > 0:
            #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
            #                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            # else:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    if vis:
        im2show = np.copy(im)
        # cv2.imshow("image", im2show)
        time.sleep(2)
    for j in xrange(1, len(classes)):
        inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          cls_boxes = pred_boxes[inds][:, 4*j:4*(j+1)]

          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
          cls_dets = cls_dets[order]
          # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          u = cls_dets.cpu().numpy()
          _save_array("cls_dets_srvc_{}.npy".format(classes[j]), u)
          print("{} : max_prob, class - {}".format(np.max(u[:,-1]), classes[j]))
          if vis:
            # print("visualizing")
            im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.5)
    im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", im2showRGB)
    # time.sleep(2)
    cv2.imwrite( "result.jpg", im2show)
            # cv2.imshow("frame", im2showRGB)
            # imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # cv2.imshow("image", imRGB)
            # time.sleep(10)

def _main(serialize=False):
    """Initialize input tensors to fasterRCNN.
    Arguments:
      None
    Returns:
      im_data (torch.FloatTensor): image data
      im_info (torch.FloatTensor): image information
      num_boxes (torch.LongTensor): number of boxes
      gt_boxes (torch.FloatTensor): ground truth boxes
    """
    thresh = 0.05
    vis = True
    _get_cfg()
    pprint(cfg)
    classes = _get_classes()
    fasterRCNN = _get_model(classes)
    image_dir, imagelist = _get_images_from_dir()
    res = _process_images_infer(image_dir, imagelist, fasterRCNN, cfg)
    for r in res:
        cls_prob = r.get("cls_prob")
        rois = r.get("rois")
        bbox_pred = r.get("bbox_pred")
        im_scales = r.get("im_scales")
        im_info = r.get("im_info")
        im = r.get("im")
        _nms(cls_prob,
             rois,
             bbox_pred,
             im,
             im_info,
             im_scales,
             classes,
             vis,
             thresh,
             cfg)

    res_fname = "fasterRCNN-results.json"
    if serialize:
        with open(res_fname) as f:
            json.dump(res, f)
    # print(res)
    _compare_results(classes)
    return res

def _load_arrays(classes, fname):
    res = []
    for k in range(len(classes)):
        if classes[k] == "__background__":
            continue
        else:
            res.append(_load_array("{}_{}.npy".format(fname,classes[k])))
    return res

def _compare_results(classes):
    # _classes = classes[1:]
    srvc_res = _load_arrays(classes, "cls_dets_srvc")
    demo_res = _load_arrays(classes, "../cls_dets_demo")
    for k in range(len(classes)):
        print(demo_res[k] == srvc_res[k])


if __name__ == "__main__":
    _main()
