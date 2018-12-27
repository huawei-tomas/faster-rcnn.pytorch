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

# Import numerical libraries.
import numpy as np
import cv2
import torch
from scipy.misc import imread

# Import modules from faster-rcnn.pytorch.
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

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
    return cfg_from_file(cfg_file)

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

def _process_image(im_file):
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
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, np.newaxis]
        im_in = np.concatenate((im_in, im_in, im_in), axis=2)
    # rgb -> bgr
    im = im_in[:, :,::-1]

    blobs, im_scales = _get_image_blob(im)
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
    return im_data, im_info, gt_boxes, num_boxes

def _collect_regions(model, im_data, gt_boxes, mode="align"):
    base_feats = model.RCNN_base(im_data)
    if mode == "align":
        pooled_feat = model.RCNN_roi_align(base_feat, gt_boxes.view(-1, 5))
    elif mode == "pooling":
        pooled_feat = model.RCN_roi_pool(base_feat, gt_boxes.view(-1, 5))
    return pooled_feat.numpy()

def _run_model_infer(model, im_data, im_info, gt_boxes, num_boxes):
    with torch.no_grad():
        return model(im_data, im_info, gt_boxes, num_boxes)

def _run_model_gt(model, im_data, gt_boxes, mode="align"):
    with torch.no_grad():
        return _collect_regions(model, im_data, mode=mode)

def _process_images_infer(image_dir, imagelist, model):
    res = []
    for image in imagelist:
        im_file = osp.join(image_dir, image)
        im_data, im_info, gt_boxes, num_boxes = _process_image(im_file)
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label  \
        = _run_model_infer(
            model, im_data, im_info, gt_boxes, num_boxes
        )
        r = {
                "image_file": im_file,
                "rois": rois,
                "cls_prob": cls_prob,
                "rpn_loss_cls": rpn_loss_cls,
                "rpn_loss_box": rpn_loss_box,
                "RCNN_loss_cls": RCNN_loss_cls,
                "RCNN_loss_bbox": RCNN_loss_bbox,
                "rois_label": rois_label
            }
        res.append(r)
    return res


def _process_images_gt(image_dir, imagelist, model):
    res = []
    for image in imagelist:
        im_file = osp.join(image_dir, image)
        im_data, im_info, gt_boxes, num_boxes = _process_image(im_file)




def _main(serialize=False):
    cfg = _get_cfg()
    classes = _get_classes()
    fasterRCNN = _get_model(classes)
    image_dir, imagelist = _get_images_from_dir()
    res = _process_images_infer(image_dir, imagelist, fasterRCNN)
    res_fname = "fasterRCNN-results.json"
    if serialize:
        with open(res_fname) as f:
            json.dump(res, f)
    print(res)
    return res

if __name__ == "__main__":
    _main()
