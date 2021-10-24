from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def cal_iou_xyxy(box1,box2):
    x1min,y1min,x1max,y1max = box1[0],box1[1],box1[2],box1[3]
    x2min,y2min,x2max,y2max = box2[0],box2[1],box2[2],box2[3]
    area_box1 = (x1max-x1min+1.)*(y1max-y1min+1.)
    area_box2 = (x2max-x2min+1.)*(y2max-y2min+1.)
    #inter
    inter_x1 = max(x1min,x2min)
    inter_y1 = max(y1min,y2min)
    inter_x2 = min(y1max,y2max)
    inter_y2 = min(y1max,y2max)
    inter_area = (inter_x2-inter_x1+1.)*(inter_y2-inter_y1+1.)
    union = (area_box1+area_box2-inter_area)
    return inter_area/union

def bbox_iou(bbox_a,bbox_b):
    if bbox_a.shape[1]!=4 or bbox_b.shape[1]!=4:
        print(bbox_a,bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:,None,:2],bbox_b[:,:2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br-tl,axis=2) * (tl<br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc



