# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每个框 (bounding box) 的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 根据得分 (score) 的大小进行降序排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 保留剩余框中得分最高的那个
        i = order[0]
        keep.append(i)

        # 计算相交区域位置，左上以及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交区域面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # 计算IoU，即 重叠面积 / (框1面积 + 框2面积 - 重叠面积)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的框
        inds = np.where(ovr <= thresh)[0]

        # 因为ovr数组的长度比order长度小1，所以这里要将所有下标后移一位
        order = order[inds + 1]

    return keep
