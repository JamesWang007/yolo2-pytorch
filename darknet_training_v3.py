import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils.cython_bbox import bbox_ious, anchor_intersections
from utils.cython_yolo import yolo_to_bbox

import utils.network as net_utils


def restore_gt_numpy(labels):
    labels_np = labels.numpy()
    gt_boxes = list()
    gt_classes = list()
    for i in range(labels_np.shape[0]):
        label = labels_np[i, ...]
        valid_id = np.where(label[:, 0] == 1)
        valid_len = np.count_nonzero(label[:, 0] == 1)
        gt_boxes.append(label[valid_id, 4:8].reshape(valid_len, -1))
        g = label[valid_id, 1][0].tolist()
        g = list(map(int, g))
        gt_classes.append(g)
    return gt_boxes, gt_classes


def _process_batch(data):
    bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, inp_size, cfg = data
    out_size = inp_size / 32
    num_gt = gt_boxes.shape[0]

    cell_w = 32
    cell_h = 32

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape
    # hw = num_cell

    # gt
    _classes = np.zeros([hw, num_anchors, cfg['num_classes']], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)
    # _class_mask = np.ones([hw, num_anchors, 1], dtype=np.float) * cfg['class_scale']

    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    # _boxes[:, :, 0:2] = 0.5
    # _boxes[:, :, 2:4] = 1.0
    # debug mask_val
    # _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    # scale pred_bbox
    anchors = np.ascontiguousarray(cfg['anchors'], dtype=np.float)
    bbox_pred_np = np.expand_dims(bbox_pred_np, 0)
    bbox_np = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
        anchors,
        out_size[1], out_size[0])
    bbox_np = bbox_np[0]  # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1
    bbox_np[:, :, 0::2] *= float(inp_size[0])  # rescale x by w
    bbox_np[:, :, 1::2] *= float(inp_size[1])  # rescale y by h

    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)

    # for each cell, compare predicted_bbox and gt_bbox
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(
        np.ascontiguousarray(bbox_np_b, dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)
    # _iou_mask[best_ious < cfg['iou_thresh']] = cfg['noobject_scale'] * 1
    iou_penalty = 0 - iou_pred_np[best_ious < cfg['iou_thresh']]
    _iou_mask[best_ious < cfg['iou_thresh']] = cfg['noobject_scale'] * iou_penalty
    ious_reshaped = np.reshape(ious, [hw, num_anchors, num_gt])

    # locate the cell of each gt_boxes
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * out_size[0] + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx  (0 ~ 1)
    target_boxes[:, 1] = cy - np.floor(cy)  # cy  (0 ~ 1)
    target_boxes[:, 2] = (gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) / cell_w  # tw
    target_boxes[:, 3] = (gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) / cell_h  # th

    # for each gt boxes, match the best anchor
    # gt_boxes_resize = [(xmin, ymin, xmax, ymax)] unit: cell px
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] /= cell_w
    gt_boxes_resize[:, 1::2] /= cell_h

    anchor_ious = anchor_intersections(
        anchors,
        np.ascontiguousarray(gt_boxes_resize, dtype=np.float)
    )
    anchor_inds = np.argmax(anchor_ious, axis=0)

    # for every gt cell
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            print('warning: invalid cell_ind, cx, cy, W, H', cell_ind, cx[i], cy[i], out_size[0], out_size[1])
            continue
        a = anchor_inds[i]

        # do not evaluate for dontcare / unknown class
        if gt_classes[i] == -1:
            continue

        iou_pred = iou_pred_np[cell_ind, a, :]  # 0 ~ 1, should be close to iou_truth
        iou_truth = ious_reshaped[cell_ind, a, i]
        _iou_mask[cell_ind, a, :] = cfg['object_scale'] * (iou_truth - iou_pred)
        _ious[cell_ind, a, :] = iou_truth

        truth_w = (gt_boxes_b[i, 2] - gt_boxes_b[i, 0]) / inp_size[0]
        truth_h = (gt_boxes_b[i, 3] - gt_boxes_b[i, 1]) / inp_size[1]
        _box_mask[cell_ind, a, :] = cfg['coord_scale'] * (2 - truth_w * truth_h)
        target_boxes[i, 2:4] /= anchors[a]
        _boxes[cell_ind, a, :] = target_boxes[i]

        _class_mask[cell_ind, a, :] = cfg['class_scale']
        _classes[cell_ind, a, gt_classes[i]] = 1.

    # _boxes[:, :, 2:4] = np.maximum(_boxes[:, :, 2:4], 0.001)
    # _boxes[:, :, 2:4] = np.log(_boxes[:, :, 2:4])

    # _boxes = (sig(tx), sig(ty), exp(tw), exp(th))
    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


def _build_target(cfg, bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, inp_size):
    """
    :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) : (sig(tx), sig(ty), exp(tw), exp(th))
    """
    bsize = bbox_pred_np.shape[0]
    data = [(bbox_pred_np[b], gt_boxes[b], gt_classes[b], iou_pred_np[b], inp_size, cfg) for b in range(bsize)]
    targets = list(map(_process_batch, data))
    _boxes = np.stack(tuple((row[0] for row in targets)))
    _ious = np.stack(tuple((row[1] for row in targets)))
    _classes = np.stack(tuple((row[2] for row in targets)))
    _box_mask = np.stack(tuple((row[3] for row in targets)))
    _iou_mask = np.stack(tuple((row[4] for row in targets)))
    _class_mask = np.stack(tuple((row[5] for row in targets)))

    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


def training_target(cfg, bbox_pred, class_pred, labels, inp_size, iou_pred):
    # inp_size = (w, h)
    gt_boxes, gt_classes = restore_gt_numpy(labels)
    bbox_pred_np = bbox_pred.data.cpu().numpy()
    iou_pred_np = iou_pred.data.cpu().numpy()
    _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = _build_target(
        cfg, bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, inp_size)

    _boxes = net_utils.np_to_variable(_boxes)
    _ious = net_utils.np_to_variable(_ious)
    _classes = net_utils.np_to_variable(_classes)
    box_mask = net_utils.np_to_variable(_box_mask, dtype=torch.FloatTensor)
    iou_mask = net_utils.np_to_variable(_iou_mask, dtype=torch.FloatTensor)
    class_mask = net_utils.np_to_variable(_class_mask, dtype=torch.FloatTensor)

    num_boxes = sum((len(boxes) for boxes in gt_boxes))
    box_mask = box_mask.expand_as(_boxes)
    class_mask = class_mask.expand_as(class_pred)

    bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
    iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes
    class_loss = nn.MSELoss(size_average=False)(class_pred * class_mask, _classes * class_mask) / num_boxes
    return bbox_loss, iou_loss, class_loss


def debug_and_vis(data):
    inputs, labels = data
    gt_boxes, gt_classes = restore_gt_numpy(labels)

    for i in range(inputs.size()[0]):
        img = inputs[i].numpy().transpose(1, 2, 0)
        # img = img[::-1]
        for b in gt_boxes[i]:
            bb = list(map(int, b))
            print(bb)
            img[bb[1]:bb[3], bb[0], :] = 1.
            img[bb[1]:bb[3], bb[2], :] = 1.
            img[bb[1], bb[0]:bb[2], :] = 1.
            img[bb[3], bb[0]:bb[2], :] = 1.
        plt.imshow(img)
        plt.show()

    print(inputs.size())
    print(labels.size())
