from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cython_bbox import bbox_ious, anchor_intersections
from utils.cython_yolo import yolo_to_bbox

import utils.network as net_utils
from layers.reorg.reorg_layer import ReorgLayer


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(net_utils.Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))
                # layers.append(net_utils.Conv2d(in_channels, out_channels, ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


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
    bbox_np[:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_np[:, :, 1::2] *= float(inp_size[1])  # rescale y

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


class Darknet19(nn.Module):
    def __init__(self, cfg):
        super(Darknet19, self).__init__()
        self.cfg = cfg

        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

        # darknet
        self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])

        stride = 2
        self.reorg = ReorgLayer(stride=2)  # stride*stride times the channels of conv1s
        # cat [conv1s, conv3]
        self.conv4, c4 = _make_layers((c1 * (stride * stride) + c3), net_cfgs[7])

        # linear
        out_channels = cfg['num_anchors'] * (cfg['num_classes'] + 5)
        self.conv5 = net_utils.Conv2d(c4, out_channels, 1, 1, relu=False)

        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.class_loss = None
        self.pool = Pool(processes=10)

    @property
    def loss(self):
        return self.bbox_loss + self.iou_loss + self.class_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, inp_size=None):
        conv1s = self.conv1s(im_data)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)  # batch_size, out_channels, h, w

        # for detection
        # bsize, c, h, w -> bsize, h, w, c -> bsize, h x w, num_anchors, 5+num_classes
        bsize, _, h, w = conv5.size()
        conv5_reshaped = conv5.permute(0, 2, 3, 1).contiguous().view(bsize, -1, self.cfg['num_anchors'],
                                                                     self.cfg['num_classes'] + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        # [batch, cell, anchor, prediction]
        xy_pred_raw = conv5_reshaped[:, :, :, 0:2]
        wh_pred_raw = conv5_reshaped[:, :, :, 2:4]
        bbox_pred_raw = torch.cat([xy_pred_raw, wh_pred_raw], 3)
        iou_pred_raw = conv5_reshaped[:, :, :, 4:5]

        xy_pred = F.sigmoid(xy_pred_raw)
        wh_pred = torch.exp(wh_pred_raw)
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = F.sigmoid(iou_pred_raw)

        class_pred_raw = conv5_reshaped[:, :, :, 5:].contiguous()
        class_pred = F.softmax(class_pred_raw.view(-1, self.cfg['num_classes'])).view_as(class_pred_raw)

        debug = False
        if debug:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            class_pred_raw_np = class_pred_raw.data.cpu().numpy()
            class_pred_np = class_pred.data.cpu().numpy()
            print(np.max(bbox_pred_np), np.max(iou_pred_np), np.max(class_pred_raw_np), np.max(class_pred_np))

        # for training
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            iou_pred_np = iou_pred.data.cpu().numpy()
            _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = self._build_target(
                bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, inp_size)

            _boxes = net_utils.np_to_variable(_boxes)
            _ious = net_utils.np_to_variable(_ious)
            _classes = net_utils.np_to_variable(_classes)
            box_mask = net_utils.np_to_variable(_box_mask, dtype=torch.FloatTensor)
            iou_mask = net_utils.np_to_variable(_iou_mask, dtype=torch.FloatTensor)
            class_mask = net_utils.np_to_variable(_class_mask, dtype=torch.FloatTensor)

            num_boxes = sum((len(boxes) for boxes in gt_boxes))
            box_mask = box_mask.expand_as(_boxes)
            class_mask = class_mask.expand_as(class_pred)

            self.bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes
            self.iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes
            self.class_loss = nn.MSELoss(size_average=False)(class_pred * class_mask, _classes * class_mask) / num_boxes

        return bbox_pred, iou_pred, class_pred

    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, iou_pred_np, inp_size):
        """
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) : (sig(tx), sig(ty), exp(tw), exp(th))
        """
        bsize = bbox_pred_np.shape[0]
        data = [(bbox_pred_np[b], gt_boxes[b], gt_classes[b], iou_pred_np[b], inp_size, self.cfg) for b in range(bsize)]
        targets = self.pool.map(_process_batch, data)

        _boxes = np.stack(tuple((row[0] for row in targets)))
        _ious = np.stack(tuple((row[1] for row in targets)))
        _classes = np.stack(tuple((row[2] for row in targets)))
        _box_mask = np.stack(tuple((row[3] for row in targets)))
        _iou_mask = np.stack(tuple((row[4] for row in targets)))
        _class_mask = np.stack(tuple((row[5] for row in targets)))

        return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask
