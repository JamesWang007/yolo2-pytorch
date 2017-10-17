import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, im_data):
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

        return bbox_pred, iou_pred, class_pred

    def get_feature_map(self, im_data, layer='conv5'):
        conv1s = self.conv1s(im_data)
        if layer == 'conv1s':
            return conv1s

        conv2 = self.conv2(conv1s)
        if layer == 'conv2':
            return conv2

        conv3 = self.conv3(conv2)
        if layer == 'conv3':
            return conv3

        conv1s_reorg = self.reorg(conv1s)
        if layer == 'conv1s_reorg':
            return conv1s_reorg

        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)
        if layer == 'cat_1_3':
            return cat_1_3

        conv4 = self.conv4(cat_1_3)
        if layer == 'conv4':
            return conv4

        conv5 = self.conv5(conv4)  # batch_size, out_channels, h, w
        if layer == 'conv5':
            return conv5

        return None