import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def imshow_fig(img, title='', **kwargs):
    h = img.shape[0]
    w = img.shape[1]
    dpi = 96
    fig = plt.figure(num=0, figsize=(w / dpi, h / dpi))
    fig.add_axes([0., 0., 1., 1.])
    fig.canvas.set_window_title(title)
    plt.imshow(img, **kwargs)
    plt.axis('off')
    return fig


def plot_feature_map(features, border=2, resize_ratio=2):
    num_channel = features.shape[1]
    feat_h = features.shape[2]
    feat_w = features.shape[3]
    map_border_num = int(math.ceil(math.sqrt(num_channel)))
    map_h = (feat_h + border) * map_border_num
    map_w = (feat_w + border) * map_border_num
    # print('create act map {:d} x {:d}'.format(map_h, map_w))
    feature_map_all = np.zeros((map_h, map_w))

    # print(features.shape)
    all_sum = 0
    idx = 0
    max_val = np.max(features.ravel())
    for i_y in range(0, map_h, feat_h+border):
        for i_x in range(0, map_w, feat_w+border):
            if idx >= num_channel:
                break
            act = features[0, idx, :, :]
            idx += 1
            if border != 0:
                act_pad = np.lib.pad(array=act,
                                     pad_width=((0, border), (0, border)),
                                     mode='constant',
                                     constant_values=max_val/6)
            else:
                act_pad = act
            feature_map_all[i_y: i_y + feat_h + border, i_x: i_x + feat_w + border] = act_pad
            act_sum = sum(act.ravel())
            all_sum += act_sum
            # print('filter-{:d}  act_sum={:f}'.format(idx, act_sum))

    # print('all_sum = {:f}'.format(all_sum))
    # min max normalization
    feature_map_all /= feature_map_all.max()
    feature_map_all = cv2.resize(feature_map_all, (feature_map_all.shape[1] * resize_ratio,
                                                   feature_map_all.shape[0] * resize_ratio))
    return feature_map_all


def draw_hsv(flow, ratio=4):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = v * ratio
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
