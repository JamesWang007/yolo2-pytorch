import matplotlib.pyplot as plt
import cv2
import numpy as np
from subprocess import check_output


def shift_filter(feature, flow):
    # feature shape = (batch, filters, h, w)
    shifted_feature = list()
    for feat in feature:
        print(feat.shape)
        for i in range(feat.shape[0]):
            act2d = feat[i, ...]
            act2d = act2d[:, :, np.newaxis]
            res = warp_flow(act2d, flow)
            shifted_feature.append(res)

            if False:
                print('act2d', act2d.shape, sum(act2d.ravel()))
                print('flow', flow.shape, sum(flow.ravel()))
                plt.figure(11)
                plt.imshow(act2d[:, :, 0], cmap='gray')
                plt.figure(12)
                plt.imshow(flow[..., 0], cmap='gray')
                plt.figure(13)
                plt.imshow(flow[..., 1], cmap='gray')
                plt.figure(14)
                plt.imshow(res, cmap='gray')
                plt.show()
                pass

    return np.asarray([shifted_feature])


def spynet_flow(image_path1, image_path2):
    flow_cmd = './run_flow.sh ' + image_path1 + ' ' + image_path2
    check_output([flow_cmd], shell=True)
    flow = np.load('./flow.npy')
    flow = flow.transpose(1, 2, 0)
    # flow.shape should be (height, width, 2)
    return flow


def dis_flow(img_path1, img_path2):
    out = check_output(['./run_of.sh ' + img_path1 + ' ' + img_path2], shell=True)
    # print(out)
    with open('flow.flo', 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        print('Reading %d x %d flo file' % (w, h))
        data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.reshape(data, (h, w, 2))
        return data2D


def get_flow_for_filter(flow):
    filter_map_height = 11
    filter_map_width = 39
    flow_ratio_y = flow.shape[0] / filter_map_height
    flow_ratio_x = flow.shape[1] / filter_map_width
    flow_small = np.asarray([cv2.resize(src=flow[:, :, 0] / flow_ratio_y,
                                        dsize=(filter_map_width, filter_map_height),
                                        interpolation=cv2.INTER_CUBIC),
                             cv2.resize(src=flow[:, :, 1] / flow_ratio_x,
                                        dsize=(filter_map_width, filter_map_height),
                                        interpolation=cv2.INTER_CUBIC)])
    flow_small = flow_small.transpose([1, 2, 0])
    print('flow_small.shape', flow_small.shape)
    return flow_small


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_map = flow.copy()
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(src=img, map1=flow_map, map2=None, interpolation=cv2.INTER_CUBIC)
    return res
