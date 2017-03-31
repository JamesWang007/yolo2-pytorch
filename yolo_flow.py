import os
import numpy as np
import cv2
import torch

import cfgs.config as cfg
import utils.network as net_utils
import utils.yolo as yolo_utils
from darknet import Darknet19
from plot_util import *
from flow_util import *
from utils.timer import Timer
from torch.autograd import Variable


def preprocess(filename):
    image = cv2.imread(filename)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
    return image, im_data


def main():
    trained_model = cfg.trained_model
    thresh = 0.5
    image_dir = '/home/cory/KITTI_Dataset/data_tracking_image_2/training/image_02/0019'

    net = Darknet19()
    net_utils.load_net(trained_model, net)
    net.eval()
    net.cuda()
    print('load model successfully')
    print(net)

    image_extensions = ['.jpg', '.JPG', '.png', '.PNG']
    image_abs_paths = sorted([os.path.join(image_dir, name)
                              for name in os.listdir(image_dir)
                              if name[-4:] in image_extensions])

    key_frame_path = ''
    detection_period = 5
    use_flow = True

    t_det = Timer()
    t_total = Timer()

    for i, image_path in enumerate(image_abs_paths):
        t_total.tic()
        image, im_data = preprocess(image_path)
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)

        # key frame
        if i % detection_period == 0:
            key_frame_path = image_path
            # conv5 feature map
            conv5_feat_gpu = net.get_feature_map(im_data=im_data, layer='conv5')
            conv5_feat = conv5_feat_gpu.data.cpu().numpy()
            feature_map_all = plot_feature_map(conv5_feat, resize_ratio=1)
            cv2.imshow('feature_map', feature_map_all)
            cv2.imwrite('output/feature_map/{:04d}.jpg'.format(i), feature_map_all * 255)

        t_det.tic()
        if use_flow:
            # flow = spynet_flow(image_path, key_frame_path)
            flow = dis_flow(image_path, key_frame_path)
            print('flow sum =', sum(flow.ravel()))

            flow_rgb = draw_hsv(flow)
            cv2.imwrite('output/flow/flow_{:04d}.jpg'.format(i), flow_rgb)

            flow_feat = get_flow_for_filter(flow)
            flow_feat_hsv = draw_hsv(flow_feat, ratio=50)
            cv2.imwrite('output/flow_feat/flow_{:04d}.jpg'.format(i), flow_feat_hsv)
            print('flow_feat sum =', sum(flow_feat.ravel()))

            img_warp = warp_flow(image, flow)
            #cv2.imshow('img_warp', img_warp)
            cv2.imwrite('output/warp/warp_{:04d}.jpg'.format(i), img_warp)

            conv5_shifted = shift_filter(conv5_feat, flow_feat)
            feat_warp = plot_feature_map(conv5_shifted, resize_ratio=10) * 255.
            cv2.imwrite('output/feat_warp/warp_{:04d}.jpg'.format(i), feat_warp)

            conv5_shifted_gpu = torch.FloatTensor(conv5_shifted).cuda()
            bbox_pred, iou_pred, prob_pred = net.feed_feature(Variable(conv5_shifted_gpu))

        else:
            bbox_pred, iou_pred, prob_pred = net.forward(im_data)

        det_time = t_det.toc()

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)

        im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)

        if im2show.shape[0] > 1100:
            im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
        cv2.imshow('detection', im2show)
        cv2.imwrite('output/detection/{:04d}.jpg'.format(i), im2show)

        total_time = t_total.toc()
        format_str = 'frame: %d, (detection: %.1f fps, %.1f ms) (total: %.1f fps, %.1f ms)'
        print(format_str % (
            i, 1. / det_time, det_time * 1000, 1. / total_time, total_time * 1000))

        t_det.clear()
        t_total.clear()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        '''if i == 0:
            # wait user press any key to start
            cv2.waitKey(0)'''

if __name__ == '__main__':
    main()
