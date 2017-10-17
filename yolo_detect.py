import numpy as np
import cv2
import os
from darknet_v3 import Darknet19
from cfgs.config_v2 import load_cfg_yamls
import utils.network as net_utils
import utils.yolo_v2 as yolo_utils

base_dir = './'


def init_network():
    dataset_yaml = os.path.join(base_dir, 'cfgs/config_kitti_demo.yaml')
    # exp_yaml = os.path.join(base_dir, 'cfgs/exps/kitti/kitti_baseline_v3.yaml')
    cfg = load_cfg_yamls([dataset_yaml])

    model = os.path.join(base_dir, 'models/training/kitti_new_2/kitti_new_2_100.h5')
    net = Darknet19(cfg)
    net_utils.load_net(model, net)
    net.eval()
    net.cuda()
    print('load model successfully')
    return net, cfg


def load_image_paths(img_list_file):
    img_files = open(img_list_file)
    image_paths = [f.strip() for f in img_files.readlines()]
    return image_paths


def preprocess(filename, inp_size):
    image = cv2.imread(filename)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, inp_size))[0], 0)
    return image, im_data


def detect_image(cfg, image_path, net, thresh):
    image, im_data = preprocess(image_path, cfg['inp_size'])
    im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
    bbox_pred, iou_pred, prob_pred = net.forward(im_data)
    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()
    bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)
    return bboxes, cls_inds, image, scores


def run():
    net, cfg = init_network()
    image_paths = load_image_paths(os.path.join('./demo/', 'demo_images_list.txt'))

    thresh = 0.6
    for i, image_path in enumerate(image_paths):
        bboxes, cls_inds, image, scores = detect_image(cfg, image_path, net, thresh)

        im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)

        cv2.imshow('detection', im2show)
        key = cv2.waitKey(100)
        if key == ord('q'):
            break


if __name__ == '__main__':
    run()
