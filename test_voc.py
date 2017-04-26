import os
import cv2
import torch
import numpy as np
import pickle

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test(image, cfg.inp_size), 0)
    return image, im_data


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.001
vis = False
# ------------


def test_net(net, imdb, max_per_image=300, thresh=0.5, vis=False):
    num_images = imdb.num_images

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):

        batch = imdb.next_batch()
        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=True, volatile=True)
        im_data = im_data.permute(0, 3, 1, 2)

        _t['im_detect'].tic()
        bbox_pred, iou_pred, prob_pred = net.forward(im_data)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, ori_im.shape, cfg, thresh)
        detect_time = _t['im_detect'].toc()
        _t['misc'].tic()

        for j in range(imdb.num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 100 == 0:
            print(i, end=' ')
            import sys
            sys.stdout.flush()
            # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis:
            im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
            cv2.imshow('test', im2show)
            cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    mAP = imdb.evaluate_detections(all_boxes, output_dir)
    return mAP


def test_ap_exp(model):
    print(model)
    imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test, processes=4, shuffle=False, dst_size=cfg.inp_size)

    net = Darknet19()
    net_utils.load_net(model, net)

    net.cuda()
    net.eval()

    mAP = test_net(net, imdb, max_per_image, thresh, vis)

    imdb.close()
    return mAP


if __name__ == '__main__':
    test_ap_exp('/home/cory/yolo2-pytorch/models/training/voc0712_trainval_exp2/voc0712_trainval_exp2_150.h5')

    # 0.7186  default (yolo-voc.weights.h5)
    # 0.6802  epoch_2 ADAM lr-6
    # 0.6754  epoch_4 ADAM lr-6
    # 0.6685  epoch_6 ADAM lr-6
    # 0.6682  epoch_8 ADAM lr-6

    # 0.7176  epoch_1 SGD lr-6
    # 0.7153  epoch_2
    # 0.7141  epoch_4
    # 0.7121  epoch_6

    # 0.4531  epoch_1 ADAM lr-3 New conv3_4_5
    # 0.4778  epoch_2
    # 0.4973  epoch_3
    # 0.5186  epoch_8
    # 0.5434  epoch_10
    # 0.5507  epoch_11
    # 0.5443  epoch_12
    # 0.5243  epoch_15
    # 0.5180  epoch_20
    # 0.5311  epoch_22
    # 0.5510  epoch_25
    # 0.5558  epoch_27
    # 0.5499  epoch_30
    # 0.5332  epoch_32
    # 0.5573  epoch_36
    # 0.5550  epoch_52
    # 0.5719  epoch_54
    # 0.5580  epoch_56
    # 0.5621  epoch_58
    # 0.5489  epoch_62


    # exp2, SGD lr=1E-3
    # 0.0594  0
    # 0.1241  1
    # 0.1820  2
    # 0.6354  91
