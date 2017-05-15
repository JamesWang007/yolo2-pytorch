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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    # model = '/home/cory/yolo2-pytorch/models/training/voc0712_new_2/voc0712_new_2_160.h5'
    # model = cfg.trained_model
    # model = '/home/cory/yolo2-pytorch/models/training/voc0712_ft/voc0712_ft_5.h5'
    # model = '/home/cory/yolo2-pytorch/models/training/voc0712_mask_val/voc0712_mask_val_14.h5'
    model = '/home/cory/yolo2-pytorch/models/training/voc0712_anchor/voc0712_anchor_94.h5'
    # model = '/home/cory/yolo2-pytorch/models/training/voc0712_template/voc0712_template_55.h5'

    test_ap_exp(model)

    # 0.6479  my trained 200 epoch
    # 0.6724  epoch 10
    # 0.6762  epoch 19
    # 0.6742  epoch 20
    # 0.6754  epoch 24
    # 0.6763  epoch 27
    # 0.6788  epoch 31
    # 0.6823  epoch 38
    # 0.6825  epoch 43
    # 0.6809  epoch 45
    # 0.6811  epoch 46

    # 0.6458  template 50
    # 0.6614  template 55

    # 0.6613  anchor 4
    # 0.6624  anchor 8
    # 0.6627  anchor 9
    # 0.6639  anchor 10
    # 0.6631  anchor 11
    # 0.6627  anchor 12
    # 0.6639  anchor 26
    # 0.7025  anchor 94

    # 0.7221  pre-trained
    # 0.7235  ft 1
    # 0.7227  ft 2
    # 0.7173  ft 5
    # 0.7160  ft 10
    # 0.7143  ft 28

    # voc0712_new
    # 0.6486  97
    # 0.6459  110
    # 0.6477  118
    # 0.6449  119

    # voc0712_new_2
    # 0.6714  133
    # 0.6717  150
    # 0.6727  155
    # 0.6727  158
    # 0.6731  160 *
    # 0.6728  162
    # 0.6727  163
    # 0.6717  164
    # 0.6721  168
    # 0.6722  171
    # 0.6725  180
    # 0.6725  190
    # 0.6725  200

    # batch_size = 16, hi-res, lr = 1e-6
    # 0.6683  201
    # 0.6711  202
    # 0.6714  203
    # 0.6713  204
    # 0.6723  205
    # 0.6712  206

    # lr = 1e-5
    # 0.6713  161
    # 0.6699  162
    # 0.6729  163
    # 0.6714  164
    # 0.6734  165
    # 0.6738  166
    # 0.6691  168
    # 0.6731  169
    # 0.6719  200
    # 0.6727  250

