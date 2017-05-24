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


def test_voc_ap(model):
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
    # voc0712_ft,
    model_dir = '/home/cory/yolo2-pytorch/models/training'
    exp_candidates = [('voc0712_template', 15),
                      ('voc0712_low_lr', 90),
                      ('voc0712_box_mask', 40),
                      ('voc0712_anchor', 100),
                      ('voc0712_baseline', 20),
                      ('voc0712_multiple_anchors', 199),
                      ('voc0712_one_anchor', 120)]
    choice = 6
    exp = exp_candidates[choice]
    model = model_dir + '/' + exp[0] + '/' + exp[0] + '_' + str(exp[1]) + '.h5'
    test_voc_ap(model)

    # one anchor (1:1 anchor)
    # 0.6212  20
    # 0.7003  37
    # 0.6962  40
    # 0.6988  45
    # 0.7017  60
    # 0.7018  70
    # 0.7027  80
    # 0.7033  120
    # 0.7033  160

    # voc_template
    # 0.5063  5
    # 0.5800  10
    # 0.6421  20
    # 0.6318  25
    # 0.6441  30
    # 0.6458  50
    # 0.6614  55
    # 0.7040  87
    # 0.7058  91
    # 0.7055  95
    # 0.7061  96
    # 0.7073  97
    # 0.7069  100 *
    # 0.7084  101
    # 0.7066  102
    # 0.7074  104
    # 0.7059  105
    # 0.7063  112
    # 0.7051  130
    # 0.7075  134
    # 0.7078  160
    # 0.7091  180
    # 0.7102  181
    # 0.7065  190
    # 0.7119  198
    # 0.7099  199

    # baseline (low_lr)
    # 0.6981  25
    # 0.6997  30 *
    # 0.7025  35
    # 0.7119  100
    # 0.7115  115

    # multiple anchors
    # 0.5556  20
    # 0.6515  30
    # 0.6725  40
    # 0.6723  50
    # 0.6729  70
    # 0.6969  153
    # 0.6958  199

    # low_lr (copy box_mask epoch 20)
    # 0.6911 21
    # 0.6934 22
    # 0.7002 25
    # 0.7009 27
    # 0.6996 28
    # 0.7033 30 *
    # 0.7077 40
    # 0.7078 42
    # 0.7072 43
    # 0.7077 45
    # 0.7082 48
    # 0.7051 50
    # 0.7090 52
    # 0.7039 55
    # 0.7063 58
    # 0.7063 60
    # 0.7073 70
    # 0.7091 80
    # 0.7059 90
    # 0.7069 100

    # 0.7221  pre-trained

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

