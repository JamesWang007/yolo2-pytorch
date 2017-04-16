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

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
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
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # data loader
    imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test, processes=2, shuffle=False, dst_size=cfg.inp_size)

    net = Darknet19()
    use_default = False
    if use_default:
        trained_model = cfg.trained_model
    else:
        trained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_4.h5')

    # 0.7186  default (yolo-voc.weights.h5)
    # 0.6040  default + FT conv5  epoch 1 trained with VOC12
    # 0.6335  default + FT conv5  epoch 1 trained with VOC07

    # SGD  lr = 1e-3
    # 0.3669  default + FT conv5  epoch 20 trained with VOC07
    # 0.3887  default + FT conv5  epoch 50 trained with VOC07
    # 0.4342  default + FT conv5  epoch 80 trained with VOC07
    # 0.4506  default + FT conv5  epoch 125 trained with VOC07

    # Adam lr = 1e-3
    # 0.5790  default + FT conv5  epoch 20 trained with VOC07
    # 0.5633  default + FT conv5  epoch 50 trained with VOC07

    # SGD  lr = 1e-4
    # 0.6400  default + FT conv5  epoch 1 trained with VOC07
    # 0.6382  default + FT conv5  epoch 5 trained with VOC07
    # 0.6449  default + FT conv5  epoch 20 trained with VOC07
    # 0.6376  default + FT conv5  epoch 70 trained with VOC07

    # Adam lr = 1e-5
    # 0.6370  default + FT conv5  epoch 20 trained with VOC07
    # 0.6473  default + FT conv5  epoch 50 trained with VOC07
    # 0.6542  default + FT conv5  epoch 110 trained with VOC07

    net_utils.load_net(trained_model, net)
    print(trained_model)

    net.cuda()
    net.eval()

    test_net(net, imdb, max_per_image, thresh, vis)

    imdb.close()
