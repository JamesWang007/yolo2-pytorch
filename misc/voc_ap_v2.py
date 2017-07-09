import os
import cv2
import numpy as np
import pickle

from darknet_v2 import Darknet19
import utils.yolo_v2 as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
from cfgs.config_v2 import add_cfg


def eval_net(net, cfg, imdb, max_per_image=300, thresh=0.001, output_dir=None, vis=False):
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


def voc_ap(model, cfg):
    imdb_name = 'voc_2007_test'
    output_dir = 'models/testing/' + cfg['exp_name']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imdb = VOCDataset(imdb_name, '../data', cfg['batch_size'],
                      yolo_utils.preprocess_test, processes=4, shuffle=False, dst_size=cfg['inp_size'])

    net = Darknet19(cfg)
    net_utils.load_net(model, net)

    net.cuda()
    net.eval()

    mAP = eval_net(net, cfg, imdb, max_per_image=300, thresh=0.001, output_dir=output_dir, vis=False)

    imdb.close()
    return mAP


def voc_ap_main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_voc.yaml'
    exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712/voc0712_baseline_v3_rand.yaml'

    cfg = dict()
    add_cfg(cfg, dataset_yaml)
    add_cfg(cfg, exp_yaml)

    epoch = 160

    model_dir = cfg['train_output_dir']
    model_name = cfg['exp_name']
    model = model_dir + '/' + model_name + '_' + str(epoch) + '.h5'
    # model = '/home/cory/yolo2-pytorch/models/yolo-voc.weights.h5'
    print(model)
    voc_ap(model, cfg)

    # baseline_v3 rand
    # 0.5663  10
    # 0.6560  20
    # 0.7197  40
    # 0.7207  80
    # 0.7203  120
    # 0.7230  160
    # 0.7221  200

    # pre-trained
    # 0.7359


if __name__ == '__main__':
    voc_ap_main()
