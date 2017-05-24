import os
import shutil
import time

import torch

os.environ['DATASET'] = 'kitti'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cfgs.config as cfg
import utils.network as net_utils
import utils.yolo as yolo_utils
from darknet import Darknet19
from plot_util import *
from misc.flow_util import *
from utils.timer import Timer
from torch.autograd import Variable


def preprocess(filename):
    image = cv2.imread(filename)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
    return image, im_data


def detection_objects(bboxes, scores, cls_inds):
    objects = list()
    for i in range(len(bboxes)):
        box = bboxes[i]
        score = scores[i]
        label = cfg.label_names[cls_inds[i]]
        objects.append((box, score, label))
    return objects


def save_as_kitti_format(frame_id, det_obj, kitti_filename, src_label='voc'):
    # 0 -1 car 0 0 0 1078 142 1126 164 0 0 0 0 0 0 0.415537
    with open(kitti_filename, 'a') as file:
        for det in det_obj:
            bbox = det[0]
            score = det[1]
            label = det[2]
            if src_label == 'voc':
                if label != 'car' and label != 'person':
                    continue
                label = label.replace('person', 'pedestrian')

            line_str = '{:d} -1 {:s} 0 0 0 {:d} {:d} {:d} {:d} 0 0 0 0 0 0 {:.4f}\n' \
                .format(frame_id, label, bbox[0], bbox[1], bbox[2], bbox[3], score)
            # print(line_str)
            file.write(line_str)


def detect_by_flow(frame_index,  conv5_feat, current_frame, image_path, key_frame_path):
    t = time.time()
    flow = spynet_flow(image_path, key_frame_path)
    # flow = dis_flow(image_path, key_frame_path)
    # print('flow sum =', sum(flow.ravel()))
    # print('flow', time.time() - t)

    t = time.time()
    flow_rgb = draw_hsv(flow)
    cv2.imwrite('output/flow/flow_{:04d}.jpg'.format(frame_index), flow_rgb)
    # print('flow_rgb', time.time() - t)

    t = time.time()
    feat_size = cfg.inp_size[1] // 32, cfg.inp_size[0] // 32
    flow_feat = get_flow_for_filter(flow, feat_size)
    flow_feat_hsv = draw_hsv(flow_feat, ratio=50)
    cv2.imwrite('output/flow_feat/flow_{:04d}.jpg'.format(frame_index), flow_feat_hsv)
    # print('flow_feat sum =', sum(flow_feat.ravel()))
    # print('flow_feat_hsv', time.time() - t)

    t = time.time()
    img_warp = warp_flow(current_frame, flow)
    # cv2.imshow('img_warp', img_warp)
    cv2.imwrite('output/warp/warp_{:04d}.jpg'.format(frame_index), img_warp)
    # print('warp_flow', time.time() - t)

    t = time.time()
    conv5_shifted = shift_filter(conv5_feat, flow_feat)
    # print('shift_filter', time.time() - t)
    t = time.time()
    # feat_warp = plot_feature_map(conv5_shifted, resize_ratio=10) * 255.
    # cv2.imwrite('output/feat_warp/warp_{:04d}.jpg'.format(frame_index), feat_warp)
    # print('feat_warp', time.time() - t)

    t = time.time()
    conv5_shifted_gpu = torch.FloatTensor(conv5_shifted).cuda()
    # print('conv5_shifted_gpu', time.time() - t)
    return conv5_shifted_gpu


def main():

    shutil.rmtree('output', ignore_errors=True)
    shutil.copytree('output_template', 'output')

    # trained_model = cfg.trained_model
    # trained_model = '/home/cory/yolo2-pytorch/models/yolo-voc.weights.h5'
    # trained_model = '/home/cory/yolo2-pytorch/models/training/kitti_new_2/kitti_new_2_100.h5'
    trained_model = '/home/cory/yolo2-pytorch/models/training/kitti_baseline/kitti_baseline_165.h5'
    # trained_model = '/home/cory/yolo2-pytorch/models/training/kitti_new_2_flow_ft/kitti_new_2_flow_ft_2.h5'
    # trained_model = '/home/cory/yolo2-pytorch/models/training/voc0712_obj_scale/voc0712_obj_scale_1.h5'
    # trained_model = '/home/cory/yolo2-pytorch/models/training/kitti_det_new_2/kitti_det_new_2_40.h5'
    # trained_model = '/home/cory/yolo2-pytorch/models/training/kitti_det_new_2/kitti_det_new_2_10.h5'
    thresh = 0.5
    use_kitti = True

    # car = 1 5
    # pedestrian = 13 17

    net = Darknet19()
    net_utils.load_net(trained_model, net)
    net.eval()
    net.cuda()
    print('load model successfully')
    # print(net)

    # img_files = open('/home/cory/yolo2-pytorch/train_data/kitti/kitti_val_images.txt')
    img_files = open('/home/cory/yolo2-pytorch/train_data/kitti/0001_images.txt')
    # img_files = open('/home/cory/yolo2-pytorch/train_data/ImageNetVID_test.txt')
    # img_files = open('/home/cory/yolo2-pytorch/train_data/vid04_images.txt')
    image_abs_paths = img_files.readlines()
    image_abs_paths = [f.strip() for f in image_abs_paths]

    key_frame_path = ''
    detection_period = 1
    use_flow = False

    kitti_filename = 'yolo_flow_kitti_det.txt'
    try:
        os.remove(kitti_filename)
    except OSError:
        pass

    t_det = Timer()
    t_total = Timer()

    for i, image_path in enumerate(image_abs_paths):
        t_total.tic()
        t0 = time.time()
        image, im_data = preprocess(image_path)
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
        t1 = time.time()
        print('t1', t1 - t0)

        layer_of_flow = 'conv4'
        # key frame
        if use_flow and i % detection_period == 0:
            key_frame_path = image_path
            # conv5 feature map
            feature = net.get_feature_map(im_data=im_data, layer=layer_of_flow)
            feature = feature.data.cpu().numpy()
            feature_map_all = plot_feature_map(feature, resize_ratio=1)
            # cv2.imshow('feature_map', feature_map_all)
            cv2.imwrite('output/feature_map/{:04d}.jpg'.format(i), feature_map_all * 255)

        t_det.tic()
        if use_flow:
            conv5_shifted_gpu = detect_by_flow(i, feature, image, image_path, key_frame_path)
            bbox_pred, iou_pred, prob_pred = net.feed_feature(Variable(conv5_shifted_gpu), layer=layer_of_flow)

        else:
            bbox_pred, iou_pred, prob_pred = net.forward(im_data)

        det_time = t_det.toc()
        t2 = time.time()
        print('t2', t2 - t1)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        t3 = time.time()
        print('t3', t3 - t2)

        # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)

        t4 = time.time()
        print('t4', t4 - t3)

        det_obj = detection_objects(bboxes, scores, cls_inds)
        save_as_kitti_format(i, det_obj, kitti_filename, src_label='kitti')
        im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)

        cv2.imshow('detection', im2show)
        cv2.imwrite('output/detection/{:04d}.jpg'.format(i), im2show)

        total_time = t_total.toc()
        format_str = 'frame: %d, (detection: %.1f fps, %.1f ms) (total: %.1f fps, %.1f ms)'
        print(format_str % (
            i, 1. / det_time, det_time * 1000, 1. / total_time, total_time * 1000))

        t5 = time.time()
        print('t5', t5 - t4)

        t_det.clear()
        t_total.clear()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
