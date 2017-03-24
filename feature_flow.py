import os
import cv2
import torch
import numpy as np
from torch.multiprocessing import Pool

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg


def preprocess(filename):
    image = cv2.imread(filename)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
    return image, im_data


def main():

    trained_model = cfg.trained_model
    thresh = 0.5
    # image_dir = '/home/cory/cedl/vid/videos/vid04'
    image_dir = '/media/cory/54604BF5604BDBFC/Project/KITTI_Dataset/data_tracking_image_2/training/image_02/0001'

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

    t_det = Timer()
    t_total = Timer()

    for i, image_path in enumerate(image_abs_paths):
        t_total.tic()
        image, im_data = preprocess(image_path)
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
        t_det.tic()
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
        cv2.imshow('test', im2show)

        total_time = t_total.toc()
        # wait_time = max(int(60 - total_time * 1000), 1)
        cv2.waitKey(1)

        format_str = 'frame: %d, (detection: %.1f fps, %.1f ms) (total: %.1f fps, %.1f ms)'
        print(format_str % (
            i, 1. / det_time, det_time * 1000, 1. / total_time, total_time * 1000))

        t_det.clear()
        t_total.clear()


if __name__ == '__main__':
    main()
