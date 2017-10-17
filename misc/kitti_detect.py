import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['DATASET'] = 'kitti'

from cfgs.config_v2 import load_cfg_yamls
import utils.network as net_utils
import utils.yolo_v2 as yolo_utils
from darknet_v3 import Darknet19
from flow.flow_util import *
from utils.timer import Timer

dataset_yaml = '/home/cory/project/yolo2-pytorch/cfgs/config_kitti.yaml'
exp_yaml = '/home/cory/project/yolo2-pytorch/cfgs/exps/kitti/kitti_baseline_v3.yaml'
gpu_id = 0

cfg = load_cfg_yamls([dataset_yaml, exp_yaml])


def preprocess(filename):
    image = cv2.imread(filename)
    im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg['inp_size']))[0], 0)
    return image, im_data


def detection_objects(bboxes, scores, cls_inds):
    objects = list()
    for i in range(len(bboxes)):
        box = bboxes[i]
        score = scores[i]
        label = cfg['label_names'][cls_inds[i]]
        objects.append((box, score, label))
    return objects


def save_as_kitti_format(frame_id, det_obj, output_dir, src_label='voc'):
    # 'Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01'
    # 0 -1 car 0 0 0 1078 142 1126 164 0 0 0 0 0 0 0.415537
    with open(output_dir + '/{:06d}.txt'.format(frame_id), 'w') as file:
        for det in det_obj:
            bbox = det[0]
            score = det[1]
            label = det[2]
            if src_label == 'voc':
                if label != 'car' and label != 'person':
                    continue
                label = label.replace('person', 'pedestrian')
            label.replace('Person', 'Person_sitting')
            line_str = '{:s} 0 0 0 {:d} {:d} {:d} {:d} 0 0 0 0 0 0 0 {:.4f}\n' \
                .format(label, bbox[0], bbox[1], bbox[2], bbox[3], score)
            # print(line_str)
            file.write(line_str)


def main():

    output_dir = '../output'
    output_template_dir = '../output_template'
    kitti_output_dir = '../kitti_det_output'
    input_file_list = '/home/cory/project/yolo2-pytorch/train_data/kitti/kitti_val_images.txt'
    # input_file_list = '/home/cory/project/yolo2-pytorch/flow/w01_imgs.txt'
    vis_enable = False
    thresh = 0.5

    trained_model = '/home/cory/project/yolo2-pytorch/models/training/kitti_new_2_flow_center_ft_half/' \
                    'kitti_new_2_flow_center_ft_half_5.h5'

    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(kitti_output_dir, ignore_errors=True)
    shutil.copytree(output_template_dir, output_dir)
    os.makedirs(kitti_output_dir)

    net = Darknet19(cfg)
    net_utils.load_net(trained_model, net)
    net.eval()
    net.cuda()
    print(trained_model)
    print('load model successfully')

    img_files = open(input_file_list)
    image_abs_paths = img_files.readlines()
    image_abs_paths = [f.strip() for f in image_abs_paths]

    t_det = Timer()
    t_total = Timer()
    for i, image_path in enumerate(image_abs_paths):
        t_total.tic()
        image, im_data = preprocess(image_path)
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)

        t_det.tic()
        bbox_pred, iou_pred, prob_pred = net.forward(im_data)
        det_time = t_det.toc()

        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)
        det_obj = detection_objects(bboxes, scores, cls_inds)
        save_as_kitti_format(i, det_obj, kitti_output_dir, src_label='kitti')

        total_time = t_total.toc()
        format_str = 'frame: %d, (detection: %.1f fps, %.1f ms) (total: %.1f fps, %.1f ms) %s'
        print(format_str % (
            i, 1. / det_time, det_time * 1000, 1. / total_time, total_time * 1000, image_path))

        t_det.clear()
        t_total.clear()

        if vis_enable:
            im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
            cv2.imshow('detection', im2show)
            cv2.imwrite(output_dir + '/detection/{:04d}.jpg'.format(i), im2show)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
