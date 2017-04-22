import os
import shutil
import numpy as np
import cv2

from datasets.imdb import ImageDataset
from datasets.voc_eval import voc_eval
from utils.im_transform import imcv2_recolor
from utils.im_transform import imcv2_affine_trans
from utils.yolo import _offset_boxes
from cfgs import config
from cfgs import config_voc


class KittiDataset(ImageDataset):
    def __init__(self, imdb_name, datadir, image_list_file, gt_list_file,
                 batch_size, im_processor, processes=2, shuffle=True, dst_size=None):
        super(KittiDataset, self).__init__(imdb_name, datadir, batch_size, im_processor, processes, shuffle, dst_size)

        self._classes = ('car', 'pedestrian', 'cyclist')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.png'
        self.dst_size = config.inp_size

        self.image_list_file = image_list_file
        self.gt_list_file = gt_list_file

        self.load_dataset()
        # self.im_processor = partial(process_im, image_names=self._image_names, annotations=self._annotations)
        # self.im_processor = preprocess_train

    def load_dataset(self):
        self._image_names = list()
        self._annotations = list()
        self._image_indexes = list()

        empty_det_id = list()

        with open(self.image_list_file) as f:
            self._image_names = [line.strip() for line in f.readlines()]

        counter = 0
        with open(self.gt_list_file) as f:
            for gt_per_file in f.readlines():
                gt_per_file = gt_per_file.strip()
                with open(gt_per_file) as gt_file:
                    gt_classes = list()
                    boxes = list()
                    is_label = False
                    for line in gt_file.readlines():
                        values = line.strip().split(' ')
                        if values[0] == 'Car':
                            gt_classes.append(config_voc.label_names.index('car'))
                            boxes.append([int(float(v)) for v in values[1:]])
                            is_label = True
                        elif values[0] == 'Pedestrian':
                            gt_classes.append(config_voc.label_names.index('person'))
                            boxes.append([int(float(v)) for v in values[1:]])
                            is_label = True
                        else:
                            # ignore other class
                            continue

                    assert len(gt_classes) == len(boxes)
                    if not is_label:
                        empty_det_id.append(counter)

                    self._annotations.append({'boxes': boxes,
                                              'gt_classes': gt_classes})
                    counter += 1

        print('all', len(self._image_names))
        print('delete', len(empty_det_id))
        self._image_names = np.delete(self._image_names, empty_det_id)
        self._annotations = np.delete(self._annotations, empty_det_id)
        print('final', len(self._image_names))
        assert len(self._image_names) == len(self._annotations)

        # DEBUG SMALL TEST
        if 0:
            self._image_names = self._image_names[:100]
            self._annotations = self._annotations[:100]

        self._image_indexes = range(len(self._image_names))

    def next_batch(self):
        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 'origin_im': []}
        i = 0
        while i < self.batch_size:
            try:
                if self.gen is None:
                    raise AttributeError
                images, gt_boxes, classes, dontcare, origin_im = next(self.gen)
                batch['images'].append(images)
                batch['gt_boxes'].append(gt_boxes)
                batch['gt_classes'].append(classes)
                batch['dontcare'].append(dontcare)
                batch['origin_im'].append(origin_im)
                i += 1
            except (StopIteration, AttributeError):
                indexes = np.arange(len(self.image_names), dtype=np.int)
                if self._shuffle:
                    np.random.shuffle(indexes)
                self.gen = self.pool.imap(self._im_processor,
                                          ([self.image_names[i], self.get_annotation(i), self.dst_size] for i in indexes),
                                          chunksize=self.batch_size)
                self._epoch += 1
                print('epoch {} start...'.format(self._epoch))
        batch['images'] = np.asarray(batch['images'])

        return batch

    @staticmethod
    def preprocess_train(data):
        im_path, blob, inp_size = data
        boxes, gt_classes = blob['boxes'], blob['gt_classes']

        im = cv2.imread(im_path)
        ori_im = np.copy(im)

        im, trans_param = imcv2_affine_trans(im)
        scale, offs, flip = trans_param
        boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

        if boxes.shape == (0,):
            return im, boxes, [], [], ori_im

        if inp_size is not None:
            w, h = inp_size
            boxes[:, 0::2] *= float(w) / im.shape[1]
            boxes[:, 1::2] *= float(h) / im.shape[0]
            im = cv2.resize(im, (w, h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = imcv2_recolor(im)
        boxes = np.asarray(boxes, dtype=np.int)
        return im, boxes, gt_classes, [], ori_im

    def get_image(self, img_path):
        # img_path = ['/home/cory/KITTI_Dataset', 'data_tracking_image_2', 'training', 'image_02', '0000', '000000.png']
        pass

    def get_detection(self, det_path):
        # det_path = ['/home/cory/KITTI_Dataset', 'tracking_label', '0000', '000000.txt']
        pass


def convert_kitti(kitti_dir, detection_output_dir):
    kitti_detection_file_names = os.listdir(kitti_dir)
    print(len(kitti_detection_file_names), kitti_detection_file_names)
    for det_file_name in kitti_detection_file_names:
        abs_det_file_name = os.path.join(kitti_dir, det_file_name)
        with open(abs_det_file_name) as det_file:
            tracklet_prefix = det_file_name.replace('.txt', '')
            lines = det_file.readlines()
            all_detections = list()
            for line in lines:
                values = line.strip().split(' ')
                # print(values)
                frame = values[0]
                label = values[2]
                xmin = values[6]
                ymin = values[7]
                xmax = values[8]
                ymax = values[9]

                frame_i = int(frame)
                while len(all_detections) <= frame_i:
                    all_detections.append(list())

                all_detections[frame_i].append((frame, label, xmin, ymin, xmax, ymax))

            print(len(all_detections))

            out_dir = os.path.join(detection_output_dir, tracklet_prefix)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)

            print(out_dir)
            for frame_id, det_per_img in enumerate(all_detections):
                det_filename = '{:06d}.txt'.format(frame_id)
                with open(os.path.join(out_dir, det_filename), 'w') as f:
                    for det in det_per_img:
                        f.write(' '.join(det[1:]) + '\n')


def iou_func(bbox1, bbox2):
    x1 = bbox1[0]
    y1 = bbox1[1]
    width1 = bbox1[2] - bbox1[0]
    height1 = bbox1[3] - bbox1[1]

    x2 = bbox2[0]
    y2 = bbox2[1]
    width2 = bbox2[2] - bbox2[0]
    height2 = bbox2[3] - bbox2[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        ratio = area * 1. / (area1 + area2 - area)
    # return IOU
    return ratio


def ap(det, gt, iou_threshold):
    det_box = det[1:]
    gt_box = gt[1:]
    iou_func(det_box, gt_box)


def run_convert_kitti():
    orig_kitt_dir = '/home/cory/KITTI_Dataset/data_tracking_label_2/training/label_02'
    detection_output_dir = '/home/cory/KITTI_Dataset/tracking_label'
    convert_kitti(orig_kitt_dir, detection_output_dir)


def run_ap():
    det_str = 'Pedestrian' '1106.137292 166.576807 1204.470628 323.876144'
    det = det_str.split(' ')
    gt_str = 'Pedestrian' '1106.137292 166.576807 1204.470628 323.876144'
    gt = gt_str.split(' ')


def test_kitti_dataset():
    dataset = KittiDataset('kitti', '/home/cory/KITTI_Dataset',
                           '/home/cory/KITTI_Dataset/kitti_tracking_images.txt',
                           '/home/cory/KITTI_Dataset/kitti_tracking_gt.txt',
                           16, KittiDataset.preprocess_train, processes=2, shuffle=True, dst_size=None)
    for i in range(1000):
        print(i)
        dataset.next_batch()

if __name__ == '__main__':
    # run_convert_kitti()
    test_kitti_dataset()
    pass
