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


class ImageFileDataset(ImageDataset):
    def __init__(self, imdb_name, datadir, image_list_file, train_labels,
                 batch_size, im_processor, processes=2, shuffle=True, dst_size=None):
        super(ImageFileDataset, self).__init__(imdb_name, datadir, batch_size, im_processor, processes, shuffle, dst_size)

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.dst_size = config.inp_size

        self.image_list_file = image_list_file
        self.label_list_file = train_labels
        if imdb_name == 'kitti':
            self.is_kitti = True
            self._classes = ('car', 'pedestrian', 'dontcare')
        else:
            self.is_kitti = False
            self._classes = config_voc.label_names

        self.load_dataset()

        self._epoch = 0
        self.current_index = 0
        self.indexes = np.arange(len(self.image_names), dtype=np.int)
        if self._shuffle:
            np.random.shuffle(self.indexes)

    def load_dataset(self):
        remove_id_list = list()

        with open(self.image_list_file) as f:
            self._image_names = [line.strip() for line in f.readlines()]

        self._annotations = list()
        with open(self.label_list_file) as f:
            for fi, label_file_name in enumerate(f.readlines()):
                label_file_name = label_file_name.strip()
                if self.is_kitti:
                    data_foramt = 'kitti'
                else:
                    data_foramt = ''
                label_dict = self.parse_label_file(label_file_name, config_voc.label_names, data_foramt)
                if not label_dict['has_label']:
                    remove_id_list.append(fi)
                self._annotations.append(label_dict)

        self._image_names = np.delete(self._image_names, remove_id_list)
        self._annotations = np.delete(self._annotations, remove_id_list)
        print('dataset size =', len(self._image_names))
        assert len(self._image_names) == len(self._annotations)
        self._image_indexes = range(len(self._image_names))

    kitti_voc_label_map = {'Car': 'car', 'Pedestrian': 'person'}
    @staticmethod
    def parse_label_file(label_file_name, label_mapping, dataset_foramt):
        gt_classes = list()
        boxes = list()
        has_label = False
        with open(label_file_name) as label_file:
            for line in label_file.readlines():
                values = line.strip().split(' ')
                label = values[0]
                if dataset_foramt == 'kitti':
                    label = ImageFileDataset.kitti_voc_label_map.get(label)
                try:
                    label_id = label_mapping.index(label)
                except ValueError:
                    label_id = -1
                gt_classes.append(label_id)
                if dataset_foramt == 'kitti':
                    bbox = [int(float(v)) for v in values[1:]]
                else:
                    bbox = [int(v) - 1 for v in values[1:]]
                boxes.append(bbox)
                has_label = True
        assert len(gt_classes) == len(boxes)
        return {'boxes': boxes, 'gt_classes': gt_classes, 'has_label': has_label}

    def next_batch(self):
        batch_end_index = min(self.current_index + self.batch_size, len(self.indexes))
        ids = [self.indexes[i] for i in range(self.current_index, batch_end_index)]
        result = self.pool.map(self._im_processor,
                                  ([self.image_names[i], self.get_annotation(i), self.dst_size]
                                   for i in ids))
        self.current_index += self.batch_size

        # reset index if reach the end of epoch
        if self.current_index >= len(self.indexes):
            self._epoch += 1
            self.current_index = 0
            if self._shuffle:
                np.random.shuffle(self.indexes)

        batch = {'images': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': [], 'origin_im': []}
        for r in result:
            images, gt_boxes, classes, dontcare, origin_im = r
            batch['images'].append(images)
            batch['gt_boxes'].append(gt_boxes)
            batch['gt_classes'].append(classes)
            batch['dontcare'].append(dontcare)
            batch['origin_im'].append(origin_im)

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

        if len(boxes) == 0:
            return im, boxes, [], [], ori_im

        if inp_size is not None:
            w, h = inp_size
            boxes[:, 0::2] *= float(w) / im.shape[1]
            boxes[:, 1::2] *= float(h) / im.shape[0]
            im = cv2.resize(im, (w, h))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = imcv2_recolor(im)
        boxes = np.asarray(boxes, dtype=np.int)

        debug = False
        if debug:
            import matplotlib.pyplot as plt
            for idx, box in enumerate(boxes):
                # box = [xmin, ymin, xmax, ymax]  with original pixel scale
                bb = [int(b) for b in box]
                label_id = gt_classes[idx]
                print(label_id, bb)
                im[bb[1]:bb[3], bb[0], :] = 1.
                im[bb[1]:bb[3], bb[2], :] = 1.
                im[bb[1], bb[0]:bb[2], :] = 1.
                im[bb[3], bb[0]:bb[2], :] = 1.
            plt.imshow(im)
            plt.show()

        return im, boxes, gt_classes, [], ori_im

    def get_image(self, img_path):
        pass

    def get_detection(self, det_path):
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
