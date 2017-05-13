import os
import shutil
import numpy as np
import cv2

from datasets.imdb import ImageDataset
from utils.im_transform import imcv2_recolor
from utils.im_transform import imcv2_affine_trans
from utils.yolo import _offset_boxes


class ImageFileDataset(ImageDataset):
    def __init__(self, cfg, im_processor, processes=2, shuffle=True, dst_size=None, mode='train'):
        if mode == 'train':
            batch_size = cfg['train_batch_size']
        else:
            batch_size = cfg['val_batch_size']
        super(ImageFileDataset, self).__init__(cfg['dataset_name'], '', batch_size,
                                               im_processor, processes, shuffle, dst_size)
        self.cfg = cfg
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.dst_size = dst_size
        if mode == 'train':
            self.image_list_file = cfg['train_images']
            self.label_list_file = cfg['train_labels']
        else:
            self.image_list_file = cfg['val_images']
            self.label_list_file = cfg['val_labels']
        self.imdb_name = cfg['dataset_name']
        # dashcam self._classes = ('car', 'bus', 'motorbike', 'bike', 'person')
        self._classes = cfg['label_names']

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
                label_dict = self.parse_label_file(label_file_name, self.cfg['label_names'], self.imdb_name)
                if not label_dict['has_label']:
                    remove_id_list.append(fi)
                self._annotations.append(label_dict)

        self._image_names = np.delete(self._image_names, remove_id_list)
        self._annotations = np.delete(self._annotations, remove_id_list)
        print('dataset size =', len(self._image_names))
        assert len(self._image_names) == len(self._annotations)
        self._image_indexes = range(len(self._image_names))

    kitti_voc_label_replacement = {'Car': 'car', 'Van': 'car',
                                   'Pedestrian': 'person', 'Person': 'person'}
    dashcam_voc_label_replacement = {'bike': 'bicycle'}
    @staticmethod
    def parse_label_file(label_file_name, label_mapping, dataset_foramt):
        gt_classes = list()
        boxes = list()
        has_label = False
        with open(label_file_name) as label_file:
            for line in label_file.readlines():
                values = line.strip().split(' ')
                label = values[0]
                # try to replace original label name with voc label name
                '''if dataset_foramt == 'kitti':
                    label = ImageFileDataset.kitti_voc_label_replacement.get(label, label)
                elif dataset_foramt == 'dashcam':
                    label = ImageFileDataset.dashcam_voc_label_replacement.get(label, label)'''

                try:
                    label_id = label_mapping.index(label)
                except ValueError:
                    # label not exist, ignore it
                    label_id = -1
                gt_classes.append(label_id)
                bbox = [int(float(v)) for v in values[1:5]]
                boxes.append(bbox)
                has_label = True
        assert len(gt_classes) == len(boxes)
        return {'boxes': boxes, 'gt_classes': gt_classes, 'has_label': has_label}

    def next_batch(self, dst_size):
        batch_end_index = min(self.current_index + self.batch_size, len(self.indexes))
        ids = [self.indexes[i] for i in range(self.current_index, batch_end_index)]
        result = self.pool.map(self._im_processor,
                                  ([self.image_names[i], self.get_annotation(i), dst_size]
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
