import numpy as np
import cv2
from utils.im_transform import imcv2_affine_trans, imcv2_recolor


def parse_label_file(label_file_path, label_map):
    gt_classes = list()
    boxes = list()
    has_label = False
    with open(label_file_path) as label_file:
        for line in label_file.readlines():
            if line == '\n':
                continue
            values = line.strip().split(' ')
            label = values[0]

            try:
                label_id = label_map.index(label)
            except ValueError:
                # label not exist, ignore it
                label_id = -1
            gt_classes.append(label_id)
            bbox = [int(float(v)) for v in values[1:5]]
            boxes.append(bbox)
            has_label = True
    assert len(gt_classes) == len(boxes)
    return {'boxes': boxes, 'gt_classes': gt_classes, 'has_label': has_label}


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x

    return boxes


def affine_transform(img, boxes, net_inp_size):
    if len(boxes) == 0:
        return
    im = np.asarray(img, dtype=np.uint8)
    w, h = net_inp_size
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im, trans_param = imcv2_affine_trans(im)
    scale, offs, flip = trans_param
    boxes = offset_boxes(boxes, im.shape, scale, offs, flip)

    boxes[:, 0::2] *= float(w) / im.shape[1]
    boxes[:, 1::2] *= float(h) / im.shape[0]
    np.clip(boxes[:, 0::2], 0, w - 1, out=boxes[:, 0::2])
    np.clip(boxes[:, 1::2], 0, h - 1, out=boxes[:, 1::2])
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
            im[bb[1]:bb[3], bb[0], :] = 1.
            im[bb[1]:bb[3], bb[2], :] = 1.
            im[bb[1], bb[0]:bb[2], :] = 1.
            im[bb[3], bb[0]:bb[2], :] = 1.
        plt.imshow(im)
        plt.show()

    # im (pixels range 0~1)
    # boxes (pos range 0~max_img_size)
    return im, boxes


def encode_to_np(gt):
    labels = gt['gt_classes']
    bboxes = gt['boxes']
    img_size = gt['img_size']
    gt_size = len(labels)

    num_type = 8  # 1 + 1 + 2 + 4
    max_label_num_per_image = 50

    data_matrix = np.zeros([max_label_num_per_image, num_type], dtype=np.float32)
    data_matrix[0:gt_size, 0] = 1  # valid mask
    data_matrix[0:gt_size, 1] = labels
    data_matrix[0:gt_size, 2:4] = img_size
    data_matrix[0:gt_size, 4:8] = bboxes
    return data_matrix

