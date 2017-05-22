import numpy as np


# trained model
h5_fname = 'yolo-voc.weights.h5'

# VOC
label_names = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(label_names)

#anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)
# anchors = np.asarray([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]])
#anchors = np.asarray([[1.08, 1.19], [1.32, 1.73], [3.19, 4.01], [3.42, 4.41], [5.05, 8.09],
#          [6.63, 11.38], [9.47, 4.84], [11.23, 10.00], [16.62, 10.52]])
anchors = np.asarray([[1., 1.], [3., 3.], [5., 5.], [9., 9.], [13., 13.]])
num_anchors = len(anchors)

