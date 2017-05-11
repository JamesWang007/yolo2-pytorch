import numpy as np


# trained model
h5_fname = 'yolo-voc.weights.h5'

# KITTI
label_names = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare')
num_classes = len(label_names)

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)
num_anchors = len(anchors)

