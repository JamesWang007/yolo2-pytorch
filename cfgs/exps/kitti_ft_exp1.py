import numpy as np

exp_name = 'kitti_ft_exp1'
dataset_name = 'kitti'
inp_size = np.array([1248, 352], dtype=np.int)   # w, h
pretrained_fname = '/home/cory/yolo2-pytorch/models/yolo-voc.weights.h5'

optimizer = 'SGD'  # 'SGD, Adam'
opt_param = 'all'  # 'all, conv345'

start_step = 0
lr_epoch = (0,)
lr_val = (1E-5,)

max_epoch = 200

# SGD only
weight_decay = 0.0005
momentum = 0.9

# for training yolo2
object_scale = 5.
noobject_scale = 1.
class_scale = 1.
coord_scale = 1.
iou_thresh = 0.6

# dataset
imdb_train = 'voc_2012_trainval'
imdb_test = 'voc_2007_test'
train_images = '/home/cory/yolo2-pytorch/train_data/kitti_train_images.txt'
train_labels = '/home/cory/yolo2-pytorch/train_data/kitti_train_labels.txt'
batch_size = 12
train_batch_size = 12

