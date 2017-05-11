import numpy as np

exp_name = 'kitti_ft_exp3_new'
dataset_name = 'kitti'
# pretrained_fname = '/home/cory/yolo2-pytorch/models/yolo-voc.weights.h5'
# pretrained_fname = '/home/cory/yolo2-pytorch/models/training/voc0712_new_2/voc0712_new_2_160.h5'
pretrained_fname = '/home/cory/yolo2-pytorch/data/darknet19.weights.npz'

network_size_rand_period = 10
inp_size_candidates = [(1216, 352)]
inp_size = np.array([1216, 352], dtype=np.int)   # w, h
out_size = inp_size / 32


optimizer = 'SGD'  # 'SGD, Adam'
opt_param = 'all'  # 'all, conv345'

start_step = 0
lr_epoch = (0, 60, 90)
lr_val = (1e-3, 1e-4, 1e-5)

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
train_images = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_train_images.txt'
train_labels = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_train_labels.txt'
val_images = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_val_images.txt'
val_labels = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_val_labels.txt'
batch_size = 1
train_batch_size = 16
