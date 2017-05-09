import numpy as np

exp_name = 'voc0712_trainval_ft_debug5'
dataset_name = 'voc'
# pretrained_fname = '/home/cory/yolo2-pytorch/models/voc0712_trainval_exp4_200.h5'
pretrained_fname = '/home/cory/yolo2-pytorch/models/yolo-voc.weights.h5'

network_size_rand_period = 10
inp_size_candidates = [(320, 320), (352, 352), (384, 384), (416, 416), (448, 448),
                       (480, 480), (512, 512), (544, 544)]
inp_size = np.array([416, 416], dtype=np.int)
out_size = inp_size / 32

optimizer = 'SGD'  # 'SGD, Adam'
opt_param = 'all'  # 'all, conv345'

start_step = 0
lr_epoch = (0, )
lr_val = (1E-5, )

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
train_images = '/home/cory/yolo2-pytorch/train_data/voc/voc_train_images.txt'
train_labels = '/home/cory/yolo2-pytorch/train_data/voc/voc_train_labels.txt'
val_images = '/home/cory/yolo2-pytorch/train_data/voc/voc_test_images.txt'
val_labels = '/home/cory/yolo2-pytorch/train_data/voc/voc_test_labels.txt'
batch_size = 1
train_batch_size = 16
