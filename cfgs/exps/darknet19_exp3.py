exp_name = 'voc0712_trainval_exp3'

pretrained_fname = '/home/cory/yolo2-pytorch/data/darknet19.weights.npz'

optimizer = 'Adam'  # 'SGD, Adam'
opt_param = 'all'   # 'all, conv345'

start_step = 0
lr_epoch = (0,)
lr_val = (1E-3,)

max_epoch = 160

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
batch_size = 1
train_batch_size = 32
