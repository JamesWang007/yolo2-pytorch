import os
from cfgs.config_voc import *
# from cfgs.config_kitti import *
from cfgs.exps.voc0712_new_2 import *
# from cfgs.exps.kitti_ft_exp3 import *

# 10.5 ~ 11 ms  yolo_flow  detection only  OpenBLAS
# 0.75 s/batch

# 16 ~ 17 ms  anaconda
# 1.55 s/batch

label_names = label_names
num_classes = len(label_names)


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# detection config
############################
thresh = 0.3


# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
TEST_DIR = os.path.join(MODEL_DIR, 'testing')

trained_model = os.path.join(MODEL_DIR, h5_fname)
pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)
train_output_dir = os.path.join(TRAIN_DIR, exp_name)
test_output_dir = os.path.join(TEST_DIR, imdb_test, h5_fname)
log_file = os.path.join(train_output_dir, 'train.log')
check_point_file = os.path.join(train_output_dir, 'check_point.txt')
mkdir(train_output_dir, max_depth=3)
mkdir(test_output_dir, max_depth=4)

rand_seed = 1024
use_tensorboard = False

log_interval = 50
disp_interval = 50