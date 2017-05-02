import os
from cfgs.config_voc import *
from cfgs.exps.kitti_ft_dontcare import *
# from cfgs.exps.darknet19_exp5 import *
# from cfgs.exps.voc_adam_exp1 import *


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# input and output size
############################
# inp_size = np.array([1248, 352], dtype=np.int)   # w, h
# inp_size = np.array([608, 608], dtype=np.int)
# inp_size = np.array([512, 288], dtype=np.int)
# inp_size = np.array([1280, 736], dtype=np.int)
# inp_size = np.array([416, 416], dtype=np.int)
# out_size = inp_size / 32


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127

base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


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