import os
import yaml

cfg = dict()

# Read YAML file
try:
    cfgs_dir = '/home/cory/yolo2-pytorch/cfgs'
    config_dataset = open(cfgs_dir + '/config_voc.yaml', 'r')
    config_exp = open(cfgs_dir + '/exps/voc0712_trainval_ft_debug2.yaml', 'r')
    cfg.update(yaml.load(config_dataset))
    cfg.update(yaml.load(config_exp))
    print('-------------------------------')
    for k, v in cfg.items():
        print(k, v)
    print('-------------------------------')
except Exception:
    print('Error: cannot parse cfg')
    raise Exception


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
'''ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
mkdir(test_output_dir, max_depth=4)'''

rand_seed = 1024
use_tensorboard = False

log_interval = 50
disp_interval = 50