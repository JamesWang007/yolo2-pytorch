import os
import numpy as np
import sys

import cfgs.config as cfg
import utils.network as net_utils
from darknet import Darknet19
from datasets.ImageFileDataset import ImageFileDataset
from utils.timer import Timer
from train_util import *

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

# data loader
imdb = ImageFileDataset(cfg.dataset_name, '',
                        cfg.val_images,
                        cfg.val_labels,
                        cfg.batch_size, ImageFileDataset.preprocess_train,
                        processes=4, shuffle=True, dst_size=None)

print('imdb load data succeeded')
net = Darknet19()

# CUDA_VISIBLE_DEVICES=1

os.makedirs(cfg.train_output_dir, exist_ok=True)
try:
    ckp = open(cfg.check_point_file)
    ckp_epoch = int(ckp.readlines()[0])
    # ckp_epoch = 20
    # raise IOError
    use_model = os.path.join(cfg.train_output_dir, cfg.exp_name + '_' + str(ckp_epoch) + '.h5')
except IOError:
    ckp_epoch = 0
    use_model = cfg.pretrained_model

net_utils.load_net(use_model, net)

net.cuda()
net.train()
print('load net succeeded')

start_epoch = ckp_epoch
imdb.epoch = start_epoch

# show training parameters
print('-------------------------------')
print('use_model', use_model)
print('exp_name', cfg.exp_name)
print('optimizer', cfg.optimizer)
print('opt_param', cfg.opt_param)
print('train_batch_size', cfg.train_batch_size)
print('start_epoch', start_epoch)
print('lr', lookup_lr(cfg, start_epoch))
print('-------------------------------')

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None

use_tensorboard = False
remove_all_log = True
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        print('remove all experiments')
        cc.remove_all_experiments()
    if start_epoch == 0:
        try:
            cc.remove_experiment(cfg.exp_name)
        except ValueError:
            pass
        exp = cc.create_experiment(cfg.exp_name)
    else:
        exp = cc.open_experiment(cfg.exp_name)

train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0

timer = Timer()

# default input size
network_size = cfg.inp_size

for step in range(start_epoch * imdb.batch_per_epoch, cfg.max_epoch * imdb.batch_per_epoch):
    timer.tic()

    prev_epoch = imdb.epoch
    batch = imdb.next_batch(network_size)

    # when go to next epoch
    if imdb.epoch > prev_epoch:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print()
        print('loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f' %
              (train_loss, bbox_loss, iou_loss, cls_loss))

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        timer.clear()

    # forward
    im_data = net_utils.np_to_variable(batch['images'], is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    x = net.forward(im_data, batch['gt_boxes'], batch['gt_classes'], batch['dontcare'], network_size)

    # loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.cls_loss.data.cpu().numpy()[0]
    train_loss += net.loss.data.cpu().numpy()[0]
    cnt += 1

    if step % cfg.disp_interval == 0:
        progress_in_epoch = (step % imdb.batch_per_epoch) / imdb.batch_per_epoch
        print('%.2f%%' % (progress_in_epoch * 100), end=' ')
        sys.stdout.flush()

imdb.close()
