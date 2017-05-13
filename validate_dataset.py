import os
import numpy as np
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from cfgs.config_v2 import add_cfg
import utils.network as net_utils
from darknet_v2 import Darknet19
from datasets.ImageFileDataset_v2 import ImageFileDataset
from utils.timer import Timer
from train_util_v2 import *


dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_kitti.yaml'
exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/kitti_new_2.yaml'

cfg = dict()
# add_cfg(cfg, '/home/cory/yolo2-pytorch/cfgs/config_voc.yaml')
add_cfg(cfg, dataset_yaml)
add_cfg(cfg, exp_yaml)

# data loader
imdb = ImageFileDataset(cfg, ImageFileDataset.preprocess_train,
                        processes=4, shuffle=False, dst_size=None, mode='val')

print('imdb load data succeeded')
net = Darknet19(cfg)

# CUDA_VISIBLE_DEVICES=1
# 20  0.68
# 40  0.60
# 45  0.56
# 50  0.58
# 55  0.55
# 60  0.59

os.makedirs(cfg['train_output_dir'], exist_ok=True)
try:
    ckp = open(cfg['train_output_dir'] + '/check_point.txt')
    ckp_epoch = int(ckp.readlines()[0])
    ckp_epoch = 60
    # raise IOError
    use_model = os.path.join(cfg['train_output_dir'], cfg['exp_name'] + '_' + str(ckp_epoch) + '.h5')
except IOError:
    ckp_epoch = 0
    use_model = cfg['pretrained_model']

net_utils.load_net(use_model, net)

net.cuda()
net.train()
print('load net succeeded')

start_epoch = ckp_epoch
imdb.epoch = start_epoch

# show training parameters
print('-------------------------------')
print('gpu_id', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('use_model', use_model)
print('exp_name', cfg['exp_name'])
print('dataset', cfg['dataset_name'])
print('optimizer', cfg['optimizer'])
print('opt_param', cfg['opt_param'])
print('train_batch_size', cfg['train_batch_size'])
print('start_epoch', start_epoch)
print('lr', lookup_lr(cfg, start_epoch))
print('-------------------------------')


train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0

timer = Timer()

# default input size
network_size = np.array(cfg['inp_size'], dtype=np.int)

for step in range(start_epoch * imdb.batch_per_epoch, (start_epoch + 5) * imdb.batch_per_epoch + 1):
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

    if step % cfg['disp_interval'] == 0:
        progress_in_epoch = (step % imdb.batch_per_epoch) / imdb.batch_per_epoch
        print('%.2f%%' % (progress_in_epoch * 100), end=' ')
        sys.stdout.flush()

imdb.close()
