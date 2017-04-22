import os

import torch

import cfgs.config as cfg
import utils.network as net_utils
from darknet import Darknet19
from datasets.kitti import KittiDataset
from utils.timer import Timer

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

# data loader
batch_size = 16
'''imdb = KittiDataset('kitti', '/home/cory/KITTI_Dataset',
                    '/home/cory/KITTI_Dataset/kitti_tracking_images.txt',
                    '/home/cory/KITTI_Dataset/kitti_tracking_gt.txt',
                    batch_size, KittiDataset.preprocess_train, processes=4, shuffle=True, dst_size=None)'''
imdb = KittiDataset('kitti', '/home/cory/KITTI_Dataset',
                    '/home/cory/yolo2-pytorch/imgs_exclude_1_19.txt',
                    '/home/cory/yolo2-pytorch/gt_exclude_1_19.txt',
                    batch_size, KittiDataset.preprocess_train, processes=4, shuffle=True, dst_size=None)
print('load data succ...')
net = Darknet19()

# CUDA_VISIBLE_DEVICES=1

use_model_type = 'exp'
use_model = ''
if use_model_type == 'default':
    use_model = cfg.trained_model
    net_utils.load_net(use_model, net)
elif use_model_type == 'exp':
    use_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_35.h5')  # 88 + 35 + 7
    # use_model = '/home/cory/yolo2-pytorch/models/training/epoch130_new.h5'
    net_utils.load_net(use_model, net)
elif use_model_type == 'conv':
    use_model = cfg.pretrained_model
    net.load_from_npz(use_model, num_conv=18)
else:
    raise AssertionError

net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
cfg.init_learning_rate = 1e-6
lr = cfg.init_learning_rate
optimizer = torch.optim.Adam([{'params': net.conv3.parameters()},
                              {'params': net.conv4.parameters()},
                              {'params': net.conv5.parameters()}], lr=lr)

# show training parameters
print('-------------------------------')
print('use_model', use_model)
print('use_model_type', use_model_type)
print('network size', cfg.inp_size)
print('batch_size', batch_size)
print('lr', lr)
print('-------------------------------')

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None

use_tensorboard = False
remove_all_log = False
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
t = Timer()
for step in range(start_epoch * imdb.batch_per_epoch, cfg.max_epoch * imdb.batch_per_epoch):
    t.tic()
    # batch
    batch = imdb.next_batch()
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    # forward
    im_data = net_utils.np_to_variable(im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    x = net.forward(im_data, gt_boxes, gt_classes, dontcare)

    # backward
    loss = net.loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.cls_loss.data.cpu().numpy()[0]
    train_loss += loss.data.cpu().numpy()[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1

    duration = t.toc()
    if step % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print('epoch: %d, step: %d, loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
            imdb.epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))

        if use_tensorboard and step % cfg.log_interval == 0:
            optimizer_lr = optimizer.param_groups[0]['lr']
            exp.add_scalar_value('loss_train', train_loss, step=step)
            exp.add_scalar_value('loss_bbox', bbox_loss, step=step)
            exp.add_scalar_value('loss_iou', iou_loss, step=step)
            exp.add_scalar_value('loss_cls', cls_loss, step=step)
            exp.add_scalar_value('learning_rate', optimizer_lr, step=step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()

    if step > 0 and (step % imdb.batch_per_epoch == 0):
        if isinstance(optimizer, torch.optim.SGD) and imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        save_name = os.path.join(cfg.train_output_dir, '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
        net_utils.save_net(save_name, net)
        print('save model: {}'.format(save_name))

imdb.close()
