import os

import torch

import cfgs.config_ft as cfg
import utils.network as net_utils
from darknet import Darknet19
from datasets.kitti import KittiDataset
from utils.timer import Timer

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

# data loader
batch_size = 32
imdb = KittiDataset('kitti', '/home/cory/KITTI_Dataset',
                    '/home/cory/KITTI_Dataset/kitti_tracking_images.txt',
                    '/home/cory/KITTI_Dataset/kitti_tracking_gt.txt',
                    batch_size, KittiDataset.preprocess_train, processes=2, shuffle=True, dst_size=None)
print('load data succ...')
print(cfg.inp_size)
net = Darknet19()

use_model = 'default'
if use_model == 'default':
    net_utils.load_net(cfg.trained_model, net)
elif use_model == 'exp':
    pretrained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_4.h5')
    net_utils.load_net(pretrained_model, net)
elif use_model == 'conv':
    net.load_from_npz(cfg.pretrained_model, num_conv=18)
else:
    raise AssertionError

net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
cfg.init_learning_rate = 1e-5
lr = cfg.init_learning_rate
optimizer = torch.optim.Adam([{'params': net.conv3.parameters()},
                              {'params': net.conv4.parameters()},
                              {'params': net.conv5.parameters()}], lr=lr)

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None

# use_tensorboard = False
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
    # DEBUG_LOSS
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