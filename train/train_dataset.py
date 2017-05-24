import os

import numpy as np

import cfgs.config as cfg
import utils.network as net_utils
from darknet import Darknet19
from datasets.ImageFileDataset import ImageFileDataset
from train.train_util import *
from utils.timer import Timer

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

# data loader
imdb = ImageFileDataset(cfg.dataset_name, '',
                        cfg.train_images,
                        cfg.train_labels,
                        cfg.train_batch_size, ImageFileDataset.preprocess_train,
                        processes=4, shuffle=True, dst_size=None)

print('imdb load data succeeded')
net = Darknet19()

# CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.makedirs(cfg.train_output_dir, exist_ok=True)
try:
    ckp = open(cfg.check_point_file)
    ckp_epoch = int(ckp.readlines()[0])
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

optimizer = get_optimizer(cfg, net, start_epoch)

# show training parameters
print('-------------------------------')
print('gpu_id', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('use_model', use_model)
print('exp_name', cfg.exp_name)
print('dataset', cfg.dataset_name)
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

for step in range(start_epoch * imdb.batch_per_epoch, cfg.max_epoch * imdb.batch_per_epoch + 1):
    timer.tic()

    # random change network size
    if step % cfg.network_size_rand_period == 0:
        rand_id = np.random.randint(0, len(cfg.inp_size_candidates))
        rand_network_size = cfg.inp_size_candidates[rand_id]
        network_size = np.array(rand_network_size, dtype=np.int)

    debug_rand = False
    if debug_rand:
        rand_network_size = cfg.inp_size_candidates[step % len(cfg.inp_size_candidates)]
        print(rand_network_size)
        network_size = np.array(rand_network_size, dtype=np.int)

    prev_epoch = imdb.epoch
    batch = imdb.next_batch(network_size)

    # when go to next epoch
    if imdb.epoch > prev_epoch:
        # save trained weights
        save_name = os.path.join(cfg.train_output_dir, '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
        net_utils.save_net(save_name, net)
        print('save model: {}'.format(save_name))

        # update check_point file
        ckp = open(os.path.join(cfg.check_point_file), 'w')
        ckp.write(str(imdb.epoch))
        ckp.close()

        # prepare optimizer for next epoch
        optimizer = get_optimizer(cfg, net, imdb.epoch)

    # forward
    im_data = net_utils.np_to_variable(batch['images'], is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    x = net.forward(im_data, batch['gt_boxes'], batch['gt_classes'], batch['dontcare'], network_size)

    # loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.cls_loss.data.cpu().numpy()[0]
    train_loss += net.loss.data.cpu().numpy()[0]
    cnt += 1
    # print('train_loss', net.loss.data.cpu().numpy()[0])

    # backward
    optimizer.zero_grad()
    net.loss.backward()
    optimizer.step()

    duration = timer.toc()
    if step % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        progress_in_epoch = (step % imdb.batch_per_epoch) / imdb.batch_per_epoch
        print('epoch: %d, step: %d (%.2f %%),'
              'loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                  imdb.epoch, step, progress_in_epoch * 100, train_loss, bbox_loss, iou_loss, cls_loss, duration))
        with open(cfg.log_file, 'a+') as log:
            log.write('%d, %d, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (
                imdb.epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))

        if use_tensorboard and step % cfg.log_interval == 0:
            exp.add_scalar_value('loss_train', train_loss, step=step)
            exp.add_scalar_value('loss_bbox', bbox_loss, step=step)
            exp.add_scalar_value('loss_iou', iou_loss, step=step)
            exp.add_scalar_value('loss_cls', cls_loss, step=step)
            exp.add_scalar_value('learning_rate', get_optimizer_lr(optimizer), step=step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        timer.clear()

imdb.close()
