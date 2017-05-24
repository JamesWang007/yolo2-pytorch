import os

import numpy as np

import utils.network as net_utils
from cfgs.config_v2 import add_cfg
from darknet_v2 import Darknet19
from datasets.ImageFileDataset_v2 import ImageFileDataset
from train.train_util_v2 import *
from utils.timer import Timer

# dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_kitti.yaml'
# exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/kitti/kitti_baseline.yaml'
dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_voc.yaml'
exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712/voc0712_baseline.yaml'
# exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712_one_anchor.yaml'
# exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712_anchor.yaml'
# exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712_adam_20.yaml'

cfg = dict()
add_cfg(cfg, dataset_yaml)
add_cfg(cfg, exp_yaml)

# data loader
imdb = ImageFileDataset(cfg, ImageFileDataset.preprocess_train,
                        processes=4, shuffle=True, dst_size=None)

print('imdb load data succeeded')
net = Darknet19(cfg)

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

os.makedirs(cfg['train_output_dir'], exist_ok=True)
try:
    ckp = open(cfg['train_output_dir'] + '/check_point.txt')
    ckp_epoch = int(ckp.readlines()[0])
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

optimizer = get_optimizer(cfg, net, start_epoch)

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

for step in range(start_epoch * imdb.batch_per_epoch, cfg['max_epoch'] * imdb.batch_per_epoch + 1):
    timer.tic()

    # random change network size
    if step % cfg['network_size_rand_period'] == 0:
        rand_id = np.random.randint(0, len(cfg['inp_size_candidates']))
        rand_network_size = cfg['inp_size_candidates'][rand_id]
        network_size = np.array(rand_network_size, dtype=np.int)

    prev_epoch = imdb.epoch
    batch = imdb.next_batch(network_size)

    # when go to next epoch
    if imdb.epoch > prev_epoch:
        if cfg['exp_name'] != 'voc0712_overfit':
            # save trained weights
            save_name = os.path.join(cfg['train_output_dir'], '{}_{}.h5'.format(cfg['exp_name'], imdb.epoch))
            net_utils.save_net(save_name, net)
            print('save model: {}'.format(save_name))

            # update check_point file
            ckp = open(os.path.join(cfg['train_output_dir'], 'check_point.txt'), 'w')
            ckp.write(str(imdb.epoch))
            ckp.close()

            # prepare optimizer for next epoch
            optimizer = get_optimizer(cfg, net, imdb.epoch)

    # forward
    im_data = net_utils.np_to_variable(batch['images'], is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    x = net.forward(im_data, batch['gt_boxes'], batch['gt_classes'], network_size)

    # loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.class_loss.data.cpu().numpy()[0]
    train_loss += net.loss.data.cpu().numpy()[0]
    cnt += 1
    # print('train_loss', net.loss.data.cpu().numpy()[0])

    # backward
    optimizer.zero_grad()
    net.loss.backward()
    optimizer.step()

    duration = timer.toc()
    if step % cfg['disp_interval'] == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        progress_in_epoch = (step % imdb.batch_per_epoch) / imdb.batch_per_epoch
        print('epoch: %d, step: %d (%.2f %%),'
              'loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                  imdb.epoch, step, progress_in_epoch * 100, train_loss, bbox_loss, iou_loss, cls_loss, duration))
        with open(cfg['train_output_dir'] + '/train.log', 'a+') as log:
            log.write('%d, %d, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (
                imdb.epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        timer.clear()

imdb.close()
