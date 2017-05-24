import os
import time

import numpy as np
from torch.autograd import Variable

import utils.network as net_utils
from cfgs.config_v2 import load_cfg_yamls
from darknet_v2 import Darknet19
from datasets.DataLoaderX import DataLoaderX
from datasets.DetectionDataset import DetectionDataset
from train.train_util_v2 import *
from utils.timer import Timer


def read_ckp(cfg):
    try:
        ckp = open(cfg['train_output_dir'] + '/check_point.txt')
        start_epoch = int(ckp.readlines()[0])
        use_model = os.path.join(cfg['train_output_dir'], cfg['exp_name'] + '_' + str(start_epoch) + '.h5')
    except IOError:
        start_epoch = 0
        use_model = cfg['pretrained_model']
    return start_epoch, use_model


def restore_gt_numpy(labels):
    labels_np = labels.numpy()
    gt_boxes = list()
    gt_classes = list()
    for i in range(labels_np.shape[0]):
        label = labels_np[i, ...]
        valid_id = np.where(label[:, 0] == 1)
        valid_len = np.count_nonzero(label[:, 0] == 1)
        gt_boxes.append(label[valid_id, 4:8].reshape(valid_len, -1))
        g = label[valid_id, 1][0].tolist()
        g = list(map(int, g))
        gt_classes.append(g)
    return gt_boxes, gt_classes


def train_main():
    dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_voc.yaml'
    exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/voc0712/voc0712_baseline_v3_rand.yaml'
    # dataset_yaml = '/home/cory/yolo2-pytorch/cfgs/config_kitti.yaml'
    # exp_yaml = '/home/cory/yolo2-pytorch/cfgs/exps/kitti/kitti_baseline.yaml'

    cfg = load_cfg_yamls([dataset_yaml, exp_yaml])

    # runtime setting
    gpu_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.makedirs(cfg['train_output_dir'], exist_ok=True)

    # data loader
    batch_size = cfg['train_batch_size']
    dataset = DetectionDataset(cfg)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    print('load dataset succeeded')
    net = Darknet19(cfg)

    start_epoch, use_model = read_ckp(cfg)

    net_utils.load_net(use_model, net)
    net.cuda()
    net.train()
    print('load net succeeded')

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
    print('inp_size', cfg['inp_size'])
    print('inp_size_candidates', cfg['inp_size_candidates'])
    print('-------------------------------')

    # default input size
    network_size = np.array(cfg['inp_size'], dtype=np.int)

    t0 = time.time()
    timer = Timer()
    try:
        for epoch in range(start_epoch, cfg['max_epoch']):

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            cnt = 0
            optimizer = get_optimizer(cfg, net, epoch)

            for step, data in enumerate(dataloader):
                timer.tic()
                inputs, labels = data
                im_data = Variable(inputs.cuda())
                gt_boxes, gt_classes = restore_gt_numpy(labels)

                x = net.forward(im_data, gt_boxes, gt_classes, network_size)

                # loss
                bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
                iou_loss += net.iou_loss.data.cpu().numpy()[0]
                cls_loss += net.class_loss.data.cpu().numpy()[0]
                train_loss += net.loss.data.cpu().numpy()[0]
                cnt += 1

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
                    progress_in_epoch = (step + 1) * batch_size / len(dataset)
                    print('epoch: %d, step: %d (%.2f %%),'
                          'loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                              epoch, step, progress_in_epoch * 100, train_loss, bbox_loss, iou_loss, cls_loss,
                              duration))
                    with open(cfg['train_output_dir'] + '/train.log', 'a+') as log:
                        log.write('%d, %d, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (
                            epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))

                    train_loss = 0
                    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
                    cnt = 0
                    timer.clear()

            # epoch_done
            # save trained weights
            ckp_epoch = epoch + 1
            save_name = os.path.join(cfg['train_output_dir'], '{}_{}.h5'.format(cfg['exp_name'], ckp_epoch))
            net_utils.save_net(save_name, net)
            print('save model: {}'.format(save_name))

            # update check_point file
            ckp = open(os.path.join(cfg['train_output_dir'], 'check_point.txt'), 'w')
            ckp.write(str(ckp_epoch))
            ckp.close()

    except KeyboardInterrupt:
        exit(1)


if __name__ == '__main__':
    train_main()
