import os
import time

from torch.autograd import Variable

from cfgs.config_v2 import load_cfg_yamls
from darknet_training_v3 import *
from darknet_v3 import Darknet19
from datasets.DataLoaderX import DataLoaderX
from datasets.DetectionDataset import DetectionDataset
from train.train_util_v2 import *
from utils.barrier import Barrier
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


def train_main():
    choice = 1
    if choice == 0:
        dataset_yaml = '/home/cory/project/yolo2-pytorch/cfgs/config_detrac.yaml'
        exp_yaml = '/home/cory/project/yolo2-pytorch/cfgs/exps/detrac/detrac_flow_center_w01.yaml'
        gpu_id = 1
    else:
        dataset_yaml = '/home/cory/project/yolo2-pytorch/cfgs/config_kitti.yaml'
        exp_yaml = '/home/cory/project/yolo2-pytorch/cfgs/exps/kitti/kitti_baseline_v3_fl.yaml'
        gpu_id = 1

    cfg = load_cfg_yamls([dataset_yaml, exp_yaml])

    # runtime setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.makedirs(cfg['train_output_dir'], exist_ok=True)

    enable_bbox_loss = True
    enable_iou_loss = True
    enable_class_loss = True

    # data loader
    batch_size = cfg['train_batch_size']
    dataset = DetectionDataset(cfg)
    print('load dataset succeeded')
    net = Darknet19(cfg)

    start_epoch, use_model = read_ckp(cfg)

    net_utils.load_net(use_model, net)
    net.cuda()
    net.train()
    print('load net succeeded')

    # show training parameters
    print('-------------------------------')
    print('pid', os.getpid())
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

    timer = Timer()
    try:
        for epoch in range(start_epoch, cfg['max_epoch']):
            time_epoch_begin = time.time()
            optimizer = get_optimizer(cfg, net, epoch)
            dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

            for step, data in enumerate(dataloader):
                timer.tic()
                barrier = Barrier()

                images, labels = data
                # debug_and_vis(data)

                im_data = Variable(images.cuda())
                barrier.add(1)

                bbox_pred, iou_pred, class_pred = net.forward(im_data)
                barrier.add(2)

                # build training target
                network_h = int(im_data.data.size()[2])
                network_w = int(im_data.data.size()[3])
                network_size_wh = np.array([network_w, network_h])  # (w, h)
                net_bbox_loss, net_iou_loss, net_class_loss = training_target(
                    cfg, bbox_pred, class_pred, labels, network_size_wh, iou_pred)
                barrier.add(3)

                # backward
                optimizer.zero_grad()

                net_loss = 0
                if enable_bbox_loss:
                    net_loss += net_bbox_loss
                if enable_iou_loss:
                    net_loss += net_iou_loss
                if enable_class_loss:
                    net_loss += net_class_loss

                net_loss.backward()
                optimizer.step()
                barrier.add(4)

                duration = timer.toc()
                if step % cfg['disp_interval'] == 0:
                    # loss for this step
                    bbox_loss = net_bbox_loss.data.cpu().numpy()[0]
                    iou_loss = net_iou_loss.data.cpu().numpy()[0]
                    cls_loss = net_class_loss.data.cpu().numpy()[0]
                    train_loss = net_loss.data.cpu().numpy()[0]
                    barrier.add(5)

                    progress_in_epoch = (step + 1) * batch_size / len(dataset)
                    print('epoch %d, step %d (%.2f %%) '
                          'loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
                              epoch, step, progress_in_epoch * 100, train_loss, bbox_loss, iou_loss, cls_loss,
                              duration))
                    with open(cfg['train_output_dir'] + '/train.log', 'a+') as log:
                        log.write('%d, %d, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (
                            epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))
                    timer.clear()
                    barrier.add(6)
                    # barrier.print()

            # epoch_done
            time_epoch_end = time.time()
            print('{:.2f} seconds for this epoch'.format(time_epoch_end - time_epoch_begin))

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
