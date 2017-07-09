import torch
from train.yellowfin import YFOptimizer


def lookup_lr(cfg, ep):
    lr_epochs = cfg['lr_epoch']
    lr_vals = cfg['lr_val']
    for i in range(len(lr_epochs) - 1):
        if lr_epochs[i] <= ep < lr_epochs[i + 1]:
            return lr_vals[i]
    return lr_vals[- 1]  # last lr


def get_optimizer_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_optimizer(cfg, net, epoch):
    lr = lookup_lr(cfg, epoch)
    optimizer = None
    if cfg['optimizer'] == 'SGD':
        if cfg['opt_param'] == 'all':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        momentum=cfg['momentum'],
                                        weight_decay=cfg['weight_decay'],
                                        nesterov=True,
                                        lr=lr)
        elif cfg['opt_param'] == 'conv345':
            optimizer = torch.optim.SGD(params=[{'params': net.conv3.parameters()},
                                                {'params': net.conv4.parameters()},
                                                {'params': net.conv5.parameters()}],
                                        momentum=cfg['momentum'],
                                        weight_decay=cfg['weight_decay'],
                                        nesterov=True,
                                        lr=lr)
    elif cfg['optimizer'] == 'Adam':
        if cfg['opt_param'] == 'all':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         weight_decay=cfg['weight_decay'],
                                         lr=lr)
        elif cfg['opt_param'] == 'conv345':
            optimizer = torch.optim.Adam(params=[{'params': net.conv3.parameters()},
                                                 {'params': net.conv4.parameters()},
                                                 {'params': net.conv5.parameters()}],
                                         weight_decay=cfg['weight_decay'],
                                         lr=lr)
    elif cfg['optimizer'] == 'YF':
        if cfg['opt_param'] == 'all':
            optimizer = YFOptimizer(var_list=net.parameters())
        elif cfg['opt_param'] == 'conv345':
            optimizer = YFOptimizer(var_list=[{'params': net.conv3.parameters()},
                                              {'params': net.conv4.parameters()},
                                              {'params': net.conv5.parameters()}])

    assert optimizer is not None

    print('optimizer_lr =', get_optimizer_lr(optimizer))
    return optimizer
