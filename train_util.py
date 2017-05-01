import torch


def lookup_lr(cfg, ep):
    for i in range(len(cfg.lr_epoch) - 1):
        if cfg.lr_epoch[i] <= ep < cfg.lr_epoch[i + 1]:
            return cfg.lr_val[i]
    return cfg.lr_val[- 1]  # last lr


def get_optimizer_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_optimizer(cfg, net, epoch):
    lr = lookup_lr(cfg, epoch)
    optimizer = None
    if cfg.optimizer == 'SGD':
        if cfg.opt_param == 'all':
            optimizer = torch.optim.SGD(params=net.parameters(),
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay,
                                        lr=lr)
        elif cfg.opt_param == 'conv345':
            optimizer = torch.optim.SGD(params=[{'params': net.conv3.parameters()},
                                                {'params': net.conv4.parameters()},
                                                {'params': net.conv5.parameters()}],
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay,
                                        lr=lr)
    elif cfg.optimizer == 'Adam':
        if cfg.opt_param == 'all':
            optimizer = torch.optim.Adam(params=net.parameters(),
                                         weight_decay=cfg.weight_decay,
                                         lr=lr)
        elif cfg.opt_param == 'conv345':
            optimizer = torch.optim.Adam(params=[{'params': net.conv3.parameters()},
                                                 {'params': net.conv4.parameters()},
                                                 {'params': net.conv5.parameters()}],
                                         weight_decay=cfg.weight_decay,
                                         lr=lr)
    assert optimizer is not None

    print('optimizer_lr =', get_optimizer_lr(optimizer))
    return optimizer
