import time

import torch
import torch.utils.data as data
from torch.autograd import Variable

from datasets.DetectionDataset import DetectionDataset
from datasets.DataLoaderIterX import DataLoaderIterX


# modify /usr/local/lib/python3.5/dist-packages/torch/utils/data/__init__.py
# add this line: from .dataloader import DataLoaderIter
# thus, let data.DataLoaderIter class become publicly available to inherent
# class DataLoaderIterX(data.DataLoaderIter):
#     pass


class DataLoaderX(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=1,
                 pin_memory=False, drop_last=False):
        super(DataLoaderX, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def __iter__(self):
        return DataLoaderIterX(self)


def test_detection_dataset():
    from cfgs.config_v2 import add_cfg
    dataset_yaml = '/home/cory/project/yolo2-pytorch/cfgs/config_detrac.yaml'
    exp_yaml = '/home/cory/project/yolo2-pytorch/cfgs/exps/detrac/detrac_baseline.yaml'
    cfg = dict()
    add_cfg(cfg, dataset_yaml)
    add_cfg(cfg, exp_yaml)
    dataset = DetectionDataset(cfg)
    num_workers = 4
    batch_size = 16
    dataloader = DataLoaderX(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    t0 = time.time()
    for i, data in enumerate(dataloader):
        if i > 100:
            break

        # get the inputs
        inputs, labels = data
        print(i, inputs.size(), labels.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), labels
        import numpy as np
        assert np.sum(inputs.data.cpu().numpy()) > 0
    t1 = time.time()
    print(t1 - t0)


if __name__ == '__main__':
    test_detection_dataset()
