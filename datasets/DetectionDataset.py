import PIL.Image as Image
import torch
import torch.utils.data as data
from torch.autograd import Variable

from datasets.DetectionDatasetHelper import *


class DetectionDataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        if self.mode == 'train':
            self.batch_size = cfg['train_batch_size']
            self.image_list_file = cfg['train_images']
            self.label_list_file = cfg['train_labels']
        else:
            self.batch_size = cfg['val_batch_size']
            self.image_list_file = cfg['val_images']
            self.label_list_file = cfg['val_labels']

        self.image_paths = list()
        self.annotations = list()
        self.image_indexes = list()
        self.classes_ = cfg['label_names']
        self.load_dataset(self.classes_)

        # use cfg for default input size, but it  will change every 10 batch (refer to DataLoaderX)
        self.inp_size = cfg['inp_size']

    def __getitem__(self, index):
        raise NotImplemented

    def __len__(self):
        return len(self.image_paths)

    def get_train_data(self, index, network_size):
        img = Image.open(self.image_paths[index]).convert('RGB')
        gt = self.annotations[index]
        gt.update({'img_size': img.size})

        # random transforms (scale, color, flip)
        im, boxes = affine_transform(img, gt['boxes'], network_size)
        gt.update({'boxes': boxes})
        target_np = encode_to_np(gt)
        im_tensor = torch.from_numpy(im.transpose((2, 0, 1))).float()
        return im_tensor, target_np

    def input_size(self):
        return self.inp_size

    def change_input_size_rand(self):
        # call this function to change input size randomly from cfg['inp_size_candidates']
        # random change network size
        rand_id = np.random.randint(0, len(self.cfg['inp_size_candidates']))
        rand_network_size = self.cfg['inp_size_candidates'][rand_id]
        self.inp_size = rand_network_size
        # print('change_input_size_rand', rand_network_size)

    def load_dataset(self, label_map):
        remove_id_list = list()
        try:
            img_file = open(self.image_list_file)
            self.image_paths = [line.strip() for line in img_file.readlines()]
            gt_file = open(self.label_list_file)
            for fi, label_file_name in enumerate(gt_file.readlines()):
                label_file_name = label_file_name.strip()
                label_dict = parse_label_file(label_file_name, label_map)
                if not label_dict['has_label']:
                    remove_id_list.append(fi)
                self.annotations.append(label_dict)
        except Exception as e:
            raise e

        self.image_paths = np.delete(self.image_paths, remove_id_list)
        self.annotations = np.delete(self.annotations, remove_id_list)
        print('dataset size =', len(self.image_paths), ' (delete', len(remove_id_list), ')')
        assert len(self.image_paths) == len(self.annotations)
        self.image_indexes = range(len(self.image_paths))


def test_detection_dataset():
    from cfgs.config_v2 import add_cfg
    dataset_yaml = '/home/cory/project/yolo2-pytorch/cfgs/config_kitti.yaml'
    exp_yaml = '/home/cory/project/yolo2-pytorch/cfgs/exps/kitti/kitti_baseline_v3.yaml'
    cfg = dict()
    add_cfg(cfg, dataset_yaml)
    add_cfg(cfg, exp_yaml)
    dataset = DetectionDataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                             shuffle=True, num_workers=4)
    for i, data in enumerate(dataloader):
        # get the inputs
        print(i)
        inputs, labels = data
        print(inputs.size(), labels.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs), labels


if __name__ == '__main__':
    test_detection_dataset()
