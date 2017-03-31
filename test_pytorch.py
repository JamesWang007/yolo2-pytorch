import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import json
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils.network import *
from visualize import make_dot

imagenet_index = json.load(open('imagenet_index.json', 'r'))


transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

dset_img = dset.ImageFolder(root='./', transform=transform)
print(dset_img.imgs)
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
resnet18.cuda()

for x, img in enumerate(dset_img):
    img = img[0]
    print(type(img), img.size())

    img0_as_batch = img.unsqueeze(0).cuda()
    ii = Variable(img0_as_batch)
    t1 = time.time()
    ret = resnet18.forward(ii)
    t2 = time.time()
    print('elapsed time {:f}'.format(t2 - t1))
    top5_val, top5_idx = ret[0, :].topk(5)

    print(dset_img.imgs[x])
    for i in range(5):
        print(imagenet_index[str(top5_idx[i].data[0])], top5_val[i].data[0])
    print()

print(resnet18)
