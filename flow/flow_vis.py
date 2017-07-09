import numpy as np
import cv2
from flow.plot_util import *
from flow.flow_util import *

pwd = '/home/cory/project/flownet2/output/'
flos = sorted(os.listdir(pwd))
for flo in flos:
    ff = read_flo_file(pwd + flo)
    flow_hsv = draw_hsv(ff, ratio=4)
    cv2.imshow('flow', flow_hsv)
    cv2.waitKey(10)
