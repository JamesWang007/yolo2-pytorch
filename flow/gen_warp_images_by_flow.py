import numpy as np
import cv2
from flow.plot_util import *
from flow.flow_util import *
from flow.gen_flow_images import find_tracklet_id


def gen_images():

    out_img_dir = 'images_flow_warp_flownet2'
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    img_files = open('/home/cory/project/yolo2-pytorch/train_data/kitti/kitti_train_images.txt')
    image_abs_paths = img_files.readlines()
    image_abs_paths = [f.strip() for f in image_abs_paths]

    pwd = '/home/cory/project/flownet2/output/'
    flo_list = list()
    for flo in sorted(os.listdir(pwd)):
        ff = pwd + flo
        flo_list.append(ff)

    for i in range(0, len(image_abs_paths)):
        curr_img_path = image_abs_paths[i]
        prev_img_path = image_abs_paths[i - 1]

        curr_tracklet_id = find_tracklet_id(curr_img_path)
        prev_tracklet_id = find_tracklet_id(prev_img_path)

        print(i, curr_img_path, curr_tracklet_id)

        tracklet_out_path = out_img_dir + '/' + curr_tracklet_id
        if not os.path.exists(tracklet_out_path):
            os.mkdir(tracklet_out_path)

        if curr_tracklet_id != prev_tracklet_id:
            prev_img_path = curr_img_path

        # w(0 -> 1) = frame(0) * flow(1 -> 0)
        print(flo_list[i])
        flo = read_flo_file(flo_list[i])
        flow_hsv = draw_hsv(flo, ratio=2)
        cv2.imshow('flow', flow_hsv)

        w01 = warp_flow(cv2.imread(prev_img_path), flo)
        out_path = curr_img_path.replace('.png', '')
        out_path = tracklet_out_path + '/' + out_path[out_path.rfind('/') + 1:]
        w01_path = out_path + '_w01.png'
        cv2.imshow('w01', w01)

        cv2.imwrite(out_path + '_flow.png', flow_hsv)
        cv2.imwrite(w01_path, w01)

        cv2.waitKey(30)


if __name__ == '__main__':
    gen_images()
