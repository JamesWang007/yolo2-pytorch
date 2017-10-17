import os

import cv2

from flow import flow_util


def find_tracklet_id(img_path):
    str_offset = img_path.rfind('/')
    tracklet_id = img_path[str_offset - 5: str_offset]
    return tracklet_id


def gen_warp(img_path_0, img_path_1):
    # img_flow = flow_util.spynet_flow(img_path_1, img_path_0)
    img_flow = flow_util.dis_flow(img_path_1, img_path_0)
    img_0 = cv2.imread(img_path_0)
    img_warp = flow_util.warp_flow(img_0, img_flow)
    return img_warp


def parse_label_file(label_file_path):
    label_file = open(label_file_path)
    vlist = list()
    for l in label_file.readlines():
        v = l.split(' ')[0:5]
        if len(v) <= 1:
            continue
        v[1:5] = list(map(float, v[1:5]))
        if v[1] < 50 or v[2] < 50 or v[3] > 900 or v[4] > 500:
            v[0] = 'DontCare'
        vlist.append(v)
        # print(v)
    return vlist


def write_to_file(labels, filename):
    curr_label = parse_label_file(labels)
    new_label_file = open(filename, 'w')
    for v in curr_label:
        line = ' '.join([str(x) for x in v])
        # print(line)
        new_label_file.write(line + '\n')


def gen_images(gen_label_only=False):

    img_files = open('/home/cory/project/yolo2-pytorch/train_data/detrac/detrac_train_images.txt')
    image_abs_paths = img_files.readlines()
    image_abs_paths = [f.strip() for f in image_abs_paths]

    label_files = open('/home/cory/project/yolo2-pytorch/train_data/detrac/detrac_train_labels.txt')
    label_abs_paths = label_files.readlines()
    label_abs_paths = [f.strip() for f in label_abs_paths]

    out_img_dir = 'images_flow_warp_detrac'
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

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

        if not gen_label_only:
            # w(0 -> 1) = frame(0) * flow(1 -> 0)
            w01 = gen_warp(prev_img_path, curr_img_path)
        out_path = curr_img_path.replace('.png', '')
        out_path = tracklet_out_path + '/' + out_path[out_path.rfind('/') + 1:]
        w01_path = out_path + '_w01.png'

        write_to_file(label_abs_paths[i], out_path + '_w01_label.txt')

        out_path = prev_img_path.replace('.png', '')
        out_path = tracklet_out_path + '/' + out_path[out_path.rfind('/') + 1:]
        w10_path = out_path + '_w10.png'
        write_to_file(label_abs_paths[i], out_path + '_w10_label.txt')

        if not gen_label_only:
            # w(1 -> 0) = frame(1) * flow(0 -> 1)
            # w10 = gen_warp(curr_img_path, prev_img_path)
            cv2.imshow('w01', w01)
            # cv2.imshow('w10', w10)

            os.makedirs(w01_path[0: w01_path.rfind('/')], exist_ok=True)
            cv2.imwrite(w01_path, w01)
            # cv2.imwrite(w10_path, w10)
            cv2.waitKey(1)


if __name__ == '__main__':
    gen_images()
