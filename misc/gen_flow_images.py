import os

import cv2

from misc import flow_util


def find_tracklet_id(img_path):
    str_offset = img_path.rfind('/')
    tracklet_id = img_path[str_offset - 4: str_offset]
    return tracklet_id


def gen_images():
    img_files = open('/home/cory/yolo2-pytorch/train_data/kitti/kitti_train_images.txt')
    image_abs_paths = img_files.readlines()
    image_abs_paths = [f.strip() for f in image_abs_paths]

    out_img_dir = 'images_flow_warp'
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

        img_flow = flow_util.spynet_flow(curr_img_path, prev_img_path)
        prev_frame = cv2.imread(prev_img_path)
        img_warp = flow_util.warp_flow(prev_frame, img_flow)

        out_path = curr_img_path.replace('.png', '_w.png')
        out_path = tracklet_out_path + '/' + out_path[out_path.rfind('/') + 1:]
        print(out_path)
        cv2.imshow('img_warp', img_warp)
        cv2.imwrite(out_path, img_warp)
        cv2.waitKey(1)


if __name__ == '__main__':
    gen_images()
