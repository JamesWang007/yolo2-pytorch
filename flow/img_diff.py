import cv2
import random


def diff(img1, img2, window_name=''):
    df = img1 - img2
    cv2.imshow(window_name, df)
    cv2.imwrite(window_name + '.jpg', df)


def main():
    img0 = cv2.imread('/media/cory/c_disk/Project/KITTI_Dataset/data_tracking_image_2/training/image_02/0003/000035.png')
    img1 = cv2.imread('/home/cory/project/yolo2-pytorch/flow/images_flow_warp/0003/000035_w01.png')
    img2 = cv2.imread('/home/cory/project/yolo2-pytorch/flow/images_flow_warp/0003/000035_w10.png')
    cv2.imshow('0', img0)
    cv2.imshow('1', img1)
    cv2.imshow('2', img2)
    diff(img1, img0, '1-0')
    diff(img2, img0, '2-0')
    diff(img1, img2, '1-2')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
