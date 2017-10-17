from yolo_detect import *


def diff_detection(img1, img2, cfg, net, thresh):
    bboxes_1, cls_inds_1, image_1, scores_1 = detect_image(cfg, img1, net, thresh)
    bboxes_2, cls_inds_2, image_2, scores_2 = detect_image(cfg, img2, net, thresh)
    is_cls_equal = False
    if len(cls_inds_1) == len(cls_inds_2):
        is_cls_equal = np.all(np.equal(cls_inds_1, cls_inds_2))

    if not is_cls_equal:
        im2show = yolo_utils.draw_detection(image_1, bboxes_1, scores_1, cls_inds_1, cfg)
        cv2.imshow('detection_1', im2show)
        im2show = yolo_utils.draw_detection(image_2, bboxes_2, scores_2, cls_inds_2, cfg)
        cv2.imshow('detection_2', im2show)
        cv2.waitKey(0)

    return is_cls_equal


def run():
    net, cfg = init_network()
    image_orig_paths = load_image_paths('/home/cory/project/yolo2-pytorch/train_data/kitti/kitti_val_images.txt')
    image_warp_paths = load_image_paths('/home/cory/project/yolo2-pytorch/flow/kitti_val_images_warp.txt')
    n_img = len(image_orig_paths)

    thresh = 0.6

    for i in range(n_img - 1):
        img_orig = image_orig_paths[i]
        img_warp = image_warp_paths[i]
        is_equal = diff_detection(img_orig, img_warp, cfg, net, thresh)


if __name__ == '__main__':
    run()
