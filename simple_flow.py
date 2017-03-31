import plot_util
import flow_util
import cv2


def run_dis_flow():
    for i in range(300, 500):
        if i % 5 == 0:
            img1 = '/home/cory/KITTI_Dataset/data_tracking_image_2/training/image_02/0019/{:06d}.png'.format(i)
        img2 = '/home/cory/KITTI_Dataset/data_tracking_image_2/training/image_02/0019/{:06d}.png'.format(i)

        data2D = flow_util.dis_flow(img2, img1)
        # data2D = data2D.transpose(1, 0, 2)
        flow_img = plot_util.draw_hsv(data2D, ratio=10)
        cv2.imshow('flow', flow_img)
        cv2.imwrite('dis_flow/flow_{:04d}.jpg'.format(i), flow_img)
        press_key = cv2.waitKey(1)
        if press_key == ord('q'):
            break


if '__main__' == __name__:
    run_dis_flow()
