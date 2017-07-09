import cv2


def plot_vis(image_path, label_path):
    img = cv2.imread(image_path)
    print(img.shape)

    label_file = open(label_path)
    for label in label_file.readlines():
        values = label.strip().split(' ')
        label = values[0]
        official_format = False
        label = label.replace('DontCare', '')
        if official_format:
            xmin = int(float(values[4]))
            ymin = int(float(values[5]))
            xmax = int(float(values[6]))
            ymax = int(float(values[7]))
        else:
            xmin = int(float(values[1]))
            ymin = int(float(values[2]))
            xmax = int(float(values[3]))
            ymax = int(float(values[4]))

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(img, label, (xmin, ymax), cv2.FORMATTER_FMT_CSV, 1, (0, 255, 0), 1, cv2.LINE_AA)
        print(values)

    cv2.imshow('img', img)
    cv2.imwrite('vis.jpg', img)
    key = cv2.waitKey()
    print(key)
    if key == ord('q'):
        return -1
    return 0


def run_vis():
    choice = 5
    if choice == 1:
        image_path = '/home/cory/cedl/dashcam/images/000900/000010.jpg'
        label_path = '/home/cory/cedl/dashcam/labels/000900/000010.txt'
    elif choice == 2:
        image_path = '/home/cory/KITTI_Dataset/data_tracking_image_2/training/image_02/0000/000000.png'
        label_path = '/home/cory/KITTI_Dataset/tracking_label/0000/000000.txt'
    elif choice == 3:
        image_path = '/home/cory/VOC/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
        label_path = '/home/cory/VOC/VOCdevkit/VOC2007/labels/000009.txt'
    elif choice == 4:
        image_path = '/home/cory/GTAV/VOCdevkit/VOC2012/JPEGImages/3384645.jpg'
        label_path = '/home/cory/GTAV/VOCdevkit/VOC2012/labels/3384645.txt'

    plot_vis(image_path, label_path)


def vis_list_file():
    # image_path = '/home/cory/yolo2-pytorch/train_data/voc/voc_train_images.txt'
    # label_path = '/home/cory/yolo2-pytorch/train_data/voc/voc_train_labels.txt'
    # image_path = '/media/cory/BackUp/ImageNet/vid_all_images.txt'
    # label_path = '/media/cory/BackUp/ImageNet/vid_all_labels.txt'
    # image_path = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_train_images.txt'
    # label_path = '/home/cory/yolo2-pytorch/train_data/kitti/kitti_train_labels.txt'

    image_path = '/home/cory/project/yolo2-pytorch/flow/warp_w01_imgs.txt'
    label_path = '/home/cory/project/yolo2-pytorch/flow/warp_center_labels.txt'

    image_file = open(image_path)
    label_file = open(label_path)
    images = [p.strip() for p in image_file.readlines()]
    labels = [p.strip() for p in label_file.readlines()]
    for i in range(len(images)):
        print(images[i], labels[i])
        r = plot_vis(images[i], labels[i])
        if r == -1:
            break

if __name__ == '__main__':
    vis_list_file()
    # run_vis()
