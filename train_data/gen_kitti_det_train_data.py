import os

kitti_det_label_path = '/home/cory/KITTI_Dataset/data_object_image_2/training/label_2'
out_label_path = '/home/cory/KITTI_Dataset/detection_label'

all_images_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_detection_images.txt'
all_labels_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_detection_labels.txt'


def copy_exclude(filename, outfilename, patterns):
    with open(outfilename, 'w') as out:
        with open(filename) as f:
            for line in f.readlines():
                pattern_found = False
                for p in patterns:
                    if line.find(p) >= 0:
                        pattern_found = True
                if not pattern_found:
                    out.write(line)
                    print(line.strip())


def copy_include(filename, outfilename, patterns):
    with open(outfilename, 'w') as out:
        with open(filename) as f:
            for line in f.readlines():
                for p in patterns:
                    if line.find(p) >= 0:
                        print(line.strip())
                        out.write(line)
                        break


def convert_file(infile_path, outfile_path):
    # 'Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01'
    infile = open(infile_path)
    outfile = open(outfile_path, 'w')
    for line in infile.readlines():
        v = line.strip().split(' ')
        bb = list(map(str, map(int, map(float, v[4:8]))))
        outfile.write(v[0] + ' ' + ' '.join(bb) + '\n')


def convert_format():
    file_list = os.listdir(kitti_det_label_path)
    file_list.sort()
    for f in file_list:
        infile_path = kitti_det_label_path + '/' + f
        outfile_path = out_label_path + '/' + f
        convert_file(infile_path, outfile_path)

        print(infile_path, outfile_path)

    print(len(file_list))


def main():
    copy_exclude(all_images_file, 'kitti/kitti_det_train_images.txt', ['/006', '/007'])
    copy_exclude(all_labels_file, 'kitti/kitti_det_train_labels.txt', ['/006', '/007'])
    copy_include(all_images_file, 'kitti/kitti_det_val_images.txt', ['/006', '/007'])
    copy_include(all_labels_file, 'kitti/kitti_det_val_labels.txt', ['/006', '/007'])

if __name__ == '__main__':
    # convert_format()
    main()
