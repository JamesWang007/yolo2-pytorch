import os
import shutil

all_images_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_tracking_images.txt'
all_labels_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_tracking_labels.txt'

tracking_raw_dir = '/media/cory/c_disk/Project/KITTI_Dataset/data_tracking_label_2/training/label_02'
tracking_label_output = '/media/cory/c_disk/Project/KITTI_Dataset/trk'


tracklet_count = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803,
                  294, 373, 78, 340, 106, 376, 209, 145, 339, 1059,
                  837]


def convert_tracking_to_detection():
    all_tracking_labels = os.listdir(tracking_raw_dir)
    if not os.path.exists(tracking_label_output):
        os.mkdir(tracking_label_output)
    for merged_file in all_tracking_labels:
        track_id = merged_file.replace('.txt', '')
        print(track_id)
        tk_out = tracking_label_output + '/' + track_id
        shutil.rmtree(tk_out, ignore_errors=True)
        os.mkdir(tk_out)
        f = open(tracking_raw_dir + '/' + merged_file)
        lines = f.readlines()
        num_image = tracklet_count[int(track_id)]
        for i in range(num_image):
            frame_id = '{:06d}'.format(i)
            open(tk_out + '/' + frame_id + '.txt', 'w')

        for line in lines:
            v = line.strip().split(' ')
            frame_id = '{:06d}'.format(int(v[0]))
            data = v[2] + ' ' + ' '.join(v[6:10]) + ' ' + ' '.join(v[3:6]) + '\n'
            of = open(tk_out + '/' + frame_id + '.txt', 'a')
            of.write(data)


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


def main():
    test_set = ['/0001/', '/0005/', '/0013/', '/0017/']
    copy_exclude(all_images_file, 'kitti/kitti_train_images.txt', test_set)
    copy_exclude(all_labels_file, 'kitti/kitti_train_labels.txt', test_set)
    copy_include(all_images_file, 'kitti/kitti_val_images.txt', test_set)
    copy_include(all_labels_file, 'kitti/kitti_val_labels.txt', test_set)

if __name__ == '__main__':
    convert_tracking_to_detection()
    # main()
