import os

all_images_file = '/home/cory/cedl/dashcam/all_images.txt'
all_labels_file = '/home/cory/cedl/dashcam/all_labels.txt'
orig_label_dir = '/home/cory/cedl/dashcam/labels_video'
output_label_dir = '/home/cory/cedl/dashcam/labels'


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


def copy_if(filename, outfilename, patterns):
    with open(outfilename, 'w') as out:
        with open(filename) as f:
            for line in f.readlines():
                matched = True
                for p in patterns:
                    if line.find(p) == -1:
                        matched = False
                        break
                if matched:
                    out.write(line)
                    print(line.strip())


# <class_label> <xmin> <ymin> <xmax> <ymax>
all_class = list()


def gen_each_label():
    for label_file in os.listdir(orig_label_dir):
        id_str = label_file[:label_file.rfind('.')]
        full_path = os.path.join(orig_label_dir, label_file)
        print(id_str, full_path)
        out_dir_video = os.path.join(output_label_dir, id_str)
        if not os.path.exists(out_dir_video):
            os.mkdir(out_dir_video)

        video_label_file = open(full_path)
        labels_per_frame = list()
        for i in range(100):
            labels_per_frame.append(list())

        for line in video_label_file.readlines():
            values = line.strip().split('\t')
            frame = int(values[0])
            classs = values[2].replace('"', '')
            if classs not in all_class:
                all_class.append(classs)
            xmin = int(values[3])
            ymin = int(values[4])
            xmax = int(values[5])
            ymax = int(values[6])
            bundle = (classs, xmin, ymin, xmax, ymax)
            labels_per_frame[frame - 1].append(bundle)

        for frame_i, labels in enumerate(labels_per_frame):
            out_file_name = out_dir_video + '/{:06d}.txt'.format(frame_i + 1)
            print(out_file_name, labels)
            out_file = open(out_file_name, 'w')
            for label in labels:
                print(label)
                out_file.write(' '.join([str(s) for s in label]) + '\n')


if __name__ == '__main__':
    # gen_each_label()

    # exclude 9xx series video
    copy_exclude(all_images_file, 'dashcam_train_images.txt', ['/0009', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'])
    copy_exclude(all_labels_file, 'dashcam_train_labels.txt', ['/0009', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'])
    copy_if(all_images_file, 'dashcam_val_images.txt', ['/0009', '0.'])
    copy_if(all_labels_file, 'dashcam_val_labels.txt', ['/0009', '0.'])
    print(all_class)
