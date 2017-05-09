

all_images_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_tracking_images.txt'
all_labels_file = '/media/cory/c_disk/Project/KITTI_Dataset/kitti_tracking_gt.txt'


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


copy_exclude(all_images_file, 'kitti/kitti_train_images.txt', ['/0001/', '/0019/'])
copy_exclude(all_labels_file, 'kitti/kitti_train_labels.txt', ['/0001/', '/0019/'])
copy_include(all_images_file, 'kitti/kitti_val_images.txt', ['/0001/', '/0019/'])
copy_include(all_labels_file, 'kitti/kitti_val_labels.txt', ['/0001/', '/0019/'])
