

all_img_file = '/media/cory/54604BF5604BDBFC/Project/KITTI_Dataset/kitti_tracking_images.txt'
all_gt_file = '/media/cory/54604BF5604BDBFC/Project/KITTI_Dataset/kitti_tracking_gt.txt'


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
                    print(line)

copy_exclude(all_img_file, 'imgs_exclude_1_19.txt', ['/0001/', '/0019/'])
copy_exclude(all_gt_file, 'gt_exclude_1_19.txt', ['/0001/', '/0019/'])
