import numpy as np
from flow.flow_util import dis_flow
from flow.gen_flow_images import parse_label_file, write_to_file
from misc.visualize_gt import plot_vis
import matplotlib.pyplot as plt


def flow_avg_in_rectangle(flow, pos):
    pos = list(map(int, pos))
    crop = flow[pos[1]: pos[3], pos[0]: pos[2]]
    avg_x = np.average(crop[:, :, 1])
    avg_y = np.average(crop[:, :, 0])
    return avg_x, avg_y


def flow_std_in_rectangle(flow, pos):
    pos = list(map(int, pos))
    crop = flow[pos[1]: pos[3], pos[0]: pos[2]]
    std_x = np.std(crop[:, :, 1])
    std_y = np.std(crop[:, :, 0])
    return std_x, std_y


def gt_save_to_file(gt, filepath):
    print(filepath)
    out_file = open(filepath, 'w')
    for g in gt:
        gs = [str(int(x)) for x in g[1: 5]]
        wline = g[0] + ' ' + ' '.join(gs) + ' 0 0 0\n'
        out_file.write(wline)


def shift_gt_by_flow():
    img_list_filename = 'w01_images.txt'
    # gt_list_filename = 'w01_center_labels.txt'
    gt_list_filename = 'kitti_train_labels.txt'
    img_list_file = open(img_list_filename)
    gt_list_file = open(gt_list_filename)

    img_paths = [f.strip() for f in img_list_file.readlines()]
    gt_paths = [f.strip() for f in gt_list_file.readlines()]

    total_num = len(img_paths)
    print(total_num)

    pt_x = list()
    pt_y = list()
    for i in range(total_num - 1):
        img_file = open(img_paths[i])
        out_gt_filepath = gt_paths[i].replace('.txt', '_shift.txt')
        gts = parse_label_file(gt_paths[i])
        # print(gts)
        print(i)

        flow = dis_flow(img_paths[i + 1], img_paths[i])
        for gt in gts:
            std_flow = flow_std_in_rectangle(flow, gt[1:5])
            pt_x.append(std_flow[0])
            pt_y.append(std_flow[1])
            if abs(std_flow[0]) > 2 or abs(std_flow[1]) > 5:
                print(gt[0], std_flow[0], std_flow[1])
                gt[0] = 'DontCare'
                print(gt)

        # r = plot_vis(img_paths[i], gts)
        gt_save_to_file(gts, out_gt_filepath)

    plt.plot(pt_x, pt_y, '*')
    plt.show()

if __name__ == '__main__':
    shift_gt_by_flow()
