import os

all_labels = 'VOC/voc0712_labels.txt'
train_images = 'VOC/train_images.txt'
train_labels = 'VOC/train_labels.txt'


prefix = '/home/cory/VOC/VOCdevkit/VOC2007/labels/'


def sort_file_line(filename):
    f = open(filename)
    sorted_line = sorted(f.readlines())
    for line in sorted_line:
        print(line, end='')

sort_file_line(train_images)


def get_filename_id(fullpath):
    filename_begin_pos = fullpath.rfind('/') + 1
    filename_end_pos = fullpath.rfind('.')
    fname = fullpath[filename_begin_pos: filename_end_pos]
    return fname


def convert_main():
    all_id = [get_filename_id(f.strip()) for f in all_labels.readlines()]
    train_id = [get_filename_id(f.strip()) for f in train_images.readlines()]

    train_counter = 0
    test_counter = 0
    for id in all_id:
        if id in train_id:
            # print(train_counter, id)
            print(prefix + id + '.txt')
            train_counter += 1
        else:
            # print(test_counter, id, 'test')
            # print(prefix + id + '.txt')
            test_counter += 1

    print('total label:', len(all_id))
    print('train', train_counter)
    print('test', test_counter)
