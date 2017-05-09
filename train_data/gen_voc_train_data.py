# modified from https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
#
# convert VOC xml to simple format
# multiple lines in file, one file for one image (with same filename prefix)
# each line in label file represents an object
# <class_label> <xmin> <ymin> <xmax> <ymax>
# the range of label's position is 0 ~ size-1, not floating point (0 ~ 1)

import os
import subprocess
import xml.etree.ElementTree as ET

# specify VOC path first!
VOC_PATH = '/home/cory/VOC/'

sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id):
    in_file = open(VOC_PATH + 'VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    out_file = open(VOC_PATH + 'VOCdevkit/VOC%s/labels_px/%s.txt' % (year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        out_file.write(cls + " " + " ".join([str(int(a) - 1) for a in (b[0], b[2], b[1], b[3])]) + '\n')


def gen_file_list():
    for year, image_set in sets:
        if not os.path.exists(VOC_PATH + 'VOCdevkit/VOC%s/labels_px/' % (year)):
            os.makedirs(VOC_PATH + 'VOCdevkit/VOC%s/labels_px/' % (year))
        image_ids = open(VOC_PATH + 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
        list_file = open(VOC_PATH + '%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write(VOC_PATH + 'VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' % (year, image_id))
            convert_annotation(year, image_id)
        list_file.close()


all_images = VOC_PATH + 'VOC0712_train_images.txt'
all_labels = VOC_PATH + 'VOC0712_all_labels.txt'
voc07_prefix = VOC_PATH + 'VOCdevkit/VOC2007/'
voc12_prefix = VOC_PATH + 'VOCdevkit/VOC2012/'

out_train_image_file_name = 'voc/voc_train_images.txt'
out_train_label_file_name = 'voc/voc_train_labels.txt'
out_test_image_file_name = 'voc/voc_test_images.txt'
out_test_label_file_name = 'voc/voc_test_labels.txt'


def fid_to_fullpath(fid):
    if len(fid) == 6:
        prefix = voc07_prefix
    else:
        prefix = voc12_prefix
    full_image_path = prefix + 'JPEGImages/' + fid + '.jpg'
    full_label_path = prefix + 'labels_px/' + fid + '.txt'
    return full_image_path, full_label_path

# open terminal and cd to VOC path
# run these commands to gen file list
# cat 2007_train.txt 2007_val.txt 2012_*.txt > train_images.txt
# realpath VOCdevkit/*/labels/*.txt > all_labels.txt
#
# test_images.txt & test_labels should both contain 4952 lines
# train_images.txt & train_images.txt should both contain 16551 lines


def gen_voc_train_data():
    subprocess.call('cd ' + VOC_PATH + ' && cat 2007_train.txt 2007_val.txt 2012_*.txt > ' + all_images,
                    shell=True)
    subprocess.call('cd ' + VOC_PATH + ' && realpath VOCdevkit/*/labels_px/*.txt > ' + all_labels,
                    shell=True)

    all_image_file = open(all_images)
    counter = 0
    train_file_id = dict()
    for f in all_image_file.readlines():
        f = f.strip()
        begin_pos = f.rfind('/') + 1
        end_pos = f.find('.jpg')
        fid = f[begin_pos: end_pos]
        train_file_id.update({fid: True})
        # print(counter, fid)
        counter += 1

    train_counter = 0
    test_counter = 0
    all_label_file = open(all_labels)
    out_train_image_file = open(out_train_image_file_name, 'w')
    out_train_label_file = open(out_train_label_file_name, 'w')
    out_test_image_file = open(out_test_image_file_name, 'w')
    out_test_label_file = open(out_test_label_file_name, 'w')

    for f in all_label_file.readlines():
        f = f.strip()
        begin_pos = f.rfind('/') + 1
        end_pos = f.find('.txt')
        fid = f[begin_pos: end_pos]
        full_image_path, full_label_path = fid_to_fullpath(fid)
        if train_file_id.get(fid):
            # print(train_counter, full_image_path, full_label_path)
            out_train_image_file.write(full_image_path + '\n')
            out_train_label_file.write(full_label_path + '\n')
            train_counter += 1
        else:
            # print(test_counter, full_image_path, full_label_path)
            out_test_image_file.write(full_image_path + '\n')
            out_test_label_file.write(full_label_path + '\n')
            test_counter += 1

if __name__ == '__main__':
    gen_file_list()
    gen_voc_train_data()
