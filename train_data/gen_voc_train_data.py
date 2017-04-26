all_images = '/home/cory/VOC/train.txt'
all_labels = '/home/cory/VOC/VOC0712_labels.txt'
voc07_prefix = '/home/cory/VOC/VOCdevkit/VOC2007/'
voc12_prefix = '/home/cory/VOC/VOCdevkit/VOC2012/'

out_train_image_file_name = 'voc_train_images.txt'
out_train_label_file_name = 'voc_train_labels.txt'
out_test_image_file_name = 'voc_test_images.txt'
out_test_label_file_name = 'voc_test_labels.txt'


def fid_to_fullpath(fid):
    if len(fid) == 6:
        prefix = voc07_prefix
    else:
        prefix = voc12_prefix
    full_image_path = prefix + 'JPEGImages/' + fid + '.jpg'
    full_label_path = prefix + 'labels/' + fid + '.txt'
    return full_image_path, full_label_path


def gen_voc_data():

    all_image_file = open(all_images)
    counter = 0
    train_file_id = dict()
    for f in all_image_file.readlines():
        f = f.strip()
        begin_pos = f.rfind('/') + 1
        end_pos = f.find('.jpg')
        fid = f[begin_pos: end_pos]
        train_file_id.update({fid: True})
        print(counter, fid)
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
            print(test_counter, full_image_path, full_label_path)
            out_test_image_file.write(full_image_path + '\n')
            out_test_label_file.write(full_label_path + '\n')
            test_counter += 1

if __name__ == '__main__':
    gen_voc_data()
