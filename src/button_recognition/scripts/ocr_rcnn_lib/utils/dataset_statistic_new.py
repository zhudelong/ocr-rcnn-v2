from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pickle
import shutil
import numpy as np
from lxml import etree
import tensorflow as tf
import matplotlib.pyplot as plt
from python.datasets import dataset_util

images_path = '/home/zhudelong/dataset/elevator_panel_database/dataset/images'
annotations_path = '/home/zhudelong/dataset/elevator_panel_database/dataset/annotations'
filtered_image_set = []
button_ratio_list = []
button_size_list = []
image_size_list = []
filter_target_count = 0

charset = {'0': 0,  '1': 1,  '2': 2,  '3': 3,  '4': 4,  '5': 5,
           '6': 6,  '7': 7,  '8': 8,  '9': 9,  'A': 10, 'B': 11,
           'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
           'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
           'O': 24, 'P': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29,
           'V': 30, 'X': 31, 'Z': 32, '<': 33, '>': 34, '(': 35,
           ')': 36, '$': 37, '#': 38, '^': 39, 's': 40, '-': 41,
           '*': 42, '%': 43, '?': 44, '!': 45, '<nul>': 46}

# elevator button category definition
filter_target = ['led', 'switch', 'CH']
standard_func_button = ['open', 'close', 'alarm', 'stop', 'call', 'up',
                        'down', 'updown', 's', '-']
special_func_button = ['fan', 'fire', 'hat', 'key', 'light',
                       'bt_keyhole', 'keyhole', 'bt_switch']
texture_less_button = ['speaker', 'indicator', 'empty', '.', '*']

number_button = re.compile(r'\d{1,3}$|-\d{1,3}$|s\d{1,3}$|'
                           r'\d[A-Z]{1,2}$|-\d[A-Z]{1,2}$|s\d[A-Z]{1,2}$|'
                           r'\d{1,2}[A-Z]$|-\d{1,2}[A-Z]$|s\d{1,2}[A-Z]$')
letter_button = re.compile(r'[A-Z]{1,3}$|-[A-Z]{1,3}$|s[A-Z]{1,3}$|'
                           r'[A-Z]{1,2}\d$|-[A-Z]{1,2}\d$|s[A-Z]{1,2}\d$|'
                           r'[A-Z]\d{1,2}$|-[A-Z]\d{1,2}$|s[A-Z]\d{1,2}$|'
                           r'[A-Z]\d[A-Z]$|-[A-Z]\d[A-Z]$|s[A-Z]\d[A-Z]$')

text_button = re.compile(r'\Atext')  # text_*
hazy_button = ['blur', 'unknown']
dict_statistic = {}


def separate_label_string(button_label, level='label'):
    if level == 'letter':
        # each label is a string and a list as well
        for letter in button_label:
            if letter in dict_statistic.keys():
                dict_statistic[letter] += 1
            else:
                dict_statistic[letter] = 1
    else:
        if button_label in dict_statistic.keys():
            dict_statistic[button_label] += 1
        else:
            dict_statistic[button_label] = 1


def categorize_button_labels(button_label, button_name):
    category_count = 0
    category_name = ''
    trans_name = ''

    if button_label in filter_target:
        category_name = 'category_name'
        category_count += 1
        return category_name, None

    if button_label in standard_func_button:
        category_name = 'standard_func_button'
        category_count += 1
        if button_label == 'open':
            trans_name = '<>'
            separate_label_string(trans_name)
        elif button_label == 'close':
            trans_name = '><'
            separate_label_string(trans_name)
        elif button_label == 'alarm':
            trans_name = '$'
            separate_label_string(trans_name)
        elif button_label == 'stop':
            trans_name = '#'
            separate_label_string(trans_name)
        elif button_label == 'call':
            trans_name = '^'
            separate_label_string(trans_name)
        elif button_label == 'up':
            trans_name = '('
            separate_label_string(trans_name)
        elif button_label == 'down':
            trans_name = ')'
            separate_label_string(trans_name)
        elif button_label == 'updown':
            trans_name = '()'
            separate_label_string(trans_name)
        elif button_label == 's':
            trans_name = 's'
            separate_label_string(trans_name)
        elif button_label == '-':
            trans_name = '-'
            separate_label_string(trans_name)
        else:
            print('{} in {} is illegal!'.format(
                button_label, button_name))

    if button_label in texture_less_button:
        category_name = 'texture_less_button'
        category_count += 1
        trans_name = '*'
        separate_label_string(trans_name)  # empty

    if button_label in special_func_button:
        category_name = 'special_func_button'
        category_count += 1
        trans_name = '%'
        separate_label_string(trans_name)

    if button_label in hazy_button:
        category_name = 'hazy_button'
        category_count += 1
        trans_name = '?'
        separate_label_string(trans_name)

    if text_button.match(button_label):
        category_name = 'text_button'
        category_count += 1
        trans_name = "!!!"
        separate_label_string(trans_name)

    if number_button.match(button_label):
        category_name = 'number_button'
        category_count += 1
        trans_name = button_label
        separate_label_string(trans_name)

    if letter_button.match(button_label):
        category_name = 'letter_button'
        category_count += 1
        trans_name = button_label
        separate_label_string(trans_name)

    if category_count != 1 and button_label != 'CH':
        print('{} in {} is illegal!'.format(button_label, button_name))
        raise IOError('CH ERROR!')
    if button_label == 'emergency':
        print('{} in {} is illegal!'.format(button_label, button_name))
        raise IOError('ILLEGAL ERROR!')
    return category_name, trans_name


def get_image_name_list(target_path):
    if target_path is None:
        raise IOError('Target path cannot be found!')
    image_name_list = []
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
        for image_name in files:
            image_name_list.append(image_name.split('.')[0])
    return image_name_list


def object_size_distribution(data):
    object_list = data['object']
    if len(object_list) == 0:
        print('object list is empty!')
    for obj in object_list:
        button_type = obj['name']
        if button_type in filter_target:
            filtered_image_set.append(button_type)
            continue
        button_width = float(obj['bndbox']['xmax']) - float(obj['bndbox']['xmin'])
        button_height = float(obj['bndbox']['ymax']) - float(obj['bndbox']['ymin'])
        button_size_list.append([button_width, button_height])
        button_ratio = 1.0 * button_width / button_height
        button_ratio_list.append(button_ratio)


def object_statistics():
    print('filtered number is {}'.format(len(filtered_image_set)))
    print('button number is {}'.format(len(button_ratio_list)))
    button_ratio_np = np.asarray(button_ratio_list)
    print("the shape, max, min of button ration is {}, {}, {}".format(
        button_ratio_np.shape, button_ratio_np.max(), button_ratio_np.min()))
    bins = np.asarray(range(0, 60, 1)) / 10
    ration_hist, ratio_bins = np.histogram(button_ratio_np, bins=bins)
    print("the ration hist and bins is {}, {}".format(ration_hist, ratio_bins))
    button_size_np = np.asarray(button_size_list)
    print("the shape, max, min of button ration is {}, {}, {}".format(
        button_size_np.shape, button_size_np.max(), button_size_np.min()))
    height_hist, height_bins = np.histogram(
        button_size_np[:, 1], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800])
    print("the height hist and bins is {}, {}".format(height_hist, height_bins))
    width_hist, width_bins = np.histogram(
        button_size_np[:, 0], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800])
    print("the width hist and bins is {}, {}".format(width_hist, width_bins))

    rations_test = np.asarray([0.8, 1, 1.3])
    scales_test = np.asarray([0.5, 1, 2, 3])
    scales_grid, aspect_ratios_grid = np.meshgrid(scales_test,
                                                  rations_test)
    scales_grid = np.reshape(scales_grid, [-1])
    aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
    aspect_ratios_grid = np.sqrt(aspect_ratios_grid)
    height_test = scales_grid / aspect_ratios_grid * 200
    width_test = scales_grid * aspect_ratios_grid * 200
    print(np.asarray([height_test, width_test]).T)


def check_buttons(data):
    object_list = data['object']
    if len(object_list) == 0:
        print('object list is empty!')
    for obj in object_list:
        button_type = obj['name']
        categorize_button_labels(button_type, data['filename'])
    return True


def parse_xml_label(image_name_list, annotation_root):
    if len(image_name_list) == 0:
        print('image name list is empty!')
    for example_name in image_name_list:
        annotation_name = os.path.join(annotation_root, example_name+'.xml')
        with tf.gfile.GFile(annotation_name, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        # filter_led_only_image(data)
        check_buttons(data)
        # object_size_distribution(data)
        # image_size_list.append([float(data['size']['height']),
        #                         float(data['size']['width'])])


def filter_led_only_image(data):
    object_list = data['object']
    if len(object_list) == 0:
        print('object list is empty!')
    obj_temp_buffer = []
    filter_target_count = 0
    for obj in object_list:
        button_type = obj['name']
        obj_temp_buffer.append(button_type)
        if button_type in filter_target:
            filter_target_count += 1
    if len(obj_temp_buffer) == 0:
        raise ValueError('found images with no labels')
    if len(obj_temp_buffer) == filter_target_count:
        filtered_image_set.append(data['filename'])
        return True
    else:
        return False


def data_washing():
    for x in filtered_image_set:
        source = os.path.join(annotations_path, x.split('.')[0]+'.jpg')
        destination = os.path.join('/home/zhudelong/Desktop/leds/annotations',
                                   x.split('.')[0]+'.jpg')
        shutil.move(source, destination)
        print(destination)


def plot_distribution(data, label):
    data_len = len(data)
    label_len = len(label)
    if data_len != label_len:
        raise ValueError('the length of data and label is not consistent!')
    plt.bar(range(data_len), list(data))
    # plt.xticks(data_len, list(label))
    plt.show()


def generate_charset():
    char_set = {}
    dict_sorted = sorted(dict_statistic.items(), key=lambda item: item[1], reverse=True)
    print(dict_sorted)
    for idx, item in enumerate(dict_sorted):
        char_set[item[0]] = idx+1
    char_set_sorted = sorted(char_set.items(), key=lambda item: item[1])
    print(char_set_sorted)


def load_dict():
    model_dir = '/home/zhudelong/dataset/elevator_panel_database/pomdp_models'
    # load category dict
    lbl_idx_dict_file = os.path.join(model_dir, 'label_category.dict')
    handle = open(lbl_idx_dict_file, 'rb')
    lbl_idx_dict = pickle.load(handle)
    idx_lbl_dict = {}
    for key in lbl_idx_dict.keys():
        idx_lbl_dict[lbl_idx_dict[key]] = key
    return lbl_idx_dict, idx_lbl_dict


def prepare_data_set(data_dir):
    # data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset/train_set'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    sample_list = get_image_name_list(label_dir)
    dataset = []
    for idx, example in enumerate(sample_list):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        dataset.append(data)
    return dataset, image_dir, label_dir

def prepare_train_set():
    data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset/train_set'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    sample_list = get_image_name_list(label_dir)
    dataset = []
    for idx, example in enumerate(sample_list):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        dataset.append(data)
    return dataset, image_dir, label_dir


def prepare_test_set():
    data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset/test_set'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    sample_list = get_image_name_list(label_dir)
    dataset = []
    for idx, example in enumerate(sample_list):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        dataset.append(data)
    return dataset, image_dir, label_dir


def prepare_dataset(name):
    assert name == 'train' or name == 'test'
    data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset'
    if name == 'train':
        dataset, _, _ = prepare_train_set()
    else:
        dataset, _, _ = prepare_test_set()

    return dataset, data_dir


def prepare_zijian_dataset():
    data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset/zijian_data'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    sample_list = get_image_name_list(label_dir)
    dataset = []
    for idx, example in enumerate(sample_list):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        dataset.append(data)
    return dataset, image_dir, label_dir


def prepare_stanford_dataset():
    data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset/stanford_data'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    sample_list = get_image_name_list(label_dir)
    dataset = []
    for idx, example in enumerate(sample_list):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with tf.gfile.GFile(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        dataset.append(data)
    return dataset, image_dir, label_dir

# old version before separating the dataset to different folder
# def prepare_dataset(name='all'):
#     data_dir = '/home/zhudelong/dataset/elevator_panel_database/dataset'
#     label_dir = os.path.join(data_dir, 'annotations')
#     image_dir = os.path.join(data_dir, 'images')
#     grids_dir = os.path.join(data_dir, 'grids')
#     eval_dir = os.path.join(data_dir, 'evaluation_train+test')
#
#     # list train and test samples
#     sample_list = get_image_name_list(label_dir)
#     random.seed(9420)
#     random.shuffle(sample_list)
#     num_samples = len(sample_list)
#     num_train = int(0.7 * num_samples)
#     train_samples = sample_list[:num_train]
#     test_samples = sample_list[num_train:]
#
#     if name == 'train':
#         dataset_type = train_samples
#     elif name == 'test':
#         dataset_type = test_samples
#     else:
#         dataset_type = sample_list
#
#     dataset = []
#     for idx, example in enumerate(dataset_type):
#         annotation_path = os.path.join(label_dir, example + '.xml')
#         with tf.gfile.GFile(annotation_path, 'r') as fid:
#             xml_str = fid.read()
#         xml = etree.fromstring(xml_str)
#         data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
#         dataset.append(data)
#     return dataset, data_dir

def bb_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)

  # return the intersection over union value
  return iou

# if __name__ == '__main__':
#     image_names = get_image_name_list(annotations_path)
#     parse_xml_label(image_names, annotations_path)
#     for idx, item in enumerate(dict_statistic.keys()):
#         dict_statistic[item] = idx
#     print(dict_statistic)
#     dict_file = '/home/zhudelong/dataset/elevator_panel_database/grids/a_label_category.dict'
#     with open(dict_file, 'wb') as handle:
#         pickle.dump(dict_statistic, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(len(dict_statistic))
#     # generate_charset()

    # plot_distribution(sorted(dict_statistic.values()), sorted(dict_statistic.keys()))
    # print(filtered_image_set)
    # data_washing()
    # x = np.asarray(image_size_list)
    # print(x.max(), x.min())
    # object_statistics()

