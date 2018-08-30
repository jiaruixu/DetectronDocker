# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
# _DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_DATA_DIR = '/mnt/fcav/self_training/object_detection/dataset'

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Optional dataset entry keys
_IM_PREFIX = 'image_prefix'
_DEVKIT_DIR = 'devkit_directory'
_RAW_DIR = 'raw_dir'

# Available datasets
_DATASETS = {
    'coco_KITTI_caronly_val1000':{
        _IM_DIR:
            _DATA_DIR + '/KITTI/image_2',
        _ANN_FN:
            _DATA_DIR + '/KITTI/annotations/instances_caronly_val1000.json'
    },
    'coco_KITTI_caronly_val':{
        _IM_DIR:
            _DATA_DIR + '/KITTI/image_2',
        _ANN_FN:
            _DATA_DIR + '/KITTI/annotations/instances_caronly_val.json'
    },
    'coco_KITTI_caronly_train':{
        _IM_DIR:
            _DATA_DIR + '/KITTI/image_2',
        _ANN_FN:
            _DATA_DIR + '/KITTI/annotations/instances_caronly_train.json'
    },
    'coco_KITTI_train_with_prediction':{
        _IM_DIR:
            _DATA_DIR + '/KITTI/image_2',
        _ANN_FN:
            _DATA_DIR + '/KITTI/annotations/instances_caronly_train_with_prediction.json'
    },
    'coco_KITTI_tracking_imagesonly':{
        _IM_DIR:
            _DATA_DIR + '/KITTI_tracking/image_2',
        _ANN_FN:
            _DATA_DIR + '/KITTI_tracking/coco_KITTI_tracking_imagesonly.json'
    },
    'voc_GTA_caronly_train':{
        _IM_DIR:
            _DATA_DIR + '/GTA_Pascal_format/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/GTA_Pascal_format/Annotations/instances_caronly_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/GTA_Pascal_format/VOCdevkit2012'
    },
    'voc_GTA_caronly_train_sample8000':{
        _IM_DIR:
            _DATA_DIR + '/GTA_Pascal_format/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/GTA_Pascal_format/Annotations/instances_caronly_train_sample8000.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/GTA_Pascal_format/VOCdevkit2012'
    },
    'voc_GTA_caronly_val_sample2000':{
        _IM_DIR:
            _DATA_DIR + '/GTA_Pascal_format/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/GTA_Pascal_format/Annotations/instances_caronly_val_sample2000.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/GTA_Pascal_format/VOCdevkit2012'
    },
    'cityscapes_caronly_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instances_caronly_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_caronly_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instances_caronly_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_caronly_train_with_prediction': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instances_caronly_train_with_prediction.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_caronly_train_with_dropannotations': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations_drop/instancesonly_filtered_gtFine_train_droprate0.6.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/COCO/train2014',
        _ANN_FN:
            _DATA_DIR + '/COCO/annotations_drop/instances_train2014_sample20000.json'
    },
    'coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/COCO/val2014',
        _ANN_FN:
            _DATA_DIR + '/COCO/annotations_drop/instances_val2014_sample20000.json'
    },
    'coco_2014_train_drop30': {
        _IM_DIR:
            _DATA_DIR + '/COCO/train2014',
        _ANN_FN:
            _DATA_DIR + '/COCO/annotations_drop/instances_train2014_droprate0.3_sample20000.json'
    },
    'coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'voc_2007_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    'voc_2012_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]


def get_im_prefix(name):
    """Retrieve the image prefix for the dataset."""
    return _DATASETS[name][_IM_PREFIX] if _IM_PREFIX in _DATASETS[name] else ''


def get_devkit_dir(name):
    """Retrieve the devkit dir for the dataset."""
    return _DATASETS[name][_DEVKIT_DIR]


def get_raw_dir(name):
    """Retrieve the raw dir for the dataset."""
    return _DATASETS[name][_RAW_DIR]