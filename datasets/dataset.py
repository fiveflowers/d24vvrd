#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-01-20
# @Author  : Yifer Huang
# @File    : dataset.py
# @Desc    : dataset register based on detectron2.data.dataset

from detectron2.data.datasets import register_coco_instances

# NOTICE: ILSVRC 2016 DET = ILSVRC 2015 DET = ILSVRC 2014 DET + ILSVRC 2013 DET

register_coco_instances(
    'vidor_train_16',
    {},
    'datasets/vidor/d2_train.json',
    'datasets/vidor/frames@16'
)
register_coco_instances(
    'vidor_test_16',
    {},
    'datasets/vidor/d2_test.json',
    'datasets/vidor/frames@16'
)
register_coco_instances(
    'vidor_coco_train',
    {},
    'datasets/coco/train_vidor.json',
    'datasets/coco/train2014'
)
register_coco_instances(
    'vidor_coco_val_minus_minival',
    {},
    'datasets/coco/val_minus_minival_vidor.json',
    'datasets/coco/val2014'
)
register_coco_instances(
    'vidor_ilsvrc_train2013',
    {},
    'datasets/ILSVRC2015/train_2013_vidor.json',
    'datasets/ILSVRC2015/Data/DET/train/ILSVRC2013_train'
)
register_coco_instances(
    'vidor_ilsvrc_train2014',
    {},
    'datasets/ILSVRC2015/train_2014_vidor.json',
    'datasets/ILSVRC2015/Data/DET/train'
)