#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-01-20
# @Author  : Yifer Huang
# @File    : annotator.py
# @Desc    : convert dataset annotations

import argparse
import os
import json
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser(description='convert dataset annotations')
    parser.add_argument('--dest', default='vidor', help='source dataset')
    parser.add_argument('--src', default='coco', help='destination dataset')
    parser.add_argument('--input', help='root path of dataset')
    parser.add_argument('--output', help='directory to store converted annotations')

    args = parser.parse_args()
    assert args.src in ['coco', 'ilsvrc']
    assert args.dest in ['vidvrd', 'vidor']
    assert args.input is not None
    args.output = args.input if args.output is None else args.output

    return args

def dump_coco_file(src, categories, annotations, images, filename):
    """dump coco annotation file

    Args:
        src (str): src video dataset type
        categories (list): categories
        annotations (list): annotations
        images (list): images
        filename (str): annotation filename
    """
    from datasets.vocab import coco_liscenses, coco_info
    coco_annotation = {
        'info': coco_info[src],
        'type': 'instances',
        'liscenses': coco_liscenses,
        'categories': categories,
        'images': images,
        'annotations': annotations,
    }
    with open(filename, 'w') as f:
        json.dump(coco_annotation, f)
        print('>>> Successfully export annotations to {}'.format(filename))

def convert_from_ilsvrc(mode, input, output):
    """convert annos from image net according to given categories
    .input path
    ├── Annotations
    │   └── DET
    ├── Data
    │   └── DET
    └── ImageSets
        └── DET

    Args:
        mode (str): dest dataset type [vidvrd, vidor]
        input (str): ILSVRC dataset location
        output (str): path to store annotation file
        categories (list): corresponding categories
    """
#   BEFORE RUN
    def get_image_id(filename, mode, class_id):
        """get image if for ILSVRC dataset image
        filename: n00007846_21106, ILSVRC2014_train_00010002
        mode: [2013, 2014]
        class_id: 021
        """
        temp_id = filename.split('_')[-1][-5:]
        return int('{}{}{}'.format(
            mode,
            str(class_id).zfill(3),
            temp_id))

    from datasets.vocab import all_categories, all_ilsvrc_map
    import xml.etree.ElementTree as ET

    categories = all_categories[mode]
    ilsvrc_map = all_ilsvrc_map[mode]
    filtered_classes = [item for item in ilsvrc_map] # 'n02691156', 'n02419796', 'n02131653'...

    annos_path = os.path.join(input, 'Annotations/DET/train') # ILSVRC2013_train, ILSVRC2014_train_0000

#   RUNNING
#   convert ILSVRC2013_train
    new_annos_train2013 = list()
    new_images_train2013 = list()
    dirty_data_train2013 = ['n02419796_3142.xml', 'n03467517_13624.xml']
    index_instance_train2013 = 20130000000
    for iter_class in tqdm(filtered_classes):
        annos_path_iter_class = os.path.join(annos_path, 'ILSVRC2013_train', iter_class)
        anno_files = os.listdir(annos_path_iter_class)
        for anno_file in anno_files:
            # filte dirty data
            if anno_file in dirty_data_train2013: continue

            anno_filename = os.path.join(annos_path_iter_class, anno_file)
            # parse and convert
            anno_xml = ET.parse(anno_filename)
            objects = anno_xml.findall('object')

            if len(objects) == 0: continue

            # image info 
            filename = anno_xml.find('filename').text
            folder = anno_xml.find('folder').text
            height = anno_xml.find('size').find('height').text
            width = anno_xml.find('size').find('width').text
            image_id = get_image_id(filename, 2013, ilsvrc_map[iter_class])
            image_info = {
                "file_name": '{}/{}.JPEG'.format(folder, filename),
                "height": int(height),
                "width": int(width),
                "id": image_id
            }
            new_images_train2013.append(image_info)

            # instance annotation
            annotations_image = []  # image = instance1 + instance2 + ...
            for obj in objects:          
                class_name = obj.find('name').text    # n00007846
                if class_name not in ilsvrc_map: continue       # 会有一些特殊情况，n00007846会随机出现
                index_instance_train2013 += 1
                category_id = ilsvrc_map[class_name] 
                xmin = obj.find('bndbox').find('xmin').text
                ymin = obj.find('bndbox').find('ymin').text
                xmax = obj.find('bndbox').find('xmax').text
                ymax = obj.find('bndbox').find('ymax').text
                annotation_instance = {
                    "category_id": category_id,
                    "area": (int(xmax) - int(xmin)) * (int(ymax) - int(ymin)),
                    "bbox": [int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)],
                    "bbox_mode": 1,
                    "image_id": image_id,
                    "id": index_instance_train2013,
                    "iscrowd": 0
                }
                annotations_image.append(annotation_instance)
            new_annos_train2013.extend(annotations_image)
#   convert ILSVRC2014_train
    new_annos_train2014 = []
    new_images_train2014 = []
    index_instance_train2014 = 20140000000
    ILSVRC2014_folders = ['ILSVRC2014_train_000' + str(i) for i in range(7)]
    for ILSVRC2014_folder in ILSVRC2014_folders:
        annos_fold = os.path.join(annos_path, ILSVRC2014_folder)
        anno_files = os.listdir(annos_fold)
        for anno_file in tqdm(anno_files):
            anno_filename = os.path.join(annos_fold, anno_file)
            # parse and convert
            anno_xml = ET.parse(anno_filename)
            objects = anno_xml.findall('object')

            if len(objects) == 0:   continue
            
            # image info 
            filename = anno_xml.find('filename').text
            folder = anno_xml.find('folder').text
            height = anno_xml.find('size').find('height').text
            width = anno_xml.find('size').find('width').text
            image_id = get_image_id(filename, 2014, 999)
            image_info = {
                "file_name": '{}/{}.JPEG'.format(folder, filename),
                "height": int(height),
                "width":int(width),
                "id": image_id
            }
            new_images_train2014.append(image_info)

            # instance annotation
            annotations_image = []  # image = instance1 + instance2 + ...
            for obj in objects:
                class_name = obj.find('name').text
                if class_name not in ilsvrc_map: continue
                index_instance_train2014 += 1
                category_id = ilsvrc_map[class_name] 
                xmin = obj.find('bndbox').find('xmin').text
                ymin = obj.find('bndbox').find('ymin').text
                xmax = obj.find('bndbox').find('xmax').text
                ymax = obj.find('bndbox').find('ymax').text
                annotation_instance = {
                    "category_id": category_id,
                    "area": (int(xmax) - int(xmin)) * (int(ymax) - int(ymin)),
                    "bbox": [int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)],
                    "bbox_mode": 1,
                    "image_id": image_id,
                    "id": index_instance_train2014,
                    "iscrowd": 0
                }
                annotations_image.append(annotation_instance)
            new_annos_train2014.extend(annotations_image)
    
#   AFTER RUN
    dump_coco_file(
        'ilsvrc-det',
        categories,
        new_annos_train2013,
        new_images_train2013,
        os.path.join(input, 'train_2013_{}.json'.format(mode))
    )
    dump_coco_file(
        'ilsvrc-det',
        categories,
        new_annos_train2014,
        new_images_train2014,
        os.path.join(input, 'train_2014_{}.json'.format(mode))
    )

def convert_from_coco(mode, input, output):
    """convert annos from coco according to given categories
    .input path
    ├── annotations (train2014, val2014, minival2014)
    ├── train2014 (images)
    └── val2014 (images)

    Args:
        mode (str) : dest dataset type [vidvrd, vidor]
        input (str): coco dataset path
        output (str): path to save annotation file
        categories (dict): corresponding categories
    """
#   BEFRORE RUN
    from pycocotools.coco import COCO
    from datasets.vocab import all_categories, all_coco_map

    categories = all_categories[mode]
    annos_path = os.path.join(input, 'annotations')
    minival2014_filename = os.path.join(annos_path, 'instances_minival2014.json')
    train2014_filename = os.path.join(annos_path, 'instances_train2014.json')
    val2014_filename = os.path.join(annos_path, 'instances_val2014.json')    
    # load coco annotations
    minival2014 = COCO(minival2014_filename)
    train2014 = COCO(train2014_filename)
    val2014 = COCO(val2014_filename)  
    coco_categories = train2014.loadCats(train2014.getCatIds())
    id2coco_categories = {item['id']: item['name'] for item in coco_categories}
    # vidor/vidvrd vocab
    categories2id = all_coco_map[mode]

    filtered_classes = [item for item in categories2id]     # for video object classes

#   RUNNING
#   convert coco train2014 annotation
    new_annos_train2014 = []
    image_ids_train2014 = train2014.getImgIds()     # image ids
    images_train2014 = train2014.loadImgs(image_ids_train2014)  # image info
    anno_ids_train2014 = train2014.getAnnIds(image_ids_train2014)  # instance anno ids
    annos_train2014 = train2014.loadAnns(anno_ids_train2014)      # instance annos
    
    for anno in tqdm(annos_train2014):
        coco_category = id2coco_categories[anno['category_id']]
        if coco_category not in filtered_classes:
            continue
        new_anno_instance = {
            "category_id": categories2id[coco_category],
            "area": anno['area'],
            "bbox": anno['bbox'],
            "bbox_mode": 1,
            "image_id": anno['image_id'],
            "id": anno['id'],
            "iscrowd": 0
        }
        new_annos_train2014.append(new_anno_instance)

#   convert coco val2014 - minival2014 annotation
    image_ids_val2014 = val2014.getImgIds()     # val2014 image ids
    image_ids_val_minus_minival2014 = image_ids_val2014.copy()
    image_ids_minival2014 = minival2014.getImgIds()  # minival2014 image ids

    for item in image_ids_minival2014:
        image_ids_val_minus_minival2014.remove(item)    # val_minus_minival
    
    images_val_minus_minival2014 = val2014.loadImgs(image_ids_val_minus_minival2014)
    anno_ids_val_minus_minival2014 = val2014.getAnnIds(image_ids_val_minus_minival2014)
    annos_val_minus_minival2014 = val2014.loadAnns(anno_ids_val_minus_minival2014)

    new_annos_val_minus_minival2014 = []
    for anno in tqdm(annos_val_minus_minival2014):
        coco_category = id2coco_categories[anno['category_id']]
        if coco_category not in filtered_classes:
            continue
        new_anno_instance = {
            "category_id": categories2id[coco_category],
            "area": anno['area'],
            "bbox": anno['bbox'],
            "bbox_mode": 1,
            "image_id": anno['image_id'],
            "id": anno['id'],
            "iscrowd": 0
        }
        new_annos_val_minus_minival2014.append(new_anno_instance)
    
#   AFTER RUN
    dump_coco_file(
        'coco',
        categories,
        new_annos_train2014,
        images_train2014,
        os.path.join(input, 'train_{}.json'.format(mode))
    )
    dump_coco_file(
        'coco',
        categories,
        new_annos_val_minus_minival2014,
        images_val_minus_minival2014,
        os.path.join(input, 'val_minus_minival_{}.json'.format(mode))
    )

if __name__ == "__main__":
    args = args_parser()
    for key, value in vars(args).items():
        print('>>> {:10}: {}'.format(key, value))

    if args.src == 'coco':
        convert_from_coco(args.dest, args.input, args.output)
    elif args.src == 'ilsvrc':
        convert_from_ilsvrc(args.dest, args.input, args.output)
    else:
        print('Coming soom...')