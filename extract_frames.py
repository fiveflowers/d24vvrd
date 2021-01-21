#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-01-19
# @Author  : Yifer Huang
# @File    : extract_frames.py
# @Desc    : extract frames from videos

import argparse
import os
import json
import cv2
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser(description='extract frames from videos')
    parser.add_argument('--dataset', default='vidvrd', help='choose dataset')
    parser.add_argument('--f', dest='frequency', type=int, default=1, help='sample frequency')
    parser.add_argument('--input', help='path of dataset')
    parser.add_argument('--output', help='path to store images extracted from video')
    parser.add_argument('--split', default='train', help='(train, test), ANNOTATOR NEED!')
    parser.add_argument('--anno-only', dest='need_anno', action='store_true')
    args = parser.parse_args()

    assert args.dataset in ['vidvrd', 'vidor']
    assert args.input is not None
    if args.output is None:
        args.output = os.path.join(args.input, 'frames@{}'.format(args.frequency))

    return args

def check_dirs(path):
    """check if the given path exists, if not, create it

    Args:
        path (string): given path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print('>>> Successfully create directory {}.'.format(path))

def dump_coco_file(filename, categories, annotations, images, dataset):
    from datasets.vocab import coco_liscenses, coco_info
    coco_annotations = {
        'info': coco_info[dataset],
        'type': 'instances',
        'liscenses': coco_liscenses,
        'categories': categories,
        'images': images,
        'annotations': annotations,
    }
    with open(filename, 'w') as f:
        json.dump(coco_annotations, f)
        print('>>> Successfully export annotations to {}'.format(filename))
    
def vidvrd_extractor(frequency, input, output):
    pass

def vidvrd_annotator(frequency, input, output):
    pass

def vidor_extractor(frequency, input, output):
    """extracte images from vidor video dataset according given frequency

    Args:
        input (string): vidor dataset directory
        output (string ): directory to store extracted images
        frequency (int): sampling frequency
    
    .input (vidvrd dataset directory)
    ├── training/                   # annotations for train split
    ├── validation/                 # annotations for test split
    ├── frames@{frequency}/         # if not exist, then mkdir
    └── videos/                     # all videos of vidvrd dataset
    """
    videos_dir = os.path.join(input, 'videos')
    frames_dir = output
    check_dirs(frames_dir)

    sub_dirs = os.listdir(videos_dir)
    for sub_dir in tqdm(sub_dirs):  # sub_dir: 0000, 0001, 0002
        sub_dir_path = os.path.join(videos_dir, sub_dir)
        video_basenames = os.listdir(sub_dir_path)

        # basename: 2401075277.mp4
        # filename: ~/datasets/vidor/videos/0000/2401075277.mp4
        # video_id: 2401075277
        for basename in video_basenames:
            filename = os.path.join(sub_dir_path, basename)
            video_id = os.path.splitext(basename)[0]
            frame_index = 0
            video = cv2.VideoCapture(filename)
            while True:
                status, frame = video.read()
                if not status:  break
                if frame_index % frequency == 0:
                     # max frame count of vidor videos is 5395
                    frame_filename = os.path.join(frames_dir, "{}_{:04d}.jpg".format(video_id, frame_index))
                    cv2.imwrite(frame_filename, frame)
                frame_index += 1
    print('>>> Successfully extract frames (1/{}) from vidor dataset.'.format(frequency))

def vidor_annotator(frequency, input, output, split):
#   BEFORE RUN
    check_dirs(output)
    prefix = 'training' if split == 'train' else "validation"
    # import statistic data
    from datasets.vocab import vidor_categories
    cat2id = {item['name']: item['id'] for item in vidor_categories}    # categories to id
    instance_index = 1      # instance counter
    annotations = list()    # all annotation (coco-style)
    images = list()
    # prepare raw vidor annotation filenames
    root_dir = os.path.join(input, prefix)  # vidor/training
    sub_dirs = os.listdir(root_dir)
    anno_vidor_fns = list()
    for sub_dir in tqdm(sub_dirs):
        sub_dir_path = os.path.join(root_dir, sub_dir)  # vidor/training/0000/
        basenames = os.listdir(sub_dir_path)
        for basename in basenames:
            filename = os.path.join(sub_dir_path, basename) # vidor/training/1021/2405668450.json
            anno_vidor_fns.append(filename)
    print('>>> Successfully prepare vidor annotation files')
    
#   RUNING
    for fn in tqdm(anno_vidor_fns):
        with open(fn, 'r') as f:
            raw_data = json.load(f)

        width = raw_data['width']
        height = raw_data['height']
        video_id = raw_data['video_id']
        trajectories = raw_data['trajectories']
        category_dict = raw_data['subject/objects']
        
        tid2index = dict()      # map tid to object class id
        for item in category_dict:
            tid = str(item['tid'])
            tid2index[tid] = cat2id[item['category']]
        
        for frame_index, trajectory in enumerate(trajectories):
            if frame_index % frequency != 0: continue
            image_id = int("{}{:04d}".format(video_id, frame_index))
            image_basename = '{}_{:04d}.jpg'.format(video_id, frame_index)
            assert os.path.exists(os.path.join(output, image_basename)) # check frame image
            if len(trajectory) == 0: continue       # pass empty anno
            image = {
                "file_name": image_basename,    # 4460320158_0000q.jpg
                "height": height,
                "width":width,
                "id": image_id
            }
            images.append(image)
            annotations_image = list()          # image = instance1 + instance2 + ...
            for instance in trajectory:
                tid = instance['tid']
                category_id = tid2index[str(tid)]
                x = instance['bbox']['xmin']
                y = instance['bbox']['ymin']
                h = instance['bbox']['ymax'] - instance['bbox']['ymin']
                w = instance['bbox']['xmax'] - instance['bbox']['xmin']
                annotation_instance = {
                    "category_id": category_id,
                    "area": w*h,
                    "bbox": [x, y, w, h],
                    "bbox_mode": 1,
                    "image_id": image_id,
                    "id": instance_index,
                    "iscrowd": 0
                }
                instance_index += 1
                annotations_image.append(annotation_instance)   
            annotations.extend(annotations_image)   # add to all annotation dict
    
#   AFTER RUN
    res_anno_fn = os.path.join(input, 'd2_{}_{}.json'.format(split, frequency))   # coco-style annotation filename
    dump_coco_file(res_anno_fn, vidor_categories, annotations, images, 'vidor')

if __name__ == "__main__":
    args = args_parser()
    for key, value in vars(args).items():
        print('>>> {:10}: {}'.format(key, value))
    
    extractor = {
        'vidvrd': vidvrd_extractor,
        'vidor': vidor_extractor
    }
    annotator = {
        'vidvrd': vidvrd_annotator,
        'vidor': vidor_annotator
    }

    if args.need_anno:
        annotator[args.dataset](args.frequency, args.input, args.output, args.split)
    else:
        extractor[args.dataset](args.frequency, args.input, args.output)