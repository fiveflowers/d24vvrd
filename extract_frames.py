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

def vidor_annotator(frequency, input, output):
    pass

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
        annotator[args.dataset](args.frequency, args.input, args.output)
    else:
        extractor[args.dataset](args.frequency, args.input, args.output)