#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-01-20
# @Author  : Yifer Huang
# @File    : train_net.py
# @Desc    : object detector (based on detectron2)
# https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py

import os

import datasets.dataset

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name=dataset_name, 
                             tasks=('bbox',), 
                             distributed=True, 
                             output_dir=output_folder)


def setup(args):
    """Create configs and perform basic setups.

    Args:
        args (*): args
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    for key, value in vars(args).items():
        print('>>> {:10}: {}'.format(key, value))

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
