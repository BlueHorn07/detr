# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)

    if args.dataset_file == 'indoor360':
        from .indoor360 import build as build_indoor360
        return build_indoor360(image_set, args)

    if args.dataset_file == 'indoor360_mollweide':
        print("==========  USE mollweide360!!  ==========")
        from .indoor360_mollweide import build as build_indoor360
        return build_indoor360(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
