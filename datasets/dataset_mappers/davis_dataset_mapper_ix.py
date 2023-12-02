# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import os

import cv2
import scipy.io
import numpy as np
from scipy.io import loadmat
from PIL import Image

import torch
from torchvision import transforms
from detectron2.structures import BitMasks, Boxes, Instances

from modeling.utils import configurable
from ..visual_sampler import build_shape_sampler

__all__ = ["DAVISDatasetMapperIX"]


# This is specifically designed for the COCO dataset.
class DAVISDatasetMapperIX:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        dataset_name='',
        min_size_test=None,
        max_size_test=None,
        shape_sampler=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC, max_size=max_size_test))
        self.transform = transforms.Compose(t)
        self.shape_sampler = shape_sampler

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=''):
        shape_sampler = build_shape_sampler(cfg, is_train=is_train, mode=dataset_name.split('_')[-1])
        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "shape_sampler": shape_sampler,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        mask_name = dataset_dict['mask_name']
        image = Image.open(file_name).convert('RGB')

        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]

        if self.is_train == False:
            image = self.transform(image)
            image = torch.from_numpy(np.asarray(image).copy())
            image = image.permute(2,0,1)
        
        instances_mask = np.max(cv2.imread(mask_name).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        instances = Instances(image.shape[-2:])
        _,h,w = image.shape
        # sbd dataset only has one gt mask.
        masks = [cv2.resize(instances_mask.astype(np.uint8), (w,h), interpolation=cv2.INTER_CUBIC)]
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        )
        instances.gt_masks = masks
        instances.gt_boxes = masks.get_bounding_boxes()
        spatial_query_utils = self.shape_sampler(instances)

        dataset_dict['spatial_query'] = spatial_query_utils
        dataset_dict['instances'] = instances
        dataset_dict['image'] = image
        dataset_dict['gt_masks_orisize'] = torch.from_numpy(instances_mask).bool()[None,] # (nm,h,w)
        return dataset_dict