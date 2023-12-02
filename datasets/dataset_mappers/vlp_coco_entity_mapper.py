# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import copy
import logging

import io
from PIL import Image
import numpy as np

import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from modeling.utils import configurable
from modeling.language import build_tokenizer

from transformers import AutoTokenizer, LlamaForCausalLM

__all__ = ["VLPreCOCOEntityDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # The scope of vlp dataset may not need any augmentation.
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])
    
    return augmentation

def build_transform_gen_se(cfg, is_train):
    min_scale = cfg['INPUT']['MIN_SIZE_TEST']
    max_scale = cfg['INPUT']['MAX_SIZE_TEST']

    augmentation = []
    augmentation.extend([
        T.ResizeShortestEdge(
            min_scale, max_size=max_scale
        ),
    ])    
    return augmentation


# This is specifically designed for the COCO dataset.
class VLPreCOCOEntityDatasetMapper:
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
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
        tokenizer=None,
        max_token_num=None,
        device=None,
        lang_model=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[PretrainDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.device = device
        self.lang_model = lang_model

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        shortest_edge = cfg['INPUT'].get('SHORTEST_EDGE', False)

        if not shortest_edge:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            tfm_gens = build_transform_gen_se(cfg, is_train)

        lang_model = cfg['MODEL']['TEXT']['NAME']
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        device = cfg['device']

        if 'llama' in lang_model:
            tokenizer = AutoTokenizer.from_pretrained(lang_model, padding_side='right')
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "device": device,
            "lang_model": lang_model,
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
        image = Image.open(file_name).convert('RGB')
        image = utils.convert_PIL_to_numpy(image, self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        captions = dataset_dict['captions']
        tokens = self.tokenizer(
            captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        return dataset_dict