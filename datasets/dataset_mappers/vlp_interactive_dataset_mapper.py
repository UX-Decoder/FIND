# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import os
import io
from PIL import Image
import numpy as np

import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from modeling.utils import configurable
from modeling.language import build_tokenizer
from ..visual_sampler.sampler import build_shape_sampler

from transformers import AutoTokenizer, LlamaForCausalLM

__all__ = ["VLPreInteractiveDatasetMapper"]


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

def get_patch_image(image, random_state, image_id):
    if random_state is not None:
        x1, y1, x2, y2 = random_state[image_id]
    else:
        # prepare crop patch coordinates
        h, w = image.shape[:2]
        # Generate width and height of the rectangle
        patch_w = np.random.randint(int(w * 0.25), int(w * 0.5))
        patch_h = np.random.randint(int(h * 0.25), int(h * 0.5))

        # Generate x1, y1 such that the rectangle fits in the array
        x1 = np.random.randint(0, w - patch_w)
        y1 = np.random.randint(0, h - patch_h)

        # Calculate x2, y2 based on x1, y1 and the rectangle's dimensions
        x2 = x1 + patch_w
        y2 = y1 + patch_h

    patch_image = image[y1:y2, x1:x2]
    return patch_image, [x1,y1,x2,y2]

# This is specifically designed for the COCO dataset.
class VLPreInteractiveDatasetMapper:
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
        shape_sampler=None,
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

        self.all_arrows = MetadataCatalog.get(dataset_name).arrows

        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.device = device
        self.lang_model = lang_model

        self.random_state_path = MetadataCatalog.get(dataset_name).random_state_pth
        self.random_state = None
        if os.path.exists(self.random_state_path):
            self.random_state = torch.load(self.random_state_path)

        self.shape_sampler = shape_sampler
        self.image_id_to_random_index = {}

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        shape_sampler = build_shape_sampler(cfg)

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
            "shape_sampler": shape_sampler,
        }
        return ret

    def get_image(self, inp):
        image_bytes = io.BytesIO(inp)
        image_bytes.seek(0)
        return Image.open(image_bytes)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        arr = self.all_arrows[dataset_dict['arr_id']]
        cur_id = dataset_dict['cur_id']
        image = self.get_image(arr['image'][cur_id].as_py())

        image = utils._apply_exif_orientation(image)
        image = utils.convert_PIL_to_numpy(image, self.img_format)
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]  # h, w

        patch_image, (x1,y1,x2,y2) = get_patch_image(image, self.random_state, arr['image_id'][cur_id].as_py())

        # self.image_id_to_random_index[arr['image_id'][cur_id].as_py()] = [x1,y1,x2,y2]
        # torch.save(self.image_id_to_random_index, self.random_state_path)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        patch_image, _ = T.apply_transform_gens(self.tfm_gens, patch_image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["patch_image"] = torch.as_tensor(np.ascontiguousarray(patch_image.transpose(2, 0, 1)))

        # prepare segmentation gt
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
            
            is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
            classes = np.array(classes)
            is_things = np.array(is_things)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.is_things = torch.tensor(is_things, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                masks = BitMasks(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])))
                instances.gt_masks = masks
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks
                instances.gt_boxes = masks.get_bounding_boxes()
            dataset_dict["instances"] = instances

        spatial_query_utils = self.shape_sampler(instances)
        spatial_query_utils['rand_shape'] = spatial_query_utils['rand_shape'][None,]
        dataset_dict['spatial_query'] = spatial_query_utils

        captions = dataset_dict['captions']
        tokens = self.tokenizer(
            captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        return dataset_dict