# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import io
from PIL import Image
import numpy as np

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from utils import prompt_engineering, Entity
from modeling.language import build_tokenizer
from modeling.utils import configurable

from ..visual_sampler.sampler import build_shape_sampler

__all__ = ["GroundingCOCOEntityDatasetMapper"]


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
class GroundingCOCOEntityDatasetMapper:
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
        shape_sampler=None,
        interleave_visual_prob=0.5,
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
        self.interleave_visual_prob = interleave_visual_prob

        self.shape_sampler = shape_sampler

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        shortest_edge = cfg['INPUT'].get('SHORTEST_EDGE', False)
        tfm_gens = build_transform_gen_se(cfg, is_train)
        shape_sampler = build_shape_sampler(cfg)

        lang_model = cfg['MODEL']['TEXT']['NAME']
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        device = cfg['device']

        if 'llama' in lang_model:
            tokenizer = AutoTokenizer.from_pretrained(lang_model, padding_side='right')
            tokenizer.pad_token = tokenizer.eos_token
        elif 'transformer' in lang_model:
            tokenizer_cfg = cfg['MODEL']['TEXT'].copy()
            tokenizer_cfg['TOKENIZER'] = 'clip-token'
            tokenizer = build_tokenizer(tokenizer_cfg)
        else:
            assert False, "Not implemented yet."

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "shape_sampler": shape_sampler,
            "interleave_visual_prob": cfg['MODEL']['DECODER']['INTERLEAVE']['VISUAL_PROB'],
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
        assert len(dataset_dict['entities']) > 0
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))        

        # prepare input mask
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict["pan_seg_file_name"], "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            inst_id = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
                    inst_id.append(segment_info["id"])
                        
            is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
            classes = np.array(classes)
            is_things = np.array(is_things)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.is_things = torch.tensor(is_things, dtype=torch.int64)
            instances.inst_id = torch.tensor(inst_id, dtype=torch.int64)

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

        # prepare gt mask
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict["pan_seg_file_name"], "RGB")
            segments_info = dataset_dict["segments_info"]

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)
            instances_gt = Instances(image_shape)
            classes = []
            masks = []
            inst_id = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]

                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
                    inst_id.append(segment_info["id"])
                        
            is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
            classes = np.array(classes)
            is_things = np.array(is_things)
            instances_gt.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances_gt.is_things = torch.tensor(is_things, dtype=torch.int64)
            instances_gt.inst_id = torch.tensor(inst_id, dtype=torch.int64)

            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances_gt.gt_masks = masks
            instances_gt.gt_boxes = masks.get_bounding_boxes()

        interleave_anno = dataset_dict['entities'][0]
        phrases = interleave_anno['phrase']
        phrases = sorted(phrases, key=lambda x: x['start_idx'])

        phrases = np.array(phrases)
        # generate random torch tensor between 0, 1 with shape len(phrases)
        interleave_mask = np.random.random(len(phrases)) < self.interleave_visual_prob
        visual_phrases = phrases[interleave_mask]
        visual_inst_indexes = [phrase['annotation_id'] for phrase in visual_phrases]

        offset = 0
        sentence = interleave_anno['sentence']
        entities = []
        _grounding_masks = []
        entity_list = []
        for phrase in phrases:
            if phrase['annotation_id'] in visual_inst_indexes:
                sentence = sentence.replace(phrase['phrase'], '[INTERACTIVE]')
                _start_idx = phrase['start_idx'] + offset
                offset += (len('[INTERACTIVE]') - (phrase['end_idx'] - phrase['_start_idx']))
                _end_idx = phrase['end_idx'] + offset
                _id = int(phrase['annotation_id'])
                _mask = instances[instances.inst_id==_id].gt_masks.tensor
                # _class = instances[instances.inst_id==_id].gt_classes
                _grounding_masks += [instances_gt[instances.inst_id==_id].gt_masks.tensor]
                _interactive = self.shape_sampler(instances[instances.inst_id==_id])['rand_shape']

                if len(_mask) == 0:
                    _mask = torch.zeros(image_shape).bool()[None, ...]
                    _interactive = torch.zeros_like(_mask).bool()
                    _grounding_masks += [torch.zeros((dataset_dict['height'], dataset_dict['width'])).bool()[None, ...]]

                _type = 'visual'
                _text = '[INTERACTIVE]'
                entities += [Entity(_id, _text, _mask, _interactive, _type, _start_idx, _end_idx)]
                entity_list += [_text]
            else:
                _start_idx = phrase['start_idx'] + offset
                _end_idx = phrase['end_idx'] + offset
                _id = int(phrase['annotation_id'])
                _mask = instances[instances.inst_id==_id].gt_masks.tensor
                # _class = instances[instances.inst_id==_id].gt_classes
                _grounding_masks += [instances_gt[instances.inst_id==_id].gt_masks.tensor]
                _interactive = torch.zeros_like(_mask).bool()

                if len(_mask) == 0:
                    _mask = torch.zeros(image_shape).bool()[None, ...]
                    _interactive = torch.zeros_like(_mask).bool()
                    _grounding_masks += [torch.zeros((dataset_dict['height'], dataset_dict['width'])).bool()[None, ...]]

                _type = 'text'
                _text = phrase['phrase']
                entities += [Entity(_id, _text, _mask, _interactive, _type, _start_idx, _end_idx)]
                entity_list += [_text]

        dataset_dict["groundings"] = {}
        dataset_dict["groundings"]["masks"] = torch.cat(_grounding_masks, dim=0)

        if 'llama' in self.lang_model:
            tokens = self.tokenizer(
                [sentence], padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt', return_offsets_mapping=True
            )
            entities_start_end_idx = torch.tensor([[entity.start_idx, entity.end_idx] for entity in entities])[:, None]
            tokens_start_end_idx = tokens.offset_mapping
            start_end_condition = (entities_start_end_idx[:, :, 0] <= tokens_start_end_idx[:, :, 0]) & (entities_start_end_idx[:, :, 1] >= tokens_start_end_idx[:, :, 1]) & (tokens_start_end_idx[:, :, 0] != tokens_start_end_idx[:, :, 1])
        else:
            tokens, start_end_condition = self.tokenizer(sentence, entity_list)


        dataset_dict['entities'] = {"entities": entities, "sentence": sentence, "tokens": tokens, "entity_to_tokens": start_end_condition}
        return dataset_dict