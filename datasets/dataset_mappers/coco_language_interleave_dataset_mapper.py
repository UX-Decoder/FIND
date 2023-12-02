# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random
import itertools

import numpy as np
import torch

from transformers import AutoTokenizer, LlamaForCausalLM

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.structures.boxes import pairwise_iou
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog
from pycocotools import mask as coco_mask

from utils import prompt_engineering, Entity, prompt_engineering_llm, get_llm_prompt_templates
from modeling.language import build_tokenizer
from modeling.language.misc import text_noun_with_prompt_all
from modeling.utils import configurable, get_class_names

from ..visual_sampler.sampler import build_shape_sampler

__all__ = ["COCOLanguageInterleaveDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        min_scale = cfg_input['MIN_SCALE']
        max_scale = cfg_input['MAX_SCALE']

        augmentation = []


        if cfg_input['RANDOM_FLIP'] != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                    vertical=cfg_input['RANDOM_FLIP'] == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])
    else:
        min_scale = cfg['INPUT']['MIN_SIZE_TEST']
        max_scale = cfg['INPUT']['MAX_SIZE_TEST']
    
        augmentation = []
        augmentation.extend([
            T.ResizeShortestEdge(
                min_scale, max_size=max_scale
            ),
        ])
    return augmentation

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

# This is specifically designed for the COCO dataset.
class COCOLanguageInterleaveDatasetMapper:
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
        *,
        tfm_gens,
        image_format,
        caption_thres,
        grounding,
        retrieval,
        interleave,
        segmentation,
        lvis,
        lvis_thres,
        max_grounding_num,
        interleave_visual_prob,
        shape_sampler,
        interleave_sampler,
        max_token_num,
        tokenizer,
        lang_model,
        class_names,
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
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.caption_thres = caption_thres
        self.grounding = grounding
        self.interleave = interleave
        self.segmentation = segmentation
        self.lvis = lvis

        self.lvis_thres = lvis_thres
        self.max_grounding_num = max_grounding_num
        self.caption_similarity = torch.load(MetadataCatalog.get('logistic').get('caption_similarity_pth'))
        self.interleave_visual_prob = interleave_visual_prob

        self.shape_sampler = shape_sampler
        self.interleave_sampler = interleave_sampler

        self.retrieval = retrieval
        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.lang_model = lang_model
        self.class_names = class_names

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        shape_sampler = build_shape_sampler(cfg)
        interleave_sampler = build_shape_sampler(cfg, is_train=False, mode='hack_train')

        retrieval = cfg['MODEL']['DECODER']['RETRIEVAL']['ENABLED']
        grounding = cfg['MODEL']['DECODER']['GROUNDING']['ENABLED']
        lvis = cfg['MODEL']['DECODER']['LVIS']['ENABLED']
        interleave = cfg['MODEL']['DECODER']['INTERLEAVE']['ENABLED']
        segmentation = cfg['MODEL']['DECODER']['MASK']['ENABLED']

        tokenizer, max_token_num = None, None
        if retrieval or interleave:
            lang_model = cfg['MODEL']['TEXT']['NAME']
            max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
            if 'llama' in lang_model:
                tokenizer = AutoTokenizer.from_pretrained(lang_model, padding_side='right')
                tokenizer.pad_token = tokenizer.eos_token
            elif 'transformer' in lang_model:
                tokenizer_cfg = cfg['MODEL']['TEXT'].copy()
                tokenizer_cfg['TOKENIZER'] = 'clip-token'
                tokenizer = build_tokenizer(tokenizer_cfg)
            else:
                assert False, "Not implemented yet."

        class_names = get_class_names(dataset_name)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "caption_thres": cfg['MODEL']['DECODER']['CAPTION']['SIM_THRES'],
            "grounding": grounding,
            "retrieval": retrieval,
            "interleave": interleave,
            "segmentation": segmentation,
            "lvis": lvis,
            "lvis_thres": cfg['MODEL']['DECODER']['LVIS']['THRES'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN'],
            "interleave_visual_prob": cfg['MODEL']['DECODER']['INTERLEAVE']['VISUAL_PROB'],
            "shape_sampler": shape_sampler,
            "interleave_sampler": interleave_sampler,
            "max_token_num": max_token_num,
            "tokenizer": tokenizer,
            "lang_model": lang_model,
            "class_names": class_names,
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
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Add caption noun that is not in coco set to target
        # HACK to comment
        # captions = dataset_dict["captions"]
        # captions_noun = []
        # for caption in captions:
        #     nouns = np.array(text_noun_with_prompt_all(caption, phrase_prob=0.0, append_text=False)[1])
        #     cap_similarity = np.array([self.caption_similarity[noun][0] for noun in nouns])
        #     captions_noun.append(nouns[cap_similarity < self.caption_thres].tolist())
        # dataset_dict["captions_noun"] = captions_noun
        # HACK to comment
        
        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict
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

        if self.lvis:
            annos = [
                utils.transform_instance_annotations({**obj, **{'bbox_mode':BoxMode.XYXY_ABS}}, transforms, image_shape)
                for obj in dataset_dict['lvis_info']
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            lvis_instances = utils.annotations_to_instances(annos, image_shape)
            lvis_instances = utils.filter_empty_instances(lvis_instances)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if len(lvis_instances) > 0:
                lvis_instances.gt_boxes = lvis_instances.gt_masks.get_bounding_boxes()
                h, w = instances.image_size
                lvis_instances_mask = convert_coco_poly_to_mask(lvis_instances.gt_masks.polygons, h, w).bool()
                coco_inter_lvis = pairwise_iou(instances.gt_boxes, lvis_instances.gt_boxes) > self.lvis_thres
                instances.gt_masks.tensor[coco_inter_lvis.max(dim=-1)[0]] = lvis_instances_mask[coco_inter_lvis.max(dim=-1)[1][coco_inter_lvis.max(dim=-1)[0]]]
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = instances

        spatial_query_utils = self.shape_sampler(instances)
        dataset_dict['spatial_query'] = spatial_query_utils

        if self.segmentation:
            rand_class_id = torch.randint(0, len(self.class_names), (1,))
            class_name = self.class_names[rand_class_id].replace('-other','').replace('-merged','').replace('-stuff','')
            prompt_class_names = []

            for prompt in get_llm_prompt_templates():
                prompt_class_names += [prompt.format(class_name)]

            if 'llama' in self.lang_model:
                tokens = self.tokenizer(
                    prompt_class_names, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
            else:
                tokens = self.tokenizer.tokenizer(
                    prompt_class_names, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
            dataset_dict['class'] = {'tokens': tokens, 'ids': [rand_class_id], 'sentences': prompt_class_names}

        if self.interleave:
            all_sentence = []
            all_input_ids = []
            all_attention_mask = []
            if len(dataset_dict['entities']) == 0:
                _text = 'N/A'
                sentence = 'N/A'
                _mask = torch.zeros(image_shape).bool()[None, ...]
                _interactive = torch.zeros_like(_mask).bool()
                _type = 'empty'
                _start_idx = 0
                _end_idx = len(_text)
                entities = [Entity(-1, _text, _mask, _interactive, _type, _start_idx, _end_idx)]

                if 'llama' in self.lang_model:
                    tokens = self.tokenizer(
                        [sentence], padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt', return_offsets_mapping=True
                    )
                    entities_start_end_idx = torch.tensor([[entity.start_idx, entity.end_idx] for entity in entities])[:, None]
                    tokens_start_end_idx = tokens.offset_mapping
                    start_end_condition = (entities_start_end_idx[:, :, 0] <= tokens_start_end_idx[:, :, 0]) & (entities_start_end_idx[:, :, 1] >= tokens_start_end_idx[:, :, 1]) & (tokens_start_end_idx[:, :, 0] != tokens_start_end_idx[:, :, 1])
                else:
                    tokens = self.tokenizer.tokenizer(
                        [sentence], padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    start_end_condition = torch.zeros(len(entities), len(tokens['input_ids'][0])).bool()

                all_sentence += [sentence]
                all_input_ids += [tokens['input_ids']]
                all_attention_mask += [tokens['attention_mask']]
            else:
                for rand_id in range(len(dataset_dict['entities'])):
                    interleave_anno = dataset_dict['entities'][rand_id]
                    phrases = interleave_anno['phrase']
                    phrases = sorted(phrases, key=lambda x: x['start_idx'])
                    # entities = [Entity(int(phrase['annotation_id']), phrase['phrase'], ) for phrase in phrases]
                    # def __init__(self, _id, _text, _mask, _interactive, _type, _start_idx, _end_idx):

                    phrases = np.array(phrases)
                    if len(phrases) > 6:
                        phrases = phrases[:6]
                        print('{} Warning: too many phrases, only use the first 6 phrases.'.format(dataset_dict['file_name']))

                    all_permutation = np.array(list(itertools.product([True, False], repeat=len(phrases))))

                    for interleave_mask in all_permutation:
                        # generate random torch tensor between 0, 1 with shape len(phrases)
                        # interleave_mask = np.random.random(len(phrases)) < self.interleave_visual_prob                    
                        visual_phrases = phrases[interleave_mask]
                        visual_inst_indexes = [phrase['annotation_id'] for phrase in visual_phrases]

                        offset = 0
                        sentence = interleave_anno['sentence']
                        entities = []
                        entity_list = []
                        for phrase in phrases:
                            if phrase['annotation_id'] in visual_inst_indexes:
                                sentence = sentence.replace(phrase['phrase'], '[INTERACTIVE]')
                                _start_idx = phrase['start_idx'] + offset
                                offset += (len('[INTERACTIVE]') - (phrase['end_idx'] - phrase['_start_idx']))
                                _end_idx = phrase['end_idx'] + offset
                                _id = int(phrase['annotation_id'])
                                _mask = instances[instances.inst_id==_id].gt_masks.tensor
                                _interactive = self.shape_sampler(instances[instances.inst_id==_id])['rand_shape']

                                if len(_mask) == 0:
                                    _mask = torch.zeros(image_shape).bool()[None, ...]
                                    _interactive = torch.zeros_like(_mask).bool()

                                _type = 'visual'
                                _text = '[INTERACTIVE]'
                                entities += [Entity(_id, _text, _mask, _interactive, _type, _start_idx, _end_idx)]
                                entity_list += [_text]
                            else:
                                _start_idx = phrase['start_idx'] + offset
                                _end_idx = phrase['end_idx'] + offset
                                _id = int(phrase['annotation_id'])
                                _mask = instances[instances.inst_id==_id].gt_masks.tensor
                                _interactive = torch.zeros_like(_mask).bool()

                                if len(_mask) == 0:
                                    _mask = torch.zeros(image_shape).bool()[None, ...]
                                    _interactive = torch.zeros_like(_mask).bool()

                                _type = 'text'
                                _text = phrase['phrase']
                                entities += [Entity(_id, _text, _mask, _interactive, _type, _start_idx, _end_idx)]
                                entity_list += [_text]

                        if 'llama' in self.lang_model:
                            tokens = self.tokenizer(
                                [sentence], padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt', return_offsets_mapping=True
                            )
                            entities_start_end_idx = torch.tensor([[entity.start_idx, entity.end_idx] for entity in entities])[:, None]
                            tokens_start_end_idx = tokens.offset_mapping
                            start_end_condition = (entities_start_end_idx[:, :, 0] <= tokens_start_end_idx[:, :, 0]) & (entities_start_end_idx[:, :, 1] >= tokens_start_end_idx[:, :, 1]) & (tokens_start_end_idx[:, :, 0] != tokens_start_end_idx[:, :, 1])
                        else:
                            tokens, start_end_condition = self.tokenizer(sentence, entity_list)

                        all_sentence += [sentence]
                        all_input_ids += [tokens['input_ids']]
                        all_attention_mask += [tokens['attention_mask']]

                # HACK for debug
                # for i in range(len(entities)):
                #     yy = tokens_start_end_idx[0][start_end_condition[i]]
                #     st = yy[0][0]
                #     ed = yy[-1][1]
                #     est = entities_start_end_idx[i,0,0]
                #     eed = entities_start_end_idx[i,0,1]
                #     print(sentence[st:ed], entities[i].text, sentence[est:eed])
            dataset_dict['entities'] = {"sentence": all_sentence, "tokens": {'input_ids': torch.cat(all_input_ids), 'attention_mask': torch.cat(all_attention_mask)}}

        if self.retrieval:
            captions = dataset_dict['captions']

            if 'llama' in self.lang_model:
                tokens = self.tokenizer(
                    captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
            else:
                tokens = self.tokenizer.tokenizer(
                    captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )

            dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

        if self.grounding:
            grounding_anno = dataset_dict['grounding_info']
            if len(grounding_anno) > 0:
                masks_grd = []
                texts_grd = []
                mode = 'text'
                random.shuffle(grounding_anno)
                for ann in grounding_anno:
                    rle = coco_mask.frPyObjects(
                        ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                    m = coco_mask.decode(rle)
                    # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)  # convert to np.uint8
                    m = transforms.apply_segmentation(m[:,:,None])[:,:,0]
                    masks_grd += [m]
                    # random select a sentence of a single annotation.
                    # rand_index = random.randint(0, len(ann['sentences'])-1)
                    texts_grd += [ann['sentences'][p]['raw'].lower() for p in range(len(ann['sentences']))]
                max_len = len(texts_grd)
                indices = np.random.permutation(max_len)
                texts_grd = list(np.array(texts_grd)[indices])
                masks_grd = None
                hash_grd = np.array([hash(txt) for txt in texts_grd])
            else:
                masks_grd = instances.gt_masks.tensor
                texts_grd = []
                mode = 'class'
                hash_grd = []

            groundings = {'masks': masks_grd, 'texts': texts_grd, 'mode': mode, 'hash': hash_grd}
            dataset_dict["groundings"] = groundings

        if not self.is_train:
            dataset_dict['interleave_sampler'] = self.interleave_sampler

        return dataset_dict