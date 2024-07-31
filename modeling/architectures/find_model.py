# --------------------------------------------------------
# FIND -- Interfacing Foundation Models' Embeddings
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kornia.contrib import distance_transform

from timm.models.layers import trunc_normal_
from nltk.stem.lancaster import LancasterStemmer
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .build import register_model
from ..utils import configurable, get_class_names, get_iou, pad_arbitrary_tensors, Spatial_ImageList, decode_entity_mask_to_indices, strict_hash, move_dict_to_cpu
from ..vision.backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity

from utils.constants import COCO_PANOPTIC_CLASSES
from utils.distributed import get_rank

st = LancasterStemmer()


class GeneralizedFIND(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        interactive_mode: str,
        interactive_iter: str,
        dilation_kernel: torch.Tensor,
        train_max_iter: int,
        class_token_length: int,
        retrieval_emsemble: bool,
        backbone_dim: int,
        dim_proj: int,
        language_embed_root: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob
        self.train_max_iter = train_max_iter

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = get_class_names(train_dataset_name)
        self.interactive_mode = interactive_mode
        self.interactive_iter = interactive_iter

        self.retrieval_emsemble = retrieval_emsemble
        self.class_token_length = class_token_length
        self.language_embed_root = language_embed_root

        # backbone itc loss
        if task_switch['retrieval'] and retrieval_emsemble:
            self.backbone_proj = nn.Parameter(torch.empty(backbone_dim, dim_proj))
            trunc_normal_(self.backbone_proj, std=.02)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.register_buffer("dilation_kernel", dilation_kernel)

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        loss_weights = {'mask': {'ce': dec_cfg['CLASS_WEIGHT'], 'dice': dec_cfg['DICE_WEIGHT'], 'bce': dec_cfg['MASK_WEIGHT']},
                        'bbox': {'l1': dec_cfg['BBOX_WEIGHT'], 'giou': dec_cfg['GIOU_WEIGHT']},
                        'spatial': {'ce': dec_cfg['SCLASS_WEIGHT'], 'dice': dec_cfg['SDICE_WEIGHT'], 'bce': dec_cfg['SMASK_WEIGHT']},
                        'interleave': {'itc': dec_cfg['IITC_WEIGHT'], 'ce': dec_cfg['ICLASS_WEIGHT'], 'dice': dec_cfg['IDICE_WEIGHT'], 'bce': dec_cfg['IMASK_WEIGHT']},
                        'retrieval': {'decoder': dec_cfg['RETRIEVAL_WEIGHT'], 'backbone': dec_cfg['BACKBONER_WEIGHT']},
                        'grounding': {'ce': dec_cfg['GCLASS_WEIGHT'], 'dice': dec_cfg['GDICE_WEIGHT'], 'bce': dec_cfg['GMASK_WEIGHT']},
                        'openimage': {'ce': dec_cfg['OCLASS_WEIGHT'], 'dice': dec_cfg['ODICE_WEIGHT'], 'bce': dec_cfg['OMASK_WEIGHT']}}

        openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get('ENABLED', False),
                            'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}

        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg.get('MASK', True),
                       'spatial': dec_cfg['SPATIAL'].get('ENABLED', False),
                       'retrieval': dec_cfg['RETRIEVAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'interleave': dec_cfg['INTERLEAVE'].get('ENABLED', False),
                       'openimage': openimage_switch}

        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'grounding': dec_cfg.get('TOP_GROUNDING_LAYERS', 10),
                        'retrieval': dec_cfg.get('TOP_RETRIEVAL_LAYERS', 10),
                        'openimage': dec_cfg.get('TOP_OPENIMAGE_LAYERS', 10),
                        'spatial': dec_cfg.get('TOP_SPATIAL_LAYERS', 10),
                        'interleave': dec_cfg.get('TOP_INTERLEAVE_LAYERS', 10)}

        spatial_cost = {"class_weight": dec_cfg['COST_SPATIAL']['CLASS_WEIGHT'],
                        "mask_weight": dec_cfg['COST_SPATIAL']['MASK_WEIGHT'],
                        "dice_weight": dec_cfg['COST_SPATIAL']['DICE_WEIGHT']}

        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=loss_weights['mask']['ce'],
            cost_mask=loss_weights['mask']['bce'],
            cost_dice=loss_weights['mask']['dice'],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            spatial_cost=spatial_cost,
        )

        # init weight dict and criterion loss functions.
        losses = {'seg': [], 'openimage': []}
        if task_switch['mask']:
            losses['seg'] += ["labels", "masks"]
        if task_switch['spatial']:
            losses['seg'] += ["spatials"]
        if task_switch['grounding']:
            losses['seg'] += ["groundings"]
        if task_switch['retrieval']:
            losses['seg'] += ["retrievals_v2"]
        if task_switch['interleave']:
            losses['seg'] += ["interleaves"]
        if task_switch['openimage']:
            losses['openimage'] += ["labels_openimage", "masks"]
        if task_switch['openimage']['grounding']:
            losses['openimage'] += ["groundings"]

        weight_dict = {}
        for key, turn_on in task_switch.items():
            if turn_on:
                if isinstance(loss_weights[key], dict):
                    # HACK it should support bbox in the future
                    for key_, weight in loss_weights[key].items():
                        weight_dict["loss_{}_{}_0".format(key, key_)] = weight # NOTE: hard code for segmentation that has multiple loss
                else:
                    weight_dict["loss_{}_0".format(key)] = loss_weights[key]

        # generate full weight dict and remove not computed layers. 
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                for k, v in weight_dict.items():
                    if (i+1) > (top_x_layers[k.split('_')[1]] - 1):
                        continue
                    aux_weight_dict.update({k.replace('_0', f"_{i+1}"): v})
            weight_dict.update(aux_weight_dict)

        grd_weight = {'text': dec_cfg['GROUNDING']['TEXT_WEIGHT'], 'class': dec_cfg['GROUNDING']['CLASS_WEIGHT']}
        # generate critenrion for loss function.
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=[],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=grd_weight,
        )

        # extra logistic
        train_dataset_name = cfg['DATASETS']['TRAIN'][0] # HACK for only one training set.
        train_max_iter = dec_cfg['SPATIAL'].get('MAX_ITER', 3)
        phrase_prob = dec_cfg['CAPTION'].get('PHRASE_PROB', 0.5)
        interactive_mode = cfg['STROKE_SAMPLER']['EVAL']['MODE']
        interactive_iter = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']
        class_token_length = cfg['MODEL']['DECODER']['MASK']['CLASS_TOKEN_LENGTH']

        dilation = 3
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        # prepare language feature folder
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        language_embed_root = os.path.join(_root, 'coco', '{}_{}'.format(sem_seg_head.predictor.lang_encoder.arch, sem_seg_head.predictor.lang_encoder.feature_layer))

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "interactive_mode": interactive_mode,
            "interactive_iter": interactive_iter,
            "dilation_kernel": dilation_kernel,
            "train_max_iter": train_max_iter,
            "class_token_length": class_token_length,
            "retrieval_emsemble": dec_cfg['RETRIEVAL']['ENSEMBLE'],
            "backbone_dim": cfg['MODEL']['BACKBONE_DIM'],
            "dim_proj": cfg['MODEL']['DIM_PROJ'],
            "language_embed_root": language_embed_root,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode='default'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            losses = {}
            if self.sem_seg_head.predictor.lang_encoder.precompute:
                return self.preprocess_language(batched_inputs)
            if self.task_switch['mask']:
                losses_seg = self.forward_seg(batched_inputs)
                losses.update(losses_seg)
            if self.task_switch['openimage'] and self.task_switch['openimage']['mask']:
                losses_openimage = self.forward_openimage(batched_inputs['openimage'])
                losses_openimage = {key.replace('mask', 'openimage'):value for key, value in losses_openimage.items()}
                losses_openimage = {key.replace('grounding', 'grounding_openimage'):value for key, value in losses_openimage.items()}
                losses.update(losses_openimage)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else: # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if mode == 'interactive':
                return self.evaluate_interactive(batched_inputs)
            elif mode == 'grounding_spatial':
                return self.evaluate_grounding_sptial(batched_inputs, mode)
            elif mode == 'retrieval':
                return self.evaluate_retrieval(batched_inputs)
            elif mode == 'retrieval_patch':
                return self.evaluate_retrieval_patch(batched_inputs)
            elif mode == 'retrieval_interactive':
                return self.evaluate_retrieval_interactive(batched_inputs)
            elif mode == 'retrieval_interleave_text':
                return self.evaluate_retrieval_interleave_text(batched_inputs)
            elif mode == 'retrieval_interleave_crossquery':
                return self.evaluate_retrieval_interleave(batched_inputs)
            elif mode == 'grounding_entity':
                return self.evaluate_interleave_grounding(batched_inputs)
            elif mode in ['grounding_phrasecut', 'grounding_refcoco']:
                return self.evaluate_grounding(batched_inputs, mode)
            else:
                return self.evaluate(batched_inputs)

        
    def forward_seg(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
        del features

        extra = {}
        # mask classification target
        if "instances" in batched_inputs[0]:
            # input bounding box is checked to be correct.
            targets = self.prepare_targets(batched_inputs, images)

            if self.task_switch['grounding']:
                grounding_tokens = [x['grounding_token_embs'] for x in targets] # need to pad for more than one grounding token
                grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens, padding_value=-1)
                non_zero_query_mask = (grounding_tokens.sum(dim=-1) == -grounding_tokens.shape[-1])
                grounding_tokens[non_zero_query_mask] = 0

                grounding_token_indices = [x['grounding_token_indices'] for x in targets]
                grounding_token_indices = nn.utils.rnn.pad_sequence(grounding_token_indices, padding_value=-2)

                grounding_query_indices = [x['grounding_query_indices'] for x in targets]
                grounding_query_indices = nn.utils.rnn.pad_sequence(grounding_query_indices, padding_value=-3)

                extra['grounding_tokens'] = grounding_tokens
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
                extra['grounding_token_indices'] = grounding_token_indices
                extra['grounding_query_indices'] = grounding_query_indices

            if self.task_switch['spatial']:
                pos_masks = [x['spatial_query']['rand_shape'].to(self.device) for x in batched_inputs]
                neg_masks = [(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs]
                fp_masks = nn.utils.rnn.pad_sequence([(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs], padding_value=False, batch_first=True)

                # prepare spatial query indcies with shape [ns, bs]
                spatial_query_indices = [x['spatial_query_indices'] for x in targets]
                spatial_query_indices = nn.utils.rnn.pad_sequence(spatial_query_indices, padding_value=-3)

                extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks, 'false_positive_mask': fp_masks, 'spatial_query_indices': spatial_query_indices})

            if self.task_switch['interleave']:
                interleave_tokens = [x['interleave_query_embs'].to(self.device) for x in targets] # need to pad for more than one grounding token
                interleave_tokens = nn.utils.rnn.pad_sequence(interleave_tokens, padding_value=-1)
                non_zero_query_mask = (interleave_tokens.sum(dim=-1) == -interleave_tokens.shape[-1]).to(self.device)
                interleave_tokens[non_zero_query_mask] = 0
                pos_masks = [x['interleave_spatial'].to(self.device) for x in targets]
                interleave_isvisual = nn.utils.rnn.pad_sequence([x['interleave_isvisual'].float().to(self.device) for x in targets], padding_value=0)
                interleave_entity_mask = pad_arbitrary_tensors([x['interleave_entity_mask'].to(self.device) for x in targets], padding_value=False).transpose(0,1)
                interleave_entity_indices = nn.utils.rnn.pad_sequence([x['interleave_entity_indices'].to(self.device) for x in targets], padding_value=-5)
                interleave_query_indices = nn.utils.rnn.pad_sequence([x['interleave_query_indices'].to(self.device) for x in targets], padding_value=-6)

                extra['interleave_tokens'] = interleave_tokens
                extra['interleave_nonzero_mask'] = non_zero_query_mask.t()
                extra['interleave_query_pos_mask'] = pos_masks
                extra['interleave_query_neg_mask'] = [x & False for x in pos_masks]
                extra['interleave_isvisual'] = interleave_isvisual
                extra['interleave_entity_mask'] = interleave_entity_mask
                extra['interleave_entity_indices'] = interleave_entity_indices
                extra['interleave_query_indices'] = interleave_query_indices

            if self.task_switch['retrieval']:
                retrieval_tokens = [x['retrieval_query_embs'] for x in targets]
                retrieval_tokens = nn.utils.rnn.pad_sequence(retrieval_tokens, padding_value=-1)
                non_zero_query_mask = (retrieval_tokens.sum(dim=-1) == -retrieval_tokens.shape[-1])
                retrieval_tokens[non_zero_query_mask] = 0

                extra['retrieval_tokens'] = retrieval_tokens
                extra['retrieval_nonzero_mask'] = non_zero_query_mask.t()

            if self.task_switch['mask']['ENABLED']:
                class_tokens = [x['class_tokens_embs'] for x in targets]
                class_tokens = nn.utils.rnn.pad_sequence(class_tokens, padding_value=-1)
                non_zero_query_mask = (class_tokens.sum(dim=-1) == -class_tokens.shape[-1])
                class_tokens[non_zero_query_mask] = 0
                class_indexes = torch.tensor([x['class_ids'] for x in targets])[:,0]

                extra['class_tokens_emb'] = class_tokens
                extra['class_tokens_mask'] = non_zero_query_mask.t()
                extra['class_indexes'] = class_indexes

        self.get_class_embeddings(self.train_class_names, is_eval=False)

        # forward spatial only without gradient
        if self.task_switch['spatial']:
            with torch.no_grad():
                # generate random integeter between [0,3]
                rand_iter_num = random.randint(0, self.train_max_iter)
                for i in range(rand_iter_num):
                    outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='spatial')
                    extra.update(outputs)
                    extra.update(self.prepare_next_spaital_mask(extra, batched_inputs))

        del batched_inputs
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='seg')
        extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.logit_scale,
                 'class_embeddings': getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('default')),
                 'false_positive_mask': extra['false_positive_mask'] if 'fp_masks' in locals() else None,
                 'lang_encoder': self.sem_seg_head.predictor.lang_encoder,
                 'training': self.training,
                 'interleave_isvisual': extra['interleave_isvisual'] if 'interleave_isvisual' in locals() else None, }
                #  'interleave_classes': interleave_classes.mean,} # HACK
        # bipartite matching-based loss
        self.criterion.losses = self.losses['seg'] # seg criterion losses
        losses = self.criterion(outputs, targets, extra)

        del outputs
        return losses

    @torch.no_grad()
    def preprocess_language(self, batched_inputs):
        targets = self.prepare_language_targets(batched_inputs)
        
        output_folder = self.language_embed_root
        # create folder if not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        for text, _hash, input_ids, attention_mask in zip(targets['texts'], targets['hash_values'], targets['input_ids'], targets['attention_mask']):
            output_pth = os.path.join(output_folder, '{}.da'.format(_hash))
            if os.path.exists(output_pth):
                _result = torch.load(output_pth)
                if _result['tokens']['input_ids'].sum() == input_ids.sum():
                    continue
            result = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({"input_ids": input_ids[None,], "attention_mask": attention_mask[None,]}, token=True, projection=False)
            result = move_dict_to_cpu(result)
            result['text'] = text
            torch.save(result, output_pth)

        return {"fake_loss": torch.tensor(0.0, device=self.device)}

    def evaluate(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        box_pred_results = outputs["pred_boxes"] if self.task_switch['bbox'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        input_size = mask_pred_results.shape[-2:]
        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, box_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, box_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # instance segmentation inference
            if self.instance_on:
                if self.task_switch['bbox']:
                    box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def evaluate_retrieval_interleave(self, batched_inputs):
        # interactive
        assert self.task_switch['spatial']
        assert self.task_switch['interleave']
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        def inference_visual(entity, extra, index):
            outputs = {}
            images = [entity.image.to(self.device)]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            features = self.backbone(images.tensor)
            mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

            _extra = {}
            _extra['interleave_entity_mask'] = extra['interleave_entity_mask'][index:index+1]
            interleave_entity_mask = _extra['interleave_entity_mask'][0].transpose(0,1)
            _extra['interleave_entity_mask'] = interleave_entity_mask[interleave_entity_mask==True][None,None]
            _extra['interleave_isvisual'] = extra['interleave_isvisual'][index:index+1]
            _extra['interleave_entity_indices'] = extra['interleave_entity_indices'][interleave_entity_mask.bool()][:,None]*0
            _extra['interleave_tokens'] = extra['interleave_tokens'][interleave_entity_mask.bool()][:,None]
            _extra['interleave_nonzero_mask'] = extra['interleave_nonzero_mask'][interleave_entity_mask.bool().transpose(0,1)][None,]
            _extra['interleave_query_indices'] = extra['interleave_query_indices'][index:index+1] * 0
            _extra['interleave_query_pos_mask'] = entity.interactive.to(self.device)[None,]
            _extra['interleave_query_neg_mask'] = entity.interactive.to(self.device)[None,].clone().detach() & False
            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=_extra, task='refint_interleave')
            outputs.update({'pred_entity_class': _outputs['pred_entity_class'], 'pred_interleave_objects': _outputs['pred_interleave_objects'],
                            'pred_imaskembs': _outputs['pred_imaskembs'], 'pred_entity_pixel': _outputs['pred_entity_pixel'],
                            'src_interleave_queries': _outputs['src_interleave_queries'], 'src_interleave_maskings': _outputs['src_interleave_maskings'], 'src_interleave_indices': _outputs['src_interleave_indices']})

            extra = {}
            extra['spatial_query_pos_mask'] = entity.interactive.to(self.device)[None,]
            extra['spatial_query_neg_mask'] = entity.interactive.to(self.device)[None,].clone().detach() & False
            extra['spatial_query_indices'] = torch.arange(1, device=self.device)[None,]
            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='refimg_spatial')
            outputs.update({'pred_smaskembs': _outputs['pred_smaskembs'], 'pred_pspatials': _outputs['pred_pspatials'], 'pred_spatials': _outputs['pred_spatials'],
                            'src_spatial_queries': _outputs['src_spatial_queries'], 'src_spatial_maskings': _outputs['src_spatial_maskings'], 'src_spatial_indices': _outputs['src_spatial_indices']})
            return outputs

        def inference_text(entity, extra, index):
            outputs = {}
            features = self.backbone(torch.zeros(1,3,640,640).to(self.device))
            mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

            _extra = {}
            interleave_entity_mask = extra['interleave_entity_mask'][index:index+1][0].transpose(0,1)
            _extra['grounding_token_indices'] = extra['interleave_entity_indices'][interleave_entity_mask.bool()][:,None]*0
            _extra['grounding_tokens'] = extra['interleave_tokens'][interleave_entity_mask.bool()][:,None]
            _extra['grounding_nonzero_mask'] = extra['interleave_nonzero_mask'][interleave_entity_mask.bool().transpose(0,1)][None,]
            _extra['grounding_query_indices'] = extra['interleave_query_indices'][index:index+1] * 0

            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=_extra, task='refint_interleave')
            outputs.update({'pred_gtexts': _outputs['pred_gtexts'], 'pred_grounding_query': _outputs['pred_grounding_query'],})
            return outputs

        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            input_ids += x['entities']['tokens']['input_ids']
            attention_mask += x['entities']['tokens']['attention_mask']
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)
        
        targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            # interleave
            target_dict = {}
            target_dict['interleave_query_embs'] = lang_results_interleave['token_emb'][idx:idx+1][lang_results_interleave['tokens']['attention_mask'][idx:idx+1].bool()]
            target_dict['interleave_entity_mask'] = batch_per_image['entities']['entity_to_tokens'][:,lang_results_interleave['tokens']['attention_mask'][idx].bool()]
            target_dict['interleave_isvisual'] = torch.tensor([x.type == 'visual' for x in batch_per_image['entities']['entities']])
            target_dict['interleave_entity_indices'] = decode_entity_mask_to_indices(target_dict['interleave_entity_mask'])
            target_dict['interleave_query_indices'] = torch.arange(len(target_dict['interleave_entity_mask']), device=target_dict['interleave_query_embs'].device)
            targets += [target_dict]

        interleave_tokens = [x['interleave_query_embs'].to(self.device) for x in targets] # need to pad for more than one grounding token
        interleave_tokens = nn.utils.rnn.pad_sequence(interleave_tokens, padding_value=-1)
        non_zero_query_mask = (interleave_tokens.sum(dim=-1) == -interleave_tokens.shape[-1]).to(self.device)
        interleave_tokens[non_zero_query_mask] = 0
        interleave_isvisual = nn.utils.rnn.pad_sequence([x['interleave_isvisual'].float().to(self.device) for x in targets], padding_value=0)
        interleave_entity_mask = pad_arbitrary_tensors([x['interleave_entity_mask'].to(self.device) for x in targets], padding_value=False).transpose(0,1)
        interleave_entity_indices = nn.utils.rnn.pad_sequence([x['interleave_entity_indices'].to(self.device) for x in targets], padding_value=-5)
        interleave_query_indices = nn.utils.rnn.pad_sequence([x['interleave_query_indices'].to(self.device) for x in targets], padding_value=-6)

        extra = {}
        extra['interleave_tokens'] = interleave_tokens
        extra['interleave_nonzero_mask'] = non_zero_query_mask.t()
        extra['interleave_isvisual'] = interleave_isvisual
        extra['interleave_entity_mask'] = interleave_entity_mask
        extra['interleave_entity_indices'] = interleave_entity_indices
        extra['interleave_query_indices'] = interleave_query_indices
        
        entity_list = batched_inputs[0]['entities']['entities']
        entity_features = {"src_interleave_queries": [], "src_interleave_maskings": [], "src_interleave_indices": []}
        spatial_features = {"src_spatial_queries": [], "src_spatial_maskings": [], "src_spatial_indices": []}
        interleave_class_proposals = []
        interleave_pixel_proposals = []
        interleave_pixel_query = []
        interleave_class_query = []

        for idx, entity in enumerate(entity_list):
            if entity.type == 'visual':
                outputs = inference_visual(entity, extra, idx)
                entity_features['src_interleave_queries'].append(outputs['src_interleave_queries'])
                entity_features['src_interleave_maskings'].append(outputs['src_interleave_maskings'])
                src_interleave_indices = []
                for x in outputs['src_interleave_indices']:
                    x[x==0] = idx
                    src_interleave_indices.append(x)
                entity_features['src_interleave_indices'].append(src_interleave_indices)

                spatial_features['src_spatial_queries'].append(outputs['src_spatial_queries'])
                spatial_features['src_spatial_maskings'].append(outputs['src_spatial_maskings'])
                src_interleave_indices = []
                for x in outputs['src_spatial_indices']:
                    x[x==0] = idx
                    src_interleave_indices.append(x)
                spatial_features['src_spatial_indices'].append(src_interleave_indices)

                interleave_class_proposals += [outputs['pred_interleave_objects']]
                interleave_pixel_proposals += [outputs['pred_imaskembs']]
                interleave_pixel_query += [outputs['pred_entity_pixel']]
                interleave_class_query += [outputs['pred_entity_class']]
            elif entity.type == 'text':
                outputs = inference_text(entity, extra, idx)

                empty_token_queries = torch.zeros(1,1,512).to(self.device)
                empty_token_maskings = torch.ones(1,1).to(self.device).bool()
                empty_token_indices = torch.zeros(1,1,1).to(self.device)

                entity_features['src_interleave_queries'].append([empty_token_queries, empty_token_queries, empty_token_queries])
                entity_features['src_interleave_maskings'].append([empty_token_maskings, empty_token_maskings, empty_token_maskings])
                entity_features['src_interleave_indices'].append([empty_token_indices-8, empty_token_indices-8, empty_token_indices-8])
        
                spatial_features['src_spatial_queries'].append([empty_token_queries, empty_token_queries, empty_token_queries])
                spatial_features['src_spatial_maskings'].append([empty_token_maskings, empty_token_maskings, empty_token_maskings])
                spatial_features['src_spatial_indices'].append([empty_token_indices-9, empty_token_indices-9, empty_token_indices-9])

                interleave_class_proposals += [None]
                interleave_pixel_proposals += [None]
                interleave_pixel_query += [None]
                interleave_class_query += [outputs['pred_grounding_query']]

        src_interleave_queries = []
        src_interleave_maskings = []
        src_interleave_indices = []
        for i in range(3):
            src_interleave_queries.append(torch.cat([x[i] for x in entity_features['src_interleave_queries']], dim=0))
            src_interleave_maskings.append(torch.cat([x[i] for x in entity_features['src_interleave_maskings']], dim=1))
            src_interleave_indices.append(torch.cat([x[i] for x in entity_features['src_interleave_indices']], dim=1))

        src_spatial_queries = []
        src_spatial_maskings = []
        src_spatial_indices = []
        for i in range(3):
            src_spatial_queries.append(torch.cat([x[i] for x in spatial_features['src_spatial_queries']], dim=0))
            src_spatial_maskings.append(torch.cat([x[i] for x in spatial_features['src_spatial_maskings']], dim=1))
            src_spatial_indices.append(torch.cat([x[i] for x in spatial_features['src_spatial_indices']], dim=1))

        extra['refint_tokens'] = {'src_interleave_queries': src_interleave_queries, 'src_interleave_maskings': src_interleave_maskings, 'src_interleave_indices': src_interleave_indices}
        extra['refimg_tokens'] = {'src_spatial_queries': src_spatial_queries, 'src_spatial_maskings': src_spatial_maskings, 'src_spatial_indices': src_spatial_indices}

        interleave_entities = []
        interleave_ids = []
        # process class embedding for each entity
        for idx, entity in enumerate(entity_list):
            if entity.type == 'visual':
                s_emb = interleave_pixel_query[idx]
                v_emb = interleave_pixel_proposals[idx]
                pred_logits = v_emb @ s_emb.transpose(1,2)
                selected_v_idx = pred_logits.max(dim=1)[1][0]

                c_emb = interleave_class_proposals[idx]
                c_emb = c_emb[:, selected_v_idx]
                interleave_entities += [c_emb]
                interleave_ids += [entity.id]
            else:
                c_emb = interleave_class_query[idx]
                interleave_entities += [c_emb]
                interleave_ids += [-1]

        interleave_ids = torch.tensor(interleave_ids, device=self.device)
        interleave_entities = torch.cat(interleave_entities, dim=0)[:,0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='seg')

        caption_results = {
                'image_embeds': outputs['pred_retrievals'],
                'object_queries_semantic': outputs['pred_captions'],
                'interleave_embeds': outputs['pred_interleave_image'],
                'interleave_entities': interleave_entities,
                'interleave_ids': interleave_ids,
                'image_ids': int(batched_inputs[0]['image_id']),
            }
        processed_results = [{"caption": caption_results}]
        return processed_results

    def demo_interleave(self, batched_inputs):
        # interactive
        assert self.task_switch['spatial']
        assert self.task_switch['interleave']
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        def inference_visual(entity, extra, index):
            outputs = {}
            images = entity.image
            img_bs = images.tensor.shape[0]

            features = self.backbone(images.tensor)
            mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

            _extra = {}
            _extra['interleave_entity_mask'] = extra['interleave_entity_mask'][index:index+1]
            interleave_entity_mask = _extra['interleave_entity_mask'][0].transpose(0,1)
            _extra['interleave_entity_mask'] = interleave_entity_mask[interleave_entity_mask==True][None,None]
            _extra['interleave_isvisual'] = extra['interleave_isvisual'][index:index+1]
            _extra['interleave_entity_indices'] = extra['interleave_entity_indices'][interleave_entity_mask.bool()][:,None]*0
            _extra['interleave_tokens'] = extra['interleave_tokens'][interleave_entity_mask.bool()][:,None]
            _extra['interleave_nonzero_mask'] = extra['interleave_nonzero_mask'][interleave_entity_mask.bool().transpose(0,1)][None,]
            _extra['interleave_query_indices'] = extra['interleave_query_indices'][index:index+1] * 0
            _extra['interleave_query_pos_mask'] = entity.interactive.to(self.device)[None,]
            _extra['interleave_query_neg_mask'] = entity.interactive.to(self.device)[None,].clone().detach() & False
            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=_extra, task='refint_interleave')
            outputs.update({'pred_entity_class': _outputs['pred_entity_class'], 'pred_interleave_objects': _outputs['pred_interleave_objects'],
                            'pred_imaskembs': _outputs['pred_imaskembs'], 'pred_entity_pixel': _outputs['pred_entity_pixel'],
                            'src_interleave_queries': _outputs['src_interleave_queries'], 'src_interleave_maskings': _outputs['src_interleave_maskings'], 'src_interleave_indices': _outputs['src_interleave_indices']})

            extra = {}
            extra['spatial_query_pos_mask'] = entity.interactive.to(self.device)[None,]
            extra['spatial_query_neg_mask'] = entity.interactive.to(self.device)[None,].clone().detach() & False
            extra['spatial_query_indices'] = torch.arange(1, device=self.device)[None,]
            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='refimg_spatial')
            outputs.update({'pred_smaskembs': _outputs['pred_smaskembs'], 'pred_pspatials': _outputs['pred_pspatials'], 'pred_spatials': _outputs['pred_spatials'],
                            'src_spatial_queries': _outputs['src_spatial_queries'], 'src_spatial_maskings': _outputs['src_spatial_maskings'], 'src_spatial_indices': _outputs['src_spatial_indices']})
            return outputs

        def inference_text(entity, extra, index):
            outputs = {}
            features = self.backbone(torch.zeros(1,3,640,640).to(self.device))
            mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

            _extra = {}
            interleave_entity_mask = extra['interleave_entity_mask'][index:index+1][0].transpose(0,1)
            _extra['grounding_token_indices'] = extra['interleave_entity_indices'][interleave_entity_mask.bool()][:,None]*0
            _extra['grounding_tokens'] = extra['interleave_tokens'][interleave_entity_mask.bool()][:,None]
            _extra['grounding_nonzero_mask'] = extra['interleave_nonzero_mask'][interleave_entity_mask.bool().transpose(0,1)][None,]
            _extra['grounding_query_indices'] = extra['interleave_query_indices'][index:index+1] * 0

            _outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=_extra, task='refint_interleave')
            outputs.update({'pred_gtexts': _outputs['pred_gtexts'], 'pred_grounding_query': _outputs['pred_grounding_query'],})
            return outputs

        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            input_ids += x['entities']['tokens']['input_ids']
            attention_mask += x['entities']['tokens']['attention_mask']
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)
        
        targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            # interleave
            target_dict = {}
            target_dict['interleave_query_embs'] = lang_results_interleave['token_emb'][idx:idx+1][lang_results_interleave['tokens']['attention_mask'][idx:idx+1].bool()]
            target_dict['interleave_entity_mask'] = batch_per_image['entities']['entity_to_tokens'][:,lang_results_interleave['tokens']['attention_mask'][idx].bool()]
            target_dict['interleave_isvisual'] = torch.tensor([x.type == 'visual' for x in batch_per_image['entities']['entities']])
            target_dict['interleave_entity_indices'] = decode_entity_mask_to_indices(target_dict['interleave_entity_mask'])
            target_dict['interleave_query_indices'] = torch.arange(len(target_dict['interleave_entity_mask']), device=target_dict['interleave_query_embs'].device)
            targets += [target_dict]

        interleave_tokens = [x['interleave_query_embs'].to(self.device) for x in targets] # need to pad for more than one grounding token
        interleave_tokens = nn.utils.rnn.pad_sequence(interleave_tokens, padding_value=-1)
        non_zero_query_mask = (interleave_tokens.sum(dim=-1) == -interleave_tokens.shape[-1]).to(self.device)
        interleave_tokens[non_zero_query_mask] = 0
        interleave_isvisual = nn.utils.rnn.pad_sequence([x['interleave_isvisual'].float().to(self.device) for x in targets], padding_value=0)
        interleave_entity_mask = pad_arbitrary_tensors([x['interleave_entity_mask'].to(self.device) for x in targets], padding_value=False).transpose(0,1)
        interleave_entity_indices = nn.utils.rnn.pad_sequence([x['interleave_entity_indices'].to(self.device) for x in targets], padding_value=-5)
        interleave_query_indices = nn.utils.rnn.pad_sequence([x['interleave_query_indices'].to(self.device) for x in targets], padding_value=-6)

        extra = {}
        extra['interleave_tokens'] = interleave_tokens
        extra['interleave_nonzero_mask'] = non_zero_query_mask.t()
        extra['interleave_isvisual'] = interleave_isvisual
        extra['interleave_entity_mask'] = interleave_entity_mask
        extra['interleave_entity_indices'] = interleave_entity_indices
        extra['interleave_query_indices'] = interleave_query_indices

        entity_list = batched_inputs[0]['entities']['entities']
        entity_features = {"src_interleave_queries": [], "src_interleave_maskings": [], "src_interleave_indices": []}
        spatial_features = {"src_spatial_queries": [], "src_spatial_maskings": [], "src_spatial_indices": []}
        interleave_class_proposals = []
        interleave_pixel_proposals = []
        interleave_pixel_query = []
        interleave_class_query = []

        for idx, entity in enumerate(entity_list):
            if entity.type == 'visual':
                outputs = inference_visual(entity, extra, idx)
                entity_features['src_interleave_queries'].append(outputs['src_interleave_queries'])
                entity_features['src_interleave_maskings'].append(outputs['src_interleave_maskings'])
                src_interleave_indices = []
                for x in outputs['src_interleave_indices']:
                    x[x==0] = idx
                    src_interleave_indices.append(x)
                entity_features['src_interleave_indices'].append(src_interleave_indices)

                spatial_features['src_spatial_queries'].append(outputs['src_spatial_queries'])
                spatial_features['src_spatial_maskings'].append(outputs['src_spatial_maskings'])
                src_interleave_indices = []
                for x in outputs['src_spatial_indices']:
                    x[x==0] = idx
                    src_interleave_indices.append(x)
                spatial_features['src_spatial_indices'].append(src_interleave_indices)

                interleave_class_proposals += [outputs['pred_interleave_objects']]
                interleave_pixel_proposals += [outputs['pred_imaskembs']]
                interleave_pixel_query += [outputs['pred_entity_pixel']]
                interleave_class_query += [outputs['pred_entity_class']]
            elif entity.type == 'text':
                outputs = inference_text(entity, extra, idx)

                empty_token_queries = torch.zeros(1,1,512).to(self.device)
                empty_token_maskings = torch.ones(1,1).to(self.device).bool()
                empty_token_indices = torch.zeros(1,1,1).to(self.device)

                entity_features['src_interleave_queries'].append([empty_token_queries, empty_token_queries, empty_token_queries])
                entity_features['src_interleave_maskings'].append([empty_token_maskings, empty_token_maskings, empty_token_maskings])
                entity_features['src_interleave_indices'].append([empty_token_indices-8, empty_token_indices-8, empty_token_indices-8])
        
                spatial_features['src_spatial_queries'].append([empty_token_queries, empty_token_queries, empty_token_queries])
                spatial_features['src_spatial_maskings'].append([empty_token_maskings, empty_token_maskings, empty_token_maskings])
                spatial_features['src_spatial_indices'].append([empty_token_indices-9, empty_token_indices-9, empty_token_indices-9])

                interleave_class_proposals += [None]
                interleave_pixel_proposals += [None]
                interleave_pixel_query += [None]
                interleave_class_query += [outputs['pred_grounding_query']]

        src_interleave_queries = []
        src_interleave_maskings = []
        src_interleave_indices = []
        for i in range(3):
            src_interleave_queries.append(torch.cat([x[i] for x in entity_features['src_interleave_queries']], dim=0))
            src_interleave_maskings.append(torch.cat([x[i] for x in entity_features['src_interleave_maskings']], dim=1))
            src_interleave_indices.append(torch.cat([x[i] for x in entity_features['src_interleave_indices']], dim=1))

        src_spatial_queries = []
        src_spatial_maskings = []
        src_spatial_indices = []
        for i in range(3):
            src_spatial_queries.append(torch.cat([x[i] for x in spatial_features['src_spatial_queries']], dim=0))
            src_spatial_maskings.append(torch.cat([x[i] for x in spatial_features['src_spatial_maskings']], dim=1))
            src_spatial_indices.append(torch.cat([x[i] for x in spatial_features['src_spatial_indices']], dim=1))

        extra['refint_tokens'] = {'src_interleave_queries': src_interleave_queries, 'src_interleave_maskings': src_interleave_maskings, 'src_interleave_indices': src_interleave_indices}
        extra['refimg_tokens'] = {'src_spatial_queries': src_spatial_queries, 'src_spatial_maskings': src_spatial_maskings, 'src_spatial_indices': src_spatial_indices}

        features = self.backbone(torch.zeros(1,3,640,640).to(self.device))
        # features = self.backbone(entity_list[0].image.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='seg')
        outputs.update({"interleave_class_proposals": interleave_class_proposals, "interleave_pixel_proposals": interleave_pixel_proposals, "interleave_pixel_query": interleave_pixel_query, "interleave_class_query": interleave_class_query})
        return outputs, extra

    def demo_interleave_grounding(self, batched_inputs, extra):
        assert self.task_switch['interleave']
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        image_shape = images.tensor.shape[-2:]

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='seg')

        # mask embedding logits
        im_emb = outputs['pred_imaskembs']
        ic_emb = outputs['pred_interleave_objects']
        sm_emb = outputs['pred_entity_pixel']
        sc_emb = outputs['pred_entity_class']

        # pos mask
        pred_logits_im = im_emb @ sm_emb.transpose(1,2)
        pred_logits_ic = ic_emb @ sc_emb.transpose(1,2)
        is_visual = extra['interleave_isvisual'].transpose(0,1)
        pred_logits = pred_logits_im*is_visual[:,None] + pred_logits_ic*(1-is_visual[:,None])
        pred_idx = pred_logits[0].max(dim=0)[1]
        pred_masks = outputs['pred_imasks'][:,pred_idx]

        height, width = batched_inputs[0]['height'], batched_inputs[0]['width']
        pred_masks = F.interpolate(pred_masks, size=image_shape, mode='bilinear', align_corners=False)[0,:,:height,:width]
        pred_masks = (pred_masks.sigmoid() > 0.5).float()
        return {"pred_masks": pred_masks}

    def demo_retrieval(self, batched_inputs):
        assert len(batched_inputs) == 1
        lang_results_retrieval = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_inputs[-1]['tokens'], token=True)
        caption_num = len(batched_inputs[-1]['captions'])

        targets = []
        for idx in range(caption_num):
            target_dict = {}
            target_dict['retrieval_query_embs'] = lang_results_retrieval['token_emb'][idx:idx+1][lang_results_retrieval['tokens']['attention_mask'][idx:idx+1].bool()]
            targets += [target_dict]

        retrieval_tokens = [x['retrieval_query_embs'] for x in targets]
        retrieval_tokens = nn.utils.rnn.pad_sequence(retrieval_tokens, padding_value=-1)
        non_zero_query_mask = (retrieval_tokens.sum(dim=-1) == -retrieval_tokens.shape[-1])
        retrieval_tokens[non_zero_query_mask] = 0

        extra = {}
        extra['retrieval_tokens'] = retrieval_tokens
        extra['retrieval_nonzero_mask'] = non_zero_query_mask.t()

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        multi_scale_features = [m.repeat(caption_num,1,1,1) for m in multi_scale_features]
        mask_features = mask_features.repeat(caption_num,1,1,1)
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='seg')

        v_emb_it = outputs['pred_retrievals']
        t_emb_it = outputs['pred_retrievals_lang']
        return {'pred_retrievals_lang': t_emb_it}

    def evaluate_retrieval_interactive(self, batched_inputs):
        # interactive
        assert self.task_switch['spatial']
        assert 'spatial_query' in batched_inputs[0]
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        extra = {}

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
        nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
        multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
        mask_features = mask_features.repeat(nm,1,1,1)

        query_index = self.sem_seg_head.predictor.query_index
        assert self.interactive_mode == 'best'
        pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)

        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
        extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='spatial_retrieval')
        v_emb_it = outputs['pred_retrievals']

        v_emb = outputs['pred_smaskembs']
        pred_smasks = outputs['pred_smasks']

        s_emb = outputs['pred_pspatials']
        pred_logits = v_emb @ s_emb.transpose(1,2)
        logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        i_emb_it = outputs['pred_spatials'][logits_idx][None,]

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            caption_ids = []
            t_emb_its = []
            processed_results.append({})
            for caption in batch_data['captions']:
                lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(caption)
                t_emb_it = lang_results['class_emb']
                caption_ids.append(batch_data['image_id'])
                t_emb_its.append(t_emb_it)

            t_emb_it = torch.cat(t_emb_its, dim=0)
            image_embeds = [v_emb_it[idx], i_emb_it[idx]]
            caption_results = {
                    'image_embeds': image_embeds,
                    'text_embeds': t_emb_it,
                    'caption_ids': caption_ids,
                    'image_ids': batch_data['image_id'],
                }
            processed_results[-1]["caption"] = caption_results            
        return processed_results

    def evaluate_retrieval_patch(self, batched_inputs):
        # image
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)
        v_emb_it = outputs['pred_retrievals']

        # patch image
        images = [x["patch_image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)
        p_emb_it = outputs['pred_retrievals']

        # compute backbone score
        # if self.task_switch['retrieval'] and self.retrieval_emsemble:
        #     _v_emb_it = features['res5']
        #     bs,nc,_,_ = _v_emb_it.shape
        #     _v_emb_it = _v_emb_it.reshape(bs,nc,-1)
        #     _v_emb_it = F.adaptive_avg_pool1d(_v_emb_it, 1).reshape(bs,nc) @ self.backbone_proj

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            caption_ids = []
            t_emb_its = []
            processed_results.append({})
            for caption in batch_data['captions']:
                lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(caption)
                t_emb_it = lang_results['class_emb']
                caption_ids.append(batch_data['image_id'])
                t_emb_its.append(t_emb_it)

            t_emb_it = torch.cat(t_emb_its, dim=0)
            image_embeds = [v_emb_it[idx], p_emb_it[idx]]
            caption_results = {
                    'image_embeds': image_embeds,
                    'text_embeds': t_emb_it,
                    'caption_ids': caption_ids,
                    'image_ids': batch_data['image_id'],
                }
            processed_results[-1]["caption"] = caption_results            
        return processed_results

    def evaluate_retrieval(self, batched_inputs):
        assert len(batched_inputs) == 1
        lang_results_retrieval = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_inputs[-1]['tokens'], token=True)
        caption_num = len(batched_inputs[-1]['captions'])

        targets = []
        for idx in range(caption_num):
            target_dict = {}
            target_dict['retrieval_query_embs'] = lang_results_retrieval['token_emb'][idx:idx+1][lang_results_retrieval['tokens']['attention_mask'][idx:idx+1].bool()]
            targets += [target_dict]

        retrieval_tokens = [x['retrieval_query_embs'] for x in targets]
        retrieval_tokens = nn.utils.rnn.pad_sequence(retrieval_tokens, padding_value=-1)
        non_zero_query_mask = (retrieval_tokens.sum(dim=-1) == -retrieval_tokens.shape[-1])
        retrieval_tokens[non_zero_query_mask] = 0

        extra = {}
        extra['retrieval_tokens'] = retrieval_tokens
        extra['retrieval_nonzero_mask'] = non_zero_query_mask.t()

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        multi_scale_features = [m.repeat(caption_num,1,1,1) for m in multi_scale_features]
        mask_features = mask_features.repeat(caption_num,1,1,1)
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='seg')

        v_emb_it = outputs['pred_retrievals']
        t_emb_it = outputs['pred_retrievals_lang']

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            caption_ids = []
            processed_results.append({})
            for caption in batch_data['captions']:
                caption_ids.append(batch_data['image_id'])

            image_embeds = [v_emb_it[idx]]
            caption_results = {
                    'image_embeds': image_embeds,
                    'text_embeds': t_emb_it[:,idx],
                    'caption_ids': caption_ids,
                    'image_ids': batch_data['image_id'],
                }
            processed_results[-1]["caption"] = caption_results            
        return processed_results

    def evaluate_retrieval_interleave_text(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            input_ids += x['tokens']['input_ids']
            attention_mask += x['tokens']['attention_mask']
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)

        targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            target_dict = {}
            target_dict['interleave_query_embs'] = lang_results_interleave['token_emb'][idx:idx+1][lang_results_interleave['tokens']['attention_mask'][idx:idx+1].bool()]
            # target_dict['interleave_class_embs'] = lang_results_interleave['class_emb'][idx:idx+1]
            target_dict['interleave_entity_mask'] = torch.ones((1, len(target_dict['interleave_query_embs'])), dtype=torch.bool, device=target_dict['interleave_query_embs'].device)
            target_dict['interleave_entity_indices'] = decode_entity_mask_to_indices(target_dict['interleave_entity_mask'])
            target_dict['interleave_query_indices'] = torch.arange(len(target_dict['interleave_entity_mask']), device=target_dict['interleave_query_embs'].device)
            target_dict['interleave_masks'] =  torch.zeros((1, images.tensor.shape[-2], images.tensor.shape[-1]), dtype=torch.bool, device=target_dict['interleave_query_embs'].device)
            target_dict['interleave_spatial'] = torch.zeros((1, images.tensor.shape[-2], images.tensor.shape[-1]), dtype=torch.bool, device=target_dict['interleave_query_embs'].device)
            target_dict['interleave_isvisual'] = torch.tensor([False])
            targets += [target_dict]

        interleave_tokens = [x['interleave_query_embs'].to(self.device) for x in targets] # need to pad for more than one grounding token
        interleave_tokens = nn.utils.rnn.pad_sequence(interleave_tokens, padding_value=-1)
        non_zero_query_mask = (interleave_tokens.sum(dim=-1) == -interleave_tokens.shape[-1]).to(self.device)
        interleave_tokens[non_zero_query_mask] = 0
        pos_masks = [x['interleave_spatial'].to(self.device) for x in targets]
        interleave_isvisual = nn.utils.rnn.pad_sequence([x['interleave_isvisual'].float().to(self.device) for x in targets], padding_value=0)
        interleave_entity_mask = pad_arbitrary_tensors([x['interleave_entity_mask'].to(self.device) for x in targets], padding_value=False).transpose(0,1)
        interleave_entity_indices = nn.utils.rnn.pad_sequence([x['interleave_entity_indices'].to(self.device) for x in targets], padding_value=-5)
        interleave_query_indices = nn.utils.rnn.pad_sequence([x['interleave_query_indices'].to(self.device) for x in targets], padding_value=-6)

        extra = {}
        extra['interleave_tokens'] = interleave_tokens
        extra['interleave_nonzero_mask'] = non_zero_query_mask.t()
        extra['interleave_query_pos_mask'] = pos_masks
        extra['interleave_query_neg_mask'] = [x & False for x in pos_masks]
        extra['interleave_isvisual'] = interleave_isvisual
        extra['interleave_entity_mask'] = interleave_entity_mask
        extra['interleave_entity_indices'] = interleave_entity_indices
        extra['interleave_query_indices'] = interleave_query_indices

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding, extra=extra)
        v_emb_it = outputs['pred_retrievals']
        _t_emb_it = outputs['pred_interleave_image']

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            caption_ids = []
            t_emb_its = []
            processed_results.append({})
            for caption in batch_data['captions']:
                t_emb_it = _t_emb_it[idx]
                caption_ids.append(batch_data['image_id'])
                t_emb_its.append(t_emb_it)

            t_emb_it = torch.cat(t_emb_its, dim=0)
            image_embeds = [v_emb_it[idx]]
            caption_results = {
                    'image_embeds': image_embeds,
                    'text_embeds': t_emb_it,
                    'caption_ids': caption_ids,
                    'image_ids': batch_data['image_id'],
                }
            processed_results[-1]["caption"] = caption_results            
        return processed_results

    def evaluate_interactive(self, batched_inputs):
        assert self.task_switch['spatial']
        assert 'spatial_query' in batched_inputs[0]
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        extra = {}

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
        nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
        # multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
        # mask_features = mask_features.repeat(nm,1,1,1)

        all_batch_shape_iou = []
        pred_smask_pointer = None
        prev_smask_pointer = None
        pred_smask_all = None

        # visualization code
        # v_pred_mask = []
        # v_pos_mask = []
        # v_neg_mask = []
        # v_pred_pos_mask = []
        # v_pred_neg_mask = []
        # v_gt_mask = batched_inputs[0]['spatial_query']['gt_masks'][0]
        query_index = self.sem_seg_head.predictor.query_index
        assert self.interactive_mode == 'best'
        pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.transpose(0,1).unbind(0)

        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.transpose(0,1).unbind(0)

        extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks, 'spatial_query_indices': torch.arange(nm)[:,None].to(self.device)})
        for i in range(self.interactive_iter):
            # v_pos_mask += [extra['spatial_query_pos_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            # v_neg_mask += [extra['spatial_query_neg_mask'][0][0][:image_sizes[0][0],:image_sizes[0][1]].float().cpu().numpy()]
            outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='spatial')
            extra.update(outputs)
            pred_smask = F.interpolate(outputs['prev_mask'], images.tensor.shape[-2:], mode='bicubic')
            # v_pred_mask += [(pred_smask[0,0][:image_sizes[0][0],:image_sizes[0][1]].sigmoid() > 0.5).float().cpu().numpy()]
            # v_pred_pos_mask += [(F.interpolate(outputs['pos_mask'], images.tensor.shape[-2:], mode='bicubic')[0,0][:image_sizes[0][0],:image_sizes[0][1]].sigmoid() > 0.5).float().cpu().numpy()]
            # v_pred_neg_mask += [(F.interpolate(outputs['neg_mask'], images.tensor.shape[-2:], mode='bicubic')[0,0][:image_sizes[0][0],:image_sizes[0][1]].sigmoid() > 0.5).float().cpu().numpy()]

            s = image_sizes[0]
            b = batched_inputs[0]
            pred_smask_all = F.interpolate(pred_smask[:,:,:s[0],:s[1]], (b['height'], b['width']), mode='bicubic')[0].sigmoid() > 0.5
            gt_smask = b['gt_masks_orisize']
            ious = get_iou(gt_smask, pred_smask_all)
            all_batch_shape_iou += [ious]
            if (ious > 0.9).sum() == len(ious):
                all_batch_shape_iou += [ious for j in range(self.interactive_iter-i-1)]
                break
            extra.update(self.prepare_next_spaital_mask(extra, batched_inputs))

        all_batch_shape_iou = torch.stack(all_batch_shape_iou)
        processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]

        # visualization
        # VL.step()
        # import cv2
        # v_mask = []
        # txt = []
        # img = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # mask_img = VL.overlay_single_mask_to_image(img[:,:,::-1], v_gt_mask.cpu().float().numpy())
        # for x,y,z,iou in zip(v_pos_mask, v_neg_mask, v_pred_mask, all_batch_shape_iou):
        #     # dilate x,y
        #     x = cv2.dilate(x, np.ones((3,3), np.uint8), iterations=3)
        #     y = cv2.dilate(y, np.ones((3,3), np.uint8), iterations=3)
        #     v_mask += [x,y,z]
        #     txt += ["pos_point", "neg_point", "pred_{}".format(str(iou[0].item())[0:5])]
        # VL.add_image(img[:,:,::-1])
        # VL.insert(mask_img, "gt_mask")
        # VL.overlay_obj_mask_to_image(img[:,:,::-1], v_mask, txt, max_len=20)
        return processed_results

    def evaluate_referring_image(self, batched_inputs, extra={}):
        assert self.task_switch['spatial']
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        assert self.interactive_mode == 'best'

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        if 'spatial_query' in batched_inputs[0]:
            image_sizes = [x["image"].shape[-2:] for x in batched_inputs]
            nm = len(batched_inputs[0]['spatial_query']['rand_shape'])
            multi_scale_features = [m.repeat(nm,1,1,1) for m in multi_scale_features]
            mask_features = mask_features.repeat(nm,1,1,1)

            query_index = self.sem_seg_head.predictor.query_index
            pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)

            neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='refimg')
        return outputs, images.tensor.shape

    def evaluate_grounding(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        # mask_pred_results = []
        # for idx, batch_per_image in enumerate(batched_inputs):
        #     grd_texts = batch_per_image['groundings']['texts']
        #     grd_masks = []
        #     for anno_text in grd_texts:
        #         gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
        #         token_emb = gtext['token_emb']
        #         tokens = gtext['tokens']
            
        #         grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
        #         extra['grounding_tokens'] = grd_emb[:,None]

        #         assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"
        #         features = self.backbone(images.tensor)
        #         outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')
                
        #         pred_gmasks = outputs['pred_masks'][idx,self.num_queries:2*self.num_queries-1]
        #         v_emb = outputs['pred_captions'][idx,self.num_queries:2*self.num_queries-1]
        #         t_emb = grd_emb[-1:]

        #         t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        #         v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        #         temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        #         out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
                
        #         matched_id = out_prob.max(0)[1]
        #         grd_masks += [pred_gmasks[matched_id,:,:]]
        #     mask_pred_results += [torch.cat(grd_masks)]

        # comment for multi object inference.
        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_texts = [x[0] for x in grd_texts]

            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

            indices = torch.arange(len(tokens['attention_mask']), device=tokens['attention_mask'].device)[:,None]
            token_indices = (tokens['attention_mask'] * indices)[tokens['attention_mask'].bool()]
            class_indices = torch.arange(len(tokens['attention_mask']), device=tokens['attention_mask'].device)

            extra['grounding_tokens'] = query_emb[:,None]
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
            extra['grounding_token_indices'] = token_indices[:,None]
            extra['grounding_query_indices'] = class_indices[:,None]

            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

            pred_gmasks = outputs['pred_gmasks'][idx]
            v_emb = outputs['pred_gtexts'][idx]
            t_emb = outputs['pred_grounding_query'][idx]

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
            matched_id = out_prob.max(0)[1]
            mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

            # compute bbox
            # bbox = BitMasks(mask_pred_result > 0).get_bounding_boxes()
            # bbox = BoxMode.convert(bbox.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            # processed_results[-1]['grounding_box'] = bbox

        return processed_results

    def evaluate_interleave_grounding(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        image_shape = images.tensor.shape[-2:]
        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        # prepare language embeddings
        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            input_ids += x['entities']['tokens']['input_ids']
            attention_mask += x['entities']['tokens']['attention_mask']
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)

        # prepare targets information
        targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            target_dict = {}
            target_dict['interleave_query_embs'] = lang_results_interleave['token_emb'][idx:idx+1][lang_results_interleave['tokens']['attention_mask'][idx:idx+1].bool()]
            # target_dict['interleave_class_embs'] = lang_results_interleave['class_emb'][idx:idx+1]
            target_dict['interleave_entity_mask'] = batch_per_image['entities']['entity_to_tokens'][:,lang_results_interleave['tokens']['attention_mask'][idx].bool()]
            target_dict['interleave_entity_indices'] = decode_entity_mask_to_indices(target_dict['interleave_entity_mask'])
            target_dict['interleave_query_indices'] = torch.arange(len(target_dict['interleave_entity_mask']), device=target_dict['interleave_query_embs'].device)
            target_dict['interleave_masks'] =  torch.cat([x.mask for x in batch_per_image['entities']['entities']])
            target_dict['interleave_spatial'] = torch.cat([x.interactive for x in batch_per_image['entities']['entities']])
            target_dict['interleave_isvisual'] = torch.tensor([x.type == 'visual' for x in batch_per_image['entities']['entities']], device=target_dict['interleave_query_embs'].device)
            targets += [target_dict]
        
        # prepare forward interleave extra
        interleave_tokens = [x['interleave_query_embs'].to(self.device) for x in targets] # need to pad for more than one grounding token
        interleave_tokens = nn.utils.rnn.pad_sequence(interleave_tokens, padding_value=-1)
        non_zero_query_mask = (interleave_tokens.sum(dim=-1) == -interleave_tokens.shape[-1]).to(self.device)
        interleave_tokens[non_zero_query_mask] = 0
        pos_masks = [x['interleave_spatial'].to(self.device) for x in targets]
        interleave_isvisual = nn.utils.rnn.pad_sequence([x['interleave_isvisual'].float().to(self.device) for x in targets], padding_value=0)
        interleave_entity_mask = pad_arbitrary_tensors([x['interleave_entity_mask'].to(self.device) for x in targets], padding_value=False).transpose(0,1)
        interleave_entity_indices = nn.utils.rnn.pad_sequence([x['interleave_entity_indices'].to(self.device) for x in targets], padding_value=-5)
        interleave_query_indices = nn.utils.rnn.pad_sequence([x['interleave_query_indices'].to(self.device) for x in targets], padding_value=-6)

        extra = {}
        extra['interleave_tokens'] = interleave_tokens
        extra['interleave_nonzero_mask'] = non_zero_query_mask.t()
        extra['interleave_query_pos_mask'] = pos_masks
        extra['interleave_query_neg_mask'] = [x & False for x in pos_masks]
        extra['interleave_isvisual'] = interleave_isvisual
        extra['interleave_entity_mask'] = interleave_entity_mask
        extra['interleave_entity_indices'] = interleave_entity_indices
        extra['interleave_query_indices'] = interleave_query_indices

        queries_grounding = None
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='seg')

        # mask embedding logits
        im_emb = outputs['pred_imaskembs']
        ic_emb = outputs['pred_interleave_objects']
        sm_emb = outputs['pred_entity_pixel']
        sc_emb = outputs['pred_entity_class']

        # pos mask
        pred_logits_im = im_emb @ sm_emb.transpose(1,2)
        pred_logits_ic = ic_emb @ sc_emb.transpose(1,2)
        is_visual = extra['interleave_isvisual'].transpose(0,1)
        pred_logits = pred_logits_im*is_visual[:,None] + pred_logits_ic*(1-is_visual[:,None])
        pred_idx = pred_logits[0].max(dim=0)[1]
        pred_masks = outputs['pred_imasks'][:,pred_idx]

        # height, width = images.image_sizes[0]
        # visual_mask = F.interpolate(pred_masks, size=image_shape, mode='bilinear', align_corners=False)[0,:,:height,:width]
        # visual_mask = (visual_mask.sigmoid() > 0.5).float().cpu().numpy()
        # visual_image = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()

        processed_results = []
        for pred_mask, input_per_image, image_size in zip(
            pred_masks, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            pred_mask = retry_if_cuda_oom(sem_seg_postprocess)(
                pred_mask, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = pred_mask
        return processed_results

    def evaluate_grounding_sptial(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        dilation = 3
        pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor
        pos_masks = (F.conv2d(pos_masks.float(), self.dilation_kernel, padding=dilation//2) > 0).unbind(0)

        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)

        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_masks = []
            for idx2, anno_text in enumerate(grd_texts):
                extra.update({'spatial_query_pos_mask': [pos_masks[idx2]], 'spatial_query_neg_mask': [neg_masks[idx2]]})

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
            
                grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
                non_zero_query_mask = torch.zeros(grd_emb[:,None].shape[:-1], dtype=torch.bool, device=grd_emb.device)
                extra['grounding_tokens'] = grd_emb[:,None]
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

                assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"
                features = self.backbone(images.tensor)
                outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')
                
                pred_gmasks = outputs['pred_gmasks'][idx]
                v_emb = outputs['pred_gtexts'][idx]
                t_emb = gtext['class_emb']

                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

                temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
                out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
                
                matched_id = out_prob.max(0)[1]
                grd_masks += [pred_gmasks[matched_id,:,:]]
            mask_pred_results += [torch.cat(grd_masks)]

        # comment for multi object inference.
        # mask_pred_results = []
        # for idx, batch_per_image in enumerate(batched_inputs):
        #     grd_texts = batch_per_image['groundings']['texts']
        #     grd_texts = [x[0] for x in grd_texts]

        #     gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
        #     token_emb = gtext['token_emb']
        #     tokens = gtext['tokens']
        #     query_emb = token_emb[tokens['attention_mask'].bool()]
        #     non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

        #     extra['grounding_tokens'] = query_emb[:,None]
        #     extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

        #     features = self.backbone(images.tensor)
        #     outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

        #     pred_gmasks = outputs['pred_gmasks'][idx]
        #     v_emb = outputs['pred_gtexts'][idx]
        #     t_emb = gtext['class_emb']

        #     t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        #     v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        #     temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        #     out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
        #     matched_id = out_prob.max(0)[1]
        #     mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

        return processed_results

    def evaluate_classification(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            processed_results.append({})
            processed_results[-1]["pred_class"] = outputs['pred_logits'][idx,-1]
        return processed_results

    def evaluate_demo(self, batched_inputs):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
        image_sizes = [x["image"].shape[-2:] for x in batched_inputs]

        extra = {}
        if 'stroke' in batched_inputs[0]:
            pos_masks = (batched_inputs[0]['stroke'].to(self.device)).unbind(0)
            pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor.unbind(0)
            neg_masks = (batched_inputs[0]['stroke'].to(self.device) & False).unbind(0)
            neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})

        if 'visual' in batched_inputs[0]:
            extra.update(batched_inputs[0]['visual'])
        
        if 'text' in batched_inputs[0]:
            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_inputs[0]['text'], name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)
            extra['grounding_tokens'] = query_emb[:,None]
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
            extra['grounding_class'] = gtext['class_emb']

        if 'audio' in batched_inputs[0]:
            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_inputs[0]['audio'], name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)
            extra['audio_tokens'] = query_emb[:,None]
            extra['audio_nonzero_mask'] = non_zero_query_mask.t()
            extra['audio_class'] = gtext['class_emb']
        
        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='demo')
        return outputs, images.tensor.shape, extra

    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]

        # prepare language embeddings for retrieval
        if self.task_switch['retrieval']:
            if not self.sem_seg_head.predictor.lang_encoder.load_tensor:
                input_ids = []
                attention_mask = []
                for cnt, x in enumerate(batched_inputs):
                    input_ids += x['tokens']['input_ids']
                    attention_mask += x['tokens']['attention_mask']

                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                lang_results_retrieval = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)
            else:
                all_token_emb = []
                input_ids = []
                attention_mask = []
                for cnt, x in enumerate(batched_inputs):
                    all_token_emb += [x['tokens']['language_embed']['token_emb']]
                    input_ids += x['tokens']['input_ids']
                    attention_mask += x['tokens']['attention_mask']
                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                all_token_emb = torch.cat(all_token_emb)
                lang_results_retrieval = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({'token_emb': all_token_emb, 'class_emb': None, 'tokens': tokens}, embed=True)

        # prepare language embeddings for retrieval
        if self.task_switch['interleave']:
            if not self.sem_seg_head.predictor.lang_encoder.load_tensor:
                input_ids = []
                attention_mask = []
                for cnt, x in enumerate(batched_inputs):
                    input_ids += x['entities']['tokens']['input_ids'].to(self.device)
                    attention_mask += x['entities']['tokens']['attention_mask'].to(self.device)
                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)
            else:
                all_token_emb = []
                input_ids = []
                attention_mask = []
                for cnt, x in enumerate(batched_inputs):
                    all_token_emb += [x['entities']['language_embed']['token_emb']]
                    input_ids += x['entities']['tokens']['input_ids'].to(self.device)
                    attention_mask += x['entities']['tokens']['attention_mask'].to(self.device)
                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                all_token_emb = torch.cat(all_token_emb)
                lang_results_interleave = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({'token_emb': all_token_emb, 'class_emb': None, 'tokens': tokens}, embed=True)

        # prepare language embeddings for segmentation
        if self.task_switch['mask']['ENABLED']:
            if not self.sem_seg_head.predictor.lang_encoder.load_tensor:
                attention_mask = []
                class_ids = []
                input_ids = []
                id_slice_to_batch = []
                start_idx = 0
                for cnt, x in enumerate(batched_inputs):
                    input_ids += x['class']['tokens']['input_ids'].to(self.device)
                    attention_mask += x['class']['tokens']['attention_mask'].to(self.device)

                    class_ids += x['class']['ids']
                    id_slice_to_batch.append([start_idx, start_idx + len(x['class']['ids'])])
                    start_idx += len(x['class']['ids'])
                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                lang_results_class = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)
            else:
                all_token_emb = []
                input_ids = []
                attention_mask = []
                class_ids = []
                id_slice_to_batch = []
                start_idx = 0
                for cnt, x in enumerate(batched_inputs):
                    all_token_emb += [x['class']['language_embed']['token_emb']]
                    input_ids += x['class']['tokens']['input_ids']
                    attention_mask += x['class']['tokens']['attention_mask']

                    class_ids += x['class']['ids']
                    id_slice_to_batch.append([start_idx, start_idx + len(x['class']['ids'])])
                    start_idx += len(x['class']['ids'])
                input_ids = torch.stack(input_ids)
                attention_mask = torch.stack(attention_mask)
                tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
                all_token_emb = torch.cat(all_token_emb)
                lang_results_class = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({'token_emb': all_token_emb, 'class_emb': None, 'tokens': tokens}, embed=True)

        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets_per_image = batch_per_image['instances'].to(self.device)
            # pad gt
            gt_masks = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_boxes = targets_per_image.gt_boxes.tensor
            ratio = torch.tensor([w_pad,h_pad,w_pad,h_pad]).to(gt_boxes.device)[None,:]
            gt_boxes = gt_boxes / ratio
            xc,yc,w,h = (gt_boxes[:,0] + gt_boxes[:,2])/2, (gt_boxes[:,1] + gt_boxes[:,3])/2, gt_boxes[:,2] - gt_boxes[:,0], gt_boxes[:,3] - gt_boxes[:,1]
            gt_boxes = torch.stack([xc,yc,w,h]).permute(1,0)

            target_dict = {
                    "labels": targets_per_image.gt_classes,
                    "is_things": targets_per_image.is_things,
                    "masks": padded_masks,
                    "boxes": gt_boxes,
                    }

            if self.task_switch['mask']['ENABLED']:
                class_tokens = lang_results_class['token_emb'][id_slice_to_batch[idx][0]:id_slice_to_batch[idx][1]]
                class_tokens_mask = lang_results_class['tokens']['attention_mask'][id_slice_to_batch[idx][0]:id_slice_to_batch[idx][1]]
                assert len(class_tokens) == 1, "We only support one class at a time now."
                class_tokens_embs = class_tokens[class_tokens_mask.bool()]
                target_dict['class_tokens_embs'] = class_tokens_embs
                target_dict['class_ids'] = class_ids[id_slice_to_batch[idx][0]:id_slice_to_batch[idx][1]]

            if self.task_switch['retrieval']:
                target_dict["caption_tokens"] = lang_results_retrieval['token_emb'][idx:idx+1]
                target_dict["caption_tokenids"] = lang_results_retrieval['tokens']['input_ids'][idx:idx+1]
                target_dict["caption_mask"] = lang_results_retrieval['tokens']['attention_mask'][idx:idx+1]
                target_dict['retrieval_query_embs'] = lang_results_retrieval['token_emb'][idx:idx+1][lang_results_retrieval['tokens']['attention_mask'][idx:idx+1].bool()]

            if self.task_switch['spatial']:
                # prepare targets for spatial query
                target_dict['gt_spatial_masks'] = batch_per_image['spatial_query']['gt_masks']
                spatial_indices = torch.arange(len(target_dict['gt_spatial_masks']), device=target_dict['gt_spatial_masks'].device)
                target_dict['spatial_query_indices'] = spatial_indices # query indices

            if self.task_switch['grounding']:
                grd_masks = batch_per_image['groundings']['masks']
                grd_texts = batch_per_image['groundings']['tokens']
                grd_hash = batch_per_image['groundings']['hash']
                grd_task = batch_per_image['groundings']['mode']
                
                # padding grounding mask to image tensor shape
                if len(grd_masks) == 0:
                    padded_masks = None
                else:
                    padded_masks = torch.zeros((grd_masks.shape[0], h_pad, w_pad), dtype=grd_masks.dtype, device=grd_masks.device)
                    padded_masks[:, : grd_masks.shape[1], : grd_masks.shape[2]] = grd_masks

                grd_texts['input_ids'] = grd_texts['input_ids'].to(self.device)                
                grd_texts['attention_mask'] = grd_texts['attention_mask'].to(self.device)
                # prepare language embeddings for grounding
                if not self.sem_seg_head.predictor.lang_encoder.load_tensor:
                    gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=True, norm=False)
                else:
                    all_token_emb = batch_per_image['groundings']['language_embed']['token_emb']
                    gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({'token_emb': all_token_emb, 'class_emb': None, 'tokens': grd_texts}, embed=True)

                token_emb = gtext['token_emb']
                tokens = gtext['tokens']

                # for each category we actually only add one name to tokens
                unique_hash_id = np.unique(grd_hash, return_index=True)[1]
                selected_mask = np.zeros(len(grd_hash)).astype(np.bool)
                selected_mask[unique_hash_id] = True
                selected_hash = grd_hash[selected_mask]

                selected_token_emb = token_emb[selected_mask]
                selected_attn_mask = tokens['attention_mask'][selected_mask]
                query_emb = selected_token_emb[selected_attn_mask.bool()]

                indices = torch.arange(len(selected_attn_mask), device=selected_attn_mask.device)[:,None]
                token_indices = (selected_attn_mask * indices)[selected_attn_mask.bool()]
                class_indices = torch.tensor([selected_hash.tolist().index(hash_id) for hash_id in grd_hash], device=selected_token_emb.device)

                target_dict['grounding_masks'] = padded_masks # masks
                target_dict['grounding_token_embs'] = query_emb # tokens
                target_dict['grounding_query_indices'] = class_indices # query indices
                target_dict['grounding_token_indices'] = token_indices # token indices
                target_dict['grounding_hash'] = grd_hash # hash
                target_dict['grounding_task'] = grd_task # text/class

            if self.task_switch['interleave']:
                target_dict['interleave_query_embs'] = lang_results_interleave['token_emb'][idx:idx+1][lang_results_interleave['tokens']['attention_mask'][idx:idx+1].bool()]
                # target_dict['interleave_class_embs'] = lang_results_interleave['class_emb'][idx:idx+1]
                # find active index for each entity
                target_dict['interleave_entity_mask'] = batch_per_image['entities']['entity_to_tokens'][:,lang_results_interleave['tokens']['attention_mask'][idx].bool()]
                target_dict['interleave_isvisual'] = torch.tensor([x.type == 'visual' for x in batch_per_image['entities']['entities']], device=target_dict['interleave_query_embs'].device)
                target_dict['interleave_entity_indices'] = decode_entity_mask_to_indices(target_dict['interleave_entity_mask'])
                target_dict['interleave_query_indices'] = torch.arange(len(target_dict['interleave_entity_mask']), device=target_dict['interleave_query_embs'].device)
                target_dict['interleave_masks'] =  torch.cat([x.mask for x in batch_per_image['entities']['entities']])
                target_dict['interleave_spatial'] = torch.cat([x.interactive for x in batch_per_image['entities']['entities']])

            new_targets.append(target_dict)
        return new_targets

    @torch.no_grad()
    def prepare_language_targets(self, batched_inputs):
        all_texts = []
        all_input_ids = []
        all_attention_mask = []

        if self.task_switch['retrieval']:
            for cnt, x in enumerate(batched_inputs):
                all_texts += x['captions']
                all_input_ids += [x['tokens']['input_ids'].to(self.device)]
                all_attention_mask += [x['tokens']['attention_mask'].to(self.device)]

        if self.task_switch['interleave']:
            for cnt, x in enumerate(batched_inputs):
                all_texts += x['entities']['sentence']
                all_input_ids += [x['entities']['tokens']['input_ids'].to(self.device)]
                all_attention_mask += [x['entities']['tokens']['attention_mask'].to(self.device)]

        if self.task_switch['mask']['ENABLED']:
            for cnt, x in enumerate(batched_inputs):
                all_texts += x['class']['sentences']
                all_input_ids += [x['class']['tokens']['input_ids'].to(self.device)]
                all_attention_mask += [x['class']['tokens']['attention_mask'].to(self.device)]

        if self.task_switch['grounding']:
            for cnt, x in enumerate(batched_inputs):
                if len(x['groundings']['texts']) > 0:
                    all_texts += x['groundings']['texts']
                    tokens = self.sem_seg_head.predictor.lang_encoder.tokenizer(
                        x['groundings']['texts'], padding='max_length', truncation=True, max_length=self.sem_seg_head.predictor.lang_encoder.max_token_num, return_tensors='pt'
                    )
                    all_input_ids += [tokens['input_ids'].to(self.device)]
                    all_attention_mask += [tokens['attention_mask'].to(self.device)]

        all_input_ids = torch.cat(all_input_ids)
        all_attention_mask = torch.cat(all_attention_mask)
        all_hash = [strict_hash(x) for x in all_texts]
        assert len(all_hash) == len(all_input_ids) == len(all_attention_mask)
        return {"texts": all_texts, "hash_values": all_hash, "input_ids": all_input_ids, "attention_mask": all_attention_mask}

    def prepare_next_spaital_mask(self, outputs, batched_inputs, mode='best'):
        gt_masks = [batched_inputs[i]['spatial_query']['gt_masks'] for i in range(len(batched_inputs))]
        gt_masks = Spatial_ImageList.from_tensors(gt_masks, self.size_divisibility).tensor

        pred_masks = (F.interpolate(outputs['prev_mask'], size=gt_masks.shape[-2:], mode='bilinear', align_corners=False).sigmoid() > 0.5)
        prev_masks = nn.utils.rnn.pad_sequence(outputs['spatial_query_pos_mask'], padding_value=False, batch_first=True) | \
                        nn.utils.rnn.pad_sequence(outputs['spatial_query_neg_mask'], padding_value=False, batch_first=True)

        fn = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks) # fn: False Negative, gt:1, pred:0, prev:0
        fp = (~gt_masks & pred_masks) & (~prev_masks) # fp: False Positive, gt:0, pred:1, prev:0

        # compute iou between gt and pred
        iou = (gt_masks & pred_masks).sum(list(range(2,len(fn.shape)))) / ((gt_masks | pred_masks).sum(dim=list(range(2,len(fn.shape)))) + 1e-8)
        fn_sum = fn.sum(dim=list(range(2,len(fn.shape))))
        fp_sum = fp.sum(dim=list(range(2,len(fp.shape))))

        is_postive = fn_sum > fp_sum
        select_mask = torch.zeros_like(fn)
        select_mask[is_postive] = fn[is_postive]
        select_mask[~is_postive] = fp[~is_postive]
        # is_postive = torch.ones(len(fn_sum), device=torch.cuda.current_device()).bool()

        # conv implementation
        bs,ns,h,w = select_mask.shape
        mask_dt = (distance_transform((~F.pad(select_mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:,:,1:-1,1:-1]).reshape(bs*ns,-1)
        if mode == 'best':
            max_xy_idx = torch.stack([torch.arange(bs*ns), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        elif mode == 'best_random':
            max_xy_idx = torch.stack([torch.arange(bs*ns), torch.cat([(mask_dt[i] > 0).nonzero()[torch.randint(0, len((mask_dt[i] > 0).nonzero()), (1,))][0] for i in range(len(mask_dt))]).cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(bs*ns,-1)
        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((bs*ns,1,h,w)).float()
        dilation = 3
        next_mask = F.conv2d(next_mask, self.dilation_kernel, padding=dilation//2).reshape(bs,ns,h,w) > 0

        # determine whether next mask is zero
        keep = (iou < 0.925)
        next_mask = next_mask & keep.view(bs,ns,1,1)

        pos_mask = []
        neg_mask = []
        for idx, ip in enumerate(is_postive):
            mask_len = len(outputs['spatial_query_pos_mask'][idx])
            pos_mask += [outputs['spatial_query_pos_mask'][idx] | (next_mask[idx][:mask_len] & ip[:mask_len,None,None])]
            neg_mask += [outputs['spatial_query_neg_mask'][idx] | (next_mask[idx][:mask_len] & (~ip[:mask_len,None,None]))]

        if 'false_positive_mask' in outputs:
            fp = outputs['false_positive_mask'] | fp
        return {'spatial_query_pos_mask': pos_mask, 'spatial_query_neg_mask': neg_mask, 'false_positive_mask': fp}

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result

    @torch.no_grad()
    def get_class_embeddings(self, class_names, is_eval):
        '''
        This function is used to extract class embeddings in query basis.
        '''
        # if the model does not have class embeddings, we need to extract them
        # This lies in the case where the model is doing inference, or first time training
        if not hasattr(self.sem_seg_head.predictor.lang_encoder, 'default_text_embeddings') or is_eval == True:

            class_embedding_list = []
            # class_names = ['background']
            for idx, _class in enumerate(class_names):
                _class = ["Describe the concept of {}.".format(_class.replace('-other','').replace('-merged','').replace('-stuff',''))]
                class_token_length = self.class_token_length if (not self.sem_seg_head.predictor.lang_encoder.load_tensor or is_eval == True) else self.sem_seg_head.predictor.lang_encoder.max_token_num

                tokens = self.sem_seg_head.predictor.lang_encoder.tokenizer(
                    _class, padding='max_length', truncation=True, max_length=class_token_length, return_tensors='pt'
                )
                tokens = {"input_ids": tokens['input_ids'].to(self.device), "attention_mask": tokens['attention_mask'].to(self.device)}

                if not self.sem_seg_head.predictor.lang_encoder.load_tensor or is_eval == True:
                    token_emb = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, name='class', token=True, norm=False)['token_emb']
                else:
                    _hash = strict_hash(_class[0])
                    in_path = os.path.join(self.language_embed_root, '{}.da'.format(_hash))
                    token_emb = torch.load(in_path)['token_emb'].to(self.device)
                    token_emb = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings({'token_emb': token_emb, 'class_emb': None, 'tokens': tokens}, embed=True)['token_emb']

                class_tokens_emb = nn.utils.rnn.pad_sequence([e[t] for e, t in zip(token_emb, tokens['attention_mask'].bool())], padding_value=-1, batch_first=True)
                class_tokens_mask = (class_tokens_emb.sum(dim=-1) == -token_emb.shape[-1])
                extra = {"class_tokens_emb": class_tokens_emb.transpose(0,1), "class_tokens_mask": class_tokens_mask}

                bs,c = class_tokens_mask.shape[0], class_tokens_emb.shape[-1]
                size_list = [20, 40, 80, 160]
                fake_multi_scale_features = [torch.zeros(bs,c,s,s, dtype=class_tokens_emb.dtype, device=class_tokens_emb.device) for s in size_list]
                outputs = self.sem_seg_head.predictor(fake_multi_scale_features[:-1], fake_multi_scale_features[-1], extra=extra, task='class')
                class_embedding_list += [outputs['pred_retrievals_class'][0]]

            class_embedding = torch.cat(class_embedding_list)
            # class_embedding = torch.load("class_embeddings_davitd5.da").to(self.device)
            # class_embedding = torch.cat([class_embedding, class_embedding_list[-1]])
            setattr(self.sem_seg_head.predictor.lang_encoder, 'default_text_embeddings', class_embedding)



@register_model
def get_find_model(cfg, **kwargs):
    return GeneralizedFIND(cfg)