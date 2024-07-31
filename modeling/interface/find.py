# --------------------------------------------------------
# FIND -- Interfacing Foundation Models' Embeddings
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d

from .operator import MLP, ContentAttention, ConditionAttention, FFNLayer
from .build import register_decoder
from .prototype.attention_data_struct_ging import AttentionDataStruct

from ..utils import prepare_vision_features, configurable
from ..utils import rand_sample_plain as rand_sample
from ..utils import rand_sample as rand_sample_multi
from ..modules import PositionEmbeddingSine
from ..modules.point_features import point_sample


class FINDDecoder(nn.Module):

    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        task_switch: dict,
        enforce_input_project: bool,
        max_spatial_len: int,
        attn_arch: dict,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.content_attention = ContentAttention(self.num_layers, hidden_dim, nheads, pre_norm)
        self.condition_attention = ConditionAttention(self.num_layers, hidden_dim, nheads, pre_norm)
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # learnable positive negative indicator
        self.pn_indicator = nn.Embedding(2, hidden_dim)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch
        self.query_index = {}

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        if task_switch['bbox']:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if task_switch['spatial'] or task_switch['interleave']:
            # spatial query
            self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
            trunc_normal_(self.mask_sptial_embed[0], std=.02)
            trunc_normal_(self.mask_sptial_embed[1], std=.02)
            trunc_normal_(self.mask_sptial_embed[2], std=.02)

            self.max_spatial_len = max_spatial_len
            # spatial memory
            num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
            self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
            self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)

        # build AttentionDataStruct
        attn_arch['NUM_LAYERS'] = self.num_layers
        self.attention_data = AttentionDataStruct(attn_arch, task_switch)

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']
        ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']

        # attn data struct
        ret["attn_arch"] = cfg['ATTENTION_ARCH']

        return ret

    def forward(self, x, mask_features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels; del mask
        demo_outputs = {}
        semantic_extra_flag = 'class_tokens_emb' in extra.keys() and task in ['class', 'seg']
        spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
        grounding_extra_flag = 'grounding_tokens' in extra.keys()
        retrieval_extra_flag = 'retrieval_tokens' in extra.keys()
        interleave_extra_flag = 'interleave_tokens' in extra.keys() or task == 'refint'
        spatial_memory_flag = 'prev_mask' in extra.keys()
        flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag, "memories_vision_spatial": spatial_memory_flag, "interleave": interleave_extra_flag, "class": semantic_extra_flag}
        self.attention_data.reset(flags, task, extra)

        src, pos, size_list = prepare_vision_features(x, self.num_feature_levels, self.pe_layer, self.input_proj, self.level_embed)
        self.attention_data.set_features('features_vision_image', 'features', self.num_feature_levels, src, pos, size_list)

        # size_list: mainly prepare for attention mask
        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        self.attention_data.set('proposals_vision_object', 'queries', output[:-1], query_embed[:-1])
        self.attention_data.set('queries_vision_image', 'queries', output[-1:], query_embed[-1:])

        if self.task_switch['mask']['ENABLED'] and semantic_extra_flag:
            class_tokens_emb = extra['class_tokens_emb']
            _class_tokens_emb = class_tokens_emb.detach().clone()
            class_tokens_mask = extra['class_tokens_mask']

            self.attention_data.set('queries_language_class', 'queries')
            self.attention_data.set('tokens_language_class', 'tokens', class_tokens_emb, _class_tokens_emb)
            self.attention_data.set_maskings('tokens_language_class', class_tokens_mask)

        if self.task_switch['spatial'] and spatial_extra_flag and task != "class":
            if 'refimg_tokens' not in extra:
                # get divisor
                c,h,w = extra['spatial_query_pos_mask'][0].shape
                divisor = torch.tensor([1,h,w], device=output.device)[None,]

                # Get mean pos spatial query
                # non_zero_pos_point = [rand_sample_multi(m, divisor, self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
                # non_zero_pos_index = [m[:,0:1].long() for m in non_zero_pos_point]
                # non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)
                # non_zero_pos_index = nn.utils.rnn.pad_sequence(non_zero_pos_index, padding_value=-1).permute(1,0,2)[:,:,0]
                # non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) < 0)
                # spatial_query_pos = point_sample(mask_features, non_zero_pos_point[:,:,1:].flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
                # num_mask_per_batch = [len(m) for m in extra['spatial_query_pos_mask']]
                # spatial_query_pos = nn.utils.rnn.pad_sequence([torch.stack([x[ns==n].mean(dim=0, keepdim=False) if (ns==n).sum() > 0 else -torch.ones((x.shape[1]), device=spatial_query_pos.device) for n in range(mb)]) for x, m, ns, mb in zip(spatial_query_pos.transpose(1,2), ~non_zero_pos_mask, non_zero_pos_index, num_mask_per_batch)], padding_value=-1).nan_to_num()

                # Get mean neg spatial query
                # non_zero_neg_point = [rand_sample_multi(m, divisor, self.max_spatial_len[-1]).t() for m in extra['spatial_query_neg_mask']]
                # non_zero_neg_index = [m[:,0:1].long() for m in non_zero_neg_point]
                # non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point, padding_value=-1).permute(1,0,2)
                # non_zero_neg_index = nn.utils.rnn.pad_sequence(non_zero_neg_index, padding_value=-1).permute(1,0,2)[:,:,0]
                # non_zero_neg_mask = (non_zero_neg_point.sum(dim=-1) < 0)
                # spatial_query_neg = point_sample(mask_features, non_zero_neg_point[:,:,1:].flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
                # num_mask_per_batch = [len(m) for m in extra['spatial_query_neg_mask']]
                # spatial_query_neg = nn.utils.rnn.pad_sequence([torch.stack([x[ns==n].mean(dim=0, keepdim=False) if (ns==n).sum() > 0 else -torch.ones((x.shape[1]), device=spatial_query_neg.device) for n in range(mb)]) for x, m, ns, mb in zip(spatial_query_neg.transpose(1,2), ~non_zero_neg_mask, non_zero_neg_index, num_mask_per_batch)], padding_value=-1).nan_to_num()

                # Get layerwise spatial query
                src_spatial_queries = []
                src_spatial_maskings = []
                src_spatial_indices = []
                for i in range(len(src)):
                    hw,_,dc = src[i].shape
                    src_mask_features = src[i].view(size_list[i][0],size_list[i][1],bs,dc)
                    src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                    non_zero_query_point_pos = [rand_sample_multi(m, divisor, self.max_spatial_len[i]).t() for m in extra['spatial_query_pos_mask']]
                    non_zero_query_point_neg = [rand_sample_multi(m, divisor, self.max_spatial_len[i]).t() for m in extra['spatial_query_neg_mask']]
                    non_zero_query_point = [torch.cat([x[:,1:],y[:,1:]], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                    non_zero_query_index = [torch.cat([x[:,0:1],y[:,0:1]], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]

                    pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                    pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)

                    non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
                    non_zero_query_index = nn.utils.rnn.pad_sequence(non_zero_query_index, padding_value=-2).permute(1,0,2)
                    non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
                    non_zero_query_point[non_zero_query_mask] = 0

                    spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                    spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                    spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]

                    src_spatial_queries += [spatial_tokens]
                    src_spatial_maskings += [non_zero_query_mask]
                    src_spatial_indices += [non_zero_query_index]

                if 'refimg' in task:
                    output_refimg = {}
                    # output_refimg['spatial_query_pos'] = spatial_query_pos
                    # output_refimg['spatial_query_neg'] = spatial_query_neg
                    output_refimg['src_spatial_queries'] = src_spatial_queries
                    output_refimg['src_spatial_maskings'] = src_spatial_maskings
                    output_refimg['src_spatial_indices'] = src_spatial_indices
                    if task == 'refimg':
                        return output_refimg
                    else:
                        demo_outputs.update(output_refimg)
            else:
                # spatial_query_pos = extra['refimg_tokens']['spatial_query_pos']
                # spatial_query_neg = extra['refimg_tokens']['spatial_query_neg']
                src_spatial_queries = extra['refimg_tokens']['src_spatial_queries']
                src_spatial_maskings = extra['refimg_tokens']['src_spatial_maskings']
                src_spatial_indices = extra['refimg_tokens']['src_spatial_indices']

            # Get object query for spatial index
            self.attention_data.set('proposals_vision_segment', 'queries')

            # set spatial memory
            spatial_output = self.spatial_featured.weight.unsqueeze(1).repeat(1, bs, 1)
            spatial_embed = self.spatial_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            self.attention_data.set('memories_vision_spatial', 'memories', spatial_output, spatial_embed)

            # set spatial query
            self.attention_data.set('queries_vision_spatial', 'queries', sample_size=len(extra['spatial_query_indices']))
            self.attention_data.set_indices('queries_vision_spatial', extra['spatial_query_indices'].transpose(0,1))

        if self.task_switch['grounding'] and grounding_extra_flag and task != "class":
            # Get grounding tokens
            grounding_tokens = extra['grounding_tokens']
            _grounding_tokens = grounding_tokens.detach().clone()

            self.attention_data.set('tokens_language_description', 'tokens', grounding_tokens, _grounding_tokens)
            self.attention_data.set('proposals_vision_grounding', 'queries')
            self.attention_data.set_maskings('tokens_language_description', extra['grounding_nonzero_mask'])
            self.attention_data.set_indices('tokens_language_description', extra['grounding_token_indices'].transpose(0,1))

            self.attention_data.set('queries_language_description', 'queries', sample_size=len(extra['grounding_query_indices']))
            self.attention_data.set_indices('queries_language_description', extra['grounding_query_indices'].transpose(0,1))

        if self.task_switch['retrieval'] and retrieval_extra_flag and task != "class":
            # Get retrieval tokens
            retrieval_tokens = extra['retrieval_tokens']
            _retrieval_tokens = retrieval_tokens.detach().clone()

            self.attention_data.set('tokens_language_caption', 'tokens', retrieval_tokens, _retrieval_tokens)
            self.attention_data.set('queries_language_caption', 'queries')
            self.attention_data.set_maskings('tokens_language_caption', extra['retrieval_nonzero_mask'])

        if self.task_switch['interleave'] and interleave_extra_flag and task != "class":
            if "refint_tokens" not in extra:
                # prepare visual tokens for interleave
                c,h,w = extra['interleave_query_pos_mask'][0].shape
                divisor = torch.tensor([1,h,w], device=output.device)[None,]

                # Get mean pos spatial query
                # inter_non_zero_pos_point = [rand_sample_multi(m, divisor, self.max_spatial_len[-1]).t() for m in extra['interleave_query_pos_mask']]
                # inter_non_zero_pos_index = [m[:,0:1].long() for m in inter_non_zero_pos_point]
                # inter_non_zero_pos_point = nn.utils.rnn.pad_sequence(inter_non_zero_pos_point, padding_value=-1).permute(1,0,2)
                # inter_non_zero_pos_index = nn.utils.rnn.pad_sequence(inter_non_zero_pos_index, padding_value=-1).permute(1,0,2)[:,:,0]
                # inter_non_zero_pos_mask = (inter_non_zero_pos_point.sum(dim=-1) < 0)
                # inter_spatial_query_pos = point_sample(mask_features, inter_non_zero_pos_point[:,:,1:].flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
                # inter_num_mask_per_batch = [len(m) for m in extra['interleave_query_pos_mask']]

                # prepare average token for the interleave visual feature, sampling from mask features
                # inter_spatial_query_pos = nn.utils.rnn.pad_sequence([torch.stack([x[ns==n].mean(dim=0, keepdim=False) if (ns==n).sum() > 0 else -torch.ones((x.shape[1]), device=inter_spatial_query_pos.device) for n in range(mb)]) for x, m, ns, mb in zip(inter_spatial_query_pos.transpose(1,2), ~inter_non_zero_pos_mask, inter_non_zero_pos_index, inter_num_mask_per_batch)], padding_value=-1).nan_to_num()

                # Get layerwise spatial query
                src_interleave_queries = []
                src_interleave_maskings = []
                src_interleave_indices = []
                for i in range(len(src)):
                    hw,_,dc = src[i].shape
                    src_mask_features = src[i].view(size_list[i][0],size_list[i][1],bs,dc)
                    src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                    int_non_zero_query_point_pos = [rand_sample_multi(m, divisor, self.max_spatial_len[i]).t() for m in extra['interleave_query_pos_mask']]
                    int_non_zero_query_point_neg = [rand_sample_multi(m, divisor, self.max_spatial_len[i]).t() for m in extra['interleave_query_neg_mask']]
                    int_non_zero_query_point = [torch.cat([x[:,1:],y[:,1:]], dim=0) for x,y in zip(int_non_zero_query_point_pos, int_non_zero_query_point_neg)]
                    int_non_zero_query_index = [torch.cat([x[:,0:1],y[:,0:1]], dim=0) for x,y in zip(int_non_zero_query_point_pos, int_non_zero_query_point_neg)]

                    pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(int_non_zero_query_point_pos, int_non_zero_query_point_neg)]
                    pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)

                    int_non_zero_query_point = nn.utils.rnn.pad_sequence(int_non_zero_query_point, padding_value=-1).permute(1,0,2)
                    int_non_zero_query_index = nn.utils.rnn.pad_sequence(int_non_zero_query_index, padding_value=-7).permute(1,0,2)
                    int_non_zero_query_mask = (int_non_zero_query_point.sum(dim=-1) < 0)
                    int_non_zero_query_point[int_non_zero_query_mask] = 0

                    int_spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), int_non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                    int_spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                    int_spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]

                    src_interleave_queries += [int_spatial_tokens]
                    src_interleave_maskings += [int_non_zero_query_mask]
                    src_interleave_indices += [int_non_zero_query_index]

                if "refint" in task:
                    output_refint = {}
                    output_refint['src_interleave_queries'] = src_interleave_queries
                    output_refint['src_interleave_maskings'] = src_interleave_maskings
                    output_refint['src_interleave_indices'] = src_interleave_indices

                    if task == 'refint':
                        return output_refint
                    else:
                        demo_outputs.update(output_refint)
            else:
                src_interleave_queries = extra['refint_tokens']['src_interleave_queries']
                src_interleave_maskings = extra['refint_tokens']['src_interleave_maskings']
                src_interleave_indices = extra['refint_tokens']['src_interleave_indices']

            # prepare language tokens for interleave
            interleave_text_tokens = extra['interleave_tokens']
            interleave_query_indices = extra['interleave_query_indices']
            interleave_entity_mask = extra['interleave_entity_mask']
            interleave_isvisual = extra['interleave_isvisual']
            interleave_isvisual_mask = (interleave_entity_mask * interleave_isvisual[:,:,None]).max(dim=0)[0].transpose(0,1)
            interleave_visual_otokens = interleave_isvisual_mask[:,:,None] * output[-1:]
            interleave_visual_qtokens = interleave_isvisual_mask[:,:,None] * query_embed[-1:]
            interleave_tokens = interleave_text_tokens + interleave_visual_otokens
            _interleave_tokens = interleave_text_tokens + interleave_visual_qtokens

            self.attention_data.set('proposals_interleave_entity', 'queries')
            self.attention_data.set('queries_interleave_legend', 'queries')
            self.attention_data.set('queries_interleave_entity', 'queries', sample_size=len(interleave_query_indices))
            self.attention_data.set_indices('queries_interleave_entity', interleave_query_indices.transpose(0,1))

        output, query_embed = self.attention_data.output_variables()
        # prediction heads on learnable query features
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        # results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
        # results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
        # results["predictions_interleave"] = interleave_matching_query_.transpose(0,1) if interleave_extra_flag else None
        self.attention_data.set_results(results)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # Prepare layerwise tokens
            if self.task_switch['spatial'] and spatial_extra_flag and task != "class":
                # get spatial tokens
                spatial_tokens = src_spatial_queries[level_index]
                _spatial_tokens = spatial_tokens.detach().clone()

                self.attention_data.set('tokens_vision_spatial', 'tokens', spatial_tokens, _spatial_tokens)
                self.attention_data.set_maskings('tokens_vision_spatial', src_spatial_maskings[level_index])
                self.attention_data.set_indices('tokens_vision_spatial', src_spatial_indices[level_index][:,:,0])

            if self.task_switch['interleave'] and interleave_extra_flag and task != "class":
                # set up interleave spatial tokens
                interleave_visual_tokens = src_interleave_queries[level_index]
                _interleave_visual_tokens = interleave_visual_tokens.detach().clone()

                self.attention_data.set('tokens_vision_spatial_', 'tokens', interleave_visual_tokens, _interleave_visual_tokens)
                self.attention_data.set_maskings('tokens_vision_spatial_', src_interleave_maskings[level_index])
                self.attention_data.set_indices('tokens_vision_spatial_', src_interleave_indices[level_index][:,:,0])

                # set up legend tokens
                self.attention_data.set('tokens_interleave_legend', 'tokens', interleave_tokens, _interleave_tokens)
                self.attention_data.set_maskings('tokens_interleave_legend', extra['interleave_nonzero_mask'])
                self.attention_data.set_indices('tokens_interleave_legend', extra['interleave_entity_indices'].transpose(0,1))

            # CROSS ATTENTION
            content_variables = self.attention_data.cross_attn_variables(level_index, self.num_heads)
            output = self.content_attention(i, content_variables)
            self.attention_data.update_variables(output, 'cross_attn')

            # SELF ATTENTION
            output, query_embed, self_attn_mask = self.attention_data.self_attn(bs, self.num_heads)
            output = self.condition_attention(i, output,
                                            tgt_mask=self_attn_mask,
                                            tgt_key_padding_mask=None,
                                            query_pos=query_embed)


            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            self.attention_data.update_variables(output, 'self_attn')
            if self.task_switch['interleave'] and interleave_extra_flag and task != "class":
                interleave_tokens = self.attention_data.attn_variables['tokens_interleave_legend'].output
                # TODO: spatial tokens is not updated

            output, query_embed = self.attention_data.output_variables()
            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i)
            # results["predictions_pos_spatial"] = spatial_query_pos.transpose(0,1) if spatial_extra_flag else None
            # results["predictions_neg_spatial"] = spatial_query_neg.transpose(0,1) if spatial_extra_flag else None
            # results["predictions_interleave"] = interleave_matching_query_.transpose(0,1) if interleave_extra_flag else None
            self.attention_data.set_results(results)

        outputs = self.attention_data.organize_output()
        if len(demo_outputs) > 0:
            outputs.update(demo_outputs)
        return outputs

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # recompute class token output.
        if (self.task_switch['retrieval'] or self.task_switch['interleave']) and ('queries_vision_image' in self.attention_data.output_names):
            assert 'proposals_vision_object' in self.attention_data.output_names
            norm_decoder_output = decoder_output / (decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
            obj_token = norm_decoder_output[:,self.attention_data.query_index['proposals_vision_object'][0]:self.attention_data.query_index['proposals_vision_object'][1]]
            cls_token = norm_decoder_output[:,self.attention_data.query_index['queries_vision_image'][0]:self.attention_data.query_index['queries_vision_image'][1]]
            sim = (cls_token @ obj_token.transpose(1,2)).softmax(-1)[:,0,:,None].detach() # TODO include class token.
            cls_token = (sim * decoder_output[:,self.attention_data.query_index['proposals_vision_object'][0]:self.attention_data.query_index['proposals_vision_object'][1]]).sum(dim=1, keepdim=True)
            decoder_output[:,self.attention_data.query_index['queries_vision_image'][0]:self.attention_data.query_index['queries_vision_image'][1]] = cls_token

        class_embed = decoder_output @ self.class_embed
        outputs_class = self.lang_encoder.compute_similarity(class_embed, self.attention_data)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        outputs_bbox = [None for i in range(len(outputs_mask))]
        if self.task_switch['bbox']:
            outputs_bbox = self.bbox_embed(decoder_output)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        outputs_caption = class_embed

        results = {
            "attn_mask": attn_mask,
            "predictions_class": outputs_class,
            "predictions_mask": outputs_mask,
            "predictions_bbox": outputs_bbox,
            "predictions_caption": outputs_caption,
            "predictions_maskemb": mask_embed,
        }
        return results


@register_decoder
def get_find_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return FINDDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)