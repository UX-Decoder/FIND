# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .build import register_model
from ..utils import configurable, strict_hash
from .LangEncoder import build_tokenizer, build_lang_encoder
from utils.prompt_engineering import prompt_engineering, get_prompt_templates
from .loss import all_gather_grad
import detectron2.utils.comm as comm

class LanguageEncoder(nn.Module):

    @configurable
    def __init__(
        self,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
        load_tensor,
        precompute,
        arch,
        lang_dict,
        text_cfg,
        verbose,
    ):
        super().__init__()
        # seg
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.feature_layer = -1

        self.load_tensor = load_tensor
        self.precompute = precompute
        self.arch = arch
        self.lang_dict = lang_dict
        self.text_cfg = text_cfg
        self.verbose = verbose

    @classmethod
    def from_config(cls, cfg):
        # build up text encoder for seg
        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        
        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['MODEL']['DIM_PROJ']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        # precompute settings
        load_tensor = cfg['MODEL']['TEXT']['LOAD_TENSOR']
        precompute = cfg['MODEL']['TEXT']['PRECOMPUTE']
        arch = cfg['MODEL']['TEXT']['ARCH']
        text_cfg = cfg['MODEL']['TEXT']
        verbose = cfg['VERBOSE']

        if not load_tensor:
            lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
            lang_dict = None
        else:
            lang_encoder = None
            lang_pth = cfg['MODEL']['TEXT']['PRETRAINED'] if cfg['MODEL']['TEXT']['PRETRAINED'] != '' else cfg['RESUME_FROM']
            pretrained_weight = torch.load(lang_pth, map_location='cpu')
            lang_dict = {}
            for key in pretrained_weight:
                if 'lang_encoder.lang_encoder.' in key:
                    new_key = key.split('lang_encoder.lang_encoder.')[-1]
                    lang_dict[new_key] = pretrained_weight[key]

        return {
            "tokenizer": tokenizer,
            "tokenizer_type": tokenizer_type,
            "lang_encoder": lang_encoder,
            "lang_projection": lang_projection,
            "max_token_num": max_token_num,
            "load_tensor": load_tensor,
            "precompute": precompute,
            "arch": arch,
            "lang_dict": lang_dict,
            "text_cfg": text_cfg,
            "verbose": verbose,
        }

    @property
    def device(self):
        return self.logit_scale.device

    def reset_text_embeddings(self, name='default'):
        delattr(self, '{}_text_embeddings'.format(name))

    def get_text_token_embeddings(self, txts, name='default', token=False, embed=False, norm=False, projection=True):

        if embed:
            tokens = txts['tokens']
            token_emb = txts['token_emb']
            class_emb = txts['class_emb']
        else:
            if not token:
                tokens = self.tokenizer(
                    txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens = {key: value.cuda() for key, value in tokens.items()}
            else:
                tokens = txts

            x = self.lang_encoder(tokens['input_ids'], tokens['attention_mask'])
            token_emb = x['last_hidden_state']

            if self.tokenizer_type == 'clip':
                class_emb = token_emb[torch.arange(token_emb.size(0)), tokens['input_ids'].argmax(dim=-1)]
            else:
                class_emb = token_emb[:, 0]

        if projection:
            class_emb = class_emb @ self.lang_proj if class_emb is not None else None
            token_emb = token_emb @ self.lang_proj

        if norm:
            class_emb = class_emb / (class_emb.norm(dim=-1, keepdim=True) + 1e-7) if class_emb is not None else None
            token_emb = token_emb / (token_emb.norm(dim=-1, keepdim=True) + 1e-7)

        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def compute_similarity(self, v_emb, attn_data, name='default', fake=False):
        if fake or hasattr(self, '{}_text_embeddings'.format(name)) is False:
            return None

        t_emb = getattr(self, '{}_text_embeddings'.format(name)).clone()
        if self.training:
            batch_t_emb = v_emb[:, attn_data.query_index['queries_language_class'][0]:attn_data.query_index['queries_language_class'][1]][:,0]
            batch_t_idx = attn_data.class_indexes

            batch_t_emb = all_gather_grad(batch_t_emb.contiguous())
            batch_t_idx = torch.cat(comm.all_gather(batch_t_idx))
            t_emb[batch_t_idx] = batch_t_emb
            setattr(self, '{}_text_embeddings'.format(name), t_emb.detach())

        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output

    def activate(self, ):
        if self.load_tensor:
            self.lang_encoder = build_lang_encoder(self.text_cfg, self.tokenizer, self.verbose).to(self.device)
            for key, tensor in self.lang_dict.items():
                self.lang_dict[key] = tensor.to(self.device)
            self.lang_encoder.load_state_dict(self.lang_dict)

    def reset(self, ):
        if self.load_tensor:
            delattr(self, 'lang_encoder')
            for key, tensor in self.lang_dict.items():
                self.lang_dict[key] = tensor.to('cpu')
            torch.cuda.empty_cache()

@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)