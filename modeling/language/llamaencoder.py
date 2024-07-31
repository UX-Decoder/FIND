# --------------------------------------------------------
# FIND -- Interfacing Foundation Model Embeddings
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from transformers import AutoTokenizer, StoppingCriteria

from .build import register_model
from ..utils import configurable
from .LangEncoder import build_tokenizer, build_lang_encoder, LlamaForCausalLM
from .loss import all_gather_grad
import detectron2.utils.comm as comm


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

class LanguageEncoder(nn.Module):

    @configurable
    def __init__(
        self,
        model,
        model_path,
        tokenizer,
        context_length,
        stop_token,
        lang_projection,
        feature_layer,
        load_tensor,
        precompute,
        arch,
    ):
        super().__init__()
        self.lang_encoder = model
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_token_num = context_length
        self.stop_token = stop_token
        self.lang_proj = lang_projection
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.feature_layer = feature_layer

        if self.lang_encoder is not None:
            self.lang_encoder.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data.float()
            self.lang_encoder.lm_head.weight.data = model.lm_head.weight.data.float()

        self.load_tensor = load_tensor
        self.precompute = precompute
        self.arch = arch

    @classmethod
    def from_config(cls, cfg):
        model_path = cfg['MODEL']['TEXT']['NAME']
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right')
        tokenizer.pad_token = tokenizer.eos_token
        context_length = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        stop_token = cfg['MODEL']['TEXT']['STOP_TOKEN']

        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['MODEL']['DIM_PROJ']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        feature_layer = cfg['MODEL']['TEXT']['FEATURE_LAYER']
        # precompute settings
        load_tensor = cfg['MODEL']['TEXT']['LOAD_TENSOR']
        precompute = cfg['MODEL']['TEXT']['PRECOMPUTE']
        arch = cfg['MODEL']['TEXT']['ARCH']

        if not load_tensor:
            model = LlamaForCausalLM.from_pretrained(model_path, 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch.float16, 
                                                    use_cache=True, 
                                                    output_hidden_states=True, 
                                                    return_dict_in_generate=True)
        else:
            model = None

        return {
            "model": model,
            "model_path": model_path,
            "tokenizer": tokenizer,
            "context_length": context_length,
            "stop_token": stop_token,
            "lang_projection": lang_projection,
            "feature_layer": feature_layer,
            "load_tensor": load_tensor,
            "precompute": precompute,
            "arch": arch,
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
            if not self.training:
                self.lang_encoder.model.embed_tokens.weight.data = self.lang_encoder.model.embed_tokens.weight.data.half()
                self.lang_encoder.lm_head.weight.data = self.lang_encoder.lm_head.weight.data.half()

            if not token:
                tokens = self.tokenizer(
                    txts, padding='max_length', truncation=True, max_length=self.context_length, return_tensors='pt'
                )
                tokens = {key: value.cuda() for key, value in tokens.items()}
            else:
                tokens = txts

            # Prepare position ids
            position_ids = tokens['attention_mask'].long().cumsum(-1) - 1
            position_ids.masked_fill_(tokens['attention_mask'] == 0, 1)

            outputs = self.lang_encoder(input_ids=tokens['input_ids'],
                                        attention_mask=tokens['attention_mask'],
                                        position_ids=position_ids,
                                        use_cache=True,
                                        output_hidden_states=True,
                                        return_dict=True)

            token_emb = outputs['hidden_states'][self.feature_layer]
            class_emb = torch.stack([x[y].mean(0) for x,y in zip(token_emb, tokens['attention_mask'].bool())])

        if projection:
            class_emb = class_emb.to(dtype=self.lang_proj.dtype) @ self.lang_proj if class_emb is not None else None
            token_emb = token_emb.to(dtype=self.lang_proj.dtype) @ self.lang_proj

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
            self.lang_encoder = LlamaForCausalLM.from_pretrained(self.model_path, 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch.float16, 
                                                    use_cache=True, 
                                                    output_hidden_states=True, 
                                                    return_dict_in_generate=True).to(self.device)

    def reset(self, ):
        if self.load_tensor:
            delattr(self, 'lang_encoder')
            torch.cuda.empty_cache()

@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)