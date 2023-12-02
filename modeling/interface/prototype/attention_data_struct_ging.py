import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.utils import flatten_dict, flatten_list

predict_name_matcher = {"predictions_class": ["pred_logits"],
                        "predictions_mask":["pred_masks", "pred_gmasks", "pred_smasks", "pred_imasks"],
                        "predictions_caption":["pred_captions", "pred_gtexts", "pred_retrievals", "pred_spatials", "pred_interleave_objects", "pred_interleave_image", "pred_retrievals_lang", "pred_retrievals_class", "pred_grounding_query", "pred_entity_class", "pred_spatial_class"],
                        "predictions_maskemb":["pred_smaskembs", "pred_imaskembs", "pred_pspatials", "pred_entity_pixel", "pred_legend_pixel", "pred_maskembs", "pred_image_pixel"],}

predict_index_matcher = {"predictions_class": ["proposals_vision_object"],
                         "predictions_mask":["proposals_vision_object", "proposals_vision_grounding", "proposals_vision_segment", "proposals_interleave_entity"],
                         "predictions_caption": ["proposals_vision_object", "proposals_vision_grounding", "queries_vision_image", "proposals_vision_segment", "proposals_interleave_entity", "queries_interleave_legend", "queries_language_caption", "queries_language_class", "queries_language_description", "queries_interleave_entity", "queries_vision_spatial"],
                         "predictions_maskemb":["proposals_vision_segment", "proposals_interleave_entity", "queries_vision_spatial", "queries_interleave_entity", "queries_interleave_legend", "proposals_vision_object", "queries_vision_image"],}

class Variable(object):
    '''
    Store dataset variable for attention
    output: embedding that accumuates during cross/self attention
    pos: positional embedding that is fixed during cross/self attention
    name: name of the variable
    type: type of the variable, e.g. queries, tokens
    attn_mask: attention mask for corss attention
    masking: masking for padding
    indexing: indexing for queries and tokens for attention
    '''
    def __init__(self, output, name, _type, pos=None):
        self.output = output
        self.pos = pos
        self.name = name
        self.type = _type
        self.attn_mask = None
        self.masking = None
        self.indexing = None
    
    def copy(self,):
        output = self.output.clone() if self.output is not None else None
        pos = self.pos.clone() if self.pos is not None else None
        return Variable(output, self.name, self.type, pos)

    def rand_sample(self, max_len):
        rand_idx = torch.randint(0, len(self.pos), (max_len,))
        self.output = self.output[rand_idx]
        self.pos = self.pos[rand_idx]
        return self

class Feature(object):
    '''
    Store dataset feature for content attention
    '''
    def __init__(self, num_feature_levels, _type, src, pos, size_list):
        self.num_feature_levels = num_feature_levels
        self.type = _type
        self.src = src
        self.pos = pos
        self.size_list = size_list

class AttentionDataStruct(nn.Module):
    '''
    Store dataset structure for cross/self attention
    task_switch: switch for different tasks

    p_attn_variables: prototype of variables that is used in cross/self attention
    p_self_attn: prototype of variables that is used in self attention
    p_cross_attn: prototype of variables that is used in cross attention
    p_iter: prototype of iteration for different queries
    p_masking: prototype of masking for different tokens
    p_duplication: prototype of duplication for different quries
    '''
    def __init__(self, attn_arch, task_switch):
        super(AttentionDataStruct, self).__init__()
        self.task_switch = task_switch

        # p stands for prototype
        self.p_attn_variables = attn_arch['VARIABLE']
        self.p_self_attn = attn_arch['SELF_ATTENTION']
        self.p_cross_attn = attn_arch['CROSS_ATTENTION']
        self.p_output = attn_arch['OUTPUT']
        self.p_masking = attn_arch['MASKING']
        self.p_duplication = attn_arch['DUPLICATION']
        self.p_ignore_cross_mask = attn_arch['IGNORE_CROSS_MASK']
        self.p_dynamics = attn_arch['DYNAMICS']

        self.num_layers = attn_arch['NUM_LAYERS']

    def reset(self, flags, task, extra):
        # reset variables
        self.attn_variables = {}
        self.cross_attn_dict = {}
        self.self_attn_dict = {}
        self.output_dict = {}
        self.duplication_dict = {}
        self.query_index = {}
        self.output = {}
        self.flags = {}
        self.spatial_memory = {}

        # initialize duplication
        self.duplication_dict = flatten_dict(self.p_duplication)

        # initialize flag
        self.flags = {"object": True}
        self.flags.update(flags)

        # initialize task
        self.task = task

        # initialize output
        if self.task_switch['mask'] and task != 'class':
            self.output['predictions_class'] = []
            self.output['predictions_mask'] = []
            self.output['predictions_caption'] = []
            self.output['predictions_maskemb'] = []

            # hard code the forwarded class index
            self.class_indexes = extra['class_indexes'] if 'class_indexes' in extra else None
        
        if self.task_switch['mask'] and task == 'class':
            self.output['predictions_caption'] = []
        
        if self.task_switch['bbox']:
            self.output['predictions_bbox'] = []
        
        if self.task_switch['retrieval']:
            self.output['predictions_caption'] = []

        if self.task_switch['spatial'] and ('spatial' in self.flags and self.flags['spatial']==True):
            self.output['predictions_maskemb'] = []

        if self.task_switch['spatial'] and ('memories_vision_spatial' in self.flags and self.flags['memories_vision_spatial']==True):
            self.spatial_memory['prev_batch_mask'] = extra['prev_mask']

        if self.task_switch['grounding'] and ('grounding' in self.flags and self.flags['grounding']==True):
            self.output['predictions_caption'] = []
        
        if self.task_switch['interleave'] and ('interleave' in self.flags and self.flags['interleave']==True):
            self.output['predictions_maskemb'] = []
            self.output['predictions_caption'] = []

        # initialize cross_attn, whether the variable is used in cross attention
        self.cross_attn_dict = flatten_dict(self.p_cross_attn)
        
        # initialize self_attn, whether the variable is used in self attention, and the interactions between queries
        self.self_attn_dict = flatten_dict(copy.deepcopy(self.p_self_attn))
        
        # initialize dynamics
        self.conditional_dynamics = flatten_dict(self.p_dynamics['CONDITIONAL_ATTENTION'])

        if self.training:
            for key, _value in self.conditional_dynamics.items():
                for value in _value:
                    if key in self.self_attn_dict and random.random() < value[2]:
                        self.self_attn_dict[key][self.self_attn_dict[key].index(value[0])] = value[1]

        # initialize output prototype
        self.output_dict = flatten_dict(self.p_output)
        
        # initialize masking
        self.masking = self.p_masking

        # initialize query_index
        self.query_index = {"all":[0, None]}


    def set(self, name, _type, output=None, pos=None, var=None, sample_size=None):
        if var is not None:
            self.attn_variables[name] = var
        elif name in self.duplication_dict:
            assert self.duplication_dict[name] in self.attn_variables, "Duplication variable {} is not initialized yet.".format(name)
            var = self.attn_variables[self.duplication_dict[name]].copy()
            if sample_size is not None:
                var = var.rand_sample(sample_size)
            self.attn_variables[name] = var
        else:
            var = Variable(output, name, _type, pos)
            self.attn_variables[name] = var
    
    def set_features(self, name, _type, num_feature_levels=None, src=None, pos=None, size_list=None):
        var = Feature(num_feature_levels, _type, src, pos, size_list)
        self.attn_variables[name] = var

    def set_results(self, results):
        for name in self.output_names:
            self.attn_variables[name].attn_mask = results['attn_mask'][:,self.query_index[name][0]:self.query_index[name][1]]
        for key in self.output:
            self.output[key].append(results[key])
    
    def set_maskings(self, name, masking):
        self.attn_variables[name].masking = masking

    def set_indices(self, name, indexing):
        self.attn_variables[name].indexing = indexing

    def output_variables(self, ):
        output_names = [key for key, value in self.output_dict.items()
                           if (value==True) and (key in self.attn_variables)
                           and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]
        self.output_names = output_names

        output = torch.cat([self.attn_variables[name].output for name in output_names])
        pos_emb = torch.cat([self.attn_variables[name].pos for name in output_names])
        
        index = 0
        for name in output_names:
            self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
            index += self.attn_variables[name].output.shape[0]
        return output, pos_emb

    def cross_attn_variables(self, i, num_heads):
        variables = {}
        # vision attention name, we hard code features_vision_image for vision feature
        vision_attn_name = [key for key, value in self.cross_attn_dict.items()
                           if ('features_vision_image' in value) and (key in self.attn_variables)
                           and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]

        # vision attention embeding
        output = torch.cat([self.attn_variables[name].output for name in vision_attn_name])
        pos_emb = torch.cat([self.attn_variables[name].pos for name in vision_attn_name])
        self.vision_attn_name = vision_attn_name
        vision_feature = self.attn_variables['features_vision_image']

        index = 0
        for name in vision_attn_name:
            self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
            index += self.attn_variables[name].output.shape[0]

        attn_mask = self.cross_attn_mask(vision_feature.size_list[i], num_heads)
        variables.update({"vision": {"output": output, "src": vision_feature.src[i], "memory_mask": attn_mask, "pos": vision_feature.pos[i], "query_pos": pos_emb}})

        # other attention, and src name, we hard code features_vision_image for vision feature
        other_attn_name = [key for key, value in self.cross_attn_dict.items() 
                           if ('features_vision_image' not in value and len(value) > 0) and (key in self.attn_variables) 
                           and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]
        other_feature_name = list(set(flatten_list([self.cross_attn_dict[key] for key in other_attn_name])))
        self.other_attn_name = other_attn_name

        if len(other_attn_name) > 0:
            output = torch.cat([self.attn_variables[name].output for name in other_attn_name])
            pos_emb = torch.cat([self.attn_variables[name].pos for name in other_attn_name])

            # add other attention index for concatentation index
            for name in other_attn_name:
                self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
                index += self.attn_variables[name].output.shape[0]

            # track local index
            index = 0
            in_other_index = {}
            for name in other_attn_name:
                in_other_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
                index += self.attn_variables[name].output.shape[0]

            src_feature = torch.cat([self.attn_variables[name].output for name in other_feature_name])
            src_pos = torch.cat([self.attn_variables[name].pos for name in other_feature_name])
            # prepare masking
            src_mask = torch.cat([self.attn_variables[name].masking
                                    if self.attn_variables[name].masking is not None
                                    else torch.zeros_like(self.attn_variables[name], dtype=torch.bool)
                                    for name in other_feature_name], dim=1)

            index = 0
            src_other_index = {}
            for name in other_feature_name:
                src_other_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
                index += self.attn_variables[name].output.shape[0]

            # initialize other content attention mask
            attn_mask = torch.ones((src_feature.shape[1], len(output), len(src_feature)), dtype=torch.bool, device=src_feature.device)
            # loop through all other_attn_name
            for key_name in other_attn_name:
                for src_name in self.cross_attn_dict[key_name]:
                    attn_mask[:,in_other_index[key_name][0]:in_other_index[key_name][1],src_other_index[src_name][0]:src_other_index[src_name][1]] = src_mask[:,None,src_other_index[src_name][0]:src_other_index[src_name][1]]        

                    # the current code does not handle where src_name and key_name are the same
                    if self.attn_variables[key_name].indexing is not None \
                        and self.attn_variables[src_name].indexing is not None:
                            index_masking = ~(self.attn_variables[key_name].indexing[:,:,None] == self.attn_variables[src_name].indexing[:,None,:])
                            attn_mask[:,in_other_index[key_name][0]:in_other_index[key_name][1],src_other_index[src_name][0]:src_other_index[src_name][1]] += index_masking

            attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
            variables.update({"other": {"output": output, "src": src_feature, "memory_mask": attn_mask, "pos": src_pos, "query_pos": pos_emb}})

        return variables
    
    def cross_attn_mask(self, size, num_heads):
        attn_mask = torch.cat([self.attn_variables[name].attn_mask for name in self.vision_attn_name], dim=1)

        # hard code memories_spatial to previous selected mask
        if 'memories_vision_spatial' in self.vision_attn_name:
            memory_attn_mask = self.spatial_memory['prev_batch_mask']
            bs,c,_,_ = memory_attn_mask.shape
            memory_attn_mask = F.interpolate(memory_attn_mask, size, mode='bilinear', align_corners=False)
            memory_attn_mask = (memory_attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, num_heads, 1, 1).flatten(0, 1) < 0.5).bool().detach()
            attn_mask[:,self.query_index['memories_vision_spatial'][0]:self.query_index['memories_vision_spatial'][1]] = memory_attn_mask
        
        # ignore predicted mask cross attention
        for key in self.vision_attn_name:
            if key in self.p_ignore_cross_mask:
                attn_mask[:,self.query_index[key][0]:self.query_index[key][1]] = False
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        return attn_mask

    def self_attn(self, bs, num_heads):
        self_attn_name = [key for key, value in self.self_attn_dict.items() 
                          if len(value)>0 and key in self.attn_variables
                          and ((key not in self.flags) or (key in self.flags and self.flags[key]==True))]
        self.self_attn_name = self_attn_name

        output = torch.cat([self.attn_variables[name].output for name in self_attn_name])
        pos_emb = torch.cat([self.attn_variables[name].pos for name in self_attn_name])

        index = 0
        for name in self_attn_name:
            self.query_index[name] = [index, index + self.attn_variables[name].output.shape[0]]
            index += self.attn_variables[name].output.shape[0]
        
        self_attn_mask = torch.ones((bs, output.shape[0], output.shape[0]), dtype=torch.bool, device=output.device)
        self_attn_pair = []
        # build self_attention mask by query interaction
        for key1, value in self.self_attn_dict.items():
            for key2 in value:
                if key1 not in self_attn_name or key2 not in self_attn_name:
                    # exclude the variables that are not used in the current layer
                    continue
                if (key1 in self.masking or key2 in self.masking) and (key1 != key2):
                    self_attn_pair += [[key1, key2]]
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1], self.query_index[key2][0]:self.query_index[key2][1]] = False

        # build self_attention mask by masking, for birectional
        for key in self.masking:
            if key in self_attn_name:
                self_attn_mask[:,self.query_index[key][0]:self.query_index[key][1],self.query_index[key][0]:self.query_index[key][1]][self.attn_variables[key].masking] = True
                self_attn_mask[:,self.query_index[key][0]:self.query_index[key][1],self.query_index[key][0]:self.query_index[key][1]].transpose(1,2)[self.attn_variables[key].masking] = True

        # build self_attention mask by masking, for uni-directional
        for key1, key2 in self_attn_pair:
            if key1 not in self_attn_name or key2 not in self_attn_name:
                # exclude the variables that are not used in the current layer
                continue
            if key1 in self.masking:
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1],self.query_index[key2][0]:self.query_index[key2][1]][self.attn_variables[key1].masking] = True # HACK, not verified
            if key2 in self.masking:
                self_attn_mask[:,self.query_index[key1][0]:self.query_index[key1][1],self.query_index[key2][0]:self.query_index[key2][1]].transpose(1,2)[self.attn_variables[key2].masking] = True

        # we do not consider self-attention indexing for now
        self_attn_mask = self_attn_mask.repeat_interleave(num_heads, dim=0)
        return output, pos_emb, self_attn_mask

    def update_variables(self, output, mode):
        name_set = self.self_attn_name if mode=='self_attn' else (self.vision_attn_name + self.other_attn_name)
        for key in name_set:
            self.attn_variables[key].output = output[self.query_index[key][0]:self.query_index[key][1]]

    def update_spatial_results(self, results):
        v_emb = results['pred_smaskembs']
        s_emb = results['pred_pspatials']
        pred_smasks = results['pred_smasks']

        pred_logits = v_emb @ s_emb.transpose(1,2)
        logits_idx_y = pred_logits[0].max(dim=0)[1]
        # logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        # logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_smasks[:,logits_idx_y]

        # s_emb = results['pred_nspatials']
        # pred_logits = v_emb @ s_emb.transpose(1,2)
        # logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        # logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        # logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        # pred_masks_neg = pred_smasks[logits_idx][:,None,]
        # # clip the negative mask to 0, and then multiply by -1
        # pred_masks_neg = (pred_masks_neg.clip(0) * -1)
        # keep_neg = (s_emb.sum(dim=list(range(1, s_emb.dim()))) != 0).float()
        # pred_masks_neg = pred_masks_neg * keep_neg[:,None,None,None]
        # extra = {"prev_mask": pred_masks_pos + pred_masks_neg}

        extra = {"prev_mask": pred_masks_pos}
        return extra

    def organize_output(self, ):
        outputs = {}
        outputs['aux_outputs'] = [{} for i in range(self.num_layers)]
        for key, values in self.output.items():
            for _key, idx_name in zip(predict_name_matcher[key], predict_index_matcher[key]):
                if idx_name not in self.query_index:
                    continue
                outputs[_key] = self.output[key][-1][:,self.query_index[idx_name][0]:self.query_index[idx_name][1]]
                for idx, aux_values in enumerate(self.output[key][:-1]):
                    outputs['aux_outputs'][idx][_key] = aux_values[:,self.query_index[idx_name][0]:self.query_index[idx_name][1]]
        if self.task == 'spatial' or self.task == 'refimg':
            outputs = self.update_spatial_results(outputs)
        return outputs