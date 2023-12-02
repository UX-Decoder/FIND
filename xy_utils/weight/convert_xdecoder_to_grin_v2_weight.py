import torch
import os

pretrained = torch.load('/nobackup3/xueyan-data/grin_data/output/xdecoder/vision_xdecoder_focall_unicl_fb_lang/default/model_state_dict.pt')
output = torch.load('output.pt')

pretrained_keys = sorted(list(pretrained.keys()))
output_keys = sorted(list(output.keys()))

# for pk in pretrained_keys:
#     print(pk)
# sem_seg_head.predictor.condition_attention.layers.4.norm.bias

output_pretrained = {}

for pk in pretrained_keys:
    if pk not in output_keys:
        if 'transformer_self_attention_layers' in pk:
            new_pk = pk.replace('transformer_self_attention_layers', 'condition_attention.layers')
            output_pretrained[new_pk] = pretrained[pk]
        elif 'transformer_cross_attention_layers' in pk:
            new_pk = pk.replace('transformer_cross_attention_layers', 'content_attention.layers')
            output_pretrained[new_pk] = pretrained[pk]
        else:
            output_pretrained[pk] = pretrained[pk]
            print(pk, 'hihihi')
    else:
        output_pretrained[pk] = pretrained[pk]

output_pth = '/nobackup3/xueyan-data/grin_data/output/xdecoder/vision_xdecoder_focall_unicl_fb_lang_grin_v1/default'

# mkae sure the output_pth exists
if not os.path.exists(output_pth):
    os.makedirs(output_pth)

torch.save(output_pretrained, os.path.join(output_pth, 'model_state_dict.pt'))