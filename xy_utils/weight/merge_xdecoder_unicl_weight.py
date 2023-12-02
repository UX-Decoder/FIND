import torch
import os

full_pth = '/nobackup3/xueyan-data/grin_data/output/mask2former_vlp_focall_enc6_fpn_dec10_lang_capgTrue_retTrue_grdTrue_topc3_topr3_topg6_capgw8_rw8_cbs32_vbs1024_ep50_lr0.0001_preuTrue_gtw2.0_gcw0.5_1122/00175900'
pre_pth = '/nobackup3/xueyan-data/grin_data/output/unicl/focal-l_bbone_focalb_lang'

full_weight = torch.load(os.path.join(full_pth, 'default', 'model_state_dict.pt'), map_location='cpu')
pre_weight = torch.load(os.path.join(pre_pth, 'default', 'model_state_dict.pt'), map_location='cpu')

output_weight = {}
for key, value in full_weight.items():
    if 'lang_encoder' in key and 'logit_scale' not in key:
        output_weight[key] = pre_weight[key]
    else:
        output_weight[key] = value

output_pth = '/nobackup3/xueyan-data/grin_data/output/xdecoder/vision_xdecoder_focall_unicl_fb_lang/default'

# create output dir
if not os.path.exists(output_pth):
    os.makedirs(output_pth)

torch.save(output_weight, os.path.join(output_pth, 'model_state_dict.pt'))