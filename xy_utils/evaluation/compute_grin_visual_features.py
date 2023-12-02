import os
import sys

pth = '/'.join(sys.path[0].split('/')[:-2])
sys.path.insert(0, pth)

import torch
import torch.nn.functional as F
from torchvision import transforms
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from utils.arguments import load_opt_command
from trainer import XDecoder_Trainer as Trainer
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half
from datasets.evaluation import GroundingEvaluator
from modeling.modules import sem_seg_postprocess
from modeling.language.loss import vl_similarity
from tqdm import tqdm

from utils.constants import COCO_PANOPTIC_CLASSES

def main(args=None):
    '''
    build args
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    pretrained_pth = opt['RESUME_FROM']
    # hard code interactive token number
    opt['DATASETS']['TEST'] = ['grounding_coco_entity_val', 'grounding_coco_entity_val_long']

    trainer = Trainer(opt)
    raw_models = trainer.pipeline.initialize_model()
    model = raw_models['default'].from_pretrained(pretrained_pth).eval()
    model = model.cuda()
    model.model.sem_seg_head.predictor.lang_encoder.activate()
    model.model.get_class_embeddings(['default', 'default'], is_eval=True)

    dataset_name = 'grounding_coco_entity_val'
    dataloader = trainer.pipeline.get_dataloaders(trainer, dataset_name, is_evaluation=True)

    class_emb_dict = {}

    def inference_visual(entity, extra, _images, height, width):
        features = model.model.backbone(_images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)

        extra = {}
        extra['spatial_query_pos_mask'] = entity.interactive.to(model.model.device)[None,]
        extra['spatial_query_neg_mask'] = entity.interactive.to(model.model.device)[None,].clone().detach() & False
        extra['spatial_query_indices'] = torch.arange(1, device=model.model.device)[None,]
        outputs = model.model.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='refimg_spatial')

        pred_sq_emb = outputs['pred_pspatials']
        pred_sp_emb = outputs['pred_smaskembs']
        pred_sc_emb = outputs['pred_spatials']

        scores = (pred_sp_emb @ pred_sq_emb.transpose(1,2))[0,:,0]
        matched_id = scores.max(0)[1]
        class_emb = pred_sc_emb[0,matched_id,:]
        _outputs = {entity.text.item(): class_emb}
        return _outputs

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for idx, batched_inputs in enumerate(tqdm(dataloader)):
                entities = batched_inputs[0]['entities']
                batched_input = batched_inputs[0]
                images = [x["image"].to(model.model.device) for x in batched_inputs]
                images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                images = ImageList.from_tensors(images, model.model.size_divisibility)
                img_bs = images.tensor.shape[0]

                entity_masks = []
                for entity in entities['entities']:
                    if entity.type == 'visual':
                        if len(entity.text) == 0:
                            continue
                        class_id = entity.text.item()
                        if class_id in class_emb_dict:
                            if len(class_emb_dict[class_id]) >= 30:
                                continue
                        else:
                            class_emb_dict[class_id] = []
                        outputs = inference_visual(entity, model, images, batched_input['height'], batched_input['width'])
                        class_id, class_emb = outputs.popitem()
                        class_emb_dict[class_id].append(class_emb)

    # class_emb_dict = torch.load('class_emb_dict_focalt.da')
    class_embeddings = []
    for i in range(len(class_emb_dict)):
        class_embeddings += [torch.stack(class_emb_dict[i], dim=0).mean(0)]
    class_embeddings = torch.stack(class_embeddings, dim=0)
    torch.save(class_embeddings.cpu(), 'class_embeddings_davitd5.da')

if __name__ == "__main__":
    main()
    sys.exit(0)