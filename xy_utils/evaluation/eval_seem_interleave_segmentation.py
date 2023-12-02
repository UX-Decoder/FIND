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
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["default", "default"], is_eval=True)

    dataset_name = 'grounding_coco_entity_val_long'
    dataloader = trainer.pipeline.get_dataloaders(trainer, dataset_name, is_evaluation=True)

    #Remove retrieval evaluator input?
    evaluator = GroundingEvaluator(dataset_name, None)
    evaluator.reset()

    def inference_visual(entity, extra, _images, height, width):
        features = model.model.backbone(_images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)
        extra = {}
        extra['spatial_query_pos_mask'] = entity.interactive.cuda()[None,]
        extra['spatial_query_neg_mask'] = entity.interactive.cuda()[None,].clone().detach() & False
        outputs = model.model.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=None, extra=extra, task='refimg')
        return outputs

    def inference_text(entity, model, _images, height, width):
        grd_texts = [entity.text]
        gtext = model.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
        token_emb = gtext['token_emb']
        tokens = gtext['tokens']
        query_emb = token_emb[tokens['attention_mask'].bool()]
        non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

        extra = {}
        extra['grounding_tokens'] = query_emb[:,None]
        extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

        features = model.model.backbone(_images.tensor)
        outputs = model.model.sem_seg_head(features, extra=extra, task='grounding_eval')

        pred_gmasks = outputs['pred_gmasks'][0]
        v_emb = outputs['pred_gtexts'][0]
        t_emb = gtext['class_emb']

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        
        matched_id = out_prob.max(0)[1]
        mask_pred_results = [pred_gmasks[matched_id,:,:]]

        # upsample masks
        mask_pred_results[0] = F.interpolate(
            mask_pred_results[0][None,],
            size=(_images.tensor.shape[-2], _images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )[0]

        mask_pred_results[0] = sem_seg_postprocess(
                mask_pred_results[0], _images.image_sizes[0], height, width
            )
        mask = (mask_pred_results[0] > 0)
        return mask

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
                        outputs = inference_visual(entity, model, images, batched_input['height'], batched_input['width'])
                        extra = {'refimg_tokens': outputs}
                        output = model.model.evaluate_interactive_single(batched_inputs, extra=extra)
                        entity_masks += [output[0]['pred_mask_ori']]

                    elif entity.type == 'text':
                        output = inference_text(entity, model, images, batched_input['height'], batched_input['width'])
                        entity_masks += [output]
                entity_masks = torch.cat(entity_masks, dim=0)
                processed_results = [{'grounding_mask': entity_masks.cpu()}]
                evaluator.process(batched_inputs, processed_results)

    print(f"{dataset_name} Results: {evaluator.evaluate()}")

if __name__ == "__main__":
    main()
    sys.exit(0)