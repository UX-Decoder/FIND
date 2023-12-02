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
from datasets.evaluation import RetrievalEvaluator
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
    database_root = "../../data/output/database"
    coco_folders = ["/nobackup3/xueyan-data/grin_data/coco/train2017", "/nobackup3/xueyan-data/grin_data/coco/val2017", database_root]
    # paragraph_path = "/nobackup3/xueyan-data/grin_data/coco/annotations/entity_val2017_long.json"
    add_image_pths = []
    add_image_id = 0

    # hard code interactive token number
    opt['DATASETS']['TEST'] = ['vlp_coco_interleave_val', 'vlp_coco_interleave_val_long']

    trainer = Trainer(opt)
    raw_models = trainer.pipeline.initialize_model()
    model = raw_models['default'].from_pretrained(pretrained_pth).eval()
    model = model.cuda()

    dataset_name = 'vlp_coco_interleave_val'
    dataloader = trainer.pipeline.get_dataloaders(trainer, dataset_name, is_evaluation=True)

    #Remove retrieval evaluator input?
    evaluator = RetrievalEvaluator(dataset_name, None)
    evaluator.reset()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for idx, batched_inputs in enumerate(tqdm(dataloader)):
                processed_results = []
                processed_results.append({})
                assert len(batched_inputs) == 1
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["default", "default"], is_eval=True)

                batched_input = batched_inputs[0]
                images = [x["image"].to(model.model.device) for x in batched_inputs]
                images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                images = ImageList.from_tensors(images, model.model.size_divisibility)
                img_bs = images.tensor.shape[0]

                targets = targets_grounding = queries_grounding = None
                features = model.model.backbone(images.tensor)
                outputs = model.model.sem_seg_head(features, target_queries=queries_grounding)
                v_emb_it = outputs['pred_captions'][:,-1]
                image_embeds = [v_emb_it]

                lang_results = model.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(batched_input['entities']['sentence'])
                t_emb_it = lang_results['class_emb']
                caption_ids = [batched_input['image_id']]

                caption_results = {
                        'image_embeds': image_embeds,
                        'text_embeds': t_emb_it,
                        'caption_ids': caption_ids,
                        'image_ids': batched_input['image_id'],
                    }
                processed_results[-1]["caption"] = caption_results
                evaluator.process(None, processed_results)

    print(f"{dataset_name} Results: {evaluator.evaluate()}")


if __name__ == "__main__":
    main()
    sys.exit(0)