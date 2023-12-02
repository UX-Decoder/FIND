import ast
import json
from pycocotools import mask as coco_mask

import os
import sys

import numpy as np
import torch

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from utils.arguments import load_opt_command
from trainer import XDecoder_Trainer as Trainer
from datasets.evaluation.grounding_evaluation import GroundingEvaluator
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half


json_path = "/nobackup3/xueyan-data/code/grin/vlcore_content/entity_val2017_gsam_bh_long.json"
annotations = json.load(open(json_path, 'r'))

def string_to_rle(rle_string):
    """
    Converts a string representation of RLE to a dictionary.

    :param rle_string: RLE string.
    :return: RLE dictionary.
    """
    try:
        rle_dict = ast.literal_eval(rle_string)
        if isinstance(rle_dict, dict) and 'counts' in rle_dict and 'size' in rle_dict:
            return rle_dict
        else:
            raise ValueError("String does not represent a valid RLE format.")
    except:
        raise ValueError("Error in converting string to RLE.")

def rle_to_mask(rle, height, width):
    """
    Converts a RLE (run length encoded) mask to a binary mask.

    :param rle: RLE dictionary.
    :param height: Height of the mask.
    :param width: Width of the mask.
    :return: Binary mask.
    """
    if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
        rle = [rle]
    else:
        raise ValueError("RLE format not recognized.")

    mask_decoded = coco_mask.decode(rle)
    return mask_decoded

def inverse_sigmoid(mask, epsilon=1e-6):
    """
    Apply inverse sigmoid (logit) transformation to a mask.

    :param mask: Binary mask.
    :param epsilon: Small value to avoid division by zero or log of zero.
    :return: Transformed mask.
    """
    # Ensure mask values are in the range (0, 1)
    mask = np.clip(mask, epsilon, 1 - epsilon)

    # Apply inverse sigmoid (logit)
    transformed_mask = np.log(mask / (1 - mask))
    return transformed_mask

def main(args=None):
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    trainer = Trainer(opt)
    dataset_name = 'grounding_coco_entity_val_long'
    opt['DATASETS']['TEST'] = [dataset_name] 
    dataloader = trainer.pipeline.get_dataloaders(trainer, dataset_name, is_evaluation=True)

    evaluator = GroundingEvaluator(dataset_name)
    evaluator.reset()

    index = 0
    for annot, batched_inputs in zip(annotations, dataloader):
        batched_inputs = move_batch_to_device(batched_inputs, 'cuda')
        height = annot['height']
        width = annot['width']
        processed_results = []
        acc_masks = []
        for phrase in annot['phrase']:
            rle = phrase['gsam_output']['mask']
            rle = string_to_rle(rle)
            mask = rle_to_mask(rle, height, width)
            mask = torch.from_numpy(mask).permute(2,0,1)
            acc_masks.append(mask.cuda())

        processed_results.append({})
        acc_masks = torch.cat(acc_masks, dim=0)
        processed_results[-1]['grounding_mask'] = acc_masks
        evaluator.process(batched_inputs, processed_results)
        index += 1
        print(index, len(annotations))
    print(evaluator.evaluate())

if __name__ == "__main__":
    main()
    sys.exit(0)