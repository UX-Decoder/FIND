import sys
import time
import logging
import datetime
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
from kornia.contrib import distance_transform

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.logger import log_every_n_seconds

from segment_anything import build_sam, build_sam_vit_b, SamAutomaticMaskGenerator, build_sam_vit_l, build_sam_vit_h
from segment_anything.utils.amg import MaskData, calculate_stability_score, batched_mask_to_box

from xy_utils.image2html.visualizer import VL
from utils.arguments import load_opt_command
from datasets import build_eval_dataloader, build_evaluator
from modeling.utils import get_iou

# mask_generator = SamAutomaticMaskGenerator(build_sam_vit_b(checkpoint="/nobackup3/xueyan-data/grin_data/output/sam/sam_vit_b_01ec64.pth").cuda())
mask_generator = SamAutomaticMaskGenerator(build_sam_vit_l(checkpoint="/nobackup3/xueyan-data/grin_data/output/sam/sam_vit_l_0b3195.pth").cuda())
# mask_generator = SamAutomaticMaskGenerator(build_sam_vit_h(checkpoint="/mnt/output/xueyanz/pretrained/sam/sam_vit_h_4b8939.pth").cuda())

def sam_interactive_mask(points, in_points, in_labels, mask_input, gt_boxes):
    masks, iou_preds, _ = mask_generator.predictor.predict_torch(
            in_points,
            in_labels,
            boxes=gt_boxes,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=True,
    )
    nm,_,h,w = masks.shape

    # Serialize predictions and store in MaskData
    data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            # points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
    )
    del masks

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
            data["masks"], mask_generator.predictor.model.mask_threshold, mask_generator.stability_score_offset
    )

    masks = data["masks"].reshape(nm, -1, h, w)
    scores = (data['iou_preds'] + data['stability_score']).reshape(nm, -1)

    index = torch.stack([torch.arange(nm).cuda(), scores.argmax(dim=1)]).tolist()
    return masks[index]

def prepare_next_spaital_mask(outputs, batched_inputs):
    gt_masks = [batched_inputs[i]['spatial_query']['gt_masks'] for i in range(len(batched_inputs))]
    gt_masks = torch.stack(gt_masks).bool().transpose(0,1).cuda()

    pred_masks = outputs[:,None] > 0.0
    prev_masks = batched_inputs[0]['spatial_query']['spatial_query_pos_mask'] | batched_inputs[0]['spatial_query']['spatial_query_neg_mask']

    fn = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks) # fn: False Negative, gt:1, pred:0, prev:0
    fp = (~gt_masks & pred_masks) & (~prev_masks) # fp: False Positive, gt:0, pred:1, prev:0

    # compute iou between gt and pred
    iou = (gt_masks & pred_masks).sum(list(range(1,len(fn.shape)))) / ((gt_masks | pred_masks).sum(dim=list(range(1,len(fn.shape)))) + 1e-8)
    fn_sum = fn.sum(dim=list(range(1,len(fn.shape))))
    fp_sum = fp.sum(dim=list(range(1,len(fp.shape))))

    is_postive = fn_sum > fp_sum
    # is_postive = torch.ones(len(fn_sum), device=torch.cuda.current_device()).bool()
    select_mask = torch.stack([fn[i] if is_postive[i] else fp[i] for i in range(len(fn))])

    # conv implementation
    n,_,h,w=select_mask.shape
    mask_dt = (distance_transform((~F.pad(select_mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:,:,1:-1,1:-1]).reshape(n,-1)
    max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]).tolist()
    next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
    next_mask = next_mask.view(n,-1)
    next_mask[max_xy_idx] = True
    next_mask = next_mask.reshape((n,1,h,w))

    pos_mask = []
    neg_mask = []
    for idx, ip in enumerate(is_postive):
        if ip:
            pos_mask += [batched_inputs[0]['spatial_query']['spatial_query_pos_mask'][idx] | next_mask[idx]]
            neg_mask += [batched_inputs[0]['spatial_query']['spatial_query_neg_mask'][idx]]
        else:
            pos_mask += [batched_inputs[0]['spatial_query']['spatial_query_pos_mask'][idx]]
            neg_mask += [batched_inputs[0]['spatial_query']['spatial_query_neg_mask'][idx] | next_mask[idx]]
    
    return {'spatial_query_pos_mask': torch.stack(pos_mask), 'spatial_query_neg_mask': torch.stack(neg_mask), 'rand_shape': next_mask, 'labels': is_postive}


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    opt['STROKE_SAMPLER']['DILATION'] = 1

    # opt['DATASETS']['TEST'] = ["openimage600_val_Box", "openimage600_val_Circle", "openimage600_val_Scribble", "openimage600_val_Polygon"]
    opt['DATASETS']['TEST'] = ["pascalvoc_val_Box"]
    # opt['DATASETS']['TEST'] = ["cocomini_val_Box", "cocomini_val_Circle", "cocomini_val_Scribble", "cocomini_val_Polygon"]

    datasets = opt['DATASETS']['TEST']
    dataloaders = build_eval_dataloader(opt)

    output_folder = opt['SAVE_DIR']
    # VL.initialize(output_folder, 'voc_best')

    with torch.no_grad():
        for dataset, dataloader in zip(datasets, dataloaders):
            evaluator = build_evaluator(opt, dataset, output_folder)
            evaluator.reset()

            total = len(dataloader)
            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
            start_data_time = time.perf_counter()

            for idx, batch in enumerate(dataloader):
                assert len(batch) == 1
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                image = batch[0]['image'].permute(1,2,0).numpy()
                gt_mask = batch[0]['gt_masks_orisize'].cuda()
                mask_generator.predictor.set_image(image)

                batch[0]['spatial_query']['spatial_query_pos_mask'] = batch[0]['spatial_query']['rand_shape'] & False
                batch[0]['spatial_query']['spatial_query_neg_mask'] = batch[0]['spatial_query']['rand_shape'] & False

                im_size = image.shape[:-1]
                all_batch_shape_iou = []
                acc_points = []
                acc_labels = []
                # v_pos_masks = []
                # v_neg_masks = []
                # v_pred_mask = []
                masks = None
                for _iter in range(0, opt['STROKE_SAMPLER']['EVAL']['MAX_ITER']):
                    rand_mask = batch[0]['spatial_query']['rand_shape']
                    gt_boxes = batch[0]['instances'].gt_boxes.tensor.cuda()
                    # v_pos_masks += [(batch[0]['spatial_query']['spatial_query_pos_mask'][0,0]).float().cpu().numpy()]
                    # v_neg_masks += [(batch[0]['spatial_query']['spatial_query_neg_mask'][0,0]).float().cpu().numpy()]
                    assert image.shape[:-1] == rand_mask.shape[2:]

                    points = rand_mask.nonzero()[:,2:].flip(dims=[1])
                    rand_idx = torch.randperm(points.shape[0])[:512]
                    points = points[rand_idx]

                    acc_points += [points]
                    _np = len(points)
                    points = torch.cat(acc_points).cpu().numpy()
                    transformed_points = mask_generator.predictor.transform.apply_coords(points, im_size)
                    in_points = torch.as_tensor(transformed_points, device=mask_generator.predictor.device).reshape(_np,-1,2).transpose(0,1)

                    if 'labels' in batch[0]['spatial_query']:
                        in_labels = batch[0]['spatial_query']['labels'].float().cuda()
                    else:
                        in_labels = torch.ones(points.shape[0], dtype=torch.int, device=mask_generator.predictor.device)

                    acc_labels += [in_labels]
                    in_labels = torch.stack(acc_labels)

                    if masks is not None:
                        masks = (F.interpolate(masks[:,None], (256, 256), mode='bicubic') > 0.0).float()

                    # masks = sam_interactive_mask(points, in_points, in_labels, masks)
                    masks = sam_interactive_mask(None,None,None,None,gt_boxes)
                    # v_pred_mask += [(masks[0]>0.0).float().cpu().numpy()]

                    pred_smask_all = F.interpolate(masks[:,None], gt_mask.shape[1:], mode='bicubic')[:,0] > 0.0
                    all_batch_shape_iou += [get_iou(gt_mask, pred_smask_all)]
                    batch[0]['spatial_query'].update(prepare_next_spaital_mask(masks, batch))

                    if (all_batch_shape_iou[-1] > 0.9).float().mean() == 1:
                        break

                all_batch_shape_iou = torch.stack(all_batch_shape_iou)
                pad_all_batch_shape_iou = torch.ones((opt['STROKE_SAMPLER']['EVAL']['MAX_ITER'], len(all_batch_shape_iou[0])))
                pad_all_batch_shape_iou[:len(all_batch_shape_iou),:] = all_batch_shape_iou
                outputs = [{"mask_iou": pad_all_batch_shape_iou[:,i]} for i in range(len(pad_all_batch_shape_iou[0]))]

                total_compute_time += time.perf_counter() - start_compute_time
                start_eval_time = time.perf_counter()

                evaluator.process(batch, outputs)

                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.WARNING,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

                # VL.step()
                # import cv2
                # v_masks = []
                # v_pos = []
                # v_neg = []
                # txt = []

                # img = batch[0]['image'].permute(1,2,0).cpu().numpy()
                # v_gt_mask = batch[0]['spatial_query']['gt_masks'][0]
                # mask_img = VL.overlay_single_mask_to_image(img[:,:,::-1], v_gt_mask.cpu().float().numpy())
                # for x,y,z,iou in zip(v_pos_masks, v_neg_masks, v_pred_mask, all_batch_shape_iou):
                #     # dilate x,y
                #     x = cv2.dilate(x, np.ones((5,5), np.uint8), iterations=3)
                #     y = cv2.dilate(y, np.ones((5,5), np.uint8), iterations=3)

                #     v_masks += [z]
                #     v_pos += [x.clip(0,1)]
                #     v_neg += [y.clip(0,1)]
                #     txt += ["pred_{}".format(str(iou[0].item())[0:5])]

                # VL.add_image(img[:,:,::-1])
                # VL.insert(mask_img, "gt_mask")
                # VL.overlay_obj_mask_to_image_withposneg(img[:,:,::-1], v_masks, v_pos, v_neg, txt, max_len=20)

            results = evaluator.evaluate()
            print(results)

if __name__ == "__main__":
        main()
        sys.exit(0)