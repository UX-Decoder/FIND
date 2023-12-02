# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from .point_features import point_sample    
from ..language.loss import vl_similarity

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0, spatial_cost = None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.num_points = num_points
        self.spatial_cost_class = cost_class
        self.spatial_cost_mask = cost_mask
        self.spatial_cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)
            
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def openimage_forward(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_captions"].shape[:2]
        if bs == 0 or len(targets) == 0:
            return None

        neg_class_emb = extra['neg_class_emb']
        neg_hash = extra['neg_hash']
        _, unique_indices = np.unique(neg_hash.cpu().numpy(), return_index=True)
        neg_class_emb = neg_class_emb[unique_indices]
        neg_hash = neg_hash[unique_indices]

        indices = []
        pred_logits = []
        # Iterate through batch size
        for b in range(bs):
            _pos_class_emb = targets[b]['pos_class_emb']
            _pos_hash = targets[b]['pos_hash']
            _neg_overlap_pos = ~(neg_hash[..., None] == _pos_hash).any(-1)
            _neg_class_emb = neg_class_emb[_neg_overlap_pos]
            t_emb = torch.cat((_pos_class_emb, _neg_class_emb))
            v_emb = outputs["pred_captions"][b]            
            del _pos_class_emb
            del _neg_class_emb

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            out_prob = vl_similarity(v_emb, t_emb, temperature=extra['lang_logit'])
            pred_logits += [out_prob]
            out_prob = out_prob.softmax(-1)
            tgt_ids = targets[b]["labels"]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)
            
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ], pred_logits

    @torch.no_grad()
    def grounding_forward(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_gmasks"].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob.softmax(dim=0)

            out_mask = outputs["pred_gmasks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["grounding_masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def spatial_forward(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_smasks"].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_smasks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["gt_spatial_masks"].to(out_mask)
            out_prob = (outputs["pred_pos_logits"][b])[:,:len(tgt_mask)] # remove redundant predictions for padding

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob.softmax(dim=0)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.spatial_cost_mask * cost_mask 
                + self.spatial_cost_class * cost_class 
                + self.spatial_cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


    @torch.no_grad()
    def interleave_forward(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_imasks'].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_imasks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["interleave_masks"].to(out_mask)
            nd,ns = outputs["pred_pos_logits"][b].shape
            out_prob = (outputs["pred_pos_logits"][b])[:,:len(tgt_mask)] # remove redundant predictions for padding
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob.softmax(dim=0)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.spatial_cost_mask * cost_mask 
                + self.spatial_cost_class * cost_class 
                + self.spatial_cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def spatial_forward_pn(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_smasks"].shape[:2]
        
        if bs == 0 or len(targets) == 0:
            return None

        fp_mask = extra['false_positive_mask']
        gt_mask = torch.stack([targets[b]["gt_spatial_masks"] for b in range(bs)])

        indices = []
        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_neg_logits"][b]
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob.softmax(dim=0)

            out_mask = outputs["pred_smasks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = fp_mask[b].to(out_mask)
            ign_mask = (gt_mask[b] | fp_mask[b]).to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            ign_mask = ign_mask[:, None]

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)

            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            ign_mask = point_sample(
                ign_mask,
                point_coords.repeat(ign_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                ign_mask = ign_mask.float()

                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask*ign_mask, tgt_mask*ign_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask*ign_mask, tgt_mask*ign_mask)
            
            # Final cost matrix
            C = (
                self.spatial_cost_mask * cost_mask 
                + self.spatial_cost_class * cost_class 
                + self.spatial_cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def caption_forward_womask(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, _ = outputs["pred_logits"].shape[:2]

        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        t_emb = torch.cat([t['captions'] for t in targets])
        v_emb = outputs['unmatched_pred_captions']
        caption_target_count = np.cumsum([0] + [len(t['captions']) for t in targets])

        # Iterate through batch size
        for b in range(bs):
            v_emb[b] = v_emb[b] / (v_emb[b].norm(dim=-1, keepdim=True) + 1e-7)
            num_queries = len(v_emb[b])
            out_prob = vl_similarity(v_emb[b][None,], t_emb, temperature=extra['temperature']).softmax(-1)[0]
            tgt_ids = [idx for idx in range(caption_target_count[b], caption_target_count[b+1])]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # Final cost matrix
            C = (self.cost_class * cost_class)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def caption_forward_wmask(self, outputs, targets, extra):
        """More memory-friendly matching"""
        bs, _ = outputs["pred_logits"].shape[:2]

        if bs == 0 or len(targets) == 0:
            return None

        indices = []
        t_emb = torch.cat([t['captions'] for t in targets])
        v_emb = outputs['unmatched_pred_captions']
        caption_target_count = np.cumsum([0] + [len(t['captions']) for t in targets])
        
        # Iterate through batch size
        for b in range(bs):
            v_emb[b] = v_emb[b] / (v_emb[b].norm(dim=-1, keepdim=True) + 1e-7)
            num_queries = len(v_emb[b])
            
            out_prob = vl_similarity(v_emb[b][None,], t_emb, temperature=extra['temperature']).softmax(-1)[0]
            tgt_ids = [idx for idx in range(caption_target_count[b], caption_target_count[b+1])]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)
            
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device, dtype=tgt_mask.dtype)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mode='default', extra={}):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if mode == 'default':
            return self.memory_efficient_forward(outputs, targets)
        elif mode == 'grounding':
            return self.grounding_forward(outputs, targets, extra)
        elif mode == 'spatial':
            return self.spatial_forward(outputs, targets, extra)
        elif mode == 'interleave':
            return self.interleave_forward(outputs, targets, extra)
        elif mode == 'spatial_pn':
            return self.spatial_forward_pn(outputs, targets, extra)            
        elif mode == 'caption_womask':
            return self.caption_forward_womask(outputs, targets, extra)
        elif mode == 'caption_wmask':
            return self.caption_forward_wmask(outputs, targets, extra)
        else:
            assert False, "Mode {} is not supported.".format(mode)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
