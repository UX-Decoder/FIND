import copy
import itertools
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class InterleaveEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name=None,
        output_dir=None,
        distributed=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._distributed = distributed

    def reset(self):
        self._image_embeds = []
        self._image_ids = []

        self._interleave_entities = []
        self._object_queries_semantic = []
        self._interleave_embeds = []
        self._interleave_ids = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for output in outputs:
            self._image_ids.append(output['caption']['image_ids'])
            self._image_embeds.append(output['caption']['image_embeds'])
            self._object_queries_semantic.append(output['caption']['object_queries_semantic'])
            self._interleave_embeds.append(output['caption']['interleave_embeds'])
            self._interleave_entities.append(output['caption']['interleave_entities'])
            self._interleave_ids.append(output['caption']['interleave_ids'])

    def evaluate(self, img_ids=None):
        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._image_embeds[0].device) for xx in x]
                return x

            image_embeds = gather(self._image_embeds, move=True) # [(1,1,512)]
            image_ids = gather(self._image_ids)

            interleave_embeds = gather(self._interleave_embeds, move=True) # [(1,1,512)]
            interleave_entities = gather(self._interleave_entities, move=True) # [(1,be,512)]
            interleave_ids = gather(self._interleave_ids, move=True) # [(1,be,512)]
            object_queries_semantic = gather(self._object_queries_semantic, move=True) # [(1,100,512)]

            if not comm.is_main_process():
                return {}
        else:
            assert False, "Not implemented"

        id_to_image_dict = {img_id: img_emb for img_id, img_emb in zip(image_ids, image_embeds)}
        fiids = torch.tensor(image_ids).view(-1).cuda()
        iids = torch.tensor(list(id_to_image_dict.keys())).cuda()

        interleave_entities = nn.utils.rnn.pad_sequence(interleave_entities, batch_first=True, padding_value=-1)
        interleave_entity_mask = interleave_entities.sum(dim=-1) == (-interleave_entities.shape[-1])
        interleave_entities = interleave_entities / interleave_entities.norm(dim=-1, keepdim=True)
        interleave_ids = nn.utils.rnn.pad_sequence(interleave_ids, batch_first=True, padding_value=-1)
        interleave_embeds = torch.cat(interleave_embeds, dim=0)[:,0]
        interleave_embeds = interleave_embeds / interleave_embeds.norm(dim=-1, keepdim=True)
        object_queries_semantic = torch.cat(object_queries_semantic, dim=0)
        object_queries_semantic = object_queries_semantic / object_queries_semantic.norm(dim=-1, keepdim=True)
        image_embeds = torch.cat(image_embeds, dim=0)[:,0]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        bs,ne,_ = interleave_entities.shape
        bs,no,_ = object_queries_semantic.shape
        interleave_entities = interleave_entities.view(bs*ne,-1)
        object_queries_semantic = object_queries_semantic.view(bs*no,-1)
        interleave_ids = interleave_ids.view(bs*ne)
        eo_scores = []
        msk_scores = []
        for i in range(len(interleave_entities)):
            eo_score = interleave_entities[i:i+1] @ object_queries_semantic.t()
            eo_score = eo_score.reshape(-1,bs,no).max(dim=-1)[0]
            eo_scores += [eo_score]
            msk_scores += [iids == interleave_ids[i]]
        msk_scores = torch.stack(msk_scores, dim=0).float() * -1e-6
        msk_scores = msk_scores.reshape(bs,ne,bs).min(dim=1)[0]
        eo_scores = torch.cat(eo_scores, dim=0)
        eo_scores = eo_scores.reshape(bs,ne,bs)
        eo_scores = (eo_scores * (1 - interleave_entity_mask[:,:,None].float())).sum(dim=1) / ((1 - interleave_entity_mask.float()).sum(dim=1)[:,None])

        ii_scores = interleave_embeds @ image_embeds.t()
        scores = ii_scores + msk_scores + 0.5 * eo_scores

        # compute image to image retrieval
        self._results = OrderedDict()
        self._results['recall'] = {}
        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]
        iir_r10 = (fiids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        iir_r5 = (fiids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        iir_r1 = (fiids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
        # Copy so the caller can do whatever with results
        self._results['recall']['interleave2ir1'] = float("{:.3f}".format(iir_r1.item() * 100))
        self._results['recall']['interleave2ir5'] = float("{:.3f}".format(iir_r5.item() * 100))
        self._results['recall']['interleave2ir10'] = float("{:.3f}".format(iir_r10.item() * 100))
        return copy.deepcopy(self._results)