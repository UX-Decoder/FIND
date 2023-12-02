import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_PRETRAIN = {
    "vlp_coco_entity_val_long": (
        "coco/val2017",
        "coco/annotations/entity_val2017_long.json",
    ),
    "vlp_coco_entity_val": (
        "coco/val2017",
        "coco/annotations/entity_val2017.json",
    ),
    "vlp_coco_entity_val_retrieval": (
        "coco/val2017",
        "coco/annotations/entity_val2017.json",
    ),
    "vlp_coco_entity_val_retrieval_long": (
        "coco/val2017",
        "coco/annotations/entity_val2017_long.json",
    ),
}

evaluator_mapper = {'vlp_coco_entity_val': 'retrieval_interleave_text', 'vlp_coco_entity_val_retrieval': 'retrieval', 'vlp_coco_entity_val_retrieval_long': 'retrieval', 'vlp_coco_entity_val_long': 'retrieval_interleave_text'}

def get_metadata(name):
    return {}

def load_pretrain_data(image_root, entity_root, meta, name):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(entity_root) as f:
        entity_info = json.load(f)
        
    # build dictionary for entity
    entity_dict = collections.defaultdict(list)
    for entity_ann in entity_info['annotations']:
        image_id = int(entity_ann["image_id"])
        entity_dict[image_id].append(entity_ann)
    
    image_dict = collections.defaultdict(list)
    for image_ann in entity_info['images']:
        image_id = int(image_ann["id"])
        image_dict[image_id] = image_ann['file_name']

    ret = []
    for image_id in entity_dict.keys():
        file_name = os.path.join(image_root, image_dict[image_id])
        ret.append({
            "file_name": file_name,
            "image_id": image_id,
            "captions": [entity_dict[image_id][i]['sentence'] for i in range(len(entity_dict[image_id]))],
        })
    return ret

def register_pretrain(
    name, metadata, image_root, entity_root,
):
    semantic_name = name
    DatasetCatalog.register(
        semantic_name,
        lambda: load_pretrain_data(image_root, entity_root, metadata, name),
    )
    MetadataCatalog.get(semantic_name).set(
        evaluator_type=evaluator_mapper[semantic_name],
        **metadata,
    )

def register_all_pretrain(root):
    for (
        prefix,
        (image_root, entity_root,),
    ) in _PREDEFINED_SPLITS_PRETRAIN.items():
        register_pretrain(
            prefix,
            get_metadata(prefix),
            os.path.join(root, image_root),
            os.path.join(root, entity_root),
        )


_root = os.getenv("DATASET", "datasets")
register_all_pretrain(_root)
