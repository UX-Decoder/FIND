import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_PRETRAIN = {
    "grounding_coco_entity_val": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/entity_val2017.json",
    ),
    "grounding_coco_entity_val_long": (
        "coco/val2017",
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/annotations/entity_val2017_long.json",
    ),
}

evaluator_mapper = {'grounding_coco_entity_val': 'grounding_entity', 'grounding_coco_entity_val_long': 'grounding_entity'}

def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

def load_pretrain_data(image_root, panoptic_root, panoptic_json, entity_json, metadata, name):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(panoptic_json) as f:
        panoptic_info = json.load(f)

    with PathManager.open(entity_json) as f:
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
    for ann in panoptic_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_root, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(panoptic_root, ann["file_name"])
        segments_info = [_convert_category_id(x, metadata) for x in ann["segments_info"]]
        entity_anno = entity_dict[image_id] if image_id in entity_dict else None

        if entity_anno is None:
            continue

        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "entities": entity_anno,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    return ret

def register_pretrain(
    name, metadata, image_root, panoptic_root, panoptic_json, entity_json,
):
    semantic_name = name
    MetadataCatalog.get(semantic_name).set(
        evaluator_type=evaluator_mapper[semantic_name],
        **metadata,
    )
    DatasetCatalog.register(
        semantic_name,
        lambda: load_pretrain_data(image_root, panoptic_root, panoptic_json, entity_json, metadata, name),
    )

def register_all_pretrain(root):
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, entity_json,),
    ) in _PREDEFINED_SPLITS_PRETRAIN.items():
        register_pretrain(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, entity_json),
        )


_root = os.getenv("DATASET", "datasets")
register_all_pretrain(_root)
