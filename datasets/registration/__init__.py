register_functions = [
    "register_refcoco_dataset",
    "register_ade20k_full",
    "register_ade20k_panoptic",
    "register_coco_stuff_10k",
    "register_coco_panoptic_annos_semseg",
    "register_coco_panoptic_annos_caption",
    "register_coco_panoptic_annos_caption_grounding",
    "register_coco_lvis_panoptic_annos_caption_grounding",
    "register_coco_lvis_panoptic_annos_caption_grounding_entity",
    "register_ade20k_instance",
    "register_vlp_datasets",
    "register_sunrgbd_semseg",
    "register_scannet_semseg",
    "register_bdd100k_semseg",
    "register_scannet_panoptic",
    "register_bdd100k_panoseg",
    "register_pascalvoc_eval",
    "register_grounding_coco_entity",
    "register_vlp_coco_entity",
    "register_vlp_coco_interleave",
    "register_davis_dataset",
    "register_ytvos_dataset",
    "register_davis_ixeval",
    "register_sbd_eval",
]

for func_name in register_functions:
    try:
        exec(f"from . import {func_name}")
    except Exception as e:
        print(f"Error with {func_name}: {e}")