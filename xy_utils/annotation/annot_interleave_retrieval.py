import os
import torch
import glob
import json
import collections
import cv2
import numpy as np
from panopticapi.utils import rgb2id
from detectron2.data import detection_utils as utils
from detectron2.utils.file_io import PathManager
# from image2html.visualizer import VL

coco_emb_root = '/nobackup3/xueyan-data/grin_data/coco/coco_panoptic_emb/val2017'
pano_anno_root = '/nobackup3/xueyan-data/grin_data/coco/panoptic_val2017'
img_root = '/nobackup3/xueyan-data/grin_data/coco/val2017'
coco_emb_pths = sorted(glob.glob(os.path.join(coco_emb_root, '*.da')))

# vl_log_dir = '/nobackup3/xueyan-data/grin_data/visual/coco_interleave_nnobject'
# vl_tag = 'seem_davitd5_deform_maskclass_prob'
# VL.initialize(vl_log_dir, vl_tag, total_step=500)

annot_id_list = []
embeddings_list = []
image_id_list = []
class_id_list = []

for coco_emb_pth in coco_emb_pths:
    coco_emb = torch.load(coco_emb_pth)

    embeddings_list += [coco_emb['embeddings']]
    image_id = coco_emb['image_id']
    annot_list = coco_emb['anno_ids'].tolist()
    annot_list = ["{}_{}".format(x, image_id) for x in annot_list]
    annot_id_list += annot_list

    image_id_list += [image_id] * len(annot_list)
    class_id_list += [coco_emb['classes']]

embeddings = torch.cat(embeddings_list, dim=0)
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
class_id_list = torch.cat(class_id_list, dim=0)
annot_id_list = np.array(annot_id_list)
image_id_list = np.array(image_id_list)

entity_file = '/nobackup3/xueyan-data/grin_data/coco/annotations/entity_val2017_long.json'
json_file = '/nobackup3/xueyan-data/grin_data/coco/annotations/panoptic_val2017.json'

def overlay_mask_on_image_with_idx(image_id, annot_id):
    pano_filename = os.path.join(pano_anno_root, "{}.png".format(str(image_id).zfill(12)))
    img_filename = os.path.join(img_root, "{}.jpg".format(str(image_id).zfill(12)))
    pan_seg_gt = utils.read_image(pano_filename, "RGB")
    pan_seg_gt = rgb2id(pan_seg_gt)
    mask = (pan_seg_gt == annot_id).astype(np.float32).clip(0, 1)
    image = utils.read_image(img_filename, "BGR")
    visual_image = VL.overlay_all_masks_to_image(image, mask[None,])
    return visual_image

# load coco panoptic annotations
with PathManager.open(json_file) as f:
    json_info = json.load(f)

with PathManager.open(entity_file) as f:
    entity_info = json.load(f)

# build dictionary for entity
entity_dict = collections.defaultdict(list)
for entity_ann in entity_info['annotations']:
    image_id = int(entity_ann["image_id"])
    entity_dict[image_id].append(entity_ann)

annot_to_iid = collections.defaultdict(list)
for ann in json_info["annotations"]:
    seg_info = ann['segments_info']
    for seg in seg_info:
        annot_to_iid[seg['id']] = ann['image_id']

# Bug 1, Class type constraint.
# Bug 2, Image id constraint.
annoid_to_annoid = {}
for image_id, entity in entity_dict.items():
    phrases = entity[0]['phrase']
    for phrase in phrases:
        annot_index = "{}_{}".format(phrase['annotation_id'], image_id)
        index = torch.from_numpy(annot_id_list == annot_index)

        if index.sum() == 0:
            print(annot_index)
            continue

        image_id_mask = (torch.from_numpy(image_id_list == image_id))
        class_id_mask = ~(class_id_list == class_id_list[index])

        cur_emb = embeddings[index]
        similarity = torch.matmul(cur_emb, embeddings.t())
        similarity[:,image_id_mask] = -1
        similarity[:,class_id_mask] = -1

        topk_prob, topk_idx = torch.topk(similarity[0], 10, dim=-1)
        selected_index = annot_id_list[topk_idx.cpu().numpy()]
        annoid_to_annoid[annot_index] = {"id": selected_index[0], "prob": topk_prob[0].item()}

        # search_image = overlay_mask_on_image_with_idx(image_id, int(phrase['annotation_id']))
        # VL.step()
        # VL.add_image(search_image)
        # for i in range(len(selected_index)):
        #     annot_id_, image_id_ = selected_index[i].split('_')
        #     candidate_image = overlay_mask_on_image_with_idx(int(image_id_), int(annot_id_))
        #     VL.insert(candidate_image, 'candidate_{}_{}'.format(str(topk_prob[i].item())[0:5], i))

for aid in range(len(entity_info['annotations'])):
    image_id = entity_info['annotations'][aid]['image_id']
    for pid in range(len(entity_info['annotations'][aid]['phrase'])):
        ref_index = "{}_{}".format(entity_info['annotations'][aid]['phrase'][pid]['annotation_id'], image_id)
        if ref_index in annoid_to_annoid:
            entity_info['annotations'][aid]['phrase'][pid]['reference'] = {"image_id": int(annoid_to_annoid[ref_index]['id'].split('_')[1]), "annotation_id": int(annoid_to_annoid[ref_index]['id'].split('_')[0]), "prob": annoid_to_annoid[ref_index]['prob']}

output_folder = '/nobackup3/xueyan-data/grin_data/coco/annotations'
with open(os.path.join(output_folder, 'interleave_val2017_long.json'), 'w') as f:
    json.dump(entity_info, f)