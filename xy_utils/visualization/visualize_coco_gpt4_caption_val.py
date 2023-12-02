import os
import glob
import torch
import re
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode

from xy_utils.image2html.visualizer import VL

import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)


colors = [
    [255, 0, 0],     # Bright Red
    [0, 255, 0],     # Bright Green
    [0, 0, 255],     # Bright Blue
    [255, 255, 0],   # Bright Yellow
    [0, 255, 255],   # Bright Cyan
    [255, 0, 255],   # Bright Magenta
    [255, 128, 0],   # Bright Orange
    [255, 0, 127],   # Bright Pink
    [0, 255, 127],   # Bright Spring Green
    [127, 255, 0],   # Bright Lime
    [127, 0, 255],   # Bright Violet
    [255, 127, 0],   # Bright Amber
    [0, 127, 255],   # Bright Sky Blue
]

def split_sentence(sentence, phrases):
    for phrase in phrases:
        sentence = sentence.replace(phrase, "SPLIT_HERE" + phrase + "SPLIT_HERE")
        
    parts = sentence.split("SPLIT_HERE")
    # Remove empty strings
    parts = [part for part in parts if part]
    return parts


def put_text(image_draw, phrases):
    # image_draw is your actual image.
    # phrases is a list of dictionaries with 'text' and 'color' keys

    # Choose your font
    font = cv2.FONT_HERSHEY_DUPLEX
    # Choose the size of your font
    font_scale = 0.4
    # Choose the thickness of the font
    thickness = 1

    # set the text start position
    text_offset_x = 10
    text_offset_y = image_draw.shape[0] + 10  # start from 10 pixels below the bottom of the image

    # Initialize some variables for the maximum text width and the total text height
    total_text_height = 10  # start with padding for the bottom of the image

    # Calculate total_text_height
    for phrase in phrases:
        _str = phrase['text']
        (_, text_height) = cv2.getTextSize(_str, font, font_scale, thickness)[0]
        total_text_height += text_height + 10  # add padding between phrases

    # Make a canvas to fit the image and the text
    canvas = np.ones((image_draw.shape[0] + total_text_height, image_draw.shape[1], 3), dtype='uint8') * 248

    # Copy the image to the canvas
    canvas[:image_draw.shape[0], :image_draw.shape[1]] = image_draw

    # Add each phrase to the canvas
    for phrase in phrases:
        _str = phrase['text']
        color = phrase['color']

        # Use getTextSize to get the width and height of the text box
        (_, text_height) = cv2.getTextSize(_str, font, font_scale, thickness)[0]

        # Add text to the canvas
        cv2.putText(canvas, _str, (text_offset_x, text_offset_y + text_height), font, font_scale, color, thickness)

        # Update the y position for next text
        text_offset_y += text_height + 10  # 10 is the vertical space between phrases

    return canvas


# def put_text(image_draw, phrases):
#     # image_draw is your actual image.
#     # phrases is a list of dictionaries with 'text' and 'color' keys

#     # Choose your font
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     # Choose the size of your font
#     font_scale = 0.4
#     # Choose the thickness of the font
#     thickness = 1

#     # set the text start position
#     text_offset_x = 10
#     text_offset_y = image_draw.shape[0] - 10  # 10 pixels from the bottom

#     # Make a canvas to fit the image and the text
#     canvas = np.zeros((image_draw.shape[0] + 20, image_draw.shape[1], 3), dtype='uint8')

#     # Copy the image to the canvas
#     canvas[:image_draw.shape[0], :image_draw.shape[1]] = image_draw

#     # Add each phrase to the canvas
#     for phrase in phrases:
#         _str = phrase['text']
#         color = phrase['color']

#         # Use getTextSize to get the width and height of the text box
#         (text_width, text_height) = cv2.getTextSize(_str, font, font_scale, thickness)[0]

#         # Make sure the canvas is wide enough for the text
#         if text_offset_x + text_width > canvas.shape[1]:
#             canvas = np.pad(canvas, ((0, 0), (0, text_offset_x + text_width - canvas.shape[1]), (0, 0)), mode='constant')

#         # Add text to the canvas
#         cv2.putText(canvas, _str, (text_offset_x, text_offset_y + text_height + 15), font, font_scale, color, thickness)
        
#         # Update the x position for next text
#         text_offset_x += text_width

#     return canvas


def draw_instances(image, instances, alpha=0.5, color_id=None):
    """
    Draw bounding boxes, overlay masks and add class labels on the image.

    Parameters:
    - image: numpy array of shape (H, W, C)
    - instances: an object with the following attributes
        - gt_masks: a tensor of shape (N, H, W), where N is the number of instances
        - gt_boxes: a tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
        - gt_classes: a list of N class labels
    """

    # For each instance
    for i in range(len(instances.gt_classes)):
        # Get a random color
        color = colors[i] if color_id is None else colors[color_id]

        # Draw the bounding box
        x1, y1, x2, y2 = instances.gt_boxes.tensor[i].int().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        color = np.array(color)[None, None, :]
        mask = instances.gt_masks.tensor[i].numpy()[:,:,None]
        overlay = mask * color
        image = image * (1 - mask) + (image * mask * alpha + overlay * alpha)
    return image[..., ::-1]

def get_panoptic_instance(image, panoptic_pth, phrases):
    pan_seg_gt = utils.read_image(panoptic_pth, "RGB")
    segments_info = imageid_to_annot[image_id]["segments_info"]

    from panopticapi.utils import rgb2id
    pan_seg_gt = rgb2id(pan_seg_gt)

    image_shape = image.shape[:2]
    instances = Instances(image_shape)

    texts = []
    masks = []
    for phrase in phrases:
        mask = (pan_seg_gt == int(phrase['annotation_id']))
        masks.append(mask)
        texts.append(phrase['phrase'])

    if len(masks) == 0:
        # Some image does not have annotation (all ignored)
        masks = BitMasks(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])))
        instances.gt_masks = masks
        instances.gt_boxes = Boxes(torch.zeros((0, 4)))
    else:
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
        )
        instances.gt_masks = masks
        instances.gt_boxes = masks.get_bounding_boxes()

    instances.gt_classes = texts
    return instances

entity_root = "/nobackup3/xueyan-data/grin_data/coco/annotations/entity_val2017.json"
panoptic_root = "/nobackup3/xueyan-data/grin_data/coco/annotations/panoptic_val2017.json"
interleave_root = "/nobackup3/xueyan-data/grin_data/coco/annotations/interleave_val2017.json"
panoptic_annot_root = "/nobackup3/xueyan-data/grin_data/coco/panoptic_val2017"
panoptic_image_root = "/nobackup3/xueyan-data/grin_data/coco/val2017"
output_folder = "/nobackup3/xueyan-data/grin_data/visual/visual_coco_{}".format(entity_root.split('/')[-1].split('.')[0])
coco_panoptic = json.load(open(panoptic_root))
coco_entity = json.load(open(entity_root))
coco_interleave = json.load(open(interleave_root))

vl_tag = 'pass5'
VL.initialize(output_folder, vl_tag, total_step=100)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

imageid_to_annot = {}
for annot in coco_panoptic['annotations']:
    imageid_to_annot[annot['image_id']] = annot

imageid_to_entity = {}
for annot in coco_entity['annotations']:
    imageid_to_entity[annot['image_id']] = annot

imageid_to_interleave = {}
for annot in coco_interleave['annotations']:
    imageid_to_interleave[annot['image_id']] = annot

for index, image_id in enumerate(list(imageid_to_annot.keys())[400:]):
    print(index, len(imageid_to_annot))
    annot = imageid_to_annot[image_id]
    entity = imageid_to_entity[image_id]
    interleave = imageid_to_interleave[image_id]

    # entity = torch.load(entity_path)
    # max_key = max(entity.keys())
    # entity = entity[max_key]
    # response = entity['response']
    # raw_sentence = entity['response']['choices'][0]['message']['content']
    raw_sentence = entity['sentence_raw']
    entities = re.findall(r'\[(\d+)\]\<(.*?)\>', raw_sentence)
    phrase = [{"annotation_id": eid, "phrase": text, "phrase_raw": "[{}]<{}>".format(eid, text)} for eid, text in entities]

    sentence = raw_sentence
    # Replace ids and tags with just the text 
    for eid, text in entities:
        sentence = sentence.replace("[{}]<{}>".format(eid, text), text)

    result = {
        "image_id": image_id,
        "sentence": sentence,
        "sentence_raw": raw_sentence,
        "phrase": phrase
    }

    image_path = os.path.join(panoptic_image_root, imageid_to_annot[image_id]['file_name'].replace('png', 'jpg'))
    image = utils.read_image(image_path, "RGB")

    vl_clock = 0
    VL.step()
    VL.add_image(image[..., ::-1])

    panoptic_pth = os.path.join(panoptic_annot_root, imageid_to_annot[image_id]['file_name'])
    instances = get_panoptic_instance(image, panoptic_pth, result['phrase'])

    image = draw_instances(image, instances)

    write_phrases = split_sentence(result['sentence_raw'], [x['phrase_raw'] for x in result['phrase']])

    if write_phrases[-1] == '.':
        write_phrases = write_phrases[:-1]

    split_phrases = [x['phrase_raw'] for x in result['phrase']]
    
    idx = 0
    _write_phrases = []
    for x in write_phrases:
        if x in split_phrases:
            _write_phrases += [{'phrase_raw': x, 'color': colors[idx][::-1]}]
            idx += 1
        else:
            _write_phrases += [{'phrase_raw': x, 'color': [0,0,0]}]

    _write_phrases = [{'text': x['phrase_raw'], 'color': x['color']} for idx, x in enumerate(_write_phrases)]
    image_text = put_text(image, _write_phrases)

    VL.insert(image, 'mask')
    VL.insert(image_text, 'mask_text')
    vl_clock += 2

    # loop through interleave image
    for j, interleave_entity in enumerate(interleave['phrase']):
        if 'reference' not in interleave_entity:
            continue
        refer_image_id = interleave_entity['reference']['image_id']
        refer_annot_id = interleave_entity['reference']['annotation_id']
        refer_phrase = interleave_entity['phrase']

        refer_image_path = os.path.join(panoptic_image_root, imageid_to_annot[refer_image_id]['file_name'].replace('png', 'jpg'))
        refer_image = utils.read_image(refer_image_path, "RGB")

        phrases = [{"annotation_id": refer_annot_id, "phrase": refer_phrase, "phrase_raw": refer_phrase}]

        refer_panoptic_pth = os.path.join(panoptic_annot_root, imageid_to_annot[refer_image_id]['file_name'])
        refer_instances = get_panoptic_instance(refer_image, refer_panoptic_pth, phrases)

        refer_image = draw_instances(refer_image, refer_instances, alpha=0.3, color_id=j)
        refer_write_phrases = [{'text': refer_phrase, 'color': colors[j][::-1]}]
        refer_image_text = put_text(refer_image, refer_write_phrases)

        VL.insert(refer_image, 'interleave_{}_mask'.format(j))
        VL.insert(refer_image_text, 'interleave_{}_mask_text'.format(j))
        vl_clock += 2

    for idx in range(vl_clock, 20):
        VL.insert(np.zeros(image.shape), 'none_{}'.format(idx))

    # image_folder = os.path.join(output_folder, str(image_id).zfill(12))
    # output_path = os.path.join(output_folder, "{}.png".format(str(image_id).zfill(12)))
    # cv2.imwrite(output_path, image)
