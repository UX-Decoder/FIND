import sys

pth = '/'.join(sys.path[0].split('/')[:-2])
sys.path.insert(0, pth)

import sys
import os
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr
from gradio import processing_utils
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data import MetadataCatalog

from utils.arguments import load_opt_command
from utils.dataset import Entity
from utils.visualizer import Visualizer
from modeling.language import build_tokenizer
from trainer import XDecoder_Trainer as Trainer
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

colors = [
    [128, 0, 0],     # Dark Red
    [0, 128, 0],     # Dark Green
    [0, 0, 128],     # Dark Blue
    [128, 128, 0],   # Dark Yellow (Olive)
    [0, 128, 128],   # Dark Cyan (Teal)
    [128, 0, 128],   # Dark Magenta (Purple)
    [128, 64, 0],    # Dark Orange
    [64, 0, 64],     # Very Dark Purple
    [0, 64, 64],     # Very Dark Cyan
    [64, 64, 0],     # Very Dark Yellow
    [64, 0, 0],      # Very Dark Red
    [0, 64, 0],      # Very Dark Green
    [0, 0, 64],      # Very Dark Blue
]


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

def draw_instances(image, instances):
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
        color = colors[i]

        # Draw the bounding box
        x1, y1, x2, y2 = instances.gt_boxes.tensor[i].int().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        color = np.array(color)[None, None, :]
        mask = instances.gt_masks.tensor[i].numpy()[:,:,None]
        overlay = mask * color
        image = image * (1 - mask) + (image * mask * 0.5 + overlay * 0.5)
    return image

def main(args=None):
    '''
    build args
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    pretrained_pth = '/nobackup3/xueyan-data/grin_data/output/grin_davitd5_enc6_fpn_dec10_unicl_lang_bsc192_cis640_ep50_fbTrue_flTrue_feTrue_mkTrue_capFalse_intTrue_inlTrue_retTrue_grdTrue_inlp0.5_maxsi0_tg6_ts6_tr6_ti6_ltTrue_qclass_qgrd_qmspa_qint/00029300/'
    # pretrained_pth = '/nobackup3/xueyan-data/grin_data/output/grin_focalt_enc6_fpn_dec10_unicl_lang_bsc192_cis640_ep50_fbTrue_flTrue_feTrue_mkTrue_capFalse_intTrue_inlTrue_retTrue_grdTrue_inlp0.5_maxsi0_tg6_ts6_tr6_ti6_qclass_qgrd_qmspa_qint_fix3/00029300/'
    database_root = "../../data/output/database"
    coco_folders = ["/nobackup3/xueyan-data/grin_data/coco/train2017", "/nobackup3/xueyan-data/grin_data/coco/val2017", database_root]
    add_image_pths = []
    add_image_id = 0

    # hard code interactive token number
    opt['MODEL']['DECODER']['MAX_SPATIAL_LEN'] = [128, 128, 128, 128]

    trainer = Trainer(opt)
    raw_models = trainer.pipeline.initialize_model()
    model = raw_models['default'].from_pretrained(os.path.join(pretrained_pth, 'default', 'model_state_dict.pt')).eval()
    model = model.cuda()
    model.model.sem_seg_head.predictor.lang_encoder.activate()
    dataloader = trainer.pipeline.get_dataloaders(trainer, 'vlp_coco_entity_val', is_evaluation=True)

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model.model.get_class_embeddings(["default", "default"], is_eval=True)
            image_ids = []
            image_class_emb_list = []
            object_class_emb_list = []
            object_pixel_emb_list = []

            for idx, batched_inputs in enumerate(dataloader):
                batched_inputs = move_batch_to_device(batched_inputs, 'cuda')
                batched_inputs = cast_batch_to_half(batched_inputs)

                images = [x["image"].to(model.model.device) for x in batched_inputs]
                images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                images = ImageList.from_tensors(images, model.model.size_divisibility)
                img_bs = images.tensor.shape[0]
                
                targets = targets_grounding = queries_grounding = None
                features = model.model.backbone(images.tensor)
                outputs = model.model.sem_seg_head(features, target_queries=queries_grounding)
                image_ids += [x['image_id'] for x in batched_inputs]
                image_class_emb_list += [outputs['pred_retrievals'][:,0]]
                object_class_emb_list += [outputs['pred_captions']]
                object_pixel_emb_list += [outputs['pred_maskembs']]

                # if idx == 30:
                #     break

            image_embs_class = torch.cat(image_class_emb_list, dim=0)
            image_embs_class = image_embs_class / image_embs_class.norm(dim=-1, keepdim=True)
            object_embs_pixel = torch.cat(object_pixel_emb_list, dim=0)
            # object_embs_pixel = object_embs_pixel / object_embs_pixel.norm(dim=-1, keepdim=True)
            object_embs_class = torch.cat(object_class_emb_list, dim=0)
            object_embs_class = object_embs_class / object_embs_class.norm(dim=-1, keepdim=True)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')

    # build language tokenizer
    tokenizer_cfg = opt['MODEL']['TEXT']
    tokenizer_cfg['TOKENIZER'] = 'clip-token'
    tokenizer = build_tokenizer(tokenizer_cfg)

    def add_image(*args):
        nonlocal model, image_embs_class, object_embs_pixel, object_embs_class, image_ids, transform, metadata, add_image_pths, add_image_id

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                image_class_emb_list = []
                object_class_emb_list = []
                object_pixel_emb_list = []
                for idx, _input in enumerate(args):
                    if _input is not None:
                        pil_image = Image.fromarray(_input)
                        save_pth = os.path.join(database_root, str(add_image_id).zfill(12) + "_add.jpg")
                        pil_image.save(save_pth)
                        add_image_pths += [save_pth]
                        image_ori = transform(pil_image)
                        image_ori = np.asarray(image_ori)
                        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
                        images = [images.to(model.model.device)]
                        images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                        images = ImageList.from_tensors(images, model.model.size_divisibility)
                        
                        targets = targets_grounding = queries_grounding = None
                        features = model.model.backbone(images.tensor)
                        outputs = model.model.sem_seg_head(features, target_queries=queries_grounding)
                        image_class_emb_list += [outputs['pred_retrievals'][:,0]]
                        object_class_emb_list += [outputs['pred_captions']]
                        object_pixel_emb_list += [outputs['pred_maskembs']]
                        image_ids += [save_pth]
                        add_image_id += 1

                if len(image_class_emb_list) > 0:
                    add_image_class_embs = torch.cat(image_class_emb_list, dim=0)
                    add_image_class_embs = add_image_class_embs / add_image_class_embs.norm(dim=-1, keepdim=True)
                    image_embs_class = torch.cat([image_embs_class, add_image_class_embs], dim=0)

                    add_object_class_embs = torch.cat(object_class_emb_list, dim=0)
                    add_object_class_embs = add_object_class_embs / add_object_class_embs.norm(dim=-1, keepdim=True)
                    object_embs_class = torch.cat([object_embs_class, add_object_class_embs], dim=0)

                    add_object_pixel_embs = torch.cat(object_pixel_emb_list, dim=0)
                    # add_object_pixel_embs = add_object_pixel_embs / add_object_pixel_embs.norm(dim=-1, keepdim=True)
                    object_embs_pixel = torch.cat([object_embs_pixel, add_object_pixel_embs], dim=0)

    def inference(image_content, text_content, *args):
        nonlocal model, image_embs_class, object_embs_pixel, object_embs_class, image_ids, transform, metadata, add_image_pths

        # offset 0 is entity_text, offset 1 is entity_image, offset 2 is connection
        sentence = ''
        char_num = 0
        entities = []
        sentence_list = []
        for index, _input in enumerate(args):
            valid = False
            if (index % 3) == 1 and _input is not None and _input != '':
                valid = True
                _text = "[INTERACTIVE]"

                _start_idx = char_num
                _end_idx = char_num + len(_text) + 1
                char_num += len(_text) + 1

                _type = 'visual'
                _mask = None

                image_ori = transform(Image.fromarray(_input['image']))
                mask_ori = Image.fromarray(_input['mask'])
                width = image_ori.size[0]
                height = image_ori.size[1]
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
                images = [images.to(model.model.device)]
                images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
                _image = ImageList.from_tensors(images, model.model.size_divisibility)
                _height, _width = _image.tensor.shape[-2:]

                mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
                mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[None,]
                mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0)[0]
                _interactive = torch.zeros((1, _height, _width), dtype=torch.bool)
                _interactive[:, :height, :width] = mask_ori
                sentence += _text + ' '
                sentence_list += [_text + ' ']

            elif (index % 3) == 0 and _input is not None and _input != '':
                valid = True
                _text = _input

                _start_idx = char_num
                _end_idx = char_num + len(_text) + 1
                char_num += len(_text) + 1

                _type = 'text'
                _image = None
                _mask = None
                _interactive = None

                sentence += _text + ' '
                sentence_list += [_text + ' ']

            elif (index % 3) == 2 and _input is not None and _input != '':
                _text = _input
                char_num += len(_text) + 1
                sentence += _text + ' '

            if valid:
                _id = index
                entities += [Entity(_id, _text, _mask, _interactive, _type, _start_idx, _end_idx, _image)]

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                sentence = sentence + '.'
                tokens, start_end_condition = tokenizer(sentence, sentence_list)
                data = {}
                data['entities'] = {"entities": entities, "sentence": sentence, "tokens": tokens, "entity_to_tokens": start_end_condition.cuda()}
                outputs, extra = model.model.demo_interleave([data])

                # interleave retrieval mode
                # qc_emb = outputs["pred_entity_class"]
                # i_emb_it = qc_emb / qc_emb.norm(dim=-1, keepdim=True)
                # object_embs_class = object_embs_class / object_embs_class.norm(dim=-1, keepdim=True)

                # bs,no,nd = object_embs_class.shape
                # nq = i_emb_it.shape[1]
                # ii_scores = (i_emb_it @ object_embs_class.reshape(1, bs*no, nd).transpose(1,2)).reshape(nq, bs, no).max(dim=-1)[0]

                # interleave proposals mode
                ii_scores_list = []
                for idx, entity in enumerate(entities):
                    if entity.type == 'visual':
                        interleave_class_proposals = outputs["interleave_class_proposals"][idx]
                        interleave_pixel_proposals = outputs["interleave_pixel_proposals"][idx]
                        interleave_pixel_query = outputs["interleave_pixel_query"][idx]
                        interleave_class_query = outputs["interleave_class_query"][idx]

                        s_emb = interleave_pixel_query
                        v_emb = interleave_pixel_proposals
                        pred_logits = v_emb @ s_emb.transpose(1,2)
                        selected_v_idx = pred_logits.max(dim=1)[1][0]

                        c_emb = interleave_class_proposals
                        i_emb_it = c_emb[:, selected_v_idx]
                        i_emb_it = i_emb_it / i_emb_it.norm(dim=-1, keepdim=True)

                        bs,no,nd = object_embs_class.shape
                        nq = i_emb_it.shape[1]
                        ii_scores_list += [(i_emb_it @ object_embs_class.reshape(1, bs*no, nd).transpose(1,2)).reshape(nq, bs, no).max(dim=-1)[0]]
                    else:
                        interleave_class_query = outputs["interleave_class_query"][idx]
                        i_emb_it = interleave_class_query / interleave_class_query.norm(dim=-1, keepdim=True)

                        bs,no,nd = object_embs_class.shape
                        nq = i_emb_it.shape[1]
                        ii_scores_list += [(i_emb_it @ object_embs_class.reshape(1, bs*no, nd).transpose(1,2)).reshape(nq, bs, no).max(dim=-1)[0]]

                ii_scores_entity = torch.cat(ii_scores_list, dim=0).mean(dim=0, keepdim=True)

                interleave_emb = outputs['pred_interleave_image'][0]
                i_emb_it = interleave_emb / interleave_emb.norm(dim=-1, keepdim=True)
                ii_scores_class = i_emb_it @ image_embs_class.t()

                ii_scores = ii_scores_class + ii_scores_entity

                topk_ids = ii_scores.topk(20, dim=-1).indices[0]
                selected_image_ids = [image_ids[x] for x in topk_ids]

                image_gallery = []
                for image_id in selected_image_ids:
                    for coco_folder in coco_folders:

                        if str(image_id).endswith(".jpg"):
                            image_path = image_id                        
                        else:
                            image_path = os.path.join(coco_folder, f'{image_id:012d}.jpg')

                        if os.path.exists(image_path):
                            # apply segmentation to selected images
                            image = Image.open(image_path).convert("RGB")
                            image = transform(image)
                            image_ori = np.asarray(image)
                            height, width = image_ori.shape[:2]
                            images = torch.from_numpy(image_ori.copy()).permute(2,0,1)

                            data = [{"image": images, "height": height, "width": width}]
                            outputs = model.model.demo_interleave_grounding(data, extra)
                            phrases = [{"text": x.text, "color": colors[idx%len(colors)]} for idx, x in enumerate(entities)]

                            pred_masks = outputs['pred_masks']

                            # visual_mask = pred_masks.cpu().numpy()
                            # for i in range(visual_mask.shape[0]):
                            #     cv2.imwrite(f"visual_mask_{i}.png", visual_mask[i] * 255)
                            # import pdb; pdb.set_trace()

                            image_shape = image_ori.shape[:2]
                            instances = Instances(image_shape)

                            if len(pred_masks) == 0:
                                # Some image does not have annotation (all ignored)
                                masks = BitMasks(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])))
                                instances.gt_masks = masks
                                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                            else:
                                masks = BitMasks(pred_masks.cpu())
                                instances.gt_masks = masks
                                instances.gt_boxes = masks.get_bounding_boxes()

                            instances.gt_classes = [x.text for x in entities]
                            canvas = draw_instances(image_ori.copy(), instances)
                            canvas = put_text(canvas, phrases)
                            image_gallery += [Image.fromarray(canvas).convert("RGB")]
                            break

        return image_gallery

    # lastidx is the index to begin the offset from
    def change_visibility(inputType, last_idx):
        if inputType == 'clear':
            return [0] + [gr.update(visible=False)] * max_item * 3
        elif inputType == 'entity_text':
            offset = 0
        elif inputType == 'entity_image':
            offset = 1
        elif inputType == 'text':
            offset = 2

        if last_idx < max_item * 3:
            #Updates nothing for previous inputs, updates the new input to visible, and updates nothing for the rest of the inputs
            return [last_idx + 3] + [gr.update()] * (last_idx + offset) + [gr.update(visible=True, interactive=True)] * (1) + [gr.update()] * (3 * max_item - last_idx - offset - 1)
        else:
            return [last_idx] + [gr.update()] * max_item * 3

    customCSS = """
    .grid-container.svelte-1b19cri.svelte-1b19cri {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }
    .thumbnail-lg.svelte-1b19cri.svelte-1b19cri{
        width: unset;
        height: 30vh;
        aspect-ratio: auto;
    }
    """

    # maxitem is the max number of visible inputs
    max_item = 9

    #Holds all the inputs
    input_list = []
    for i in range(max_item):
        entity_text = gr.Textbox(visible=False, label="Text Entity")
        input_list.append(entity_text)
        entity_image = gr.Image(visible=False, tool='sketch', label="Image Entity")
        input_list.append(entity_image)
        text = gr.Textbox(visible=False, label="Connection")
        input_list.append(text)

    # initialize gallery
    gallery_output = gr.Gallery(label="Image Gallery.")

    with gr.Blocks(css=customCSS) as demo:
        gr.Markdown(f"# Grounded Interleaved Visual Understanding")
        gr.Markdown(f"The front-end is powered by [Arul Aravinthan](https://www.linkedin.com/in/arul-aravinthan-414509218).")

        example = gr.Examples(
            examples=[
                    ["A green and white leafy potted plant", None, None, None, None, "sits near", "a cozy bed with a blue comforter", None, None, None, None, "creating a relaxing atmosphere in the bedroom"]
                    # ["Some fries", None, None, None, None, "is placed next to", None, "xy_utils/fries1.png", None],
                    ],
            inputs=[*input_list],
            outputs=gallery_output,
            cache_examples=False,
        )


        gr.Markdown(f"### 👉 Query.")
        with gr.Row():
            last_idx = gr.State(value=0)
            entity_text_state = gr.State(value='entity_text')
            entity_image_state = gr.State(value='entity_image')
            text_state = gr.State(value='text')
            clear_state = gr.State(value='clear')

            for component in input_list:
                component.render()

        with gr.Row():
            addtxt_entity = gr.Button("Add Text Entity")
            addimg_entity = gr.Button("Add Image Entity")
            addtxt = gr.Button("Add Connection")
            clear = gr.ClearButton(value="Clear all inputs", components=input_list)
            addtxt_entity.click(change_visibility, [entity_text_state, last_idx], [
                            last_idx, *input_list])
            addimg_entity.click(change_visibility, [entity_image_state, last_idx], [
                            last_idx, *input_list])
            addtxt.click(change_visibility, [text_state, last_idx], [
                            last_idx, *input_list])
            clear.click(change_visibility, [clear_state, last_idx], [
                            last_idx, *input_list])

        gr.Markdown(f"### 👉 Content.")
        with gr.Row():
            image_content = gr.Image(label="Image Content.")
            text_content = gr.Textbox(label="Text Content.", lines=7)

        gr.Markdown(f"### 👉 Search Space.")
        with gr.Row():
            gr.CheckboxGroup(["Image", "Text", "Dataset"], label="Search Space."),
            run = gr.Button("Run")
            run.click(inference, [image_content, text_content] + [*input_list], [gallery_output])

        gr.Markdown(f"### 👉 Results.")
        gallery_output.render()

        gr.Markdown(f"### 👉 Upload new images to the database.")
        with gr.Row():
            input_image1 = gr.Image(label="Add Image 1.")
            input_image2 = gr.Image(label="Add Image 2.")
            input_image3 = gr.Image(label="Add Image 3.")
            input_image4 = gr.Image(label="Add Image 4.")
            input_image5 = gr.Image(label="Add Image 5.")
            input_image6 = gr.Image(label="Add Image 6.")
            addimage = gr.Button("Upload")
            addimage.click(add_image, [input_image1, input_image2, input_image3, input_image4, input_image5, input_image6])

    demo.launch(server_port=6036)

if __name__ == "__main__":
    main()
    sys.exit(0)