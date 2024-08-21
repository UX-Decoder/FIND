import sys
import os

from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data import MetadataCatalog
from utils.dataset import Entity
from transformers import AutoTokenizer

from utils.arguments import load_opt_command
from trainer import XDecoder_Trainer as Trainer
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils import draw_text_on_image_with_score, put_text, draw_instances, colors


def main(args=None):
    '''
    build args
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    pretrained_pth = opt["RESUME_FROM"]
    debug = opt["FAKE_UPDATE"]
    data_root = os.getenv('DETECTRON2_DATASETS')
    coco_folders = [os.path.join(data_root, "coco/train2017"), os.path.join(data_root, "coco/val2017")]

    # The code support add your own database, feel free to uncomment and figure it out.
    add_image_pths = []
    # database_root = "/tmp/database"
    # add_image_id = 0

    # hard code interactive token number
    opt['DATASETS']['TEST'] += ['vlp_coco_entity_val_long']

    trainer = Trainer(opt)
    raw_models = trainer.pipeline.initialize_model()
    model = raw_models['default'].from_pretrained(pretrained_pth).eval()
    model = model.cuda()
    
    # build language tokenizer
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    interleave_text_list = []
    interleave_entity_list = []

    dataloader = trainer.pipeline.get_dataloaders(trainer, 'vlp_coco_entity_val_long', is_evaluation=True)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model.model.get_class_embeddings(["default", "default"], is_eval=True)

            for bidx, batched_inputs in enumerate(dataloader):
                sentence = batched_inputs[0]['captions'][0]
                tokens = tokenizer(
                    [sentence], padding='max_length', truncation=True, max_length=150, return_tensors='pt', return_offsets_mapping=True
                )

                tokens_start_end_idx = tokens.offset_mapping
                masked_tokens_start_end_idx = tokens_start_end_idx.sum(dim=-1) > 0
                selected_tokens_start_end_idx = tokens_start_end_idx[masked_tokens_start_end_idx]

                # build pseudo entities
                interval, char_num = 4, 0
                entities = []
                for idx in range(0, len(selected_tokens_start_end_idx), interval):
                    _start_idx = selected_tokens_start_end_idx[idx][0].item()
                    if idx + interval - 1 >= len(selected_tokens_start_end_idx):
                        _end_idx = selected_tokens_start_end_idx[-1][1].item()
                    else:
                        _end_idx = selected_tokens_start_end_idx[idx+interval-1][1].item()
                    _text = sentence[_start_idx:_end_idx]
                    _type = 'text'
                    _image = None
                    _mask = None
                    _interactive = None
                    entities += [Entity(idx, _text, _mask, _interactive, _type, _start_idx, _end_idx, _image)]
                
                entities_start_end_idx = torch.tensor([[entity.start_idx, entity.end_idx] for entity in entities])[:, None]
                start_end_condition = (entities_start_end_idx[:, :, 0] <= tokens_start_end_idx[:, :, 0]) & (entities_start_end_idx[:, :, 1] >= tokens_start_end_idx[:, :, 1]) & (tokens_start_end_idx[:, :, 0] != tokens_start_end_idx[:, :, 1])

                data = {}
                data['entities'] = {"entities": entities, "sentence": sentence, "tokens": tokens, "entity_to_tokens": start_end_condition.cuda()}
                outputs, extra = model.model.demo_interleave([data])
                pred_entity_class = outputs["pred_entity_class"]
                pred_interleave_class = outputs["pred_interleave_image"]

                interleave_entity_list += [{"pred_entity_class": pred_entity_class, "sentence": sentence, "entities": entities}]
                interleave_text_list += [pred_interleave_class]

                if bidx == 40 and debug:
                    break

    interleave_text_long = torch.cat(interleave_text_list, dim=0)[:,0]
    interleave_text_long = interleave_text_long / interleave_text_long.norm(dim=-1, keepdim=True)

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

                # if idx == 40 and debug:
                #     break

            image_embs_class = torch.cat(image_class_emb_list, dim=0)
            image_embs_class = image_embs_class / image_embs_class.norm(dim=-1, keepdim=True)
            object_embs_pixel = torch.cat(object_pixel_emb_list, dim=0)
            # object_embs_pixel = object_embs_pixel / object_embs_pixel.norm(dim=-1, keepdim=True)
            object_embs_class = torch.cat(object_class_emb_list, dim=0)
            object_embs_class = object_embs_class / object_embs_class.norm(dim=-1, keepdim=True)

    t = []
    t.append(transforms.Resize(720, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')

    # def add_image(*args):
    #     nonlocal model, image_embs_class, object_embs_pixel, object_embs_class, image_ids, transform, metadata, add_image_pths, add_image_id

    #     with torch.no_grad():
    #         with torch.autocast(device_type='cuda', dtype=torch.float16):
    #             image_class_emb_list = []
    #             object_class_emb_list = []
    #             object_pixel_emb_list = []
    #             for idx, _input in enumerate(args):
    #                 if _input is not None:
    #                     pil_image = Image.fromarray(_input)
    #                     save_pth = os.path.join(database_root, str(add_image_id).zfill(12) + "_add.jpg")
    #                     pil_image.save(save_pth)
    #                     add_image_pths += [save_pth]
    #                     image_ori = transform(pil_image)
    #                     image_ori = np.asarray(image_ori)
    #                     images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    #                     images = [images.to(model.model.device)]
    #                     images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
    #                     images = ImageList.from_tensors(images, model.model.size_divisibility)
                        
    #                     targets = targets_grounding = queries_grounding = None
    #                     features = model.model.backbone(images.tensor)
    #                     outputs = model.model.sem_seg_head(features, target_queries=queries_grounding)
    #                     image_class_emb_list += [outputs['pred_retrievals'][:,0]]
    #                     object_class_emb_list += [outputs['pred_captions']]
    #                     object_pixel_emb_list += [outputs['pred_maskembs']]
    #                     image_ids += [save_pth]
    #                     add_image_id += 1

    #             if len(image_class_emb_list) > 0:
    #                 add_image_class_embs = torch.cat(image_class_emb_list, dim=0)
    #                 add_image_class_embs = add_image_class_embs / add_image_class_embs.norm(dim=-1, keepdim=True)
    #                 image_embs_class = torch.cat([image_embs_class, add_image_class_embs], dim=0)

    #                 add_object_class_embs = torch.cat(object_class_emb_list, dim=0)
    #                 add_object_class_embs = add_object_class_embs / add_object_class_embs.norm(dim=-1, keepdim=True)
    #                 object_embs_class = torch.cat([object_embs_class, add_object_class_embs], dim=0)

    #                 add_object_pixel_embs = torch.cat(object_pixel_emb_list, dim=0)
    #                 # add_object_pixel_embs = add_object_pixel_embs / add_object_pixel_embs.norm(dim=-1, keepdim=True)
    #                 object_embs_pixel = torch.cat([object_embs_pixel, add_object_pixel_embs], dim=0)

    def inference(search_space, *args):
        nonlocal model, image_embs_class, object_embs_pixel, object_embs_class, image_ids, transform, metadata, add_image_pths

        # offset 0 is entity_text, offset 1 is entity_image, offset 2 is connection
        sentence = ''
        char_num = 0
        entities = []
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
                tokens = tokenizer(
                    [sentence], padding='max_length', truncation=True, max_length=77, return_tensors='pt', return_offsets_mapping=True
                )
                entities_start_end_idx = torch.tensor([[entity.start_idx, entity.end_idx] for entity in entities])[:, None]
                tokens_start_end_idx = tokens.offset_mapping
                start_end_condition = (entities_start_end_idx[:, :, 0] <= tokens_start_end_idx[:, :, 0]) & (entities_start_end_idx[:, :, 1] >= tokens_start_end_idx[:, :, 1]) & (tokens_start_end_idx[:, :, 0] != tokens_start_end_idx[:, :, 1])

                data = {}
                image_gallery = []
                data['entities'] = {"entities": entities, "sentence": sentence, "tokens": tokens, "entity_to_tokens": start_end_condition.cuda()}
                outputs, extra = model.model.demo_interleave([data])

                if 'caption' in search_space:
                    interleave_emb = outputs['pred_interleave_image'][0]
                    interleave_emb = interleave_emb / interleave_emb.norm(dim=-1, keepdim=True)
                    pred_entity_class = outputs['pred_entity_class'][0]
                    pred_entity_class = pred_entity_class / pred_entity_class.norm(dim=-1, keepdim=True)

                    ii_scores_text = interleave_emb @ interleave_text_long.t()
                    topk_ids = ii_scores_text.topk(30, dim=-1).indices[0]

                    for select_topk_id in topk_ids:
                        interleave_entity_tokenemb = interleave_entity_list[select_topk_id]['pred_entity_class']
                        interleave_entity_tokenemb = interleave_entity_tokenemb / interleave_entity_tokenemb.norm(dim=-1, keepdim=True)
                        ei_scores_text = pred_entity_class @ interleave_entity_tokenemb[0].t()
                        entities = interleave_entity_list[select_topk_id]['entities']
                        sentence = interleave_entity_list[select_topk_id]['sentence']

                        image = draw_text_on_image_with_score(sentence, entities, ei_scores_text)
                        image_gallery += [image]

                if 'image' in search_space:
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

                                image_shape = image_ori.shape[:2]
                                instances = Instances(image_shape)

                                if len(pred_masks) == 0:
                                    # Some image does not have annotation (all ignored)
                                    masks = BitMasks(torch.zeros((0, image_shape[0], image_shape[1])))
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
    gallery_output2 = gr.Textbox(label="Text Gallery.")

    with gr.Blocks(css=customCSS) as demo:
        gr.Markdown(f"# &#x1F50E; FIND: Interfacing Foundation Models' Embeddings")

        gr.Markdown(f"### 👉 Examples.")
        example1 = gr.Examples(
            examples=[
                    ["A green and white leafy potted plant", None, None, None, None, "sits near", "a cozy bed with a blue comforter", None, None, None, None, "creating a relaxing atmosphere in the bedroom"]
                    ],
            inputs=[*input_list],
            outputs=[gallery_output, None],
            cache_examples=False,
            label='Image Retrieval & Grounding'
        )

        example2 = gr.Examples(
            examples=[
                    [None, "assets/images/dog.jpg", None, None, None, "is sitting on", "a wooden bench", None, None, None, None, None]
                    ],
            inputs=[*input_list],
            outputs=[gallery_output, None],
            cache_examples=False,
            label='Interleave Retrieval & Grounding'
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
            run = gr.Button("Run")

        with gr.Row():
            search_space = gr.Radio(["image", "caption"], label="Retrieval & Grounding Search Space.", value="image")
            run.click(inference, [search_space] + [*input_list], [gallery_output])

        gr.Markdown(f"### 👉 Results.")
        gallery_output.render()

        gr.Markdown(f"### 👉 Usage.")
        with gr.Row():
            gr.Video(
                value="assets/videos/example1.mp4",
                label='Image Retrieval & Grounding',
                width=400,
            )
            gr.Video(
                value="assets/videos/example2.mp4",
                label='Interleave Retrieval & Grounding',
                width=400,
            )

        # gr.Markdown(f"### 👉 Upload new images to the database.")
        # with gr.Row():
        #     input_image1 = gr.Image(label="Add Image 1.")
        #     input_image2 = gr.Image(label="Add Image 2.")
        #     input_image3 = gr.Image(label="Add Image 3.")
        #     input_image4 = gr.Image(label="Add Image 4.")
        #     input_image5 = gr.Image(label="Add Image 5.")
        #     input_image6 = gr.Image(label="Add Image 6.")
        #     addimage = gr.Button("Upload")
        #     addimage.click(add_image, [input_image1, input_image2, input_image3, input_image4, input_image5, input_image6])

    demo.launch(share=True)

if __name__ == "__main__":
    main()
    # debug()
    sys.exit(0)