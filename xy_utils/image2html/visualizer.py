import os
from re import X

import cv2
import torch
import numpy as np

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from utils.constants import IMAGENET_CLASSES, COCO_PANOPTIC_CLASSES

from .utils import *
from utils.distributed import get_world_size


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Visualizer(object):

    def __init__(self, ):
        '''
        output_folder: the output folder for storing html file.
        col: the number of column of the html webpage
        size: image size
        demo_name: file name of the html file
        '''
        # Parameters that will update through time
        self._step = None
        self._imgs_dict = None
        self._extra = None

        # Parameters that will fixed after initialize
        self.size = None
        self.log_dir = None
        self.total_step = None
        self.turn_on = False
    
    def initialize(self, log_dir, tag, total_step=20, size=(224,224), track_color=True):
        assert get_world_size() <=1, "Visualization only applies on single GPU."
        self._step = -1
        self._imgs = []
        self._extra = {}

        self.log_dir = os.path.join(log_dir, 'visual', tag)
        self.html_pth = os.path.join(self.log_dir, 'demo.html')
        self.size = size
        self.total_step = total_step
        self.turn_on = True
        self.track_color = track_color
        if track_color:
            self._extra['track_color'] = {}            

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def step(self,):
        if self._step > -1 and self.track_color:
            self.write_color_platte()
            self._extra['track_color'] = {}

        self._step += 1
        self._imgs.append({})
        
        if self._step >= self.total_step:
            self.write_and_close()
        
    def add_image(self, img, name=None):
        name = str(self._step).zfill(5) if name is None else name
        self._imgs[self._step]['image_name'] = name
        out_folder = os.path.join(self.log_dir, name)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        self._imgs[self._step]['image_folder'] = out_folder
        out_pth = os.path.join(out_folder, 'img.png')
        self._imgs[self._step][name] = out_pth.replace('{}/'.format(self.log_dir),'')
        cv2.imwrite(out_pth, img)
    
    def insert(self, img, name, resize=False):
        assert 'image_name' in self._imgs[self._step].keys(), "Please add image before insert visualization."
        out_pth = os.path.join(self._imgs[self._step]['image_folder'], '{}.png'.format(name))
        if resize:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_pth, img)
        self._imgs[self._step][name] = out_pth.replace('{}/'.format(self.log_dir),'')

    def write_and_close(self, skip_list=['image_name', 'image_folder']):
        row_num = len(self._imgs)
        col_num = len(self._imgs[0].keys()) - len(skip_list)

        dir_lst = []
        caption_lst = []
        for i in range(row_num):
            for key, value in self._imgs[i].items():
                if key not in skip_list:
                    dir_lst.append(value)
                    caption_lst.append(key)
        
        writeSeqHTML(self.html_pth, dir_lst, caption_lst, col_num, self.size[0], self.size[1])
        assert False, "Visualized sample number: {}".format(self.total_step)
        
    def fetch_unnormalized_image(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean, device=x.device)[:,None,None]
        std = torch.tensor(std, device=x.device)[:,None,None]
        
        img = (((x[0] * std) + mean) * 255).byte().permute(1,2,0).cpu().numpy()
        return img[...,::-1]
    
    def fetch_heat_map(self, x):
        x = ((x - x.min())/(x.max() - x.min()) * 255).byte().cpu().numpy()
        x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
        return x

    def overlay_x_mask_to_single_image(self, img, masks, categories):
        for i in range(0, len(masks)):
            mask = masks[i].astype(np.float32)[:,:,None]
            cat_name = categories[i]
            # cat_id = ADE_CLASSES[cat_name]['id']
            cat_id = hash(cat_name) + i
            color = np.array(COCO_CATEGORIES[cat_id%133]['color'])[None,None,:]
            overlay = mask * color
            img = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat_id%133]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][cat_name] = COCO_CATEGORIES[cat_id%133]['color']

        visual_name = 'gt'
        self.insert(img, visual_name)

    def overlay_obj_mask_to_image(self, img, masks, texts, max_len=20):
        classes = COCO_PANOPTIC_CLASSES
        classes += ['other']

        for i in range(0, min(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            color = np.array([0,255,0])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            visual_name = '{}_{}'.format(texts[i], i)
            self.insert(x, visual_name)
        return x

    def overlay_x_mask_to_single_image_sam(self, img, masks):
        for i in range(0, len(masks)):
            mask = masks[i].astype(np.float32)[:,:,None]
            # cat_name = categories[i]
            # cat_id = ADE_CLASSES[cat_name]['id']
            cat_id = i
            color = np.array(COCO_CATEGORIES[cat_id%133]['color'])[None,None,:]
            overlay = mask * color
            img = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat_id%133]['color'], thickness=2, lineType=cv2.LINE_AA)
        return img

    def overlay_obj_mask_to_image_withposneg(self, img, masks, pos_masks, neg_masks, texts, max_len=20):
        classes = COCO_PANOPTIC_CLASSES
        classes += ['other']

        for i in range(0, max(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            color = np.array([0,255,0])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)

            pos_mask = pos_masks[i].astype(np.float32)[:,:,None]
            color = np.array([255,0,0])[None,None,:]
            overlay = pos_mask * color
            x = x * (1 - pos_mask) + (x * pos_mask * 0.5 + overlay * 0.5)

            neg_mask = neg_masks[i].astype(np.float32)[:,:,None]
            color = np.array([0,0,255])[None,None,:]
            overlay = neg_mask * color
            x = x * (1 - neg_mask) + (x * neg_mask * 0.5 + overlay * 0.5)

            visual_name = '{}_{}'.format(texts[i], i)
            self.insert(x, visual_name)
        return x

    def overlay_x_recall_to_image(self, img, masks, gts, categories, scores, max_len=20, color_idx=0, prefix_name=None):
        for i in range(0, max_len):
            if i < len(masks):
                mask = masks[i].astype(np.float32)[:,:,None]
                gt = gts[i].astype(np.float32)[:,:,None]
                cat = categories[i]
                score = scores[i]
            else:
                mask = np.zeros(img.shape)
                gt = mask
                cat = 'none'
                score = 0

            color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)

            color = np.array(COCO_CATEGORIES[3]['color'])[None,None,:]
            overlay = gt * color
            x = x * (1 - mask) + (x * mask * 0.7 + overlay * 0.3)

            visual_name = 'objtoken_{}_{}_{:.4f}'.format(i, cat, score)
            self.insert(x, visual_name)
        return x

    def overlay_coco_mask_to_image(self, img, masks, categories, scores, cls_, max_len=20):
        classes = COCO_PANOPTIC_CLASSES
        classes += ['other']

        for i in range(0, max(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            cat = categories[i]
            score = scores[i]
            color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            visual_name = 'objtoken_{}_{}_{:.4f}_{}'.format(i, classes[cat], score, cls_)
            self.insert(x, visual_name)
        return x
    
    def overlay_grounding_mask_to_image(self, img, masks, names, max_len=20):
        classes = COCO_PANOPTIC_CLASSES
        classes += ['other']

        for i in range(0, min(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            name = names[i]
            color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            visual_name = 'objtoken_{}_{}'.format(i, name)
            self.insert(x, visual_name)
        return x

    def overlay_mask_to_image(self, x, masks, categories):
        for i in range(0, len(masks)):
            mask = masks[i].astype(np.float32)[:,:,None]
            cat = categories[i]
            color = np.array(COCO_CATEGORIES[cat]['color'])[None,None,:]
            overlay = mask * color
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
        return x

    def overlay_all_masks_to_image(self, x, masks):
        for i in range(0, len(masks)):
            mask = masks[i].astype(np.float32)[:,:,None]
            color = np.array(COCO_CATEGORIES[i%133]['color'])[None,None,:]
            overlay = mask * color
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
        return x

    def overlay_single_mask_to_image(self, x, mask):
        mask = mask[:,:,None]
        color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
        overlay = mask * color
        x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
        return x

    def overlay_box_to_image(self, x, boxes, categories):
        h,w,c = x.shape
        boxes = box_cxcywh_to_xyxy(boxes).cpu()
        scale = torch.tensor([w,h,w,h])[None,:]
        boxes = (boxes*scale).int().numpy()
        for i in range(0, len(boxes)):            
            cat = categories[i]
            color = COCO_CATEGORIES[cat]['color']
            x = cv2.rectangle(x.copy(), (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), color, 2)
        return x

    def overlay_box_to_image_withtext(self, x, boxes, categories, names):
        h,w,c = x.shape
        boxes = boxes.int()
        # boxes = box_cxcywh_to_xyxy(boxes).cpu()
        # scale = torch.tensor([w,h,w,h])[None,:]
        # boxes = (boxes*scale).int().numpy()
        for i in range(0, len(boxes)):            
            cat = categories[i]
            color = COCO_CATEGORIES[cat%133]['color']
            x = cv2.rectangle(x.copy(), (boxes[i,0].item(), boxes[i,1].item()), (boxes[i,2].item(), boxes[i,3].item()), color, 2)
            if self.track_color:
                self._extra['track_color'][names[cat]] = color
        self.insert(x, 'gt_bbox')

    def overlay_inet_mask_to_image(self, img, masks, categories, scores, max_len=20):
        for i in range(0, max(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            cat = categories[i]
            score = scores[i]
            color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            visual_name = 'objtoken_{}_{}_{:.4f}'.format(i, IMAGENET_CLASSES[cat], score)
            self.insert(x, visual_name)
        return x

    def overlay_inst_to_image(self, x, labels):
        scores = labels['scores']
        keep = scores > 0.5
        pred_classes = labels['pred_classes'][keep]
        pred_masks = labels['pred_masks'][keep]

        for i in range(0, len(pred_masks)):
            mask = pred_masks[i].cpu().numpy()[:,:,None]
            cat = pred_classes[i]
            color = np.array(COCO_CATEGORIES[cat]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][COCO_CATEGORIES[cat]['name']] = COCO_CATEGORIES[cat]['color']

        return x

    def overlay_gtinst_to_image(self, x, masks, labels):
        for i in range(0, len(masks)):
            mask = masks[i][:,:,None]
            cat = labels[i]
            color = np.array(COCO_CATEGORIES[i]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[i]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][cat] = COCO_CATEGORIES[i]['color']

        return x

    def overlay_pano_to_x(self, x, labels, categories):
        masks, labels = labels

        for i in range(0, len(labels)):
            cat = labels[i]['category_id']
            id_ = labels[i]['id']
            mask = (masks == id_).float().cpu().numpy()[:,:,None]
            color = np.array(COCO_CATEGORIES[cat%133]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat%133]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][categories[cat]] = COCO_CATEGORIES[cat%133]['color']

        return x

    def overlay_seg_to_x(self, x, labels, categories):
        unique_id = torch.unique(labels).int().tolist()

        for i in range(0, len(unique_id)):
            cat = unique_id[i]
            mask = (labels == cat).float().cpu().numpy()[:,:,None]
            color = np.array(COCO_CATEGORIES[cat%133]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat%133]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][categories[cat]] = COCO_CATEGORIES[cat%133]['color']

        return x

    def overlay_x_mask_to_image(self, img, masks, categories, scores, class_name, max_len=20):
        for i in range(0, min(len(masks), max_len)):
            mask = masks[i].astype(np.float32)[:,:,None]
            cat = categories[i]
            score = scores[i]
            color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
            overlay = mask * color
            x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
            visual_name = 'objtoken_{}_{}_{:.4f}'.format(i, class_name[cat], score)
            self.insert(x, visual_name)

        return x


    def overlay_x_mask_to_image_bytxt(self, img, masks, txts, max_len=20):
        for i in range(0, max_len):
            if i < len(masks):
                mask = masks[i].astype(np.float32)[:,:,None]
                color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
                overlay = mask * color
                x = img * (1 - mask) + (img * mask * 0.5 + overlay * 0.5)
                visual_name = 'objtoken_{}_{}'.format(i, txts[i])
                self.insert(x, visual_name)
            else:
                x = np.zeros((256, 256))
                self.insert(x, 'test_{}'.format(i))
        return x


    def overlay_pano_to_image(self, x, labels):
        masks, labels = labels

        for i in range(0, len(labels)):
            cat = labels[i]['category_id']
            id_ = labels[i]['id']
            mask = (masks == id_).float().cpu().numpy()[:,:,None]
            color = np.array(COCO_CATEGORIES[cat]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][COCO_CATEGORIES[cat]['name']] = COCO_CATEGORIES[cat]['color']

        return x

    def overlay_pano_to_imagenet(self, x, labels):
        masks, labels = labels

        for i in range(0, len(labels)):
            cat = labels[i]['category_id']
            id_ = labels[i]['id']
            mask = (masks == id_).float().cpu().numpy()[:,:,None]
            color = np.array(COCO_CATEGORIES[cat%133]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat%133]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][IMAGENET_CLASSES[cat]] = COCO_CATEGORIES[cat%133]['color']

        return x

    def overlay_seg_to_imagenet(self, x, labels, imagenet_classes=None):
        if imagenet_classes is None:
            imagenet_classes = IMAGENET_CLASSES

        unique_id = torch.unique(labels).int().tolist()

        for i in range(0, len(unique_id)):
            cat = unique_id[i]
            mask = (labels == cat).float().cpu().numpy()[:,:,None]
            color = np.array(COCO_CATEGORIES[cat%133]['color'])[None,None,:]
            overlay = mask * color
            h,w,_ = mask.shape
            x = cv2.resize(x, (w,h))
            x = x * (1 - mask) + (x * mask * 0.5 + overlay * 0.5)        
            contours, _ = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=x, contours=contours, contourIdx=-1, color=COCO_CATEGORIES[cat%133]['color'], thickness=2, lineType=cv2.LINE_AA)
            if self.track_color:
                self._extra['track_color'][imagenet_classes[cat]] = COCO_CATEGORIES[cat%133]['color']

        return x

    def overlay_grounding_to_image(self, image, masks, texts, num=20):
        masks = np.array(masks)
        for idx in range(len(masks)):
            mask = masks[idx][:,:,None]
            cur_texts = [texts[idx]]
            for text in cur_texts:
                if num == 0:
                    break
                color = np.array(COCO_CATEGORIES[0]['color'])[None,None,:]
                overlay = mask * color
                x = image * (1 - mask) + (image * mask * 0.5 + overlay * 0.5)        
                self.insert(x, '{}_{}'.format(text, num))
                num -= 1

        for idx in range(0, num):
            self.insert(np.zeros(image.shape), 'none_{}'.format(idx))

    def overlay_phrasecut_to_image(self, image, masks, name):
        masks = np.array(masks)
        for idx in range(len(masks)):
            mask = masks[idx][:,:,None]
            color = np.array(COCO_CATEGORIES[idx]['color'])[None,None,:]
            overlay = mask * color
            image = image * (1 - mask) + (image * mask * 0.5 + overlay * 0.5)        
        self.insert(image, '{}'.format(name))

    def write_color_platte(self, ):
        image = (np.ones((512,512,3)) * 255).astype(np.uint8)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        thickness              = 2
        lineType               = 2

        index_j = 1
        index_i = 0
        for name, color in self._extra['track_color'].items():
            bottomLeftCornerOfText = (10 + index_i*256,index_j * 30)
            fontColor              = color
            cv2.putText(image, name[0:12],
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            index_j += 1
            if index_j % 16 == 0:
                index_i += 1

        name = 'color_platte'
        self.insert(image, name)

VL = Visualizer()