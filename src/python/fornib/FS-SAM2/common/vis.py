r""" Visualize model predictions """
import os

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

from . import utils
import random


class Visualizer:

    @classmethod
    def initialize(cls, visualize=False, path=None):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (0, 0, 255), 'green': (0, 255, 0),}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = path if path else './vis/'
        if not os.path.exists(cls.vis_path):
            os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, qry_name_b, spt_name_b, iou_predictions_b=None, iou_b=None, pred_spt_masks=None):
        spt_img_b = utils.to_cpu(spt_img_b)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)
        if pred_spt_masks is not None:
            pred_spt_masks = [utils.to_cpu(pred_spt_mask) for pred_spt_mask in pred_spt_masks]

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, qry_name, spt_name) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, qry_name_b, spt_name_b)):
            iou_prediction = iou_predictions_b[sample_idx][0] if iou_predictions_b is not None else None
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, True, qry_name, spt_name, iou_prediction, iou, pred_spt_masks)

    @classmethod
    def visualize_prediction_batch_demo(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b, batch_idx, sub_idx, iou_b=None):
        spt_img_b = utils.to_cpu(spt_img_b)
        spt_mask_b = utils.to_cpu(spt_mask_b)
        qry_img_b = utils.to_cpu(qry_img_b)
        qry_mask_b = utils.to_cpu(qry_mask_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)
        cls_id_b = utils.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, cls_id_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction_demo(spt_img, spt_mask, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, sub_idx, True, iou)
    
    @classmethod  
    def visualize_sub_prediction_batch(cls, qry_img_b, batch_idx, pred_mask_b):
        
        qry_img_b = utils.to_cpu(qry_img_b)
        pred_mask_b = utils.to_cpu(pred_mask_b)

        for sample_idx, (qry_img, pred_mask) in enumerate(zip(qry_img_b, pred_mask_b)):
            for idx in range(pred_mask.shape[0]):
                sub_merge_img = cls.visualize_sub_prediction(qry_img, pred_mask[idx,:,:])
                sub_path = cls.vis_path + '%d_img/'% (batch_idx)
                if not os.path.exists(sub_path): os.makedirs(sub_path)
                sub_merge_img.save(sub_path + '%d_sub-%d' % (sample_idx, idx) + '.jpg')

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sample_idx, label, qry_name='', spt_name='', iou_prediction=None, iou=None, pred_spt_masks=None):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs_ori = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img.astype(np.uint8), spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        mask_gt = [Image.fromarray(cls.apply_mask_gt(spt_mask, spt_color)) for spt_mask in spt_masks]
        
        pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(), qry_img.size()[-2:], mode='nearest').squeeze()  # resize to 512x512
        pred_mask = (pred_mask > 0.0).float()
        qry_mask = F.interpolate(qry_mask.unsqueeze(0).unsqueeze(0).float(), qry_img.size()[-2:], mode='nearest').squeeze()  # resize to 512x512
        
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_img_ori = Image.fromarray(qry_img)
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        # pred_masked_pil = cls.apply_point(pred_masked_pil, postivate_pos)
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        
        if pred_spt_masks is not None:  ### test learnable prompts
            #pred_spt_masks = F.interpolate(pred_spt_masks.unsqueeze(0).unsqueeze(0).float(), qry_img.size()[-2:], mode='nearest').squeeze()  # resize to 512x512
            pred_spt_masks = [(torch.sigmoid(pred_spt_mask) > 0.5).float() for pred_spt_mask in pred_spt_masks]
            pred_spt_masks = [cls.to_numpy(pred_spt_mask, 'mask') for pred_spt_mask in pred_spt_masks]
            pred_spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img.astype(np.uint8), pred_spt_mask.astype(np.uint8), spt_color)) for spt_img, pred_spt_mask in zip(spt_imgs, pred_spt_masks)]

            merged_pil = cls.merge_image_pair([item for i in zip(pred_spt_masked_pils,spt_masked_pils) for item in i] + [pred_masked_pil, qry_masked_pil])
        
        else:
            merged_pil = cls.merge_image_pair(spt_masked_pils + [pred_masked_pil, qry_masked_pil])  #supp + query + query_GT
            # # add titles on top of the images
            # draw = ImageDraw.Draw(merged_pil)
            # for i, spt_img in enumerate(spt_imgs):
            #     draw.text((spt_img.shape[0] * i + 10, 10), f'Support (GT):  {spt_name}', fill=(255, 255, 255, 128), font_size=30)
            # draw.text((qry_img.shape[0] * len(spt_pils) + 10, 10), f'Query:  {qry_name}', fill=(255, 255, 255, 128), font_size=30)
            # draw.text((qry_img.shape[0] * (len(spt_pils) + 1) + 10, 10), 'Query GT', fill=(255, 255, 255, 128), font_size=30)

        iou = iou.item() if iou else 0.0
        random_number = random.randint(0, 10000)
        merged_pil.save(cls.vis_path + f'{qry_name}_class{cls_id}_{spt_name}_iou{iou:.2f}.jpg')  #_iou{iou:.2f}

        # save original images
        save_path_1 = os.path.join(cls.vis_path, '_refer/')
        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)
        spt_masked_pils[0].save(save_path_1 + f'{qry_name}_class{cls_id}_{spt_name}_support.jpg')
        qry_masked_pil.save(save_path_1 + f'{qry_name}_class{cls_id}_{spt_name}_query.jpg')
        pred_masked_pil.save(save_path_1 + f'{qry_name}_class{cls_id}_{spt_name}_pred.jpg')


    @classmethod
    def visualize_prediction_demo(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, cls_id, batch_idx, sub_idx, sample_idx, label, iou=None):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_imgs_ori = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]
        mask_gt = [Image.fromarray(cls.apply_mask_gt(spt_mask, spt_color)) for spt_mask in spt_masks]
        
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_img_ori = Image.fromarray(qry_img)
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        # pred_masked_pil = cls.apply_point(pred_masked_pil, postivate_pos)
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        save_path_1 = os.path.join(cls.vis_path,'%d_refer/'%(batch_idx))
        if not os.path.exists(save_path_1): os.makedirs(save_path_1)
        
        mask_gt[0].save(save_path_1 + ' 0_mask_gt'+'.jpg')
        spt_imgs_ori[0].save(save_path_1 + ' 0_qry_img'+'.jpg')
        spt_masked_pils[0].save(save_path_1 + ' 1_re_mask_img'+'.jpg')
        pred_masked_pil.save(save_path_1 + ' 2_tgt_mask_img_%d'%(sample_idx)+'.jpg')

    @classmethod
    def visualize_sub_prediction(cls, qry_img, pred_mask):
        
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        qry_img = cls.to_numpy(qry_img, 'img')
        qry_pil = cls.to_pil(qry_img)
        # qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        
        return pred_masked_pil


    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
    
    @classmethod
    def apply_mask_gt(cls, mask, color, alpha=1):
        r""" Apply mask to the given image. """
        image = Image.new('RGB', mask.shape, (255,255,255))
        # image_copy = image.copy()
        image = np.array(image)
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def apply_point(cls, image, points):
        r""" Apply mask to the given image. """
        draw = ImageDraw.Draw(image)
        if points.shape[0]==0:
            return image
        else:
            for point in points:
                draw.rectangle((point[0]-5, point[1]-5, point[0]+5, point[1]+5), fill=(102, 140, 255))
            return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
