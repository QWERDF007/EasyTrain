import os
import sys
from pathlib import Path

# add facebookresearch/sam2 to path so `import sam2` works without pip install -e
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'facebookresearch' / 'sam2'))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def iou_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    intersection = (inputs * targets).sum(-1)
    union = inputs.sum(-1) + targets.sum(-1) - intersection
    loss = 1 - (intersection + 1) / (union + 1)
    return loss.sum() / num_masks   


class SAM2_pred(nn.Module):
    def __init__(self, checkpoint=None, model_cfg=None):
        super().__init__()
        from sam2.build_sam import build_sam2  # defer import for CLI help

        if checkpoint is None:
            checkpoint = "./checkpoint/sam2.1_hiera_base_plus.pt"
        if model_cfg is None:
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.model = build_sam2(model_cfg, checkpoint)

        # loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def compute_objective(self, logit_mask, gt_mask):
        """ BCE + Dice loss"""
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice
    
    
    def forward(self, img_batch, mask_inputs=None, prev_out=None, query_mask=None):
        B, ch, h, w = img_batch.shape  # batch, 3, 1024, 1024
        #C = self.model.hidden_dim  #=256
        current_out = {}
        assert h == w == self.model.image_size, f"input image size must be {self.model.image_size}x{self.model.image_size}"
        # if img_batch.shape[-2:] != (self.model.image_size,self.model.image_size):  # resize img & mask to 1024x1024, SAM2 training dimension
        #     img_batch = F.interpolate(img_batch, (self.model.image_size,self.model.image_size), align_corners=False, mode='bilinear', antialias=True)
        # if mask is not None:
        #     mask = F.interpolate(mask, (self.model.image_size,self.model.image_size), mode='bilinear')
        if prev_out:  # accumulate previous outputs
            current_out = prev_out

        # 1. image encoder
        backbone_out = self.model.forward_image(img_batch)  # SAM2 features
        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels :]  # 3 levels (default), other levels used in mask decoder
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels :]
        for i in range(len(feature_maps)):  # clone to help torch.compile
            feature_maps[i] = feature_maps[i].clone()
            vision_pos_embeds[i] = vision_pos_embeds[i].clone()
        high_res_features = feature_maps[:-1]  # shape=[32x256x256, 64x128x128]
        pix_feat = feature_maps[-1]
        #vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]  # BxCxHxW -> HWxBxC
        #vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]  # BxCxHxW -> HWxBxC


        # 2. memory encoder (for support img, using features and masks)
        if mask_inputs is not None:  # we got mask from support, encode attention

            mask_inputs = mask_inputs[:, None, ...]  # add channel dimension
            assert mask_inputs.shape[-2:] == (self.model.image_size, self.model.image_size), f"mask size must be {self.model.image_size}x{self.model.image_size}"
            # if mask_inputs.shape[-2:] != (1024, 1024):  # make sure its same shape as img
            #     mask_inputs = F.interpolate(mask_inputs, size=(1024, 1024), align_corners=False, mode="bilinear", antialias=True)

            # turn GT binary mask into output logits mask without using SAM2 (for support)
            high_res_masks = mask_inputs.float() * 20.0 - 10.0  # convert to logits: [0,1] -> [-10,10], sigmoid(-10.0)=4.5398e-05
            low_res_masks = F.interpolate(  # resize to 256x256
                high_res_masks,
                size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()  # dummy IoU of all 1
            obj_ptr = torch.zeros(  # dummy object pointer of all 0
                mask_inputs.size(0), self.model.hidden_dim, device=mask_inputs.device  # BxC
            )
            
            maskmem_out = self.model.memory_encoder(
                pix_feat, high_res_masks, skip_mask_sigmoid=True  # sigmoid already applied (and scaled)
            )
            
            current_out.setdefault("mask_inputs", []).append(mask_inputs)
            current_out.setdefault("maskmem_features", []).append(maskmem_out["vision_features"].clone())  # (clone to help torch.compile)
            if "maskmem_pos_enc" not in current_out:
                current_out["maskmem_pos_enc"] = [m.clone() for m in maskmem_out["vision_pos_enc"]]  # list of len=1, same for all support imgs
            
            current_out.setdefault("obj_ptr", []).append(obj_ptr)  # use dummy obj_ptr
            return current_out  # and skip obj_ptr calculation from mask decoder


        # 3. memory attention
        if prev_out and mask_inputs is None:  # we got memory from support, run attention on query
            #C' = self.model.mem_dim  #= 64
            to_cat_memory = [maskmem.flatten(2).permute(2, 0, 1) for maskmem in prev_out["maskmem_features"]]  # BxC'xHxW -> HWxBxC'
            to_cat_memory_pos_embed = len(prev_out["maskmem_features"]) * [prev_out["maskmem_pos_enc"][0].flatten(2).permute(2, 0, 1)]  # BxC'xHxW -> HWxBxC'
            # for i in range(len(to_cat_memory_pos_embed)):  # add temporal encoding to memory
            #     to_cat_memory_pos_embed[i] = to_cat_memory_pos_embed[i] + self.model.maskmem_tpos_enc[i]  # temporal encoding (it adds it to all the elements in the list, since they reference the same tensor)
            #to_cat_memory_pos_embed[0] += self.model.maskmem_tpos_enc[6]  # temporal encoding (it adds it to all the elements in the list, since they reference the same tensor), 6 -> input frame
            num_obj_ptr_tokens = 0
            # if self.model.use_obj_ptrs_in_encoder:
            #     obj_ptrs = torch.stack(prev_out["obj_ptr"], dim=0)
            #     # split C = 256 into 4 parts
            #     obj_ptrs = obj_ptrs.reshape(-1, B, 4, 64).permute(0, 2, 1, 3).flatten(0, 1)  # `len_obj_ptr`xBx256 -> `len_obj_ptr`*4xBx64
            #     # if self.model.add_tpos_enc_to_obj_ptrs:  # set temporal encoding to 1/15 (minimum distance, see sam2_base.py line 628)
            #     #     from sam2.modeling.sam2_utils import get_1d_sine_pe
            #     #     obj_pos = torch.tensor([1] * len(prev_out["obj_ptr"]), device=self.model.device)
            #     #     obj_pos = get_1d_sine_pe(obj_pos / 15, dim=256)
            #     #     obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)  # linear projection to dim=64
            #     #     obj_pos = obj_pos.unsqueeze(1).expand(-1, B, 64)  # shape: `len_obj_ptr`xBx64
            #     # else:
            #     #     obj_pos = obj_ptrs.new_zeros(len(prev_out["obj_ptr"]), B, 64)  # NO temporal position embedding (shape: `len_obj_ptr`xBx64)
            #     obj_pos = obj_ptrs.new_zeros(len(prev_out["obj_ptr"]), B, 64)  # NO temporal position embedding (shape: `len_obj_ptr`xBx64)
            #     obj_pos = obj_pos.repeat_interleave(4, dim=0)  # shape: `len_obj_ptr`*4xBx64
                
            #     to_cat_memory.append(obj_ptrs)
            #     to_cat_memory_pos_embed.append(obj_pos)
            #     num_obj_ptr_tokens = obj_ptrs.shape[0]  # = kshot*4
            memory = torch.cat(to_cat_memory, dim=0)  # (HW+4)xBxC, C = 64
            memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

            pix_feat_with_mem = self.model.memory_attention(  # (self- + cross-attention)
                curr=feature_maps[-1].flatten(2).permute(2, 0, 1),  # BxCxHxW -> HWxBxC, C = 256
                curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
                )
            pix_feat_with_mem = pix_feat_with_mem.clone()  # clone to help torch.compile
            
        else:  # no memory from support, skip the memory encoder
            # add no_mem_embed to the last feature map
            pix_feat_with_mem = feature_maps[-1].flatten(2).permute(2, 0, 1) + self.model.no_mem_embed

            # OR: dummy token to avoid empty memory input to transformer encoder
            #to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            #to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
            #pix_feat_with_mem = self.model.memory_attention(...)
        
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(*pix_feat.shape)  # (HW)BC => BCHW, C = 256
        

        # 4. prompt encoder
        sam_point_coords = torch.zeros(B, 1, 2, device=self.model.device)  # pad with empty point
        sam_point_labels = - torch.ones(B, 1, dtype=torch.int32, device=self.model.device)  # -1 label
        sam_mask_prompt = None
        if mask_inputs is not None:  # support img with GT mask
            sam_mask_prompt = low_res_masks  # shape = Bx1x256x256

        # OR: use mask from previous iteration
        #prev_sam_mask_logits = prev_out["pred_masks"]  # previous mask logits of same img
        #prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)  # clamp to avoid rare numerical issues
        #sam_mask_prompt = ...

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # clone to help torch.compile
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.model.sam_prompt_encoder.get_dense_pe().clone()

        
        # 5. mask decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,  # downsampled to 1x64x64 using CNN
            multimask_output=False,  # output 3 or 1 masks
            repeat_image=False,  # if multiple mask, single img (repeat batchwise)
            high_res_features=high_res_features,
        )  # 6 tokens used: [object, ious, mask[0,1,2,3]]
        low_res_multimasks = low_res_multimasks.clone()  # clone to help torch.compile
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()
        # is_obj_appearing = object_score_logits > 0

        low_res_multimasks = low_res_multimasks.float()  # convert from bfloat16
        high_res_multimasks = F.interpolate(low_res_multimasks, size=(1024, 1024), mode="bilinear", align_corners=False)

        sam_output_token = sam_output_tokens[:, 0]  # take only the first mask
        low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        obj_ptr = self.model.obj_ptr_proj(sam_output_token)
        #obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr


        # resize back to original size
        logit_mask = F.interpolate(low_res_masks, size=(h,w), mode='bilinear', align_corners=False)


        # save state
        current_out["logit_mask"] = logit_mask
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["object_score_logits"] = object_score_logits  # ! if not self.training
        current_out["ious"] = ious
        current_out.setdefault("obj_ptr", []).append(obj_ptr)

        if query_mask is not None:  # compute loss
            loss = self.compute_objective(logit_mask, query_mask)
            return current_out, loss

        return current_out
