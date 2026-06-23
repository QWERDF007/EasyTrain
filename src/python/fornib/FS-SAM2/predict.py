""" FS-SAM2 prediction script: segment query images given support examples """
import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from torchvision import tv_tensors
from torchvision.transforms import v2


def load_support(support_dir, img_size, transform, device):
    """Load all support images and masks from a directory with images/ and masks/ subdirs."""
    img_dir = os.path.join(support_dir, 'images')
    mask_dir = os.path.join(support_dir, 'masks')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Support images directory not found: {img_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Support masks directory not found: {mask_dir}')

    imgs, masks, names = [], [], []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for img_path in sorted(glob.glob(os.path.join(img_dir, ext))):
            name = Path(img_path).stem
            mask_path = os.path.join(mask_dir, name + '.png')
            if not os.path.exists(mask_path):
                continue
            img = Image.open(img_path).convert('RGB')
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            mask[mask < 128] = 0
            mask[mask >= 128] = 1

            img = img.resize((img_size, img_size))
            mask = tv_tensors.Mask(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (img_size, img_size), mode='nearest').squeeze())
            img, mask = transform(img, mask)
            imgs.append(img)
            masks.append(mask)
            names.append(name)

    if not imgs:
        raise FileNotFoundError(f'No support images found in {img_dir} with matching masks in {mask_dir}')
    print(f'Loaded {len(imgs)} support images')
    return torch.stack(imgs).to(device), torch.stack(masks).to(device), names


def load_queries(query_dir):
    """List all query image paths."""
    queries = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        queries.extend(sorted(glob.glob(os.path.join(query_dir, ext))))
    if not queries:
        raise FileNotFoundError(f'No query images found in {query_dir}')
    return queries


def setup_model(checkpoint_path, device, sam2_checkpoint=None, sam2_cfg=None):
    """Build FS-SAM2 model with LoRA and load trained weights."""
    from peft import LoraConfig, get_peft_model
    from sam2_pred import SAM2_pred

    sam_model = SAM2_pred(checkpoint=sam2_checkpoint, model_cfg=sam2_cfg)

    # LoRA image_encoder
    peft_config_encoder = LoraConfig(inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1,
                                     target_modules=['qkv', 'proj'], bias="none")
    sam_model.model.image_encoder = get_peft_model(sam_model.model.image_encoder, peft_config_encoder)

    # LoRA memory_attention
    peft_config_mem = LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
                                 target_modules=['q_proj', 'v_proj', 'k_proj', 'out_proj'], bias="none")
    sam_model.model.memory_attention = get_peft_model(sam_model.model.memory_attention, peft_config_mem)

    # LoRA memory_encoder
    peft_config_mem_enc = LoraConfig(inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
                                     target_modules=['out_proj'], bias="none")
    sam_model.model.memory_encoder = get_peft_model(sam_model.model.memory_encoder, peft_config_mem_enc)

    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    sam_model.load_state_dict(state_dict)
    sam_model.to(device)
    sam_model.eval()
    return sam_model


def main():
    parser = argparse.ArgumentParser(description='FS-SAM2 Prediction')
    parser.add_argument('--support_dir', type=str, required=True, help='Dir with images/ and masks/ subdirs (support examples)')
    parser.add_argument('--query_dir', type=str, required=True, help='Dir with query images to segment')
    parser.add_argument('--output_dir', type=str, required=True, help='Dir to save predicted masks')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained .pt checkpoint')
    parser.add_argument('--sam2_checkpoint', type=str, default='./checkpoint/sam2.1_hiera_base_plus.pt')
    parser.add_argument('--sam2_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml')
    parser.add_argument('--kshot', type=int, default=1, help='Number of support images to use (default 1)')
    parser.add_argument('--img_size', type=int, default=1024, help='Image size for inference (default 1024)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    os.makedirs(args.output_dir, exist_ok=True)

    # Image transform (same normalization as training)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(img_mean, img_std),
    ])

    # Load model
    print(f'Loading checkpoint: {args.checkpoint}')
    model = setup_model(args.checkpoint, device, args.sam2_checkpoint, args.sam2_cfg)

    # Load support images
    support_imgs, support_masks, _ = load_support(args.support_dir, args.img_size, transform, device)

    # Use only kshot support images
    total = len(support_imgs)
    if args.kshot < total:
        support_imgs = support_imgs[:args.kshot]
        support_masks = support_masks[:args.kshot]
        print(f'Using {args.kshot} of {total} support images (controlled by --kshot)')

    # Pre-compute support memory once
    print('Encoding support images...')
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        current_out = {}
        for i in range(len(support_imgs)):
            current_out = model(support_imgs[i].unsqueeze(0), support_masks[i].unsqueeze(0), prev_out=current_out)
    print('Support encoding done.')

    # Build query list — use query.txt if present, else all images in dir
    query_txt = os.path.join(args.query_dir, 'query.txt')
    if os.path.exists(query_txt):
        query_list = []
        with open(query_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qid, qname = line.split(',', 1)
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                    p = os.path.join(args.query_dir, qname + ext)
                    if os.path.exists(p):
                        query_list.append((qid, p))
                        break
                else:
                    print(f'  Skip (no image): {qname}')
    else:
        query_list = [(Path(p).stem, p) for p in load_queries(args.query_dir)]

    print(f'Found {len(query_list)} query images')
    for qid, qpath in query_list:
        print(f'  Processing [{qid}]: {Path(qpath).stem}')

        img = Image.open(qpath).convert('RGB')
        orig_size = img.size  # (W, H)
        img = img.resize((args.img_size, args.img_size))
        img = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            prev = {}
            for k, v in current_out.items():
                prev[k] = [t.clone() for t in v] if isinstance(v, list) else v.clone() if isinstance(v, torch.Tensor) else v
            out = model(img, prev_out=prev)

        logit_mask = out['logit_mask']
        logit_mask = F.interpolate(logit_mask, (orig_size[1], orig_size[0]), mode='bilinear', align_corners=True)
        pred_mask = (logit_mask.squeeze() > 0.0).float().cpu().numpy() * 255

        out_path = os.path.join(args.output_dir, f'{qid}.png')
        Image.fromarray(pred_mask.astype(np.uint8)).save(out_path)
        print(f'    Saved: {out_path}')


if __name__ == '__main__':
    main()
