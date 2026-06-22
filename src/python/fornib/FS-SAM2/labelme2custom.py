"""Convert LabelMe JSON annotations to FS-SAM2 custom dataset format."""
import os
import sys
import json
import argparse
import glob
import random
import re
from pathlib import Path
from collections import defaultdict

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as np


def sanitize(name):
    """Replace non-alphanumeric chars with underscore for safe directory names."""
    return re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', name)


def find_image(json_path, image_dir):
    """Find the image file by matching JSON stem."""
    base = Path(json_path).stem
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        p = os.path.join(image_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def render_mask(shapes, height, width):
    """Render LabelMe shapes to a binary PIL mask."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        pts = [(p[0], p[1]) for p in shape['points']]
        t = shape['shape_type']
        if t == 'polygon' and len(pts) >= 3:
            draw.polygon(pts, fill=255)
        elif t == 'rectangle' and len(pts) >= 2:
            draw.rectangle(pts, fill=255)
        elif t == 'circle' and len(pts) >= 2:
            cx, cy = pts[0]
            ex, ey = pts[1]
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return mask


def main():
    parser = argparse.ArgumentParser(description='LabelMe to FS-SAM2 custom dataset converter')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory containing LabelMe .json annotation files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for custom dataset')
    parser.add_argument('--support_ratio', type=float, default=0.5,
                        help='Fraction of images per class for support set (default 0.5)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(args.json_dir, '*.json')))
    if not json_files:
        print(f'No .json files found in {args.json_dir}')
        sys.exit(1)
    print(f'Found {len(json_files)} LabelMe JSON files')

    # Group: label -> [(image_path, mask_pil)]
    label_items = defaultdict(list)
    skip_count = 0

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_path = find_image(jf, args.image_dir)
        if img_path is None:
            skip_count += 1
            print(f'  Skip (no image): {os.path.basename(jf)}')
            continue

        h, w = data['imageHeight'], data['imageWidth']

        # Group shapes by label
        shapes_by_label = defaultdict(list)
        for shape in data.get('shapes', []):
            lbl = shape.get('label', '').strip()
            if lbl and shape['shape_type'] in ('polygon', 'rectangle', 'circle'):
                shapes_by_label[lbl].append(shape)

        for lbl, shapes in shapes_by_label.items():
            mask = render_mask(shapes, h, w)
            # Skip if mask is all background
            if not mask.getextrema()[1]:
                continue
            label_items[lbl].append((img_path, mask))

    if skip_count:
        print(f'Skipped {skip_count} JSON files (image not found)')

    if not label_items:
        print('No valid annotations found.')
        sys.exit(1)

    all_classes = sorted(label_items.keys())
    print(f'Found {len(all_classes)} classes: {", ".join(all_classes)}')

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for cls in all_classes:
        items = label_items[cls]
        rng.shuffle(items)
        n_support = max(1, int(len(items) * args.support_ratio))
        support_items = items[:n_support]
        query_items = items[n_support:]

        safe_cls = sanitize(cls)
        cls_dir = os.path.join(args.output_dir, safe_cls)
        os.makedirs(os.path.join(cls_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(cls_dir, 'masks'), exist_ok=True)

        def write_items(item_list, lst_file):
            names = []
            for img_path, mask in item_list:
                stem = Path(img_path).stem
                ext = Path(img_path).suffix
                # Dedup: include original filename stem as prefix
                safe_name = sanitize(stem)
                dst_img = os.path.join(cls_dir, 'images', safe_name + ext)
                # Only copy if not already there (same image may have multiple labels)
                if not os.path.exists(dst_img):
                    Image.open(img_path).save(dst_img)
                mask.save(os.path.join(cls_dir, 'masks', safe_name + '.png'))
                names.append(safe_name)
            with open(os.path.join(cls_dir, lst_file), 'w') as f:
                f.write('\n'.join(names) + '\n')
            return names

        sn = write_items(support_items, 'support.txt')
        qn = write_items(query_items, 'query.txt')
        print(f'  {safe_cls}: {len(sn)} support, {len(qn)} query')

    print(f'\nDone. Output: {args.output_dir}')


if __name__ == '__main__':
    main()
