"""Fixed prediction-time augmentation and IoU voting helpers for FS-SAM2."""

import numpy as np
import PIL.Image as Image
from PIL import ImageEnhance


def bounded(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def fixed_prediction_augmentations(args):
    """Build deterministic variants from the configured prediction augmentations."""
    augmentations = []
    if getattr(args, 'prediction_horizontal_flip', False):
        augmentations.append(
            {
                'name': 'horizontal_flip',
                'horizontal_flip': True,
                'vertical_flip': False,
                'scale': 1.0,
                'brightness': 1.0,
                'contrast': 1.0,
                'hue': 0.0,
            }
        )
    if getattr(args, 'prediction_vertical_flip', False):
        augmentations.append(
            {
                'name': 'vertical_flip',
                'horizontal_flip': False,
                'vertical_flip': True,
                'scale': 1.0,
                'brightness': 1.0,
                'contrast': 1.0,
                'hue': 0.0,
            }
        )

    scale = 1.0 + bounded(getattr(args, 'prediction_scale', 0.0), -0.9, 1.0)
    if abs(scale - 1.0) >= 1e-6:
        augmentations.append(
            {
                'name': 'scale',
                'horizontal_flip': False,
                'vertical_flip': False,
                'scale': scale,
                'brightness': 1.0,
                'contrast': 1.0,
                'hue': 0.0,
            }
        )

    brightness = 1.0 + bounded(getattr(args, 'prediction_brightness', 0.0), -1.0, 1.0)
    if abs(brightness - 1.0) >= 1e-6:
        augmentations.append(
            {
                'name': 'brightness',
                'horizontal_flip': False,
                'vertical_flip': False,
                'scale': 1.0,
                'brightness': brightness,
                'contrast': 1.0,
                'hue': 0.0,
            }
        )

    contrast = 1.0 + bounded(getattr(args, 'prediction_contrast', 0.0), -1.0, 1.0)
    if abs(contrast - 1.0) >= 1e-6:
        augmentations.append(
            {
                'name': 'contrast',
                'horizontal_flip': False,
                'vertical_flip': False,
                'scale': 1.0,
                'brightness': 1.0,
                'contrast': contrast,
                'hue': 0.0,
            }
        )

    hue = bounded(getattr(args, 'prediction_hue', 0.0), -0.5, 0.5)
    if abs(hue) >= 1e-6:
        augmentations.append(
            {
                'name': 'hue',
                'horizontal_flip': False,
                'vertical_flip': False,
                'scale': 1.0,
                'brightness': 1.0,
                'contrast': 1.0,
                'hue': hue,
            }
        )

    rotation = bounded(getattr(args, 'prediction_rotation', 0.0), -180.0, 180.0)
    if abs(rotation) >= 1e-6:
        augmentations.append(
            {
                'name': 'rotation',
                'horizontal_flip': False,
                'vertical_flip': False,
                'scale': 1.0,
                'brightness': 1.0,
                'contrast': 1.0,
                'hue': 0.0,
                'rotation': rotation,
            }
        )
    return augmentations


def scale_layout(size, scale):
    """Return the centered crop/pad layout used for a scale augmentation."""
    width, height = size
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    visible_width = min(width, scaled_width)
    visible_height = min(height, scaled_height)
    source_left = max(0, (scaled_width - width) // 2)
    source_top = max(0, (scaled_height - height) // 2)
    destination_left = max(0, (width - scaled_width) // 2)
    destination_top = max(0, (height - scaled_height) // 2)
    return (
        (scaled_width, scaled_height),
        source_left,
        source_top,
        destination_left,
        destination_top,
        visible_width,
        visible_height,
    )


def resize_centered(image, scale, resample, fill):
    """Scale an image around its center while keeping the original canvas size."""
    if abs(scale - 1.0) < 1e-6:
        return image

    original_size = image.size
    scaled_size, source_left, source_top, destination_left, destination_top, visible_width, visible_height = scale_layout(
        original_size, scale
    )
    scaled = image.resize(scaled_size, resample)
    canvas = Image.new(image.mode, original_size, fill)
    visible = scaled.crop((source_left, source_top, source_left + visible_width, source_top + visible_height))
    canvas.paste(visible, (destination_left, destination_top))
    return canvas


def transform_prediction_box(box, augmentation, image_size):
    """Transform an xyxy box with the same fixed geometry as its augmented image."""
    width, height = image_size
    points = [
        (float(box[0]), float(box[1])),
        (float(box[2]), float(box[1])),
        (float(box[2]), float(box[3])),
        (float(box[0]), float(box[3])),
    ]

    transformed = []
    for x, y in points:
        if augmentation['horizontal_flip']:
            x = width - x
        if augmentation['vertical_flip']:
            y = height - y

        scale = augmentation['scale']
        if abs(scale - 1.0) >= 1e-6:
            _, source_left, source_top, destination_left, destination_top, _, _ = scale_layout(image_size, scale)
            x = x * scale - source_left + destination_left
            y = y * scale - source_top + destination_top

        rotation = augmentation.get('rotation', 0.0)
        if abs(rotation) >= 1e-6:
            angle = np.deg2rad(rotation)
            cos_angle = float(np.cos(angle))
            sin_angle = float(np.sin(angle))
            dx = x - width / 2.0
            dy = y - height / 2.0
            x = width / 2.0 + cos_angle * dx + sin_angle * dy
            y = height / 2.0 - sin_angle * dx + cos_angle * dy
        transformed.append((x, y))

    return [
        min(point[0] for point in transformed),
        min(point[1] for point in transformed),
        max(point[0] for point in transformed),
        max(point[1] for point in transformed),
    ]


def apply_prediction_augmentation(image, augmentation):
    """Apply one fixed geometry/color augmentation to an image."""
    flip = getattr(Image, 'Transpose', Image)
    if augmentation['horizontal_flip']:
        image = image.transpose(flip.FLIP_LEFT_RIGHT)
    if augmentation['vertical_flip']:
        image = image.transpose(flip.FLIP_TOP_BOTTOM)

    image = resize_centered(image, augmentation['scale'], Image.BILINEAR, (0, 0, 0))
    rotation = augmentation.get('rotation', 0.0)
    if abs(rotation) >= 1e-6:
        image = image.rotate(rotation, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    if abs(augmentation['brightness'] - 1.0) >= 1e-6:
        image = ImageEnhance.Brightness(image).enhance(augmentation['brightness'])
    if abs(augmentation['contrast'] - 1.0) >= 1e-6:
        image = ImageEnhance.Contrast(image).enhance(augmentation['contrast'])
    if abs(augmentation['hue']) >= 1e-6:
        hsv = np.asarray(image.convert('HSV'), dtype=np.uint8).copy()
        hue_delta = int(round(augmentation['hue'] * 255.0))
        hsv[..., 0] = np.mod(hsv[..., 0].astype(np.int16) + hue_delta, 256).astype(np.uint8)
        image = Image.fromarray(hsv, mode='HSV').convert('RGB')
    return image


def inverse_prediction_geometry(mask, augmentation, original_size):
    """Map an augmented prediction mask back to the original image coordinates."""
    result = Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), mode='L')
    rotation = augmentation.get('rotation', 0.0)
    if abs(rotation) >= 1e-6:
        result = result.rotate(-rotation, resample=Image.NEAREST, expand=False, fillcolor=0)
    scale = augmentation['scale']
    if abs(scale - 1.0) >= 1e-6:
        scaled_size, source_left, source_top, destination_left, destination_top, visible_width, visible_height = scale_layout(
            original_size, scale
        )
        visible = result.crop(
            (destination_left, destination_top, destination_left + visible_width, destination_top + visible_height)
        )
        scaled = Image.new('L', scaled_size, 0)
        scaled.paste(visible, (source_left, source_top))
        result = scaled.resize(original_size, Image.NEAREST)

    flip = getattr(Image, 'Transpose', Image)
    if augmentation['vertical_flip']:
        result = result.transpose(flip.FLIP_TOP_BOTTOM)
    if augmentation['horizontal_flip']:
        result = result.transpose(flip.FLIP_LEFT_RIGHT)

    if result.size != original_size:
        result = result.resize(original_size, Image.NEAREST)
    return np.where(np.asarray(result, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)


def mask_area(mask):
    return int(np.count_nonzero(mask))


def mask_iou(left, right):
    """Calculate IoU for two masks in the same original-image coordinate system."""
    left_foreground = left > 0
    right_foreground = right > 0
    intersection = np.logical_and(left_foreground, right_foreground).sum()
    union = np.logical_or(left_foreground, right_foreground).sum()
    return 1.0 if union == 0 else float(intersection) / float(union)


def prediction_components(predictions, iou_threshold):
    """Group predictions whose pairwise IoU reaches the configured threshold."""
    components = []
    visited = set()
    for start in range(len(predictions)):
        if start in visited:
            continue
        component = []
        pending = [start]
        visited.add(start)
        while pending:
            current = pending.pop()
            component.append(current)
            for candidate in range(len(predictions)):
                if candidate in visited:
                    continue
                if mask_iou(predictions[current], predictions[candidate]) >= iou_threshold:
                    visited.add(candidate)
                    pending.append(candidate)
        components.append(component)
    return components


def select_voted_prediction(predictions, iou_threshold, min_vote_count):
    """Select a consensus mask, preferring the original prediction when its group is valid."""
    if not predictions:
        raise ValueError('predictions is empty')
    if len(predictions) == 1:
        return predictions[0]

    components = prediction_components(predictions, bounded(iou_threshold, 0.0, 1.0))
    required_votes = int(min_vote_count)
    valid_components = [component for component in components if len(component) >= required_votes]
    if not valid_components:
        return np.zeros_like(predictions[0], dtype=np.uint8)

    for component in valid_components:
        if 0 in component:
            return predictions[0].copy()

    selected_component = max(
        valid_components,
        key=lambda component: max(mask_area(predictions[index]) for index in component),
    )
    selected_index = max(selected_component, key=lambda index: mask_area(predictions[index]))
    return predictions[selected_index].copy()


def predict_mask_with_augmentation(
    image,
    args,
    predict_original,
    predict_augmented,
    enabled_attribute='prediction_enhancement_enabled',
    check_stopped=None,
    progress=-1,
):
    """Run the original prediction, fixed TTA, and IoU voting.

    ``predict_original`` predicts the unmodified image. ``predict_augmented``
    receives an augmented image and its geometry description, allowing the
    caller to transform prompts together with the image when necessary.
    """
    original_mask = predict_original(image)
    if not getattr(args, enabled_attribute, False):
        return original_mask

    augmentations = fixed_prediction_augmentations(args)
    if not augmentations:
        return original_mask

    predictions = [original_mask]
    for augmentation in augmentations:
        if check_stopped is not None:
            check_stopped(progress)
        augmented_image = apply_prediction_augmentation(image, augmentation)
        augmented_mask = predict_augmented(augmented_image, augmentation)
        if augmented_mask is not None:
            predictions.append(inverse_prediction_geometry(augmented_mask, augmentation, image.size))

    return select_voted_prediction(
        predictions,
        getattr(args, 'prediction_iou_threshold', 0.5),
        getattr(args, 'prediction_min_vote_count', 2),
    )
