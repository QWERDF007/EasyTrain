"""The three Dinomaly2 losses used by mask-constraint training in DLTool.

``good_values`` / ``anomaly_values`` / ``ignore_values`` are lists of class
values that may appear in a training mask: good pixels are pulled toward the
encoder, anomaly pixels are pushed away, and ignore pixels are excluded from
every mask-constraint loss.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from utils import global_cosine, global_cosine_hm_percent


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape [B, 1, H, W], got {mask.shape}")
    return mask.float()


def _zero_from(features: Sequence[torch.Tensor]) -> torch.Tensor:
    return features[0].sum() * 0.0


def _isin(mask: torch.Tensor, values: Sequence[int]) -> torch.Tensor:
    if not values:
        return torch.zeros_like(mask, dtype=torch.bool)
    value_tensor = torch.as_tensor(
        list(values), dtype=mask.dtype, device=mask.device
    )
    return torch.isin(mask, value_tensor)


def masked_global_cosine(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the default global cosine loss only to pixels selected by mask."""

    mask = _as_mask(mask)
    losses = []
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
    ):
        feature_mask = F.interpolate(
            mask,
            size=encoder_feature.shape[-2:],
            mode="nearest",
        )
        valid = feature_mask.flatten(1).sum(dim=1) > 0
        if not torch.any(valid):
            continue
        encoder_selected = (encoder_feature * feature_mask).flatten(1)
        decoder_selected = (decoder_feature * feature_mask).flatten(1)
        cosine = F.cosine_similarity(
            encoder_selected.detach(),
            decoder_selected,
            dim=1,
        )
        losses.append((1.0 - cosine[valid]).mean())

    if not losses:
        return _zero_from(decoder_features)
    return torch.stack(losses).mean()


def masked_global_cosine_hm_percent(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
    mask: torch.Tensor,
    p: float,
    factor: float,
) -> torch.Tensor:
    """Masked equivalent of Dinomaly2's loose cosine loss.

    The global cosine value is unchanged for the selected region. The
    original low-distance gradient suppression is applied only to selected
    feature-map positions, so zeroed-out pixels cannot affect top-k logic.
    """

    mask = _as_mask(mask)
    losses = []
    p = float(min(max(p, 0.0), 0.999999))
    for encoder_feature, decoder_feature in zip(
        encoder_features,
        decoder_features,
    ):
        feature_mask = F.interpolate(
            mask,
            size=encoder_feature.shape[-2:],
            mode="nearest",
        )
        valid_pixels = feature_mask[:, 0] > 0
        valid_samples = valid_pixels.flatten(1).any(dim=1)
        if not torch.any(valid_samples):
            continue

        encoder_selected = encoder_feature * feature_mask
        decoder_selected = decoder_feature * feature_mask
        point_distance = 1.0 - F.cosine_similarity(
            encoder_feature.detach(),
            decoder_feature,
            dim=1,
        )

        thresholds = []
        for sample_index in range(point_distance.shape[0]):
            values = point_distance[sample_index][valid_pixels[sample_index]]
            if values.numel() == 0:
                thresholds.append(point_distance.new_tensor(float("inf")))
                continue
            keep_count = max(1, int(values.numel() * (1.0 - p)))
            thresholds.append(torch.topk(values, keep_count).values[-1])
        threshold = torch.stack(thresholds).view(-1, 1, 1)
        suppressed = (
            (point_distance < threshold)
            & valid_pixels
            & valid_samples.view(-1, 1, 1)
        )

        if decoder_selected.requires_grad:
            def modify_gradient(gradient):
                scale = torch.where(
                    suppressed.unsqueeze(1),
                    gradient.new_tensor(float(factor)),
                    gradient.new_tensor(1.0),
                )
                return gradient * scale

            decoder_selected.register_hook(modify_gradient)

        encoder_flat = encoder_selected.flatten(1)
        decoder_flat = decoder_selected.flatten(1)
        cosine = F.cosine_similarity(
            encoder_flat.detach(),
            decoder_flat,
            dim=1,
        )
        losses.append((1.0 - cosine[valid_samples]).mean())

    if not losses:
        return _zero_from(decoder_features)
    return torch.stack(losses).mean()


def default_dinomaly_loss(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
    use_loose_loss: bool,
    p: float,
    factor: float,
) -> torch.Tensor:
    """Call the exact existing Dinomaly2 loss for the full image."""

    if use_loose_loss:
        return global_cosine_hm_percent(
            encoder_features,
            decoder_features,
            p=p,
            factor=factor,
        )
    return global_cosine(encoder_features, decoder_features)


def masked_default_dinomaly_loss(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
    valid_mask: torch.Tensor,
    use_loose_loss: bool,
    p: float,
    factor: float,
) -> torch.Tensor:
    """Compute the Dinomaly2 loss only on pixels selected by ``valid_mask``.

    ``valid_mask`` contains one for pixels that may contribute to the default
    loss and zero for excluded pixels. A missing-mask sample is represented by
    an all-one mask, so it keeps contributing the full image.
    """

    if use_loose_loss:
        return masked_global_cosine_hm_percent(
            encoder_features,
            decoder_features,
            valid_mask,
            p=p,
            factor=factor,
        )
    return masked_global_cosine(
        encoder_features,
        decoder_features,
        valid_mask,
    )


def calculate_mask_constraint_losses(
    encoder_features: Sequence[torch.Tensor],
    decoder_features: Sequence[torch.Tensor],
    masks: torch.Tensor,
    has_mask: torch.Tensor,
    good_values: Sequence[int],
    anomaly_values: Sequence[int],
    use_loose_loss: bool,
    p: float,
    factor: float,
    ignore_values: Sequence[int] = (255,),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(L_dinomaly, L_good, L_anomaly)`` for one batch.

    Masks without ignore values use the exact full-image Dinomaly2 loss. When
    a mask contains any ignore value, only its valid pixels contribute to
    ``L_dinomaly``. ``L_good`` and ``L_anomaly`` select their values
    explicitly and therefore never include background or ignored pixels.
    """

    masks = masks.to(device=decoder_features[0].device)
    has_mask = has_mask.to(device=masks.device, dtype=torch.bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape [B, H, W], got {masks.shape}")

    has_mask_per_pixel = has_mask.view(-1, 1, 1)
    ignore_pixels = _isin(masks, ignore_values) & has_mask_per_pixel
    if torch.any(ignore_pixels):
        # Samples without a mask use an all-one mask; samples with a mask use
        # every annotated value except the explicit ignore values.
        valid_mask = torch.where(
            has_mask_per_pixel,
            ~ignore_pixels,
            torch.ones_like(ignore_pixels),
        ).float()
        loss_dinomaly = masked_default_dinomaly_loss(
            encoder_features,
            decoder_features,
            valid_mask,
            use_loose_loss=use_loose_loss,
            p=p,
            factor=factor,
        )
    else:
        # Preserve the original Dinomaly2 path bit-for-bit when no ignored
        # value is present in this batch.
        loss_dinomaly = default_dinomaly_loss(
            encoder_features,
            decoder_features,
            use_loose_loss=use_loose_loss,
            p=p,
            factor=factor,
        )

    if not torch.any(has_mask):
        zero = _zero_from(decoder_features)
        return loss_dinomaly, zero, zero

    good_mask = (
        _isin(masks, good_values)
        & has_mask_per_pixel
    ).float()
    anomaly_mask = (
        _isin(masks, anomaly_values)
        & has_mask_per_pixel
    ).float()

    if use_loose_loss:
        region_loss = masked_global_cosine_hm_percent
        region_kwargs = {"p": p, "factor": factor}
    else:
        region_loss = masked_global_cosine
        region_kwargs = {}
    loss_good = region_loss(
        encoder_features,
        decoder_features,
        good_mask,
        **region_kwargs,
    )
    loss_anomaly = region_loss(
        encoder_features,
        decoder_features,
        anomaly_mask,
        **region_kwargs,
    )
    return loss_dinomaly, loss_good, loss_anomaly
