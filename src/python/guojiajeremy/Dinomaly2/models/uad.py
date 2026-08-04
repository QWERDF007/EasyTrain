import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.init import trunc_normal_
import math


class Dinomaly(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_bottleneck=[0, 1, 2, 3, 4, 5, 6, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            context_aware_recenter=True,
            use_get_intermediate=False
    ) -> None:
        super(Dinomaly, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.fuse_layer_bottleneck = fuse_layer_bottleneck
        self.remove_class_token = remove_class_token
        self.context_aware_recenter = context_aware_recenter
        self.use_get_intermediate = use_get_intermediate
        if not hasattr(self.encoder, 'num_register_tokens'):
            if hasattr(self.encoder, 'n_storage_tokens'):
                self.encoder.num_register_tokens = self.encoder.n_storage_tokens
            else:
                self.encoder.num_register_tokens = 0

        self.mask_neighbor_size = mask_neighbor_size

    def init_weights(self):
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        if self.use_get_intermediate:
            with torch.no_grad():
                en_list = self.encoder._get_intermediate_layers_not_chunked(x, self.target_layers)
        else:
            with torch.no_grad():
                x = self.encoder.prepare_tokens(x)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list_bn = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = self.fuse_feature([en_list_bn[idx] for idx in self.fuse_layer_bottleneck]).detach()
        else:
            x = self.fuse_feature([en_list[idx] for idx in self.fuse_layer_bottleneck]).detach()

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        if self.context_aware_recenter:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] - e[:, :1, :] for e in en]
            en = [F.layer_norm(e, normalized_shape=(e.shape[-1],), eps=1e-8) for e in en]
        else:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class DinomalyRGBD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_bottleneck=[0, 1, 2, 3, 4, 5, 6, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            add_class_token_weight=0.,
            norm_encoder_token=False,
            encoder_require_grad_layer=[],
            rgb_ratio=0.5,
    ) -> None:
        super(DinomalyRGBD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.fuse_layer_bottleneck = fuse_layer_bottleneck
        self.remove_class_token = remove_class_token
        self.add_class_token_weight = add_class_token_weight
        self.norm_encoder_token = norm_encoder_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.rgb_ratio = rgb_ratio
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def init_weights(self):
        for m in self.bottleneck.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        assert x.shape[1] == 6, "The channel of RGBD input should be 6 (3 RGB, 3 depth)"
        bs = x.shape[0]
        x_rgb = x[:, :3]
        x_depth = x[:, 3:]
        x = torch.cat([x_rgb, x_depth], dim=0)
        with torch.no_grad():
            x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        "Merge RGB feature and depth feature"
        en_list = [self.rgb_ratio * e[:bs] + (1 - self.rgb_ratio) * e[bs:] for e in en_list]

        x = self.fuse_feature([en_list[idx] for idx in self.fuse_layer_bottleneck]).detach()
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            if self.add_class_token_weight != 0:
                en = [e[:, :1, :] * self.add_class_token_weight + e[:, 1 + self.encoder.num_register_tokens:, :] for
                      e in en]
            else:
                en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        if self.norm_encoder_token:
            # en = [F.normalize(e, dim=1) for e in en]
            en = [F.layer_norm(e, normalized_shape=(e.shape[-1],), eps=1e-8) for e in en]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all
