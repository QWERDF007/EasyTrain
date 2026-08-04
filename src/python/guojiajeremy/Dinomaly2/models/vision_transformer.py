"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 grad=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = self.grad * x + (1 - self.grad) * x.detach()
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DropKey(nn.Module):
    """DropKey
    """

    def __init__(self, p=0.):
        super(DropKey, self).__init__()
        self.p = p

    def forward(self, attn):
        if self.training:
            m_r = torch.ones_like(attn) * self.p
            attn = attn + torch.bernoulli(m_r) * -1e12
        return attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Glu(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)
        self.split_features = hidden_features // 2

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x[..., :self.split_features] * x[..., self.split_features:]
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoE(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., num_experts=8,
                 noise=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_weights = nn.Parameter(torch.empty((num_experts, hidden_features, in_features)))
        self.act = act_layer()
        self.fc2_weights = nn.Parameter(torch.empty((num_experts, out_features, hidden_features)))
        self.drop = nn.Dropout(drop)
        self.num_experts = num_experts
        self.route = nn.Linear(in_features, num_experts, bias=False)
        if noise == True:
            self.noise = nn.Linear(in_features, num_experts, bias=False)
        else:
            self.noise = None

    def forward(self, x):
        route_logits = self.route(x[:, 0, :])
        if self.noise is not None and self.training:
            raw_noise_stddev = self.noise(x[:, 0, :])
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            route_logits = route_logits + (torch.randn_like(route_logits) * noise_stddev)
        route_weight = torch.softmax(route_logits, dim=-1)
        fc1_weights = torch.einsum('ne,eab->nab', route_weight, self.fc1_weights)
        x = self.conditional_linear(x, fc1_weights)
        x = self.act(x)
        x = self.drop(x)
        fc2_weights = torch.einsum('ne,eab->nab', route_weight, self.fc2_weights)
        x = self.conditional_linear(x, fc2_weights)
        x = self.drop(x)
        return x

    def conditional_linear(self, input_tensor, weight_tensor):
        # input_tensor shape: [N, L, C]
        # weight_tensor shape: [N, C_in, C_out]

        N, L, C_in = input_tensor.shape
        _, C_out, C_in = weight_tensor.shape

        input_tensor = input_tensor.permute(0, 2, 1).reshape(1, N * C_in, L)
        weight_tensor = weight_tensor.reshape(N * C_out, C_in, 1)

        output = F.conv1d(
            input_tensor, weight=weight_tensor, bias=None, stride=1, padding=0, groups=N)

        output = output.reshape(N, C_out, L).permute(0, 2, 1)

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., return_attn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn = return_attn

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.] = 0.

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        if self.return_attn:
            return x, attn
        else:
            return x



class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        attn = (q @ k.transpose(-2, -1))
        attn = self.attn_drop(attn)

        if attn_mask is not None:
            attn[:, :, attn_mask == 0.] = 0.

        attn = attn / (torch.sum(attn, dim=-1, keepdim=True))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LinearAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., eps=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.eps = eps

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / (torch.einsum('...sd,...d->...s', q, k.sum(dim=-2)) + self.eps)
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Attention, ffn=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, attn_mask=None):
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class MABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Attention, ffn=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.act = act_layer()

    def forward(self, x, return_attention=False, attn_mask=None):
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(self.act(y))
        x = x + self.drop_path(self.mlp(self.act(self.norm2(x))))
        if return_attention:
            return x, attn
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.conv = SepConv(dim, kernel_size=kernel_size, act1_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, attn_mask=None):
        y = self.conv(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, None
        else:
            return x


class FeatureJitter(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, feature_tokens):
        if self.training:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel  # B x N x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * self.scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens


class PartialJitter(nn.Module):
    def __init__(self, scale=1., ratio=0.5):
        super().__init__()
        self.scale = scale
        self.ratio = ratio

    def forward(self, feature_tokens, mask=None):
        if self.training:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel  # B x N x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * self.scale

            mask = torch.bernoulli(torch.ones([batch_size, num_tokens, 1],
                                              device=feature_tokens.device) * self.ratio) if mask is None else mask

            feature_tokens = feature_tokens + jitter * mask
        return feature_tokens


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=nn.GELU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        b, hxw, c = x.shape
        h = int(math.sqrt(hxw))
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(b, hxw, -1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


from .utils import get_2d_sincos_pos_embed
import numpy as np


class RandomMask(nn.Module):
    def __init__(self, p=0.2, embed_dim=768, num_cls_token=5, num_patches=784):
        super(RandomMask, self).__init__()
        self.p = p
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_cls_token = num_cls_token
        self.mlp = Mlp(embed_dim, hidden_features=embed_dim * 4)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_cls_token, embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(num_patches ** .5), cls_token=False)
        decoder_pos_embed = np.concatenate([np.zeros([num_cls_token, embed_dim]), decoder_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.proj(x)
        if self.training:
            x_ = x[:, self.num_cls_token:]
            x_, mask, ids_restore = self.random_masking(x_, self.p)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_.shape[1], 1)
            x_ = torch.cat([x_, mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :self.num_cls_token], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        x = self.mlp(x)

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
