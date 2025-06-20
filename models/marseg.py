import math
from functools import partial

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from tqdm import tqdm

from models.mar import mar_layers
from models.mar.diffloss import DiffLoss
from models.mar.vae import Encoder


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        scale = self.sigmoid(out)
        return scale


class FPNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, feature_strides, align_corners=False):
        super(FPNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.feature_strides = feature_strides
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, channels, kernel_size=1))

        self.fpn_convs = nn.ModuleList()
        for _ in in_channels:
            self.fpn_convs.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        laterals = [l_conv(x) for l_conv, x in zip(self.lateral_convs, inputs)]
        out = laterals[0]
        for i in range(1, len(laterals)):
            laterals[i] = F.interpolate(laterals[i],
                                        size=out.shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
            out = out + laterals[i]
        out = self.fpn_convs[0](out)
        out = self.cls_seg(out)
        return out


class MARSeg(nn.Module):
    def __init__(self,
                 mar_model,
                 pretrained_vae_path,
                 embed_dim,
                 ch_mult,
                 num_classes,
                 resolution=256,
                 layer_indices=(2, 5, 8, 11),
                 fuse_dim=256,
                 reduction_ratio=16,
                 pool_types=['avg','max'],
                 num_cbam_layers=4):
        super(MARSeg, self).__init__()
        self.mar = mar_model
        self.embed_dim = embed_dim
        self.resolution = resolution
        self.en_dim = mar_model.encoder_embed_dim
        self.fuse_dim = fuse_dim
        self.layer_indices = layer_indices
        self.num_layers = len(layer_indices)
        self.num_cbam_layers = num_cbam_layers
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.quant_conv = nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1)
        
        en_dim = mar_model.encoder_embed_dim  
        self.initial_projection_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(en_dim * 2, en_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(en_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_cbam_layers)
        ])
        self.spatial_attention_modules = nn.ModuleList([
            SpatialAttention(kernel_size=7) for _ in range(num_cbam_layers)
        ])
        self.channel_attention_modules = nn.ModuleList([
            ChannelAttention(in_channels=en_dim, ratio=16) for _ in range(num_cbam_layers)
        ])
        self.fusion_projection_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(en_dim * 2, fuse_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(fuse_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_cbam_layers)
        ])
        
        if pretrained_vae_path:
            checkpoint = torch.load(pretrained_vae_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            encoder_state_dict = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items() if k.startswith("encoder.")
            }
            msg_enc = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print("[Info] Loaded pretrained VAE checkpoint for encoder.")
            print("Encoder load message:", msg_enc)
        else:
            print("[Info] No pretrained VAE checkpoint provided. Using random init.")

        in_channels = [fuse_dim] * self.num_layers
        feature_strides = [1] * self.num_layers
        self.decoder = FPNHead(
            in_channels=in_channels,
            channels=fuse_dim,
            num_classes=num_classes,
            feature_strides=feature_strides,
            align_corners=False
        )

    def forward(self, x, class_label=None):

        z = self.encoder(x)
        z = self.quant_conv(z)

        if class_label is not None:
            class_embedding = self.mar.class_emb(class_label)
        else:
            class_embedding = self.mar.fake_latent.repeat(z.size(0), 1)

        z = z * 0.2325

        z_enc, encoder_feats_all = self.mar.forward_mae_encoder(z, class_embedding, return_all=True)
        decoder_feats_all = self.mar.forward_mae_decoder(z_enc, return_all=True)

        encoder_feats = [encoder_feats_all[i] for i in self.layer_indices]
        decoder_feats = [decoder_feats_all[i] for i in self.layer_indices]

        fused_feats = []

        for i in range(self.num_cbam_layers):
            enc_feat = encoder_feats[i]
            dec_feat = decoder_feats[i]
            bs, seq_len, dim = enc_feat.shape
            h = w = int(seq_len ** 0.5)
            if h * w != seq_len:
                raise ValueError(f"Layer {i}: Invalid seq_len {seq_len}. Must be a perfect square.")

            enc_feat_4d = enc_feat.permute(0, 2, 1).contiguous().reshape(bs, dim, h, w)
            dec_feat_4d = dec_feat.permute(0, 2, 1).contiguous().reshape(bs, dim, h, w)
            concat_feat = torch.cat([enc_feat_4d, dec_feat_4d], dim=1)
            proj_feat = self.initial_projection_modules[i](concat_feat)

            sa_map = self.spatial_attention_modules[i](proj_feat)
            sa_out = proj_feat * sa_map
            ca_map = self.channel_attention_modules[i](proj_feat)
            ca_out = proj_feat * ca_map

            merged_feat = torch.cat([sa_out, ca_out], dim=1)
            fused_feat = self.fusion_projection_modules[i](merged_feat)
            fused_feats.append(fused_feat)

        seg_logits = self.decoder(fused_feats)

        scale_factor = self.resolution / seg_logits.shape[-1]
        seg_logits = F.interpolate(seg_logits, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return seg_logits
