# This code is from the original repository:
# https://github.com/LTH14/mar

# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_)  # b, c,hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):

        temb = None

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# class ModifiedEncoder(nn.Module):
#     def __init__(
#         self,
#         *,
#         ch=128,
#         out_ch=3,
#         ch_mult=(1, 1, 2, 2, 4),
#         num_res_blocks=2,
#         attn_resolutions=(16,),
#         dropout=0.0,
#         resamp_with_conv=True,
#         in_channels=3,
#         resolution=256,
#         z_channels=16,
#         double_z=True,
#         **ignore_kwargs,
#     ):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#         self.z_channels = z_channels

#         self.conv_in = torch.nn.Conv2d(
#             in_channels, self.ch, kernel_size=3, stride=1, padding=1
#         )

#         curr_res = resolution
#         in_ch_mult = (1,) + tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch * in_ch_mult[i_level]
#             block_out = ch * ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(
#                     ResnetBlock(
#                         in_channels=block_in,
#                         out_channels=block_out,
#                         temb_channels=self.temb_ch,
#                         dropout=dropout,
#                     )
#                 )
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions - 1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(
#             in_channels=block_in,
#             out_channels=block_in,
#             temb_channels=self.temb_ch,
#             dropout=dropout,
#         )
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(
#             in_channels=block_in,
#             out_channels=block_in,
#             temb_channels=self.temb_ch,
#             dropout=dropout,
#         )

#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(
#             block_in,
#             2 * z_channels if double_z else z_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#     def init_from_ckpt(self, path):
#         checkpoint = torch.load(path, map_location="cpu")
        
#         if "model" in checkpoint:
#             sd = checkpoint["model"]
#         elif "model_state_dict" in checkpoint:
#             sd = checkpoint["model_state_dict"]
#         else:
#             sd = checkpoint
        
#         sd = {k.replace('encoder.', ''): v for k, v in sd.items()}
        
#         msg = self.load_state_dict(sd, strict=False)
#         print("Loading pre-trained KL-VAE")
#         print("Missing keys:", msg.missing_keys)
#         print("Unexpected keys:", msg.unexpected_keys)
#         print(f"Restored from {path}")

#     def forward(self, x):

#         temb = None

#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions - 1:
#                 h = self.down[i_level].downsample(hs[-1])
#                 hs.append(h)

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         z = self.conv_out(h)
#         hs.append(z)

#         return (z, hs)


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, current_resolution=None, attn_resolutions=None):
        super().__init__()
        self.upsample = Upsample(in_channels, with_conv=True)
        self.resblock = ResnetBlock(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            temb_channels=0,
            dropout=dropout
        )
        if current_resolution is not None and attn_resolutions is not None:
            self.attn = AttnBlock(out_channels) if current_resolution in attn_resolutions else nn.Identity()
        else:
            self.attn = nn.Identity()

    def forward(self, x, skip):
        
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, temb=None)
        x = self.attn(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SegmentationModel(nn.Module):
    def __init__(self, encoder, mar_model, embed_dim, ch_mult, num_res_blocks, attn_resolutions, dropout=0.0, num_classes=21, resolution=256):
        super().__init__()
        self.encoder = encoder
        self.mar = mar_model

        self.decoder_blocks = nn.ModuleList()
        self.num_resolutions = len(ch_mult)
        self.ch_mult = ch_mult
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions

        self.skip_indices = [2 + i*(self.num_res_blocks +1) for i in range(self.num_resolutions -1)]
        self.skip_indices = self.skip_indices[::-1]  # [11,8,5,2]

        decoder_ch_mult = [2, 2, 1, 1]

        in_ch = encoder.z_channels * 2
        current_resolution = resolution // (2 ** (len(ch_mult) -1 ))

        for mult in decoder_ch_mult:
            out_ch = encoder.ch * mult
            decoder_block = UNetDecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout=dropout,
                current_resolution=current_resolution,
                attn_resolutions=attn_resolutions
            )
            self.decoder_blocks.append(decoder_block)
            in_ch = out_ch
            current_resolution *=2

        self.segmentation_head = SegmentationHead(in_channels=encoder.ch, num_classes=num_classes)
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(768, embed_dim*2, kernel_size=1)
        self.embed_dim = embed_dim

    def forward(self, x, class_label):
        z, hs = self.encoder(x)
        if class_label is not None:
            class_embedding = self.mar.class_emb(class_label)
        else:
            class_embedding = self.mar.fake_latent.repeat(z.size(0), 1)
        z = self.quant_conv(z)
        z = z.mul_(0.2325)
        z_mae = self.mar.forward_mae_encoder(z, class_embedding)
        z_mae = z_mae / 0.2325

        bs, seq_len, embed_dim = z_mae.shape
        h = w = int(seq_len**0.5)
        if h * w != seq_len:
            raise ValueError(f"Invalid seq_len {seq_len}. It must be a perfect square.")
        z_mae = z_mae.permute(0, 2, 1).reshape(bs, embed_dim, h, w)
        
        z_mae = self.post_quant_conv(z_mae)
        
        skips = [hs[i] for i in self.skip_indices]
        
        x = z_mae
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                pass
            else:
                pass
            x = decoder_block(x, skip)

        seg_map = self.segmentation_head(x)
        return seg_map


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
