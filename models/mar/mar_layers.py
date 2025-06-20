# This code is modified from the original repository:
# https://github.com/LTH14/mar

from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from timm.models.vision_transformer import Block

from models.mar.diffloss import DiffLoss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]
        p = self.patch_size
        c = self.vae_embed_dim

        h_ = w_ = int(math.sqrt(seq_len))
        if h_ * w_ != seq_len:
            raise ValueError(f"Invalid seq_len: {seq_len}. Expected a perfect square.")

        if embed_dim != c * p ** 2:
            raise ValueError(f"Invalid embed_dim: {embed_dim}. Expected {c * p ** 2}.")

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x
    
    def interpolate_pos_embed(self, pos_embed, target_len):
        buffer_size = self.buffer_size
        orig_patch_len = pos_embed.shape[1] - buffer_size
        new_patch_len = target_len - buffer_size
        if new_patch_len == orig_patch_len:
            return pos_embed

        buffer_embed = pos_embed[:, :buffer_size, :]
        patch_embed = pos_embed[:, buffer_size:, :]

        orig_size = int(math.sqrt(orig_patch_len))
        new_size = int(math.sqrt(new_patch_len))
        patch_embed = patch_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_patch_len, -1)
        new_pos_embed = torch.cat([buffer_embed, patch_embed], dim=1)
        return new_pos_embed
    
    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = 0
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                     src=torch.ones(bsz, seq_len, device=x.device))
        print(f"mask in random masking. everything should be 0: {mask}")
        return mask


    def forward_mae_encoder(self, x, class_embedding, return_all=False):
        x = self.patchify(x)
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        buffer_tokens = torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device)
        x = torch.cat([buffer_tokens, x], dim=1)

        if self.training:
            drop_latent_mask = (torch.rand(bsz, device=x.device) < self.label_drop_prob).unsqueeze(-1).to(x.dtype)
            class_embedding = class_embedding.to(x.device)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        current_len = x.shape[1]
        if self.img_size == 256:
            pos_embed = self.encoder_pos_embed_learned
        elif self.img_size == 224:
            if current_len != self.encoder_pos_embed_learned.shape[1]:
                pos_embed = self.interpolate_pos_embed(self.encoder_pos_embed_learned, current_len)
            else:
                pos_embed = self.encoder_pos_embed_learned
        else:
            if current_len != self.encoder_pos_embed_learned.shape[1]:
                pos_embed = self.interpolate_pos_embed(self.encoder_pos_embed_learned, current_len)
            else:
                pos_embed = self.encoder_pos_embed_learned

        x = x + pos_embed
        x = self.z_proj_ln(x)

        if return_all:
            outputs = []
            if self.grad_checkpointing and not torch.jit.is_scripting():
                for block in self.encoder_blocks:
                    x = checkpoint(block, x)
                    outputs.append(x)
            else:
                for block in self.encoder_blocks:
                    x = block(x)
                    outputs.append(x)
            outputs = [self.encoder_norm(o) for o in outputs]
            outputs = [o[:, self.buffer_size:] for o in outputs]
            x = self.encoder_norm(x)
            x_with_buffer = x
            return x_with_buffer, outputs
        else:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                for block in self.encoder_blocks:
                    x = checkpoint(block, x)
            else:
                for block in self.encoder_blocks:
                    x = block(x)
            x = self.encoder_norm(x)
            x_with_buffer = x
            x_without_buffer = x[:, self.buffer_size:]
            return x_with_buffer, x_without_buffer


    def forward_mae_decoder(self, x, return_all=False):
        if isinstance(x, list):
            outputs_list = []
            for xi in x:
                y = self.decoder_embed(xi)
                current_len = y.shape[1]
                if self.img_size == 256:
                    pos_embed = self.decoder_pos_embed_learned
                elif self.img_size == 224:
                    if current_len == self.seq_len:
                        base_embed = self.decoder_pos_embed_learned[:, self.buffer_size:]
                        if current_len != base_embed.shape[1]:
                            pos_embed = self.interpolate_pos_embed(base_embed, current_len)
                        else:
                            pos_embed = base_embed
                    else:
                        if current_len != self.decoder_pos_embed_learned.shape[1]:
                            pos_embed = self.interpolate_pos_embed(self.decoder_pos_embed_learned, current_len)
                        else:
                            pos_embed = self.decoder_pos_embed_learned
                else:
                    if current_len != self.decoder_pos_embed_learned.shape[1]:
                        pos_embed = self.interpolate_pos_embed(self.decoder_pos_embed_learned, current_len)
                    else:
                        pos_embed = self.decoder_pos_embed_learned

                y = y + pos_embed
                if return_all:
                    outs = []
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        for block in self.decoder_blocks:
                            y = checkpoint(block, y)
                            outs.append(y)
                    else:
                        for block in self.decoder_blocks:
                            y = block(y)
                            outs.append(y)
                    outs = [self.decoder_norm(o) for o in outs]
                    final_output = outs[-1][:, self.buffer_size:]
                    outputs_list.append(final_output)
                else:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        for block in self.decoder_blocks:
                            y = checkpoint(block, y)
                    else:
                        for block in self.decoder_blocks:
                            y = block(y)
                    y = self.decoder_norm(y)
                    y = y[:, self.buffer_size:]
                    outputs_list.append(y)
            return outputs_list
        else:
            y = self.decoder_embed(x)
            current_len = y.shape[1]

            if self.img_size == 256:
                pos_embed = self.decoder_pos_embed_learned
            elif self.img_size == 224:
                if current_len == self.seq_len:
                    base_embed = self.decoder_pos_embed_learned[:, self.buffer_size:]
                    if current_len != base_embed.shape[1]:
                        pos_embed = self.interpolate_pos_embed(base_embed, current_len)
                    else:
                        pos_embed = base_embed
                else:
                    if current_len != self.decoder_pos_embed_learned.shape[1]:
                        pos_embed = self.interpolate_pos_embed(self.decoder_pos_embed_learned, current_len)
                    else:
                        pos_embed = self.decoder_pos_embed_learned
            else:
                if current_len != self.decoder_pos_embed_learned.shape[1]:
                    pos_embed = self.interpolate_pos_embed(self.decoder_pos_embed_learned, current_len)
                else:
                    pos_embed = self.decoder_pos_embed_learned

            y = y + pos_embed

            if return_all:
                outputs = []
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    for block in self.decoder_blocks:
                        y = checkpoint(block, y)
                        outputs.append(y)
                else:
                    for block in self.decoder_blocks:
                        y = block(y)
                        outputs.append(y)
                outputs = [self.decoder_norm(o) for o in outputs]
                outputs = [o[:, self.buffer_size:] for o in outputs]
                return outputs
            else:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    for block in self.decoder_blocks:
                        y = checkpoint(block, y)
                else:
                    for block in self.decoder_blocks:
                        y = block(y)
                y = self.decoder_norm(y)
                y = y[:, self.buffer_size:]
                return y



    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):
        x = self.patchify(imgs)
        
        # mae encoder
        x = self.forward_mae_encoder(x, class_embedding=None)

        # mae decoder
        x = self.unpatchify(x)
        return x

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        for step in indices:
            cur_tokens = tokens.clone()


            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x)


            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()



            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            mask_next = mask_by_order(mask_len, orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            z = z[mask_to_pred.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens
    
def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
