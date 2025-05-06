"""
This module is an optimized implementation of the Shadow Diffusion Model by Lanqing Guo et al. described in their paper
"ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal."

Author: Deja S.
Version: 1.1.0
Edited: 03-05-2025
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from inspect import isfunction
import torch.nn.functional as F
from einops import rearrange, repeat


# Helper functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    device = a.device
    t = t.to(device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# Positional encodings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# U-Net building blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return F.gelu(self.double_conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


# Memory-efficient implementation of self-attention
class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, size, heads=4, head_dim=None, downsample_factor=4):
        super(EfficientSelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.heads = heads
        self.downsample_factor = downsample_factor

        # Reduce spatial dimensions to save memory
        self.spatial_size = size // downsample_factor if size > downsample_factor else 1

        # Define head dimensions
        if head_dim is None:
            head_dim = channels // heads

        # Multi-head attention with linear projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

        # Spatial reduction
        self.spatial_reduction = nn.AvgPool2d(downsample_factor) if downsample_factor > 1 else nn.Identity()
        self.spatial_restore = nn.Upsample(scale_factor=downsample_factor,
                                           mode='bilinear') if downsample_factor > 1 else nn.Identity()

    def forward(self, x):
        # Store original shape
        B, C, H, W = x.shape
        residual = x

        # Reshape for attention
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # B, H*W, C
        x_norm = self.norm1(x_flat)

        original_seq_len = x_norm.shape[1]

        if self.downsample_factor > 1:
            # Reshape to spatial format for downsampling
            x_spatial = x_norm.permute(0, 2, 1).view(B, C, H, W)
            x_down = self.spatial_reduction(x_spatial)
            x_norm = x_down.view(B, C, -1).permute(0, 2, 1)  # B, (H/d)*(W/d), C

            down_seq_len = x_norm.shape[1]

        # Split into heads
        q = self.q_proj(x_norm).view(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)  # B, h, L, d

        # Use downsampled features for keys and values
        k = self.k_proj(x_norm).view(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)  # B, h, L, d
        v = self.v_proj(x_norm).view(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)  # B, h, L, d

        with torch.cuda.amp.autocast(enabled=True):
            # Calculate attention scores
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(C // self.heads)
            attention_probs = F.softmax(attention_scores, dim=-1)

            # Apply attention to values
            context = torch.matmul(attention_probs, v).permute(0, 2, 1, 3).contiguous()
            context = context.view(B, -1, C)
            context = self.out_proj(context)

            if self.downsample_factor > 1:
                down_h, down_w = H // self.downsample_factor, W // self.downsample_factor
                context = context.permute(0, 2, 1).view(B, C, down_h, down_w)
                context = self.spatial_restore(context)
                context = context.view(B, C, H * W).permute(0, 2, 1)

            context = context + x_flat

        context = self.norm2(context)
        ff_output = self.ff(context)
        output = ff_output + context
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output + residual


# U-Net with timestep embedding and attention
class ShadowDiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, time_emb_dim=256, base_channels=32, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        c = base_channels

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        # Encoder with reduced channels
        self.inc = DoubleConv(c, c, residual=True)
        self.down1 = Down(c, c * 2, emb_dim=time_emb_dim)
        self.sa1 = EfficientSelfAttention(c * 2, 128, downsample_factor=4)
        self.down2 = Down(c * 2, c * 4, emb_dim=time_emb_dim)
        self.sa2 = EfficientSelfAttention(c * 4, 64, downsample_factor=4)
        self.down3 = Down(c * 4, c * 8, emb_dim=time_emb_dim)
        self.sa3 = EfficientSelfAttention(c * 8, 32, downsample_factor=2)
        self.down4 = Down(c * 8, c * 16, emb_dim=time_emb_dim)
        self.sa4 = EfficientSelfAttention(c * 16, 16, downsample_factor=2)

        # Bottleneck with reduced channels
        self.bot1 = DoubleConv(c * 16, c * 16)
        self.bot2 = DoubleConv(c * 16, c * 16)
        self.bot3 = DoubleConv(c * 16, c * 16)
        self.bot_sa = EfficientSelfAttention(c * 16, 16, downsample_factor=1)

        # Decoder with reduced channels
        self.up1 = Up(c * 16 + c * 8, c * 8, emb_dim=time_emb_dim)
        self.sa5 = EfficientSelfAttention(c * 8, 32, downsample_factor=2)
        self.up2 = Up(c * 8 + c * 4, c * 4, emb_dim=time_emb_dim)
        self.sa6 = EfficientSelfAttention(c * 4, 64, downsample_factor=4)
        self.up3 = Up(c * 4 + c * 2, c * 2, emb_dim=time_emb_dim)
        self.sa7 = EfficientSelfAttention(c * 2, 128, downsample_factor=4)
        self.up4 = Up(c * 2 + c, c, emb_dim=time_emb_dim)

        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(1, c),
            nn.GELU(),
            nn.Conv2d(c, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, shadow_mask, t):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # Lower precision for efficient computation
        with torch.cuda.amp.autocast(enabled=True):
            x = torch.cat([x, shadow_mask], dim=1)
            x = self.conv0(x)
            t = self.time_mlp(t)
            x1 = self.inc(x)

            # Apply gradient checkpointing
            if self.use_checkpoint and self.training:
                x2 = torch.utils.checkpoint.checkpoint(create_custom_forward(self.down1), x1, t)
                x2 = torch.utils.checkpoint.checkpoint(self.sa1, x2)
                x3 = torch.utils.checkpoint.checkpoint(create_custom_forward(self.down2), x2, t)
                x3 = torch.utils.checkpoint.checkpoint(self.sa2, x3)
                x4 = torch.utils.checkpoint.checkpoint(create_custom_forward(self.down3), x3, t)
                x4 = torch.utils.checkpoint.checkpoint(self.sa3, x4)
                x5 = torch.utils.checkpoint.checkpoint(create_custom_forward(self.down4), x4, t)
                x5 = torch.utils.checkpoint.checkpoint(self.sa4, x5)

                # Bottleneck
                x5 = torch.utils.checkpoint.checkpoint(self.bot1, x5)
                x5 = torch.utils.checkpoint.checkpoint(self.bot2, x5)
                x5 = torch.utils.checkpoint.checkpoint(self.bot3, x5)
                x5 = torch.utils.checkpoint.checkpoint(self.bot_sa, x5)

                # Decoder with skip connections
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.up1), x5, x4, t)
                x = torch.utils.checkpoint.checkpoint(self.sa5, x)
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.up2), x, x3, t)
                x = torch.utils.checkpoint.checkpoint(self.sa6, x)
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.up3), x, x2, t)
                x = torch.utils.checkpoint.checkpoint(self.sa7, x)
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.up4), x, x1, t)
            else:
                x2 = self.down1(x1, t)
                x2 = self.sa1(x2)
                x3 = self.down2(x2, t)
                x3 = self.sa2(x3)
                x4 = self.down3(x3, t)
                x4 = self.sa3(x4)
                x5 = self.down4(x4, t)
                x5 = self.sa4(x5)

                # Bottleneck
                x5 = self.bot1(x5)
                x5 = self.bot2(x5)
                x5 = self.bot3(x5)
                x5 = self.bot_sa(x5)

                # Decoder with skip connections
                x = self.up1(x5, x4, t)
                x = self.sa5(x)
                x = self.up2(x, x3, t)
                x = self.sa6(x)
                x = self.up3(x, x2, t)
                x = self.sa7(x)
                x = self.up4(x, x1, t)

            # Output projection
            output = self.outc(x)
            return output


# Diffusion model core components
class GaussianDiffusion(nn.Module):
    def __init__(self, model, betas, image_size=512, channels=3):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels

        # Define beta schedule
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))
        self.register_buffer('sqrt_recipm1_alphas', torch.sqrt(1.0 / self.alphas - 1))
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             torch.cat([torch.ones(1, device=betas.device), self.sqrt_alphas_cumprod[:-1]]))
        posterior_variance = self.betas * (1. - self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             self.betas * torch.sqrt(self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - self.sqrt_alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                                         1. - self.alphas_cumprod))

    # Forward diffusion
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_recipm1_alphas_t = extract(self.sqrt_recipm1_alphas, t, x_t.shape)

        return sqrt_recip_alphas_t * x_t - sqrt_recipm1_alphas_t * noise

    def q_posterior_mean(self, x_0, x_t, t):
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_t.shape)

        mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        return mean

    def q_posterior_variance_and_log(self, t):
        posterior_variance_t = extract(self.posterior_variance, t, (t.shape[0],))
        posterior_log_variance_t = extract(self.posterior_log_variance_clipped, t, (t.shape[0],))
        return posterior_variance_t, posterior_log_variance_t

    def forward(self, shadow_mask, x_0):
        with torch.cuda.amp.autocast(enabled=True):
            t = torch.randint(0, len(self.betas), (x_0.shape[0],), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            predicted_noise = self.model(x_t, shadow_mask, t)

            loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, shadow_mask, x, t, t_index):
        with torch.cuda.amp.autocast(enabled=True):
            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
            model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * self.model(x, shadow_mask, t) / sqrt_one_minus_alphas_cumprod_t
            )

            if t_index == 0:
                return model_mean
            else:
                posterior_variance_t = extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shadow_mask, image_size=None, batch_size=1, channels=None, device=None, steps=100):
        if image_size is None:
            image_size = self.image_size
        if channels is None:
            channels = self.channels
        if device is None:
            device = next(self.model.parameters()).device

        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=device)
        time_steps = torch.linspace(len(self.betas) - 1, 0, steps, dtype=torch.long, device=device)

        for i, step in enumerate(time_steps):
            if i % 10 == 0 and i > 0:
                torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=True):
                img = self.p_sample(
                    shadow_mask,
                    img,
                    step.expand(batch_size),
                    i
                )

        return img

    def __call__(self, shadow_mask, shadow_img, steps=50):
        batch_size = shadow_img.shape[0]
        device = shadow_img.device
        steps = min(steps, len(self.betas))
        x = torch.randn(shadow_img.shape, device=device)
        time_steps = torch.linspace(len(self.betas) - 1, 0, steps, dtype=torch.long, device=device)

        for i, step in enumerate(time_steps):
            if i % 10 == 0 and i > 0:
                torch.cuda.empty_cache()

            t = step.expand(batch_size)
            x = self.p_sample(shadow_mask, x, t, i)

        x = torch.clamp(x, -1, 1)
        return x


def create_model(opts, mode="train"):
    # Create diffusion schedule
    if mode == "train":
        timesteps = opts['model']['beta_schedule']['train']['n_timestep']
        beta_start = opts['model']['beta_schedule']['train']['linear_start']
        beta_end = opts['model']['beta_schedule']['train']['linear_end']
        betas = torch.linspace(beta_start, beta_end, timesteps)
    else:
        timesteps = opts['model']['beta_schedule']['val']['n_timestep']
        beta_start = opts['model']['beta_schedule']['val']['linear_start']
        beta_end = opts['model']['beta_schedule']['val']['linear_end']
        betas = torch.linspace(beta_start, beta_end, timesteps)

    model = ShadowDiffusionUNet(
        in_channels=4,
        out_channels=3,
        time_emb_dim=256,
        base_channels=32,
        use_checkpoint=True
    )

    # Create diffusion model
    diffusion = GaussianDiffusion(
        model=model,
        betas=betas,
        image_size=512,
        channels=3
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = diffusion.to(device)

    return diffusion


def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint)
    except Exception as e:
        # Print warning and try loading with strict=False
        print(f"Warning: Could not load checkpoint strictly. Error: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(checkpoint, strict=False)
    return model

def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
    device = self.betas.device

    if schedule_phase == 'train':
        timesteps = schedule_opt['n_timestep']
        beta_start = schedule_opt['linear_start']
        beta_end = schedule_opt['linear_end']
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        self.betas = betas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1)
        self.sqrt_alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.sqrt_alphas_cumprod[:-1]])
        posterior_variance = self.betas * (1. - self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.sqrt_alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)
    else:
        raise NotImplementedError(f"Schedule phase {schedule_phase} not recognized.")

# Add the method to the GaussianDiffusion class
GaussianDiffusion.set_new_noise_schedule = set_new_noise_schedule