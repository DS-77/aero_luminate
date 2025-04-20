"""
This module is a modified implementation of the Shadow Diffusion Model by Lanqing Guo et al. described in their paper
"ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal."

Author: Deja S.
Version: 1.0.0
Edited: 19-04-2025
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


# def extract(a, t, x_shape):
#     batch_size = t.shape[0]
#     out = a.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    device = a.device  # Get the device of tensor 'a'
    t = t.to(device) # Move 't' to the same device as 'a'
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


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


# U-Net with timestep embedding and attention
class ShadowDiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, time_emb_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Encoder
        self.inc = DoubleConv(64, 64, residual=True)
        self.down1 = Down(64, 128, emb_dim=time_emb_dim)
        # Size after down1 for 512x512: 256x256
        self.sa1 = SelfAttention(128, 128)
        self.down2 = Down(128, 256, emb_dim=time_emb_dim)
        # Size after down2 for 512x512: 128x128
        self.sa2 = SelfAttention(256, 64)
        self.down3 = Down(256, 512, emb_dim=time_emb_dim)
        # Size after down3 for 512x512: 64x64
        self.sa3 = SelfAttention(512, 32)
        self.down4 = Down(512, 1024, emb_dim=time_emb_dim)
        # Size after down4 for 512x512: 32x32
        self.sa4 = SelfAttention(1024, 16)

        # Bottleneck
        self.bot1 = DoubleConv(1024, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 1024)
        # Size at bottleneck for 512x512: 32x32
        self.bot_sa = SelfAttention(1024, 16)

        # Decoder
        self.up1 = Up(1024 + 512, 512, emb_dim=time_emb_dim)
        # Size after up1 for 512x512: 64x64
        self.sa5 = SelfAttention(512, 32)
        self.up2 = Up(512 + 256, 256, emb_dim=time_emb_dim)
        # Size after up2 for 512x512: 128x128
        self.sa6 = SelfAttention(256, 64)
        self.up3 = Up(256 + 128, 128, emb_dim=time_emb_dim)
        # Size after up3 for 512x512: 256x256
        self.sa7 = SelfAttention(128, 128)
        self.up4 = Up(128 + 64, 64, emb_dim=time_emb_dim)

        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, shadow_mask, t):
        # Concatenate shadow image with shadow mask
        x = torch.cat([x, shadow_mask], dim=1)

        # Initial projection
        x = self.conv0(x)

        # Time embedding
        t = self.time_mlp(t)

        # Encoder path with skip connections
        x1 = self.inc(x)
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

        # Return shadow-free image
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

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.cat([torch.ones(1), self.sqrt_alphas_cumprod[:-1]]))
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = self.betas * (1. - self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             self.betas * torch.sqrt(self.sqrt_alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - self.sqrt_alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))

    # Forward diffusion (add noise)
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the predicted x_0 from x_t and predicted noise
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_recipm1_alphas_t = extract(self.sqrt_recipm1_alphas, t, x_t.shape)

        return sqrt_recip_alphas_t * x_t - sqrt_recipm1_alphas_t * noise

    # Get the mean for posterior distribution q(x_{t-1} | x_t, x_0)
    def q_posterior_mean(self, x_0, x_t, t):
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_t.shape)

        mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        return mean

    # Get the variance and log variance for q(x_{t-1} | x_t, x_0)
    def q_posterior_variance_and_log(self, t):
        posterior_variance_t = extract(self.posterior_variance, t, (t.shape[0],))
        posterior_log_variance_t = extract(self.posterior_log_variance_clipped, t, (t.shape[0],))
        return posterior_variance_t, posterior_log_variance_t

    # Forward pass
    def forward(self, shadow_mask, x_0):
        t = torch.randint(0, len(self.betas), (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Predict the noise
        predicted_noise = self.model(x_t, shadow_mask, t)

        # Loss is MSE between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    # Sampling
    @torch.no_grad()
    def p_sample(self, shadow_mask, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, shadow_mask, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Generate samples with progressive denoising
    @torch.no_grad()
    def sample(self, shadow_mask, image_size=None, batch_size=1, channels=None, device=None):
        if image_size is None:
            image_size = self.image_size
        if channels is None:
            channels = self.channels
        if device is None:
            device = next(self.model.parameters()).device

        shape = (batch_size, channels, image_size, image_size)
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, len(self.betas))):
            img = self.p_sample(
                shadow_mask,
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i
            )

        return img

    # Remove shadows from input image with shadow mask
    def __call__(self, shadow_mask, shadow_img):
        batch_size = shadow_img.shape[0]
        device = shadow_img.device

        # Generate shadow-free image through denoising process
        steps = 100

        # Start from a random noise image
        x = torch.randn(shadow_img.shape, device=device)

        # Progressive denoising
        for i in reversed(range(0, len(self.betas), len(self.betas) // steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(shadow_mask, x, t, i)

        # Ensure output is in valid image range
        x = torch.clamp(x, -1, 1)
        return x


# Create the complete model
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

    # Create U-Net model
    model = ShadowDiffusionUNet(
        in_channels=4,
        out_channels=3,
        time_emb_dim=256
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
    model.load_state_dict(torch.load(checkpoint_path))
    return model


# Method to set new noise schedules
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

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.sqrt_alphas_cumprod[:-1]])
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
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