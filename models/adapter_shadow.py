"""
This module is an implementation of AdapterShadow by Leiping Jie et al. as described in their paper "AdapterShadow:
Adapting Segment Anything Model for Shadow Detection."

Author: Deja S.
Version: 1.0.0
Edited: 19-04-2025
"""

import math
import copy
global math
import torch
import numpy as np
import torch.nn as nn
from sympy import false
from functools import partial
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d
from typing import Dict, List, Optional, Tuple, Type, Any
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.image_encoder import ImageEncoderViT, Block, PatchEmbed


class AdapterBlock(nn.Module):
    """
    Adapter module
    """

    def __init__(
            self,
            dim: int,
            adapter_dim: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run adapter forward pass"""
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x + residual


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for efficient fine-tuning
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            rank: int = 4,
            alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # LoRA components
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        # Initialise with near-zero weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


class VPTLayer(nn.Module):
    """
    Visual Prompt Tuning (VPT) module
    """

    def __init__(
            self,
            embedding_dim: int,
            num_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.vpt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, embedding_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.vpt_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        vpt_embeddings = self.vpt_embeddings.expand(batch_size, -1, -1)
        return torch.cat([vpt_embeddings, x], dim=1)


class ShadowMultiBranchAttention(nn.Module):
    """
    Multi-branch attention module for shadow detection
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            mb_ratio: float = 0.25
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Main branch
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Shadow branch
        shadow_dim = int(dim * mb_ratio)
        self.shadow_dim = shadow_dim
        self.shadow_qkv = nn.Linear(dim, shadow_dim * 3, bias=qkv_bias)
        self.shadow_proj = nn.Linear(shadow_dim, dim)

        # Branch fusion parameters
        self.branch_fusion = nn.Parameter(torch.ones(2))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Main branch processing
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x_main = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_main = self.proj(x_main)

        # Shadow branch processing
        shadow_qkv = self.shadow_qkv(x).reshape(B, N, 3, self.num_heads, self.shadow_dim // self.num_heads).permute(2,
                                                                                                                    0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    4)
        shadow_q, shadow_k, shadow_v = shadow_qkv.unbind(0)

        # Shadow attention
        shadow_attn = (shadow_q @ shadow_k.transpose(-2, -1)) * self.scale
        shadow_attn = shadow_attn.softmax(dim=-1)

        # Apply shadow attention to values
        x_shadow = (shadow_attn @ shadow_v).transpose(1, 2).reshape(B, N, self.shadow_dim)
        x_shadow = self.shadow_proj(x_shadow)

        # Combine branches with learned weights
        fusion_weights = self.softmax(self.branch_fusion)
        x = fusion_weights[0] * x_main + fusion_weights[1] * x_shadow

        return x


class AdapterShadowBlock(nn.Module):
    """
    Modified transformer block with adapters and multi-branch attention
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path: float = 0.0,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            mb_ratio: float = 0.25,
            use_adapter: bool = True,
            adapter_dim: int = 64,
            use_lora: bool = False,
            lora_rank: int = 4
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        # Shadow-specific multi-branch attention
        self.attn = ShadowMultiBranchAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mb_ratio=mb_ratio
        )

        self.use_lora = use_lora
        if use_lora:
            self.attn_lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.attn_lora_B = nn.Linear(lora_rank, dim, bias=False)
            self.scale = 1.0 / lora_rank

            # Initialise with near-zero weights
            nn.init.kaiming_uniform_(self.attn_lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.attn_lora_B.weight)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # Drop path for regularization
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

        # Adapter modules
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter1 = AdapterBlock(dim, adapter_dim)
            self.adapter2 = AdapterBlock(dim, adapter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with adapter and LoRA
        residual = x
        x = self.norm1(x)
        x = self.attn(x)

        # Apply LoRA if enabled
        if self.use_lora:
            x = x + self.attn_lora_B(self.attn_lora_A(self.norm1(residual))) * self.scale

        # Apply adapter if enabled
        if self.use_adapter:
            x = x + self.adapter1(x)

        x = residual + self.drop_path(x)

        # MLP block with adapter
        residual = x
        x = self.mlp(self.norm2(x))

        # Apply adapter if enabled
        if self.use_adapter:
            x = x + self.adapter2(x)

        x = residual + self.drop_path(x)
        return x


class ShadowSkipAdapter(nn.Module):
    """
    Skip connection adapter for shadow features
    """

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    """

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# class AdaptedImageEncoder(nn.Module):
#     """
#     Modified SAM Image Encoder with AdapterShadow components
#     """
#
#     def __init__(self, args, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.args = args
#
#         # Original SAM image encoder attributes
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         # Patch embedding
#         self.patch_embed = PatchEmbed(
#             kernel_size=(patch_size, patch_size),
#             stride=(patch_size, patch_size),
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#         )
#
#         # VPT (Visual Prompt Tuning)
#         self.use_vpt = getattr(args, "vpt", False)
#         if self.use_vpt:
#             self.vpt = VPTLayer(embed_dim, num_tokens=10)
#             num_patches = (img_size // patch_size) ** 2 + 10
#         else:
#             num_patches = (img_size // patch_size) ** 2
#
#         # Position embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#
#         # Initialise configurations for different architecture components
#         self.mb_ratio = getattr(args, "mb_ratio", 0.25)  # Multi-branch ratio
#         self.use_adapter = getattr(args, "all", False)
#         self.adapter_dim = embed_dim // 4
#         self.use_lora = getattr(args, "lora", False)
#         self.lora_rank = 4
#         self.use_skip_adapter = getattr(args, "skip_adapter", False)
#
#         # Number of encoder layers -- Default: 12 for ViT-B
#         depth = 12
#
#         # Create encoder blocks with adapter components
#         self.blocks = nn.ModuleList([
#             AdapterShadowBlock(
#                 dim=embed_dim,
#                 num_heads=12,
#                 mlp_ratio=4.0,
#                 qkv_bias=True,
#                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                 mb_ratio=self.mb_ratio,
#                 use_adapter=self.use_adapter,
#                 adapter_dim=self.adapter_dim,
#                 use_lora=self.use_lora,
#                 lora_rank=self.lora_rank
#             )
#             for _ in range(depth)
#         ])
#
#         # Skip connections with adaptation
#         if self.use_skip_adapter:
#             self.skip_adapters = nn.ModuleList([
#                 ShadowSkipAdapter(embed_dim, embed_dim)
#                 for _ in range(depth // 4)  # Create skip connections every 4 blocks
#             ])
#
#         # Final norm layer
#         self.norm = nn.LayerNorm(embed_dim)
#
#         # Shadow-specific head
#         self.shadow_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, embed_dim // 4),
#             nn.GELU(),
#             nn.Linear(embed_dim // 4, 1)
#         )
#
#         # Upsampling layers to generate full-resolution mask
#         self.neck = nn.Sequential(
#             nn.Conv2d(embed_dim, 256, kernel_size=1),
#             LayerNorm2d(256),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             LayerNorm2d(256)
#         )
#
#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             LayerNorm2d(128),
#             nn.GELU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             LayerNorm2d(64),
#             nn.GELU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             LayerNorm2d(32),
#             nn.GELU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#             LayerNorm2d(16),
#             nn.GELU(),
#             nn.Conv2d(16, 1, kernel_size=1)
#         )
#
#         # Initialise weights
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # Initialise patch embedding
#         nn.init.normal_(self.pos_embed, std=0.02)
#
#         # Initialise shadow head
#         for m in self.shadow_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.shape[0]
#
#         # Patch embedding
#         x = self.patch_embed(x)
#
#         # Add visual prompt tokens
#         if self.use_vpt:
#             x = self.vpt(x)
#
#         # Add positional embedding
#         print(x.shape)
#         print(self.pos_embed.shape)
#         exit()
#         x = x + self.pos_embed
#
#         # Process through transformer blocks with skip connections
#         skip_features = []
#         skip_idx = 0
#
#         for i, block in enumerate(self.blocks):
#             x = block(x)
#
#             # Save features for skip connections
#             if self.use_skip_adapter and i % 4 == 3:
#                 skip_features.append(self.skip_adapters[skip_idx](x))
#                 skip_idx += 1
#
#         # Apply final normalisation
#         x = self.norm(x)
#
#         # Reshape for convolutional processing
#         # Remove VPT tokens
#         if self.use_vpt:
#             x = x[:, 10:]
#
#         # Reshape to 2D feature map
#         H = W = int(math.sqrt(x.shape[1]))
#         x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
#
#         # Apply neck
#         x = self.neck(x)
#
#         # Add skip connections
#         if self.use_skip_adapter and len(skip_features) > 0:
#             # Reshape skip features to 2D
#             skip_feat = torch.stack(skip_features, dim=1).mean(dim=1)
#             skip_feat = skip_feat.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
#             skip_feat = self.neck(skip_feat)
#             x = x + skip_feat
#
#         return x
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Get features
#         features = self.forward_features(x)
#
#         # Upsample to original resolution and generate shadow mask
#         shadow_mask = self.upsample(features)
#
#         # Apply sigmoid for binary mask
#         shadow_mask = torch.sigmoid(shadow_mask)
#
#         return shadow_mask

class AdaptedImageEncoder(nn.Module):
    """
    Modified SAM Image Encoder with AdapterShadow components
    """

    def __init__(self, args, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.args = args

        # Original SAM image encoder attributes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Calculate expected feature map size
        feature_size = img_size // patch_size

        # Initialise position embeddings for 4D tensors
        self.pos_embed = nn.Parameter(torch.zeros(1, feature_size, feature_size, embed_dim))

        # Initialise configurations for different architecture components
        self.mb_ratio = getattr(args, "mb_ratio", 0.25)
        self.use_adapter = getattr(args, "all", False)
        self.adapter_dim = embed_dim // 4
        self.use_lora = getattr(args, "lora", False)
        self.lora_rank = 4
        self.use_skip_adapter = getattr(args, "skip_adapter", False)
        self.use_vpt = getattr(args, "vpt", False)

        # Number of encoder blocks
        depth = 12

        # Create encoder blocks with adapter components
        self.blocks = nn.ModuleList([
            AdapterShadowBlock(
                dim=embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                mb_ratio=self.mb_ratio,
                use_adapter=self.use_adapter,
                adapter_dim=self.adapter_dim,
                use_lora=self.use_lora,
                lora_rank=self.lora_rank
            )
            for _ in range(depth)
        ])

        # Skip connections with adaptation
        if self.use_skip_adapter:
            self.skip_adapters = nn.ModuleList([
                ShadowSkipAdapter(embed_dim, embed_dim)
                for _ in range(depth // 4)
            ])

        # Final norm layer
        self.norm = nn.LayerNorm(embed_dim)

        # Shadow-specific head
        self.shadow_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

        # Upsampling layers to generate full-resolution mask
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            LayerNorm2d(256)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Initialise weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialise patch embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialise shadow head
        for m in self.shadow_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Check shapes of x and pos_embed to fix a few errors
        if x.shape[1:3] != self.pos_embed.shape[1:3]:
            # Resize positional embeddings to match feature map size
            H, W = x.shape[1:3]
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = self.pos_embed

        # Add positional embedding
        x = x + pos_embed

        # Reshape for transformer processing
        # [B, H, W, C] -> [B, H*W, C]
        H, W = x.shape[1:3]
        x = x.reshape(B, H * W, self.embed_dim)

        # Process through transformer blocks with skip connections
        skip_features = []
        skip_idx = 0

        for i, block in enumerate(self.blocks):
            x = block(x)

            # Save features for skip connections
            if self.use_skip_adapter and i % 4 == 3:
                skip_features.append(self.skip_adapters[skip_idx](x))
                skip_idx += 1

        # Apply final normalisation
        x = self.norm(x)

        # Reshape to 2D feature map for convolutional processing
        x = x.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Apply neck
        x = self.neck(x)

        # Add skip connections
        if self.use_skip_adapter and len(skip_features) > 0:
            # Reshape skip features to 2D
            skip_feat = torch.stack(skip_features, dim=1).mean(dim=1)
            skip_feat = skip_feat.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
            skip_feat = self.neck(skip_feat)
            x = x + skip_feat

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features
        features = self.forward_features(x)

        # Upsample to original resolution and generate shadow mask
        shadow_mask = self.upsample(features)

        # Apply sigmoid for binary mask
        shadow_mask = torch.sigmoid(shadow_mask)

        return shadow_mask

class AdapterShadow(nn.Module):
    """
    AdapterShadow main model adapting SAM for shadow detection
    """

    def __init__(self, args, checkpoint=None):
        super().__init__()
        self.args = args

        # Load pre-trained SAM model
        if checkpoint is not None:
            print(f"Loading SAM checkpoint from {checkpoint}")
            sam = sam_model_registry['vit_b'](checkpoint=checkpoint)

            # Initialise our modified encoder with SAM weights
            self.image_encoder = AdaptedImageEncoder(args)

            # Copy SAM image encoder weights
            sam_img_encoder = sam.image_encoder
            # Transfer patch embedding weights
            self.image_encoder.patch_embed.load_state_dict(sam_img_encoder.patch_embed.state_dict())
            if self.image_encoder.use_vpt:
                pos_embed = sam_img_encoder.pos_embed.data
                # Insert zeros for the new VPT token positions
                new_pos_embed = torch.zeros(1, self.image_encoder.pos_embed.shape[1], pos_embed.shape[2])
                new_pos_embed[0, 10:] = pos_embed[0]
                self.image_encoder.pos_embed.data = new_pos_embed
            else:
                self.image_encoder.pos_embed.data = sam_img_encoder.pos_embed.data

            # Transfer transformer block weights
            for i, (sam_block, our_block) in enumerate(zip(sam_img_encoder.blocks, self.image_encoder.blocks)):
                # Transfer basic transformer components
                our_block.norm1.load_state_dict(sam_block.norm1.state_dict())
                our_block.norm2.load_state_dict(sam_block.norm2.state_dict())

                # Transfer attention weights
                # Copy Q, K, V projections from SAM to main branch
                our_block.attn.qkv.weight.data = sam_block.attn.qkv.weight.data
                our_block.attn.qkv.bias.data = sam_block.attn.qkv.bias.data
                our_block.attn.proj.weight.data = sam_block.attn.proj.weight.data
                our_block.attn.proj.bias.data = sam_block.attn.proj.bias.data

                # MLP weights
                # our_block.mlp[0].load_state_dict(sam_block.mlp[0].state_dict())
                # our_block.mlp[1].load_state_dict(sam_block.mlp[1].state_dict())
                our_block.mlp[0].load_state_dict(sam_block.mlp.lin1.state_dict())
                our_block.mlp[2].load_state_dict(sam_block.mlp.lin2.state_dict())

            # Transfer final norm layer
            # self.image_encoder.norm.load_state_dict(sam_img_encoder.neck[0].state_dict())
            self.image_encoder.norm.load_state_dict(sam_img_encoder.blocks[-1].norm1.state_dict())
        else:
            self.image_encoder = AdaptedImageEncoder(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass to get shadow mask
        shadow_mask = self.image_encoder(x)
        return shadow_mask


def adapt_sam_model_registry(model_type, args=None, checkpoint=None):
    """
    Build AdapterShadow model
    """
    if args is None:
        # Default arguments
        class DefaultArgs:
            def __init__(self):
                self.vpt = False
                self.all = True
                self.lora = True
                self.mb_ratio = 0.25
                self.skip_adapter = True

        args = DefaultArgs()

    if model_type == 'vit_b':
        return AdapterShadow(args, checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")