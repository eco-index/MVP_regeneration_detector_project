from __future__ import annotations

import json
import importlib.util
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PrithviSegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.repo_id = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
        self._load_backbone()
        self.auto_pos_embed = True

        self.projector = nn.Sequential(
            ConvBNReLU(self.embed_dim, 256, kernel_size=1, padding=0),
            ConvBNReLU(256, 128),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        prepared = self._prepare_input(pixel_values)
        if self.auto_pos_embed:
            self._ensure_backbone_pos_embed(prepared.shape[-3:], prepared.device)
        feature_tokens = self.backbone.forward_features(prepared)
        feature_maps = self.backbone.prepare_features_for_image_model(feature_tokens)
        feat = self._reduce_temporal(feature_maps[-1])
        proj = self.projector(feat)
        logits = self.classifier(proj)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    def _load_backbone(self):
        config_path = hf_hub_download(self.repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pretrained_cfg = cfg.get("pretrained_cfg", {})

        module_path = hf_hub_download(self.repo_id, "prithvi_mae.py")
        spec = importlib.util.spec_from_file_location("prithvi_mae_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules.setdefault("prithvi_mae_module", module)

        vit_cls = getattr(module, "PrithviViT", None)
        if vit_cls is not None and not getattr(vit_cls, "_pos_embed_cache_installed", False):
            original_get_pos_embed = vit_cls._get_pos_embed

            def cached_pos_embed(self, x):
                cache = getattr(self, "_pos_embed_cache", None)
                if cache is None:
                    cache = {}
                    setattr(self, "_pos_embed_cache", cache)

                device = x.device
                device_index = device.index if device.type == "cuda" else -1
                key = (tuple(x.shape[-3:]), device.type, device_index)

                cached = cache.get(key)
                if cached is not None:
                    return cached

                pos = original_get_pos_embed(self, x).detach()
                cache[key] = pos
                return pos

            vit_cls._get_pos_embed = cached_pos_embed
            vit_cls._pos_embed_cache_installed = True

        img_size = pretrained_cfg.get("img_size", 224)
        patch_size = tuple(pretrained_cfg.get("patch_size", (1, 16, 16)))
        num_frames = pretrained_cfg.get("num_frames", 1)
        in_chans = pretrained_cfg.get("in_chans", 3)

        self.input_scale = 10000.0
        mean = pretrained_cfg.get("mean", [0.0] * in_chans)
        std = pretrained_cfg.get("std", [1.0] * in_chans)
        self.register_buffer("channel_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("channel_std", torch.tensor(std, dtype=torch.float32))

        backbone = module.PrithviMAE(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=pretrained_cfg.get("embed_dim", cfg.get("num_features", 768)),
            depth=pretrained_cfg.get("depth", 12),
            num_heads=pretrained_cfg.get("num_heads", 12),
            decoder_embed_dim=pretrained_cfg.get("decoder_embed_dim", 512),
            decoder_depth=pretrained_cfg.get("decoder_depth", 8),
            decoder_num_heads=pretrained_cfg.get("decoder_num_heads", 16),
            mlp_ratio=pretrained_cfg.get("mlp_ratio", 4.0),
            coords_encoding=pretrained_cfg.get("coords_encoding", []),
            coords_scale_learn=pretrained_cfg.get("coords_scale_learn", False),
            drop_path=pretrained_cfg.get("drop_path", 0.0),
            mask_ratio=pretrained_cfg.get("mask_ratio", 0.0),
        )

        weights_path = hf_hub_download(self.repo_id, "Prithvi_EO_V1_100M.pt")
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=False)

        backbone_encoder = backbone.encoder
        backbone_encoder.patch_embed.proj = backbone_encoder.patch_embed.proj.to(torch.float32)
        self.backbone = backbone_encoder
        self.embed_dim = self.backbone.embed_dim
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.effective_time_dim = self.backbone.patch_embed.input_size[0] // self.backbone.patch_embed.patch_size[0]
        self.backbone._active_pos_key = None

    def _prepare_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        r = pixel_values[:, 0]
        g = pixel_values[:, 1]
        b = pixel_values[:, 2]

        bands: List[torch.Tensor] = []
        mapping = [b, g, r, r, g, b]
        for i in range(self.in_chans):
            bands.append(mapping[i % len(mapping)])

        stacked = torch.stack(bands, dim=1) * self.input_scale
        mean = self.channel_mean.view(1, -1, 1, 1)
        std = self.channel_std.view(1, -1, 1, 1)
        stacked = (stacked - mean) / (std + 1e-6)

        if self.num_frames > 1:
            stacked = stacked.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        else:
            stacked = stacked.unsqueeze(2)
        return stacked

    def _reduce_temporal(self, feature_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature_map.shape
        frames = max(self.effective_time_dim, 1)
        embed_dim = C // frames
        feature_map = feature_map.view(B, frames, embed_dim, H, W)
        feature_map = feature_map.mean(dim=1)
        return feature_map

    def _ensure_backbone_pos_embed(self, spatial_shape: tuple[int, int, int], device: torch.device) -> None:
        device_index = device.index if device.type == "cuda" else -1
        key = (spatial_shape, device.type, device_index)
        active_key = getattr(self.backbone, "_active_pos_key", None)
        if active_key == key:
            return

        t, h, w = spatial_shape
        dummy = torch.zeros(1, self.in_chans, t, h, w, device=device)
        pos_embed = self.backbone._get_pos_embed(dummy).detach()

        patch = self.backbone.patch_embed.patch_size
        grid = (
            max(1, t // max(1, patch[0])),
            max(1, h // max(1, patch[1])),
            max(1, w // max(1, patch[2])),
        )

        self.backbone.pos_embed = pos_embed
        self.backbone.patch_embed.input_size = spatial_shape
        self.backbone.patch_embed.grid_size = grid
        setattr(self.backbone, "_active_pos_key", key)
