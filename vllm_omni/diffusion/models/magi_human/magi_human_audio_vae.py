# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Ported from daVinci-MagiHuman inference/model/sa_audio/

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def vae_sample(mean, scale):
    stdev = F.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):
    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        info["kl"] = kl
        return (x, info) if return_info else x

    def decode(self, x):
        return x


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(activation: Literal["elu", "snake", "none"], antialias: bool = False, channels=None) -> nn.Module:
    if antialias:
        raise NotImplementedError("antialias activation not supported")
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels, out_channels, kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return (_checkpoint(self.layers, x) if self.training else self.layers(x)) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, 9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False
    ):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=1, bias=False, padding="same"),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            )
        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, 9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        c_mults = [1] + c_mults
        depth = len(c_mults)
        layers = [WNConv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(depth - 1):
            layers.append(
                EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels, strides[i], use_snake=use_snake)
            )
        layers.extend(
            [
                get_activation(
                    "snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels
                ),
                WNConv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        c_mults = [1] + c_mults
        depth = len(c_mults)
        layers = [WNConv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    c_mults[i] * channels,
                    c_mults[i - 1] * channels,
                    strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            )
        layers.extend(
            [
                get_activation(
                    "snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels
                ),
                WNConv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Audio autoencoder
# ---------------------------------------------------------------------------
class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck=None,
        in_channels=None,
        out_channels=None,
        soft_clip=False,
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels if in_channels is not None else io_channels
        self.out_channels = out_channels if out_channels is not None else io_channels
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.soft_clip = soft_clip

    def encode(self, audio, skip_bottleneck=False, return_info=False, **kwargs):
        info = {}
        latents = self.encoder(audio)
        info["pre_bottleneck_latents"] = latents
        if self.bottleneck is not None and not skip_bottleneck:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)
        return (latents, info) if return_info else latents

    def decode(self, latents, skip_bottleneck=False, **kwargs):
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _create_encoder_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "oobleck", f"Only 'oobleck' encoder supported, got: {cfg.get('type')}"
    enc = OobleckEncoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in enc.parameters():
            p.requires_grad = False
    return enc


def _create_decoder_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "oobleck", f"Only 'oobleck' decoder supported, got: {cfg.get('type')}"
    dec = OobleckDecoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in dec.parameters():
            p.requires_grad = False
    return dec


def _create_bottleneck_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "vae", f"Only 'vae' bottleneck supported, got: {cfg.get('type')}"
    bn = VAEBottleneck()
    if not cfg.get("requires_grad", True):
        for p in bn.parameters():
            p.requires_grad = False
    return bn


def _create_autoencoder_from_config(config: dict[str, Any]):
    ae_config = config["model"]
    if ae_config.get("pretransform") is not None:
        raise NotImplementedError("Nested pretransform not supported")
    encoder = _create_encoder_from_config(ae_config["encoder"])
    decoder = _create_decoder_from_config(ae_config["decoder"])
    bottleneck_cfg = ae_config.get("bottleneck")
    bottleneck = _create_bottleneck_from_config(bottleneck_cfg) if bottleneck_cfg else None
    return AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=ae_config["latent_dim"],
        downsampling_ratio=ae_config["downsampling_ratio"],
        sample_rate=config["sample_rate"],
        io_channels=ae_config["io_channels"],
        bottleneck=bottleneck,
        in_channels=ae_config.get("in_channels"),
        out_channels=ae_config.get("out_channels"),
        soft_clip=ae_config["decoder"].get("soft_clip", False),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
class SAAudioFeatureExtractor:
    def __init__(self, device, model_path):
        self.device = device
        self.vae_model, self.sample_rate = self._load_vae(model_path)
        self.resampler = None

    def _load_vae(self, model_path):
        if not (isinstance(model_path, str) and Path(model_path).is_dir()):
            raise ValueError("model_path must be a local directory")

        model_config_path = os.path.join(model_path, "model_config.json")
        with open(model_config_path) as f:
            full_config = json.load(f)

        vae_config = full_config["model"]["pretransform"]["config"]
        sample_rate = full_config["sample_rate"]

        autoencoder_config = {
            "model_type": "autoencoder",
            "sample_rate": sample_rate,
            "model": vae_config,
        }
        vae_model = _create_autoencoder_from_config(autoencoder_config)

        weights_path = Path(model_path) / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weight file does not exist: {weights_path}")

        full_state_dict = load_file(weights_path, device=str(self.device))
        vae_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith("pretransform.model."):
                vae_state_dict[key[len("pretransform.model.") :]] = value

        model_keys = set(vae_model.state_dict().keys())
        vae_keys = set(vae_state_dict.keys())
        missing = model_keys - vae_keys
        extra = vae_keys - model_keys
        if missing:
            logger.warning("Audio VAE missing keys (%d): %s", len(missing), list(missing)[:5])
        if extra:
            logger.warning("Audio VAE unexpected keys (%d): %s", len(extra), list(extra)[:5])

        vae_model.load_state_dict(vae_state_dict)
        vae_model.to(self.device)
        return vae_model, sample_rate

    def decode(self, latents):
        with torch.no_grad():
            return self.vae_model.decode(latents)

    def encode(self, waveform):
        with torch.no_grad():
            return self.vae_model.encode(waveform)
