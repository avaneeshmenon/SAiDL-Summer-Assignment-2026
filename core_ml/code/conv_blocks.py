"""
conv_blocks.py
──────────────
Four plug-and-play 1-D convolutional components for the hybrid Transformer.

Design 1 – Conv1DBefore
    A depthwise-separable Conv1D applied *before* the attention sublayer.
    Acts as a local n-gram feature extractor that feeds richer tokens into
    attention. Residual connection keeps the original signal intact.

Design 2 – InterleavedConvBlock
    A complete block that *replaces* a TransformerBlock when interleaving is
    active.  It runs:   LayerNorm → DepthwiseConv → Pointwise → GELU → residual
    followed by the standard FFN sublayer.
    Used every other layer (even indices get ConvBlock, odd get TransformerBlock).

Design 3 – DepthwiseSeparableReplace
    A drop-in module that *replaces the attention sublayer* for a configurable
    fraction of layers (default: the first half).  This is the depthwise
    separable 1-D convolution option from the assignment.

Design 4 – GatedConvFFN
    Replaces the standard FFN with a Gated Convolutional Feed-Forward Network.
    Uses a split-gate design:
        gate  = sigmoid(Conv1d(x))
        value = GELU(Conv1d(x))
        out   = gate * value   → projected back to d_model
    This is analogous to the gated linear units used in modern LLMs.

Usage
──────
All modules take (B, T, C) tensors and return (B, T, C) tensors so they slot
seamlessly into the existing TransformerBlock / TransformerLM pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise + pointwise Conv1D.
    Input / output: (B, T, C)  [channel-last, as used throughout the codebase]
    Internally transposes to (B, C, T) for Conv1D, then transposes back.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding for odd kernels
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=pad, groups=d_model, bias=False
        )
        self.pw = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.transpose(1, 2)          # (B, C, T)
        h = self.dw(h)
        h = self.pw(h)
        h = self.act(h)
        h = self.drop(h)
        return h.transpose(1, 2)       # (B, T, C)


# ─────────────────────────────────────────────────────────────────────────────
# Design 1 – Conv1D before attention
# ─────────────────────────────────────────────────────────────────────────────

class Conv1DBefore(nn.Module):
    """
    Prepended to every attention sublayer.
    Flow:  x → DepthwiseSepConv → residual add → [then normal attn + FFN]

    The TransformerBlock checks cfg.conv_type == "conv_before_attn" and
    instantiates this module; forward() is called at the top of the block.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.ln(x))


# ─────────────────────────────────────────────────────────────────────────────
# Design 2 – Interleaved Conv block (replaces TransformerBlock at even layers)
# ─────────────────────────────────────────────────────────────────────────────

class InterleavedConvBlock(nn.Module):
    """
    Pure-conv block used on every other layer when conv_type == "interleaved".
    Mirrors the pre-LN TransformerBlock structure but replaces attention with
    a depthwise-separable convolution.

    Layer structure:
        x = x + DWSConv(LN(x))       ← local context (replaces attention)
        x = x + FFN(LN(x))           ← position-wise transform
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # Standard FFN reused
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Design 3 – Depthwise-separable replacement for attention sublayer
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableReplace(nn.Module):
    """
    Used when conv_type == "depthwise_subset".
    Replaces the attention sublayer in the first (n_layers // 2) blocks.
    The containing TransformerBlock instantiates this instead of build_attention().

    The FFN sublayer is kept unchanged, so these 'conv layers' still do:
        x = x + DWSConv(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, cfg):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TransformerBlock already wraps this in LN + residual,
        # so just return the transformed tensor.
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Design 4 – Gated Convolutional Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class GatedConvFFN(nn.Module):
    """
    Replaces the standard MLP FFN when conv_type == "gated_conv_ff".

    Architecture (GLU-style with convolution):
        h_gate  = sigmoid( Conv1d_expand(x) )
        h_value = GELU(    Conv1d_expand(x) )
        h       = h_gate * h_value            ← gated activation
        out     = Conv1d_project(h)           ← project back to d_model

    The two Conv1d_expand projections are fused into one 2*d_ff projection
    then split, matching the efficiency of GeLU-gated variants.
    Kernel size 1 gives a pointwise conv equivalent to a linear layer;
    kernel size 3 adds local context inside the FFN.
    """

    def __init__(self, cfg):
        super().__init__()
        k = getattr(cfg, "conv_kernel_size", 3)
        pad = k // 2

        # Fused gate + value projection  (d_model → 2 * d_ff)
        self.expand = nn.Conv1d(
            cfg.d_model, 2 * cfg.d_ff, k, padding=pad, bias=False
        )
        # Project back  (d_ff → d_model)
        self.project = nn.Conv1d(cfg.d_ff, cfg.d_model, 1, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.transpose(1, 2)                    # (B, C, T)
        h = self.expand(h)                        # (B, 2*d_ff, T)
        gate, value = h.chunk(2, dim=1)           # each (B, d_ff, T)
        h = torch.sigmoid(gate) * F.gelu(value)  # gated activation
        h = self.project(h)                       # (B, d_model, T)
        h = self.drop(h)
        return h.transpose(1, 2)                  # (B, T, C)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

CONV_REGISTRY = {
    "conv_before_attn":   Conv1DBefore,
    "interleaved":        InterleavedConvBlock,   # used at the model level
    "depthwise_subset":   DepthwiseSeparableReplace,
    "gated_conv_ff":      GatedConvFFN,
    "none":               None,
}