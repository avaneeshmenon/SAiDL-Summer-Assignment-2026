"""
conv_blocks.py
──────────────
Four plug-and-play 1-D convolutional components for the hybrid Transformer.

CRITICAL: All Conv1D operations use causal (left-only) padding to prevent
future token leakage. Standard symmetric padding allows each position to
see kernel_size//2 future tokens, which causes catastrophically low PPL
by leaking ground truth information into predictions.

Causal padding: pad (kernel_size - 1) on the LEFT, 0 on the RIGHT.
This ensures position t only sees positions <= t.

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
    fraction of layers (default: the first half).

Design 4 – GatedConvFFN
    Replaces the standard FFN with a Gated Convolutional Feed-Forward Network.
    Uses a split-gate design:
        gate  = sigmoid(Conv1d(x))
        value = GELU(Conv1d(x))
        out   = gate * value   → projected back to d_model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """
    Causal depthwise + pointwise Conv1D.
    Input / output: (B, T, C)  [channel-last]

    Uses left-only padding of (kernel_size - 1) to ensure causality:
    position t sees only positions [t - (kernel_size-1), ..., t].
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.causal_pad = kernel_size - 1   # pad left only

        self.dw  = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=0,              # no built-in padding — we do it manually
            groups=d_model, bias=False
        )
        self.pw   = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = x.transpose(1, 2)                      # (B, C, T)
        h = F.pad(h, (self.causal_pad, 0))         # pad left only → causal
        h = self.dw(h)                             # (B, C, T)
        h = self.pw(h)                             # (B, C, T)
        h = self.act(h)
        h = self.drop(h)
        return h.transpose(1, 2)                   # (B, T, C)


# ─────────────────────────────────────────────────────────────────────────────
# Design 1 – Conv1D before attention
# ─────────────────────────────────────────────────────────────────────────────

class Conv1DBefore(nn.Module):
    """
    Prepended to every attention sublayer.
    Flow:  x → CausalDWSConv → residual add → [then normal attn + FFN]
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln   = nn.LayerNorm(cfg.d_model)
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.ln(x))


# ─────────────────────────────────────────────────────────────────────────────
# Design 2 – Interleaved Conv block
# ─────────────────────────────────────────────────────────────────────────────

class InterleavedConvBlock(nn.Module):
    """
    Pure-conv block used on every other layer when conv_type == "interleaved".
    Replaces attention with causal depthwise-separable convolution.

    Layer structure:
        x = x + CausalDWSConv(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ff   = nn.Sequential(
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
    Replaces the attention sublayer in the first (n_layers // 2) blocks.
    TransformerBlock wraps this in LN + residual, so forward() just
    returns the causal conv output.
    """

    def __init__(self, cfg):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            cfg.d_model,
            kernel_size=getattr(cfg, "conv_kernel_size", 3),
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Design 4 – Gated Convolutional Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class GatedConvFFN(nn.Module):
    """
    Replaces the standard MLP FFN with a causal Gated Conv FFN.

    Architecture (GLU-style):
        h_gate  = sigmoid( CausalConv1d_expand(x) )
        h_value = GELU(    CausalConv1d_expand(x) )
        h       = h_gate * h_value
        out     = Conv1d_project(h)

    Both the expand conv and project conv are causal.
    """

    def __init__(self, cfg):
        super().__init__()
        k = getattr(cfg, "conv_kernel_size", 3)
        self.causal_pad = k - 1                    # left-only padding

        # Fused gate + value projection (d_model → 2 * d_ff)
        self.expand  = nn.Conv1d(
            cfg.d_model, 2 * cfg.d_ff, k,
            padding=0, bias=False                  # manual causal padding
        )
        # Project back (d_ff → d_model) — kernel=1 is always causal
        self.project = nn.Conv1d(cfg.d_ff, cfg.d_model, 1, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                     # (B, C, T)
        h = F.pad(h, (self.causal_pad, 0))        # pad left only → causal
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
    "conv_before_attn": Conv1DBefore,
    "interleaved":      InterleavedConvBlock,
    "depthwise_subset": DepthwiseSeparableReplace,
    "gated_conv_ff":    GatedConvFFN,
    "none":             None,
}