"""
model.py
────────
Decoder-only Transformer LM with pluggable:
  • attention variants   (attention.py   → build_attention)
  • positional encodings (positional.py  → build_pos_encoding)
  • conv hybrid designs  (conv_blocks.py → CONV_REGISTRY)

Conv routing summary
────────────────────
conv_type = "none"
    Pure Transformer.  Same behaviour as the original codebase.

conv_type = "conv_before_attn"   [Design 1]
    Every TransformerBlock gets a Conv1DBefore module inserted at the top.
    Flow per block:  x → Conv1DBefore(x) → Attn → FFN

conv_type = "interleaved"        [Design 2]
    Even-indexed blocks (0, 2, 4, …) become InterleavedConvBlock (no attn).
    Odd-indexed blocks  (1, 3, 5, …) remain standard TransformerBlocks.

conv_type = "depthwise_subset"   [Design 3]
    The first (n_layers // 2) TransformerBlocks replace their attention
    sublayer with DepthwiseSeparableReplace; the remaining blocks keep
    standard attention.  Both halves keep their FFN sublayer.

conv_type = "gated_conv_ff"      [Design 4]
    All TransformerBlocks keep their attention sublayer but replace the
    standard MLP FFN with GatedConvFFN.
"""

import math
import torch
import torch.nn as nn

from attention import build_attention
from positional import build_pos_encoding
from conv_blocks import (
    Conv1DBefore,
    InterleavedConvBlock,
    DepthwiseSeparableReplace,
    GatedConvFFN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Networks
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Standard position-wise FFN: Linear → GELU → Linear."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

    Behaviour is controlled by cfg.conv_type at construction time:

    "none" / "conv_before_attn" / "gated_conv_ff"
        → standard block with optional additions (see below)

    "depthwise_subset"
        → pass use_dw_replace=True to replace attn with DWSConv

    The "interleaved" case never instantiates this class for even layers;
    TransformerLM handles that routing directly.
    """

    def __init__(self, cfg, use_dw_replace: bool = False):
        super().__init__()

        conv_type = getattr(cfg, "conv_type", "none")

        # ── Design 1: prepend conv before attention ───────────────────────
        self.pre_conv = (
            Conv1DBefore(cfg)
            if conv_type == "conv_before_attn"
            else None
        )

        # ── Attention sublayer (or DWSConv replacement for Design 3) ─────
        self.ln1 = nn.LayerNorm(cfg.d_model)
        if use_dw_replace:
            # Design 3: depthwise separable conv instead of self-attention
            self.attn = DepthwiseSeparableReplace(cfg)
        else:
            self.attn = build_attention(cfg)

        # ── FFN sublayer (or Gated Conv FFN for Design 4) ─────────────────
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if conv_type == "gated_conv_ff":
            # Design 4: replace MLP FFN with gated conv FFN
            self.ff = GatedConvFFN(cfg)
        else:
            self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Design 1 pre-conv (no-op when self.pre_conv is None)
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        # Attention (or DWSConv replacement) sublayer
        x = x + self.attn(self.ln1(x))

        # FFN sublayer
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer LM
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Block construction is delegated to _build_blocks() which reads cfg.conv_type
    and assembles the correct mix of TransformerBlock / InterleavedConvBlock.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = build_pos_encoding(cfg)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = self._build_blocks(cfg)

        self.ln_f = nn.LayerNorm(cfg.d_model)

        # ── Weight tying ──────────────────────────────────────────────────
        # lm_head projects d_model → vocab_size; tie its weight to token_emb
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # ── Parameter initialisation ──────────────────────────────────────
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("weight") and ("attn" in name or "ff" in name):
                nn.init.normal_(p, mean=0.0, std=0.02 /
                                math.sqrt(2 * cfg.n_layers))

    # ──────────────────────────────────────────────────────────────────────
    # Block factory
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_blocks(cfg) -> nn.ModuleList:
        """
        Returns an nn.ModuleList of n_layers blocks according to cfg.conv_type.

        Design   conv_type            Even layers         Odd layers
        ───────  ───────────────────  ──────────────────  ──────────────────
        none     "none"               TransformerBlock    TransformerBlock
        1        "conv_before_attn"   TransformerBlock*   TransformerBlock*
                                      (* pre_conv active)
        2        "interleaved"        InterleavedConvBlock TransformerBlock
        3        "depthwise_subset"   TransformerBlock    TransformerBlock
                                      (first half: DWSConv replaces attn)
        4        "gated_conv_ff"      TransformerBlock**  TransformerBlock**
                                      (** GatedConvFFN replaces FFN)
        """
        conv_type = getattr(cfg, "conv_type", "none")
        n = cfg.n_layers
        n_dw_layers = n // 2  # used by Design 3

        blocks = []
        for i in range(n):
            if conv_type == "interleaved":
                # Design 2: alternate Conv / Attn blocks
                if i % 2 == 0:
                    blocks.append(InterleavedConvBlock(cfg))
                else:
                    blocks.append(TransformerBlock(cfg))

            elif conv_type == "depthwise_subset":
                # Design 3: first half replace attn with DWSConv
                use_dw = (i < n_dw_layers)
                blocks.append(TransformerBlock(cfg, use_dw_replace=use_dw))

            else:
                # Designs 1, 4, and baseline "none":
                # TransformerBlock reads conv_type internally
                blocks.append(TransformerBlock(cfg))

        return nn.ModuleList(blocks)

    # ──────────────────────────────────────────────────────────────────────
    # Weight init
    # ──────────────────────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(idx)   # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)    # (B, T, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
