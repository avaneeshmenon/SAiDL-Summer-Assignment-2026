import math
import torch
import torch.nn as nn

# IMPORTANT: these must exist in other files
from attention import build_attention
from positional import build_pos_encoding


# ─────────────────────────────────────────────────────────────────────────────
# FeedForward
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
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = build_attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

        # Optional conv hybrid
        self.conv = None
        if getattr(cfg, "use_conv_hybrid", False):
            self.conv = nn.Sequential(
                nn.Conv1d(cfg.d_model, cfg.d_model, 3, padding=1,
                          groups=cfg.d_model, bias=False),
                nn.Conv1d(cfg.d_model, cfg.d_model, 1, bias=False),
                nn.GELU(),
            )

    def forward(self, x):
        if self.conv is not None:
            x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer LM
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = build_pos_encoding(cfg)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Init
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name.endswith("weight"):
                if "attn" in name or "ff" in name:
                    nn.init.normal_(p, mean=0.0, std=0.02 /
                                    math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.token_emb(idx)
        x = self.pos_enc(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
