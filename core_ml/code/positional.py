import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# Learned Positional Embedding
# ─────────────────────────────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.context_length, cfg.d_model)

    def forward(self, x):
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dropout = nn.Dropout(getattr(cfg, "dropout", 0.0))

        pe = torch.zeros(cfg.context_length, cfg.d_model)
        pos = torch.arange(0, cfg.context_length).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, cfg.d_model, 2).float()
            * (-math.log(10000.0) / cfg.d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))

# ─────────────────────────────────────────────────────────────────────────────
# No-Op Positional Encoding
# Used for RoPE / ALiBi / Relative
# ─────────────────────────────────────────────────────────────────────────────


class NoPositionalEncoding(nn.Module):
    """
    Identity module.
    Used when positional information is handled inside attention.
    """

    def forward(self, x):
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

POS_ENCODING_REGISTRY = {
    "learned": LearnedPositionalEmbedding,
    "sinusoidal": SinusoidalPositionalEncoding,

    # 🔥 These use attention-based positional handling
    "rope": NoPositionalEncoding,
    "rope_interp": NoPositionalEncoding,
    "alibi": NoPositionalEncoding,
    "relative": NoPositionalEncoding,
}


def build_pos_encoding(cfg):
    if cfg.pos_encoding_type not in POS_ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown pos_encoding_type '{cfg.pos_encoding_type}'. "
            f"Available: {list(POS_ENCODING_REGISTRY.keys())}"
        )
    cls = POS_ENCODING_REGISTRY[cfg.pos_encoding_type]

    # NoOp doesn't take cfg
    if cls is NoPositionalEncoding:
        return cls()
    else:
        return cls(cfg)
