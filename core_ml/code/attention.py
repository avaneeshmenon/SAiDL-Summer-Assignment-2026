import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Standard Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class StandardMultiHeadAttention(nn.Module):
    """
    Vanilla causal multi-head self-attention.
    Uses PyTorch scaled_dot_product_attention (Flash Attention when available).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = getattr(cfg, "dropout", 0.0)

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ATTENTION_REGISTRY = {
    "standard": StandardMultiHeadAttention,
    # "mqa": MultiQueryAttention,
    # "linear": LinearAttention,
    # "sliding": SlidingWindowAttention,
}


def build_attention(cfg):
    if cfg.attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention_type '{cfg.attention_type}'. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[cfg.attention_type](cfg)

# print("ATTENTION_REGISTRY:", list(ATTENTION_REGISTRY.keys()))