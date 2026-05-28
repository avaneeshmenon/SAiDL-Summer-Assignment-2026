import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def reshape_heads(x, B, T, n_heads, d_head):
    return x.view(B, T, n_heads, d_head).transpose(1, 2)


# ─────────────────────────────────────────────
# Positional Bias Module
# ─────────────────────────────────────────────

class PositionalBias(nn.Module):
    """
    Decoupled positional bias that can be injected into any attention class.

    Handles:
        "rope"       – rotates Q and K before dot product
        "rope_interp"– same as rope but with position scaling
        "alibi"      – adds distance-based linear bias to scores
        "relative"   – adds learned relative embedding bias to scores
        "learned"    – no-op (handled in positional.py)
        "sinusoidal" – no-op (handled in positional.py)
        "none"       – no-op
    """

    def __init__(self, cfg):
        super().__init__()
        self.pos_type = getattr(cfg, "pos_encoding_type", "none")
        self.d_head   = cfg.d_model // cfg.n_heads
        self.n_heads  = cfg.n_heads

        # ── RoPE buffers ──────────────────────────────────────────────────
        if self.pos_type in ("rope", "rope_interp"):
            base     = 10000.0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.d_head, 2).float() / self.d_head)
            )
            self.register_buffer("inv_freq", inv_freq)
            self.rope_scale = getattr(cfg, "rope_scale", 1.0)

        # ── ALiBi slopes ──────────────────────────────────────────────────
        elif self.pos_type == "alibi":
            slopes = torch.tensor(
                [2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)]
            )
            self.register_buffer("slopes", slopes)

        # ── Relative positional embeddings ────────────────────────────────
        elif self.pos_type == "relative":
            self.max_dist = 4096
            self.rel_emb  = nn.Embedding(self.max_dist, self.d_head)

    # ── RoPE helpers ──────────────────────────────────────────────────────

    def _get_cos_sin(self, T, device):
        scale = self.rope_scale if self.pos_type == "rope_interp" else 1.0
        t     = torch.arange(T, device=device) * scale
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    # ── Main interface ────────────────────────────────────────────────────

    def apply_to_qk(self, q, k, T, device):
        """
        For RoPE variants: rotate Q and K in-place before dot product.
        For all other types: no-op, returns q and k unchanged.
        """
        if self.pos_type in ("rope", "rope_interp"):
            cos, sin = self._get_cos_sin(T, device)
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
        return q, k

    def apply_to_scores(self, scores, q, T, device):
        """
        For ALiBi / Relative: add positional bias to attention scores.
        For all other types: no-op, returns scores unchanged.
        scores shape: (B, n_heads, T, T)
        q      shape: (B, n_heads, T, d_head)
        """
        if self.pos_type == "alibi":
            pos  = torch.arange(T, device=device)
            dist = (pos[None, :] - pos[:, None]).clamp(min=0)
            bias = -self.slopes[:, None, None] * dist   # (n_heads, T, T)
            scores = scores + bias.unsqueeze(0)

        elif self.pos_type == "relative":
            idx   = torch.arange(T, device=device)
            dist  = (idx[:, None] - idx[None, :]).clamp(0, self.max_dist - 1)
            rel   = self.rel_emb(dist)                  # (T, T, d_head)
            # q: (B, n_heads, T, d_head) → rel_scores: (B, n_heads, T, T)
            rel_scores = torch.einsum("bhid,ijd->bhij", q, rel)
            scores = scores + rel_scores

        return scores


# ─────────────────────────────────────────────
# 1. STANDARD
# ─────────────────────────────────────────────

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.qkv    = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out    = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.posbias = PositionalBias(cfg)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        q, k = self.posbias.apply_to_qk(q, k, T, x.device)

        pos_type = getattr(self.posbias, "pos_type", "none")
        if pos_type in ("alibi", "relative"):
            scale  = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = self.posbias.apply_to_scores(scores, q, T, x.device)
            mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn   = F.softmax(scores, dim=-1)
            out    = attn @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 2. SLIDING WINDOW
# ─────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.window  = getattr(cfg, "window_size", 128)
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv     = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out     = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.posbias = PositionalBias(cfg)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        q, k = self.posbias.apply_to_qk(q, k, T, x.device)

        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window)
            mask[i, start:i+1] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)

        scale  = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = self.posbias.apply_to_scores(scores, q, T, x.device)
        scores = scores + mask

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores      = scores.masked_fill(causal_mask, float("-inf"))
        attn        = F.softmax(scores, dim=-1)
        out         = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 3. SPARSE BLOCK
# ─────────────────────────────────────────────

class SparseBlockAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block   = getattr(cfg, "block_size", 64)
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv     = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out     = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.posbias = PositionalBias(cfg)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        q, k = self.posbias.apply_to_qk(q, k, T, x.device)

        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            block_id = i // self.block
            start    = block_id * self.block
            mask[i, start:i+1] = 0
            for b in range(block_id):
                mask[i, (b+1)*self.block - 1] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)

        scale  = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = self.posbias.apply_to_scores(scores, q, T, x.device)
        scores = scores + mask

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores      = scores.masked_fill(causal_mask, float("-inf"))
        attn        = F.softmax(scores, dim=-1)
        out         = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 4. GQA
# ─────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv    = max(1, cfg.n_heads // 4)
        self.d_head  = cfg.d_model // cfg.n_heads

        self.q   = nn.Linear(cfg.d_model, cfg.d_model,          bias=False)
        self.k   = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.v   = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model,          bias=False)

        self.posbias = PositionalBias(cfg)

    def forward(self, x):
        B, T, C = x.shape

        q = reshape_heads(self.q(x), B, T, self.n_heads, self.d_head)
        k = reshape_heads(self.k(x), B, T, self.n_kv,    self.d_head)
        v = reshape_heads(self.v(x), B, T, self.n_kv,    self.d_head)

        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)

        q, k = self.posbias.apply_to_qk(q, k, T, x.device)

        pos_type = getattr(self.posbias, "pos_type", "none")
        if pos_type in ("alibi", "relative"):
            scale  = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = self.posbias.apply_to_scores(scores, q, T, x.device)
            mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn   = F.softmax(scores, dim=-1)
            out    = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 5. MQA
# ─────────────────────────────────────────────

class MultiQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv    = 1
        self.d_head  = cfg.d_model // cfg.n_heads

        self.q   = nn.Linear(cfg.d_model, cfg.d_model,          bias=False)
        self.k   = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.v   = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model,          bias=False)

        self.posbias = PositionalBias(cfg)

    def forward(self, x):
        B, T, C = x.shape

        q = reshape_heads(self.q(x), B, T, self.n_heads, self.d_head)
        k = reshape_heads(self.k(x), B, T, self.n_kv,    self.d_head)
        v = reshape_heads(self.v(x), B, T, self.n_kv,    self.d_head)

        k = k.expand(B, self.n_heads, T, self.d_head)
        v = v.expand(B, self.n_heads, T, self.d_head)

        q, k = self.posbias.apply_to_qk(q, k, T, x.device)

        pos_type = getattr(self.posbias, "pos_type", "none")
        if pos_type in ("alibi", "relative"):
            scale  = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = self.posbias.apply_to_scores(scores, q, T, x.device)
            mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn   = F.softmax(scores, dim=-1)
            out    = attn @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 8. RoPE ATTENTION
# kept for backward compat with Part 2/3 runs
# (attention_type = "rope" still works)
# ─────────────────────────────────────────────

class RoPEAttention(nn.Module):
    def __init__(self, cfg, base=10000.0):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.d_head, 2).float() / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _get_cos_sin(self, T, device):
        t     = torch.arange(T, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        cos, sin = self._get_cos_sin(T, x.device)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 9. RoPE + INTERPOLATION
# ─────────────────────────────────────────────

class RoPEWithInterpolation(RoPEAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = getattr(cfg, "rope_scale", 1.0)

    def _get_cos_sin(self, T, device):
        t     = torch.arange(T, device=device) * self.scale
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


# ─────────────────────────────────────────────
# 10. ALiBi
# kept for backward compat with Part 2/3 runs
# ─────────────────────────────────────────────

class ALiBiAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)

        slopes = torch.tensor(
            [2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)]
        )
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        scale  = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        pos    = torch.arange(T, device=x.device)
        dist   = (pos[None, :] - pos[:, None]).clamp(min=0)
        bias   = -self.slopes[:, None, None] * dist
        scores = scores + bias.unsqueeze(0)

        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        out    = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 11. RELATIVE POSITIONAL ATTENTION
# kept for backward compat with Part 2/3 runs
# ─────────────────────────────────────────────

class RelativePositionalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv      = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out      = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.max_dist = 4096
        self.rel_emb  = nn.Embedding(self.max_dist, self.d_head)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        scale  = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        idx        = torch.arange(T, device=x.device)
        dist       = (idx[:, None] - idx[None, :]).clamp(0, self.max_dist - 1)
        rel        = self.rel_emb(dist)
        rel_scores = torch.einsum("bhid,ijd->bhij", q, rel)
        scores     = scores + rel_scores

        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        out    = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

ATTENTION_REGISTRY = {
    "standard":       StandardMultiHeadAttention,
    "sliding_window": SlidingWindowAttention,
    "sparse_block":   SparseBlockAttention,
    "gqa":            GroupedQueryAttention,
    "mqa":            MultiQueryAttention,

    # backward compat — Part 2/3 runs unchanged
    "rope":           RoPEAttention,
    "rope_interp":    RoPEWithInterpolation,
    "alibi":          ALiBiAttention,
    "relative":       RelativePositionalAttention,
}


def build_attention(cfg):
    return ATTENTION_REGISTRY[cfg.attention_type](cfg)