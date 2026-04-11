import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def reshape_heads(x, B, T, n_heads, d_head):
    return x.view(B, T, n_heads, d_head).transpose(1, 2)


# ─────────────────────────────────────────────
# 1. STANDARD
# ─────────────────────────────────────────────
class StandardMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

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
        self.window = getattr(cfg, "window_size", 128)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        mask = torch.full((T, T), float("-inf"), device=x.device)
        for i in range(T):
            start = max(0, i - self.window)
            mask[i, start:i+1] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 3. SPARSE BLOCK
# ─────────────────────────────────────────────
class SparseBlockAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block = getattr(cfg, "block_size", 64)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        mask = torch.full((T, T), float("-inf"), device=x.device)

        for i in range(T):
            block_id = i // self.block
            start = block_id * self.block
            mask[i, start:i+1] = 0

            for b in range(block_id):
                mask[i, (b+1)*self.block - 1] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 4. LINEAR
# ─────────────────────────────────────────────
class LinearAttention(nn.Module):
    """
    Causal Linear Attention (stable version)

    Fixes:
    - better denominator stabilization (no hard clamp to 1.0)
    - proper scaling (like softmax attention)
    - avoids numerical explosion
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.drop = nn.Dropout(getattr(cfg, "dropout", 0.0))

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.eps = 1e-4  # 🔥 stability

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # 🔥 feature map + scaling (IMPORTANT)
        q = self.phi(q) * (self.d_head ** -0.5)
        k = self.phi(k)

        d = self.d_head

        S = torch.zeros(B, self.n_heads, d, d, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, self.n_heads, d,    device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            # update prefix sums
            S = S + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
            z = z + k_t

            # numerator
            num = (q_t.unsqueeze(-2) @ S).squeeze(-2)

            # 🔥 FIXED denominator (no hard clamp)
            den = (q_t * z).sum(dim=-1, keepdim=True) + self.eps

            outputs.append(num / den)

        out = torch.stack(outputs, dim=2)

        out = self.drop(out)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)

# ─────────────────────────────────────────────
# 5. GQA
# ─────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = max(1, cfg.n_heads // 4)
        self.d_head = cfg.d_model // cfg.n_heads

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.v = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)

        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = reshape_heads(self.q(x), B, T, self.n_heads, self.d_head)
        k = reshape_heads(self.k(x), B, T, self.n_kv, self.d_head)
        v = reshape_heads(self.v(x), B, T, self.n_kv, self.d_head)

        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 6. MQA
# ─────────────────────────────────────────────
class MultiQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.n_kv = 1  # ✅ FIX: define early
        self.d_head = cfg.d_model // cfg.n_heads

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)
        self.v = nn.Linear(cfg.d_model, self.n_kv * self.d_head, bias=False)

        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        q = reshape_heads(self.q(x), B, T, self.n_heads, self.d_head)
        k = reshape_heads(self.k(x), B, T, self.n_kv, self.d_head)
        v = reshape_heads(self.v(x), B, T, self.n_kv, self.d_head)

        # expand KV to match heads
        k = k.expand(B, self.n_heads, T, self.d_head)
        v = v.expand(B, self.n_heads, T, self.d_head)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


# ─────────────────────────────────────────────
# 7. SOFTMAX-FREE
# ─────────────────────────────────────────────
class SoftmaxFreeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)

        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)

        return self.out(out.transpose(1, 2).reshape(B, T, C))
    
# ─────────────────────────────────────────────
# 8. RoPE ATTENTION
# ─────────────────────────────────────────────
class RoPEAttention(nn.Module):
    def __init__(self, cfg, base=10000.0):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def _get_cos_sin(self, T, device):
        t = torch.arange(T, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
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
        t = torch.arange(T, device=device) * self.scale
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    
# ─────────────────────────────────────────────
# 10. ALiBi
# ─────────────────────────────────────────────
class ALiBiAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        slopes = torch.tensor([2 ** (-8 * i / self.n_heads) for i in range(self.n_heads)])
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        scale = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        pos = torch.arange(T, device=x.device)
        dist = (pos[None, :] - pos[:, None]).clamp(min=0)
        bias = -self.slopes[:, None, None] * dist
        bias = bias.unsqueeze(0)

        scores = scores + bias

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))
    
# ─────────────────────────────────────────────
# 11. RELATIVE POSITIONAL ATTENTION
# ─────────────────────────────────────────────
class RelativePositionalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.max_dist = cfg.context_length
        self.rel_emb = nn.Embedding(self.max_dist, self.d_head)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        scale = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        idx = torch.arange(T, device=x.device)
        dist = (idx[:, None] - idx[None, :]).clamp(0, self.max_dist - 1)
        rel = self.rel_emb(dist)

        rel_scores = torch.einsum("bhid,ijd->bhij", q, rel)
        scores = scores + rel_scores

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        return self.out(out.transpose(1, 2).reshape(B, T, C))



# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────
ATTENTION_REGISTRY = {
    "standard": StandardMultiHeadAttention,
    "sliding_window": SlidingWindowAttention,
    "sparse_block": SparseBlockAttention,
    "linear": LinearAttention,
    "gqa": GroupedQueryAttention,
    "mqa": MultiQueryAttention,
    "softmax_free": SoftmaxFreeAttention,

    "rope": RoPEAttention,
    "rope_interp": RoPEWithInterpolation,
    "alibi": ALiBiAttention,
    "relative": RelativePositionalAttention,
}


def build_attention(cfg):
    return ATTENTION_REGISTRY[cfg.attention_type](cfg)
