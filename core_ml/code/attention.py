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
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def phi(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = reshape_heads(q, B, T, self.n_heads, self.d_head)
        k = reshape_heads(k, B, T, self.n_heads, self.d_head)
        v = reshape_heads(v, B, T, self.n_heads, self.d_head)

        q = self.phi(q)
        k = self.phi(k)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)

        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)

        return self.out(out.transpose(1, 2).reshape(B, T, C))


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
class MultiQueryAttention(GroupedQueryAttention):
    def __init__(self, cfg):
        cfg.n_heads = cfg.n_heads
        super().__init__(cfg)
        self.n_kv = 1


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
}

def build_attention(cfg):
    return ATTENTION_REGISTRY[cfg.attention_type](cfg)