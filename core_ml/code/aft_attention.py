"""
aft_attention.py
────────────────
Attention-Free Transformer (AFT) variants from:
  "An Attention Free Transformer" — Zhai et al. (2021)

Variants implemented:
  1. AFTFull        — Full T×T learned position bias
  2. AFTLocal       — Position bias only within a local window
  3. AFTSimple      — No position bias; O(T) global aggregation
  4. AFTConv        — Relative-offset position bias (convolutional)
  5. AFTRoPESimple  — Our own variant: AFT-Simple + RoPE on Keys
                      to restore positional awareness lost in AFT-Simple

Mathematical summary
────────────────────
AFT core formula (for position t):

    Y[t] = sigmoid(Q[t])  ⊙  Σ_s  exp(w[t,s] + K[s]) ⊙ V[s]  
                            ─────────────────────────────────────────
                                Σ_s  exp(w[t,s] + K[s])

Where:
    Q, K, V ∈ R^{T × d}   — projections of input x
    w       ∈ R^{T × T}   — learned position bias (variant-dependent)
    ⊙                      — element-wise multiply

Causal masking: for autoregressive LM, w[t, s] = -inf for s > t.
This ensures position t can only attend to positions 0..t.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. AFT-Full
# ─────────────────────────────────────────────────────────────────────────────

class AFTFull(nn.Module):
    """
    AFT-Full: Full T×T learned position bias.

    The bias table w is of shape (context_length, context_length).
    For autoregressive use we apply a causal mask (upper triangle → -inf).

    Complexity: O(T² · d) — same asymptotic as standard attention but no
    matrix multiply; only element-wise operations.

    Key difference from standard attention:
      - No QK dot-product; Q is used as a sigmoid gate only
      - w[t,s] is a *scalar* per position pair, shared across d dimensions
        → far fewer parameters than d-dimensional attention scores
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        T = cfg.context_length

        # Projections — no bias needed (position bias handles that role)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Learned T×T position bias — initialized small
        # w[t, s] = how much position s contributes to position t
        self.w = nn.Parameter(torch.zeros(T, T))
        nn.init.normal_(self.w, std=0.02)

        # Causal mask (registered as buffer — not a parameter, moves with .to(device))
        # upper_tri[t, s] = True means s > t (future position)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x)   # (B, T, d)
        k = self.k(x)   # (B, T, d)
        v = self.v(x)   # (B, T, d)

        # Slice position bias to actual sequence length (T may < context_length)
        w = self.w[:T, :T]  # (T, T)

        # Apply causal mask: future positions get -inf so exp(-inf) = 0
        w = w.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # ── Core AFT computation ─────────────────────────────────────────
        # exp(w[t,s] + K[s]) for all t, s  →  shape (T, T, d)
        #
        # We want:  exp_wk[t, s, d] = exp(w[t,s] + k[b, s, d])
        # w:  (T, T)       → unsqueeze last dim → (T, T, 1)
        # k:  (B, T, d)    → need (B, T, T, d):  k[:, None, :, :]
        #
        # exp_wk[b, t, s, d] = exp(w[t, s] + k[b, s, d])

        exp_wk = torch.exp(
            w.unsqueeze(0).unsqueeze(-1)       # (1, T, T, 1)
            + k.unsqueeze(1)                   # (B, 1, T, d)
        )                                       # → (B, T, T, d)

        # Numerator: Σ_s  exp_wk[b, t, s, d] * v[b, s, d]
        # v: (B, T, d) → (B, 1, T, d) for broadcasting over t
        numerator = (exp_wk * v.unsqueeze(1)).sum(dim=2)   # (B, T, d)

        # Denominator: Σ_s  exp_wk[b, t, s, d]  — sum over s, then expand d
        denominator = exp_wk.sum(dim=2)                    # (B, T, d)

        # AFT aggregation + query gate
        aft_out = torch.sigmoid(q) * (numerator / (denominator + 1e-9))  # (B, T, d)

        return self.out(aft_out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AFT-Local
# ─────────────────────────────────────────────────────────────────────────────

class AFTLocal(nn.Module):
    """
    AFT-Local: Position bias is non-zero only within a window of size s.

    For |t - s| > window_size, w[t,s] is fixed at 0 (not -inf — the position
    simply contributes exp(0 + K[s]) = exp(K[s]) uniformly).

    To truly zero out distant positions we mask them to -inf instead, which
    gives the model *local-only* attention, similar to sliding window attention
    but with a different mechanism.

    Complexity: O(T · window_size · d)
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.window = getattr(cfg, "aft_local_window", 64)
        T = cfg.context_length

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Only learn biases for pairs within window
        # For pairs outside window we'll use w=0 (uniform contribution)
        # The original paper learns a (T, 2*window+1) table and indexes it
        # We implement it as a full T×T but with a locality mask
        self.w = nn.Parameter(torch.zeros(T, T))
        nn.init.normal_(self.w, std=0.02)

        # Build locality + causality mask
        # True = this (t, s) pair should be masked to -inf
        idx = torch.arange(T)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)          # (T, T): dist[t,s] = t - s
        # Mask: future positions OR positions too far in the past
        locality_mask = (dist < 0) | (dist > self.window)   # causal + local
        self.register_buffer("locality_mask", locality_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        w = self.w[:T, :T].masked_fill(self.locality_mask[:T, :T], float("-inf"))

        exp_wk = torch.exp(
            w.unsqueeze(0).unsqueeze(-1) + k.unsqueeze(1)
        )  # (B, T, T, d)

        numerator = (exp_wk * v.unsqueeze(1)).sum(dim=2)
        denominator = exp_wk.sum(dim=2)

        aft_out = torch.sigmoid(q) * (numerator / (denominator + 1e-9))
        return self.out(aft_out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AFT-Simple
# ─────────────────────────────────────────────────────────────────────────────

class AFTSimple(nn.Module):
    """
    AFT-Simple: Set w[t,s] = 0 for all t, s.

    When all biases are zero, the weighted sum doesn't depend on t:
        Σ_s exp(K[s]) ⊙ V[s]   and   Σ_s exp(K[s])
    are the SAME for every output position t.

    We compute them once and reuse → O(T · d) instead of O(T² · d).

    Trade-off: Zero positional awareness. The model sees a "bag of tokens".
    Causal masking still applies: at position t, only sum over s ≤ t.

    This is surprisingly competitive for short contexts because the
    query sigmoid gate provides some position-specific modulation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x)   # (B, T, d)
        k = self.k(x)   # (B, T, d)
        v = self.v(x)   # (B, T, d)

        # exp(K[s]) — shape (B, T, d)
        exp_k = torch.exp(k)

        # Causal cumulative sum: at position t, sum over s = 0..t
        # torch.cumsum along the T dimension gives exactly this
        cumsum_expk_v = torch.cumsum(exp_k * v, dim=1)  # (B, T, d)  numerator
        cumsum_expk   = torch.cumsum(exp_k,     dim=1)  # (B, T, d)  denominator

        aft_out = torch.sigmoid(q) * (cumsum_expk_v / (cumsum_expk + 1e-9))
        return self.out(aft_out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AFT-Conv  (relative position bias)
# ─────────────────────────────────────────────────────────────────────────────

class AFTConv(nn.Module):
    """
    AFT-Conv: Position bias depends only on relative offset (t - s),
    not absolute positions t and s independently.

    w[t, s] = w_rel[t - s]   for s ≤ t  (causal)

    This is like a 1D convolution kernel of length T applied to the
    key-value products. Parameters: O(T) scalars instead of O(T²).

    The name "Conv" comes from the fact that computing Σ_s w_rel[t-s] * f(s)
    is literally a 1D convolution of f with kernel w_rel.

    We implement it by:
      1. Build w[t,s] = w_rel[t-s] on the fly from the 1D parameter vector
      2. Apply causal mask
      3. Proceed as AFT-Full
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        T = cfg.context_length

        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Only T parameters for relative offsets 0, 1, 2, ..., T-1
        # w_rel[0] = bias for same position, w_rel[1] = one step back, etc.
        self.w_rel = nn.Parameter(torch.zeros(T))
        nn.init.normal_(self.w_rel, std=0.02)

        # Build index matrix: offset[t, s] = t - s, clamped to [0, T-1]
        idx = torch.arange(T)
        offset = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp(min=0)  # (T, T)
        self.register_buffer("offset", offset)

        # Causal mask
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Build full T×T bias from relative offsets
        w = self.w_rel[self.offset[:T, :T]]           # (T, T)
        w = w.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        exp_wk = torch.exp(
            w.unsqueeze(0).unsqueeze(-1) + k.unsqueeze(1)
        )  # (B, T, T, d)

        numerator   = (exp_wk * v.unsqueeze(1)).sum(dim=2)
        denominator = exp_wk.sum(dim=2)

        aft_out = torch.sigmoid(q) * (numerator / (denominator + 1e-9))
        return self.out(aft_out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AFT-RoPE-Simple  (our own variant)
# ─────────────────────────────────────────────────────────────────────────────

class AFTRoPESimple(nn.Module):
    """
    Our own variant: AFT-Simple + RoPE on Keys.

    ── Motivation ──────────────────────────────────────────────────────────
    AFT-Simple is O(T) and elegant, but has *zero* positional awareness:
    the cumulative sums treat all past tokens identically regardless of
    where they appear.

    The fix: rotate the Key vectors with RoPE before the cumulative sum.
    RoPE encodes position by rotating pairs of key dimensions:
        k_rot[s, 2i]   = k[s, 2i]   * cos(s * theta_i) - k[s, 2i+1] * sin(...)
        k_rot[s, 2i+1] = k[s, 2i]   * sin(s * theta_i) + k[s, 2i+1] * cos(...)

    After rotation, exp(k_rot[s]) carries position-s information.
    The cumulative sum then naturally weights recent tokens differently
    from distant ones — without any learned T×T table.

    ── Complexity ──────────────────────────────────────────────────────────
    O(T · d) — same as AFT-Simple. No T×T matrix anywhere.

    ── Parameters ──────────────────────────────────────────────────────────
    No extra parameters vs AFT-Simple. RoPE frequencies are fixed buffers.

    ── Why this is a genuine improvement ───────────────────────────────────
    AFT-Simple's main weakness on language modeling is that without position
    information in the keys, the model can't distinguish "the word that came
    right before me" from "the word that came 500 tokens ago". RoPE-rotated
    keys break this symmetry, giving the model relative distance information
    at zero parameter cost.
    """

    def __init__(self, cfg, base: float = 10000.0):
        super().__init__()
        self.d_model = cfg.d_model
        d = cfg.d_model

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

        # RoPE inverse frequencies — same as RoPEAttention in your codebase
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ...] → [-x_{d/2+1}, ..., x1, x2, ...]"""
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to tensor of shape (B, T, d)."""
        B, T, d = x.shape
        t = torch.arange(T, device=x.device)
        freqs = torch.outer(t, self.inv_freq)          # (T, d/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (T, d)
        cos   = emb.cos().unsqueeze(0)                  # (1, T, d)
        sin   = emb.sin().unsqueeze(0)                  # (1, T, d)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape

        q = self.q(x)           # (B, T, d)
        k = self.k(x)           # (B, T, d)
        v = self.v(x)           # (B, T, d)

        # Apply RoPE to keys — encodes absolute position into the key vectors
        k = self._apply_rope(k)  # (B, T, d)

        # AFT-Simple with RoPE keys
        exp_k = torch.exp(k)                                     # (B, T, d)
        cumsum_expk_v = torch.cumsum(exp_k * v, dim=1)           # (B, T, d)
        cumsum_expk   = torch.cumsum(exp_k,     dim=1)           # (B, T, d)

        aft_out = torch.sigmoid(q) * (cumsum_expk_v / (cumsum_expk + 1e-9))
        return self.out(aft_out)


# ─────────────────────────────────────────────────────────────────────────────
# 6. AFT-Decay  (our improved variant)
# ─────────────────────────────────────────────────────────────────────────────

class AFTDecay(nn.Module):
    """
    AFT-Decay: AFT-Simple + data-dependent exponential decay gate.

    ── The genuine gap in AFT-Simple ───────────────────────────────────────
    AFT-Simple's cumulative sum weights every past token equally (up to key
    magnitude). Language has strong locality bias: nearby tokens are far more
    relevant than tokens from 500 positions ago. AFT-Simple cannot learn this.

    RoPE on keys (AFT-RoPE-Simple) was our first attempt, but it failed
    because RoPE works through the Q·K dot-product interaction in standard
    attention. In AFT, Q is only a sigmoid gate — there is no Q·K inner
    product for RoPE to encode relative distance through. The rotation just
    adds noise to exp(K) without creating the expected position structure.

    ── Fix: content-dependent decay gate ───────────────────────────────────
    Replace the cumulative sum with a recurrent update that has a learned,
    input-dependent decay gate — like an RNN forget gate applied to AFT's
    key-value aggregation:

        gate[t] = σ(W_g · x[t])  ∈ (0,1)^d     ← how much to retain history
        h[t]    = gate[t] ⊙ h[t-1] + exp(K[t]) ⊙ V[t]
        z[t]    = gate[t] ⊙ z[t-1] + exp(K[t])
        Y[t]    = σ(Q[t]) ⊙ h[t] / (z[t] + ε)

    When gate ≈ 1: full history retained (long-range dependencies).
    When gate ≈ 0: history forgotten (focus on current token).

    The gate is content-dependent — the model learns WHEN to forget and
    when to remember based on what it's currently reading. This gives:
      1. Implicit recency bias (decayed history → recent tokens weighted more)
      2. Selective memory (some positions trigger forgetting, others don't)
      3. Implicit positional awareness through the recurrent state

    ── Complexity ──────────────────────────────────────────────────────────
    O(T · d) — no T×T matrix anywhere. Sequential scan (parallelisable
    with associative scan, but sequential here for simplicity).

    ── Parameters ──────────────────────────────────────────────────────────
    One extra linear layer W_g: d×d + d parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.q    = nn.Linear(d, d, bias=False)
        self.k    = nn.Linear(d, d, bias=False)
        self.v    = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=True)
        self.out  = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape

        q    = self.q(x)                      # (B, T, d)
        k    = self.k(x)                      # (B, T, d)
        v    = self.v(x)                      # (B, T, d)
        gate = torch.sigmoid(self.gate(x))    # (B, T, d) ∈ (0, 1)
        exp_k = torch.exp(k)                  # (B, T, d)

        # Parallel prefix scan — no Python loop over T.
        # Recurrence h[t] = gate[t]*h[t-1] + u[t] has closed form:
        #   h[t] = cp[t] * cumsum(u / cp)[t]
        # where cp[t] = gate[0]*...*gate[t] (cumulative product).
        # Computed in log-space for numerical stability.
        log_cp  = torch.cumsum(torch.log(gate.clamp(min=1e-8)), dim=1)
        cp      = torch.exp(log_cp.clamp(-30, 30))      # (B, T, d)
        inv_cp  = torch.exp(-log_cp.clamp(-30, 30))     # (B, T, d)

        h = cp * torch.cumsum(exp_k * v * inv_cp, dim=1)   # (B, T, d)
        z = cp * torch.cumsum(exp_k     * inv_cp, dim=1)   # (B, T, d)

        return self.out(torch.sigmoid(q) * (h / (z + 1e-9)))


# ─────────────────────────────────────────────────────────────────────────────
# Registry — plug into your existing build_attention() system
# ─────────────────────────────────────────────────────────────────────────────

AFT_REGISTRY = {
    "aft_full":        AFTFull,
    "aft_local":       AFTLocal,
    "aft_simple":      AFTSimple,
    "aft_conv":        AFTConv,
    "aft_rope_simple": AFTRoPESimple,
    "aft_decay":       AFTDecay,
}

# ─────────────────────────────────────────────────────────────────────────────
# Self-registration
# ─────────────────────────────────────────────────────────────────────────────
# When this file is imported anywhere, it silently injects AFT variants into
# the existing ATTENTION_REGISTRY. No edits needed to attention.py or model.py.
# The only requirement: import aft_attention before building any AFT model.

from attention import ATTENTION_REGISTRY
ATTENTION_REGISTRY.update(AFT_REGISTRY)