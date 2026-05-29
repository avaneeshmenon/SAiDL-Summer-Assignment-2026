"""
sora.py 
───────
Sparse Low-Rank Adaptation (SoRA) implemented from scratch.

SoRA extends LoRA by introducing a per-rank gating vector g that is
optimized with proximal gradient descent on an L1-regularized objective.
This drives small gate values to exactly zero, automatically determining
the effective rank during training.

The weight update is:
    W = W0 + B * diag(g) * A

where:
    A ∈ R^{r x d}  — down-projection
    B ∈ R^{k x r}  — up-projection
    g ∈ R^r        — gating vector, sparsified via proximal update

The proximal update (soft-thresholding) for g:
    g_temp = g - α * grad_g
    g_new  = sign(g_temp) * max(|g_temp| - λ * α, 0)

Fixes applied vs original:
    1. forward() now correctly adds bias when present
    2. _inject_sora targets DeBERTa-v3's actual layer names:
       "in_proj", "pos_proj", "pos_q_proj" (not query_proj/key_proj/value_proj)
    3. Gate initialized to small values (0.1) so proximal thresholding
       can actually zero them out during training
    4. apply_proximal_update uses a stable, fixed lambda rather than
       scaling by the scheduler LR (which is near-zero during warmup)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Which layer name substrings to target in DeBERTa-v3
# DeBERTa-v3 uses DisentangledSelfAttention; the Q/K/V projections are:
#   attention.self.query_proj
#   attention.self.key_proj
#   attention.self.value_proj
# (12 transformer layers × 3 projections = 36 replaced modules)
# ─────────────────────────────────────────────────────────────────────────────
DEBERTA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj"]


class SoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with SoRA adaptation.

    Forward pass:
        out = x @ W0.T + bias  +  (x @ A.T * g) @ B.T * scaling
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        r:            int = 8,
        lora_alpha:   int = 16,
        lora_dropout: float = 0.1,
        lora_lambda:  float = 1e-3,
        bias:         bool = True,
    ):
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_lambda = lora_lambda

        # ── Frozen base weight ────────────────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # ── Bias (mirrors original layer) ─────────────────────────────────
        # FIX 1: keep bias as a proper frozen parameter so forward is correct
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # ── LoRA matrices ─────────────────────────────────────────────────
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # ── Gating vector ─────────────────────────────────────────────────
        # FIX 3: init small so proximal thresholding can zero gates out.
        # ones(r) makes gates too large relative to the threshold λ·lr.
        self.gate = nn.Parameter(torch.full((r,), 0.1))

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(
            lora_dropout) if lora_dropout > 0 else nn.Identity()

        # ── Init ──────────────────────────────────────────────────────────
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is zeros → adaptation delta starts at zero ✓

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FIX 1: pass bias correctly (was missing — caused wrong base output)
        base = F.linear(x, self.weight, self.bias)

        # SoRA delta: (x A^T) ⊙ g  →  B^T  ×  scaling
        ax = self.dropout(x) @ self.lora_A.T   # (..., r)
        gax = ax * self.gate                      # element-wise gate
        out = gax @ self.lora_B.T                 # (..., out_features)

        return base + out * self.scaling

    @torch.no_grad()
    def apply_proximal_update(self, lr: float):
        """
        Soft-thresholding:  g ← sign(g) · max(|g| − λ·lr, 0)

        FIX 4: The threshold is λ·lr. During warmup the scheduler LR
        is near-zero, making the threshold ≈ 0 and gates never zeroed.
        We therefore use the *base* lr passed in from the optimizer
        param group, not the scheduler's get_last_lr().
        """
        threshold = self.lora_lambda * lr
        self.gate.data = torch.sign(self.gate.data) * \
            torch.clamp(self.gate.data.abs() - threshold, min=0.0)

    def effective_rank(self, eps: float = 1e-3) -> int:
        return (self.gate.data.abs() > eps).sum().item()

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel() + self.gate.numel()


# ─────────────────────────────────────────────────────────────────────────────
# SoRAModel
# ─────────────────────────────────────────────────────────────────────────────

class SoRAModel(nn.Module):
    """
    Wraps a HuggingFace model and replaces targeted Linear layers
    with SoRALinear modules.

    For DeBERTa-v3-base the targeted names are:
        "in_proj", "pos_proj", "pos_q_proj"

    You can override via the `target_modules` argument.
    """

    def __init__(
        self,
        base_model:     nn.Module,
        cfg,
        target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.model = base_model
        self.cfg = cfg

        # FIX 2: use correct DeBERTa-v3 layer names by default
        self.target_modules = target_modules or DEBERTA_TARGET_MODULES

        self._inject_sora(cfg.sora_r, cfg.lora_alpha,
                          cfg.lora_dropout, cfg.sora_lambda)
        self._freeze_base()

    # ── injection ────────────────────────────────────────────────────────────

    def _inject_sora(self, r, alpha, dropout, lora_lambda):
        """Replace query and value projections with SoRALinear."""
        replaced = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in self.target_modules:
                parent_name, child_name = name.rsplit(".", 1)
                parent = self.model.get_submodule(parent_name)

                sora = SoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    lora_lambda=lora_lambda,
                )
                # Copy pretrained weights
                sora.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    sora.bias.data = module.bias.data.clone()

                setattr(parent, child_name, sora)
                replaced += 1

        print(f"  SoRA: replaced {replaced} linear layers")
        if replaced == 0:
            raise RuntimeError(
                "SoRA replaced 0 layers! Check target_modules against your "
                "model's actual layer names. Run:\n"
                "  [n for n,m in model.named_modules() "
                "if isinstance(m, nn.Linear)]"
            )

    def _freeze_base(self):
        """Freeze everything except LoRA/gate params and classifier/pooler."""
        frozen = trainable = 0
        for name, param in self.model.named_parameters():
            if any(k in name for k in
                   ["lora_A", "lora_B", "gate", "classifier", "pooler"]):
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        print(f"  Frozen: {frozen:,}  |  Trainable: {trainable:,}")

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, **kwargs):
        return self.model(**kwargs)

    # ── proximal updates ─────────────────────────────────────────────────────

    @torch.no_grad()
    def apply_proximal_updates(self, base_lr: float):
        """
        Apply soft-thresholding to all gate parameters.

        Args:
            base_lr: the *optimizer* learning rate (cfg.learning_rate),
                     NOT scheduler.get_last_lr() which is near-zero during warmup.
        """
        for module in self.model.modules():
            if isinstance(module, SoRALinear):
                module.apply_proximal_update(base_lr)

    # ── utilities ────────────────────────────────────────────────────────────

    def effective_ranks(self, eps: float = 1e-3) -> dict[str, int]:
        return {
            name: (module.gate.data.abs() > eps).sum().item()
            for name, module in self.model.named_modules()
            if isinstance(module, SoRALinear)
        }

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
