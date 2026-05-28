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
    A ∈ R^{r x d}  — down-projection (frozen after init in standard LoRA,
                      but trained here)
    B ∈ R^{k x r}  — up-projection
    g ∈ R^r        — gating vector, sparsified via proximal update

The proximal update (soft-thresholding) for g:
    g ← prox_{λα}(g - α * ∂L/∂g)
      = sign(g - α * ∂L/∂g) * max(|g - α * ∂L/∂g| - λα, 0)

This is equivalent to:
    g_temp = g - α * grad_g
    g_new  = sign(g_temp) * max(|g_temp| - λ * α, 0)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with SoRA adaptation.

    The forward pass computes:
        out = x @ W0.T + x @ A.T @ diag(g) @ B.T
            = base_out  +  lora_out

    g is maintained as a raw parameter and soft-thresholded
    after each optimizer step via apply_proximal_update().
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        r:            int   = 8,
        lora_alpha:   int   = 16,
        lora_dropout: float = 0.1,
        lora_lambda:  float = 1e-3,
    ):
        super().__init__()

        self.r           = r
        self.lora_alpha  = lora_alpha
        self.scaling     = lora_alpha / r
        self.lora_lambda = lora_lambda

        # ── Frozen base weight ────────────────────────────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # ── LoRA matrices ─────────────────────────────────────────────────
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # ── Gating vector ─────────────────────────────────────────────────
        self.gate = nn.Parameter(torch.ones(r))

        # ── Dropout ───────────────────────────────────────────────────────
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # ── Init ──────────────────────────────────────────────────────────
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B already zeros → adaptation starts at zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen forward
        base = F.linear(x, self.weight)

        # SoRA adaptation: x A^T diag(g) B^T * scaling
        ax   = self.dropout(x) @ self.lora_A.T          # (..., r)
        gax  = ax * self.gate                            # element-wise gate
        out  = gax @ self.lora_B.T                       # (..., out_features)

        return base + out * self.scaling

    @torch.no_grad()
    def apply_proximal_update(self, lr: float):
        """
        Soft-thresholding proximal update for the gate parameter.
        Called after optimizer.step() to induce sparsity.

        prox_{λlr}(g) = sign(g) * max(|g| - λ*lr, 0)
        """
        threshold = self.lora_lambda * lr
        self.gate.data = torch.sign(self.gate.data) * \
            torch.clamp(self.gate.data.abs() - threshold, min=0.0)

    def effective_rank(self) -> int:
        """Number of non-zero gate values = effective rank."""
        return (self.gate.abs() > 1e-6).sum().item()

    def trainable_parameters(self) -> int:
        return (
            self.lora_A.numel() +
            self.lora_B.numel() +
            self.gate.numel()
        )


class SoRAModel(nn.Module):
    """
    Wraps a HuggingFace model and replaces all query/value Linear layers
    in attention blocks with SoRALinear modules.

    Usage:
        base_model = AutoModelForSequenceClassification.from_pretrained(...)
        model = SoRAModel(base_model, cfg)
    """

    def __init__(self, base_model: nn.Module, cfg):
        super().__init__()
        self.model = base_model
        self.cfg   = cfg

        self._inject_sora(cfg.sora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.sora_lambda)
        self._freeze_base()

    def _inject_sora(self, r, alpha, dropout, lora_lambda):
        """Replace query and value projections with SoRALinear."""
        replaced = 0
        for name, module in self.model.named_modules():
            # Target attention query and value projections
            if isinstance(module, nn.Linear) and any(
                k in name for k in ["query_proj", "value_proj", "q_proj", "v_proj",
                                    "query", "value"]
            ):
                parent_name, child_name = name.rsplit(".", 1)
                parent = self.model.get_submodule(parent_name)

                sora = SoRALinear(
                    in_features  = module.in_features,
                    out_features = module.out_features,
                    r            = r,
                    lora_alpha   = alpha,
                    lora_dropout = dropout,
                    lora_lambda  = lora_lambda,
                )
                # Copy pretrained weights
                sora.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    sora.bias = nn.Parameter(module.bias.data.clone())

                setattr(parent, child_name, sora)
                replaced += 1

        print(f"  SoRA: replaced {replaced} linear layers")

    def _freeze_base(self):
        """Freeze everything except LoRA/gate parameters."""
        for name, param in self.model.named_parameters():
            if not any(k in name for k in ["lora_A", "lora_B", "gate", "classifier"]):
                param.requires_grad = False

    def forward(self, **kwargs):
        return self.model(**kwargs)

    @torch.no_grad()
    def apply_proximal_updates(self, lr: float):
        """Apply soft-thresholding to all gate parameters."""
        for module in self.model.modules():
            if isinstance(module, SoRALinear):
                module.apply_proximal_update(lr)

    def effective_ranks(self) -> dict:
        """Returns effective rank for each SoRALinear layer."""
        ranks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, SoRALinear):
                ranks[name] = module.effective_rank()
        return ranks

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)