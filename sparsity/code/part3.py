"""
part3.py
────────
Part 3 — SoRA-style sparse low-rank adaptation applied to
recurrent sequence architectures: xLSTM and Mamba, on CoLA (GLUE).

Key methodological difference vs Transformer (Part 1):
    - Transformers have attention weight matrices (Q/K/V) that are square
      and used in parallel — easy targets for low-rank adaptation.
    - xLSTM and Mamba are recurrent: their parameters are gate projection
      matrices (input→hidden). We apply SoRA to these projections instead.
    - The proximal update logic (soft-thresholding on gate vector g) is
      identical — only the target modules change.

Architecture notes:
    xLSTM  — We use the xlstm pip package (Beck et al. 2024).
              Target modules: input projections of mLSTM blocks (z_proj, o_proj)
              and sLSTM blocks (Wz, Wi, Wf, Wo via their linear layers).

    Mamba  — We use the mamba-ssm pip package.
              Target modules: in_proj, out_proj, x_proj, dt_proj
              (the four linear layers in each Mamba block).

Both models are wrapped as sequence classifiers by adding a mean-pool +
linear head on top of the backbone, same as Part 1.

Same evaluation metrics as Part 1:
    MCC, trainable parameters, effective rank, training time.
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef

# ── reuse SoRALinear from Part 1 ─────────────────────────────────────────────
from methods.sora import SoRALinear


# ─────────────────────────────────────────────────────────────────────────────
# Target modules per architecture
# ─────────────────────────────────────────────────────────────────────────────

XLSTM_TARGET_MODULES = [
    # mLSTM cell projections (matrix memory)
    "q_proj", "k_proj", "v_proj",
    # input/output projections in the block
    "proj_up", "proj_down",
]

MAMBA_TARGET_MODULES = [
    "in_proj",   # expands hidden → 2*d_inner
    "out_proj",  # projects back d_inner → hidden
    "x_proj",    # projects d_inner → dt_rank + 2*d_state
    "dt_proj",   # projects dt_rank → d_inner
]


# ─────────────────────────────────────────────────────────────────────────────
# Generic SoRA injection (same logic as Part 1, parameterised by targets)
# ─────────────────────────────────────────────────────────────────────────────

def inject_sora(model: nn.Module, target_modules: list[str],
                r: int, alpha: int, dropout: float, lora_lambda: float) -> int:
    """
    Walk model, replace any nn.Linear whose attribute name is in
    target_modules with a SoRALinear. Returns count of replaced layers.

    Uses child-name matching (same as peft) so it works regardless of
    how deeply nested the layers are.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        child_name = name.rsplit(".", 1)[-1]
        if child_name not in target_modules:
            continue

        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)

        has_bias = module.bias is not None
        sora = SoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            lora_lambda=lora_lambda,
            bias=has_bias,
        )
        sora.weight.data = module.weight.data.clone()
        if has_bias:
            sora.bias.data = module.bias.data.clone()

        setattr(parent, child, sora)
        replaced += 1

    return replaced


def freeze_base(model: nn.Module):
    """Freeze all params except SoRA trainable ones and classifier head."""
    for name, param in model.named_parameters():
        if any(k in name for k in ["lora_A", "lora_B", "gate", "classifier", "head", "embed", "norm"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_proximal_updates(model: nn.Module, base_lr: float):
    for module in model.modules():
        if isinstance(module, SoRALinear):
            module.apply_proximal_update(base_lr)


def get_effective_ranks(model: nn.Module, eps: float = 1e-3) -> dict:
    return {
        name: (module.gate.data.abs() > eps).sum().item()
        for name, module in model.named_modules()
        if isinstance(module, SoRALinear)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cola(cfg, tokenizer):
    dataset = load_dataset("nyu-mll/glue", "cola")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=[
                       "input_ids", "attention_mask", "labels"])
    return dataset["train"], dataset["validation"]


def compute_mcc(preds, labels):
    return matthews_corrcoef(labels, preds)


# ─────────────────────────────────────────────────────────────────────────────
# Generic sequence classifier wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SequenceClassifier(nn.Module):
    """
    Wraps any backbone that returns (batch, seq, hidden) into a
    binary sequence classifier via mean pooling + linear head.

    Works for both xLSTM and Mamba since both output
    (batch, seq_len, d_model) hidden states.
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # backbone forward — returns (B, T, D)
        hidden = self.backbone(input_ids)           # (B, T, D)

        # mean pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            hidden = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            hidden = hidden.mean(dim=1)             # (B, D)

        logits = self.classifier(self.dropout(hidden))   # (B, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        # mimic HuggingFace output so train loop is identical to Part 1
        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=logits)


# ─────────────────────────────────────────────────────────────────────────────
# xLSTM backbone
# ─────────────────────────────────────────────────────────────────────────────

def build_xlstm_backbone(cfg):
    """
    Build a small xLSTM model using the `xlstm` package.
    Falls back to a minimal pure-PyTorch mLSTM if package unavailable.
    """
    try:
        from xlstm import xLSTMLMModel, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                )
            ),
            num_blocks=4,
            embedding_dim=cfg.xlstm_d_model,
            add_post_blocks_norm=True,
        )

        class xLSTMBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(cfg.vocab_size, cfg.xlstm_d_model)
                self.xlstm = xLSTMLMModel(xlstm_cfg)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                return self.xlstm(x)

        print("  Using xlstm package")
        return xLSTMBackbone(), cfg.xlstm_d_model, XLSTM_TARGET_MODULES

    except ImportError:
        print("  xlstm package not found — using minimal mLSTM implementation")
        return build_minimal_mlstm(cfg), cfg.xlstm_d_model, [
            "Wq",
            "Wk",
            "Wv",
            "Wi",
            "Wf",
            "proj_out"
        ]


def build_minimal_mlstm(cfg):
    """
    Minimal mLSTM (matrix memory LSTM from xLSTM paper) in pure PyTorch.

    State: C ∈ R^{d x d} (matrix memory), n ∈ R^d (normalizer), m (scalar max)
    Update:
        q = Wq * x,  k = Wk * x,  v = Wv * x
        i = exp(Wi*x + bi),  f = exp(Wf*x + bf)   (input/forget gates, log-space)
        m_new = max(log_f + m_old, log_i)
        i' = exp(log_i - m_new),  f' = exp(log_f + m_old - m_new)
        C = f'*C + i' * (v ⊗ k)
        n = f'*n + i'*k
        h = (C*q) / max(|n^T q|, 1)
    """

    class mLSTMCell(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d = d_model
            # These are the target modules for SoRA injection
            self.Wq = nn.Linear(d_model, d_model, bias=False)
            self.Wk = nn.Linear(d_model, d_model, bias=False)
            self.Wv = nn.Linear(d_model, d_model, bias=False)
            self.Wi = nn.Linear(d_model, 1)
            self.Wf = nn.Linear(d_model, 1)
            self.proj_out = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward_sequence(self, x):
            # x: (B, T, D)
            B, T, D = x.shape
            C = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
            n = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            m = torch.full((B, 1), -1e9, device=x.device, dtype=x.dtype)
            outputs = []

            for t in range(T):
                xt = x[:, t, :]
                q = self.Wq(xt)
                k = self.Wk(xt) / (D ** 0.5)
                v = self.Wv(xt)

                log_i = self.Wi(xt)           # (B, 1)
                log_f = F.logsigmoid(self.Wf(xt))  # (B, 1)  — stabilized

                m_new = torch.maximum(log_f + m, log_i)
                i_prime = torch.exp(log_i - m_new)
                f_prime = torch.exp(log_f + m - m_new)

                # C: (B, D, D), outer product v⊗k: (B, D, D)
                C = f_prime.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.bmm(
                    v.unsqueeze(2), k.unsqueeze(1)
                )
                n = f_prime * n + i_prime * k
                m = m_new

                # retrieve
                h_num = torch.bmm(C, q.unsqueeze(2)).squeeze(2)   # (B, D)
                denom = torch.clamp(
                    (n * q).sum(-1, keepdim=True).abs(), min=1.0)
                h = h_num / denom

                outputs.append(h)

            return torch.stack(outputs, dim=1)   # (B, T, D)

    class MinimalMLSTMBackbone(nn.Module):
        def __init__(self, d_model, vocab_size, n_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList(
                [mLSTMCell(d_model) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = x + layer.forward_sequence(self.norm(x))
            return x

    return MinimalMLSTMBackbone(
        d_model=cfg.xlstm_d_model,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.xlstm_n_layers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mamba backbone
# ─────────────────────────────────────────────────────────────────────────────

def build_mamba_backbone(cfg):
    """
    Build Mamba using the `mamba-ssm` package.
    Falls back to a minimal SSM if package unavailable.
    """
    try:
        from mamba_ssm import Mamba
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig

        mamba_cfg = MambaConfig(
            d_model=cfg.mamba_d_model,
            n_layer=cfg.mamba_n_layers,
            vocab_size=cfg.vocab_size,
            ssm_cfg={},
            rms_norm=True,
            fused_add_norm=False,
        )

        class MambaBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MambaLMHeadModel(mamba_cfg)

            def forward(self, input_ids):
                # returns CausalLMOutput; we want hidden states
                out = self.model(input_ids, return_dict=True,
                                 output_hidden_states=True)
                # last hidden state before LM head
                return out.hidden_states[-1]

        print("  Using mamba-ssm package")
        return MambaBackbone(), cfg.mamba_d_model, MAMBA_TARGET_MODULES

    except ImportError:
        print("  mamba-ssm not found — using minimal S6 implementation")
        return build_minimal_mamba(cfg), cfg.mamba_d_model, ["in_proj", "out_proj", "x_proj", "dt_proj"]


def build_minimal_mamba(cfg):
    """
    Minimal Mamba block in pure PyTorch (selective SSM / S6).

    The selective scan mechanism:
        x  → expand → (u, z)          via in_proj
        u  → conv1d → SSM(u) → y      selective state space
        y  → gate(z) → out_proj → h
    """

    class S6(nn.Module):
        """Minimal selective state space model."""

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_inner = d_model * expand
            self.d_state = d_state

            # These are the SoRA target modules
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.x_proj = nn.Linear(self.d_inner, 16 + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(16, self.d_inner)

            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner,
                kernel_size=d_conv, padding=d_conv - 1,
                groups=self.d_inner, bias=True,
            )

            # SSM parameters A, D
            A = torch.arange(
                1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            B, T, D = x.shape
            xz = self.in_proj(x)                          # (B, T, 2*d_inner)
            # each (B, T, d_inner)
            u, z = xz.chunk(2, dim=-1)

            # conv over sequence
            u_conv = self.conv1d(u.transpose(1, 2))[..., :T].transpose(1, 2)
            u_act = F.silu(u_conv)                       # (B, T, d_inner)

            # selective parameters
            # (B, T, dt_rank+2*d_state)
            x_dbl = self.x_proj(u_act)
            dt, B_ssm, C_ssm = x_dbl.split(
                [16, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))             # (B, T, d_inner)

            # discretize A
            A = -torch.exp(self.A_log)                    # (d_inner, d_state)
            # (B, T, d_inner, d_state)
            dA = torch.exp(dt.unsqueeze(-1) * A)
            # (B, T, d_inner, d_state)
            dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)

            # sequential scan
            h = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=x.dtype)
            ys = []
            for t in range(T):
                h = dA[:, t] * h + dB[:, t] * u_act[:, t].unsqueeze(-1)
                y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)
                ys.append(y)
            y = torch.stack(ys, dim=1)                    # (B, T, d_inner)
            y = y + u_act * self.D

            # gate and project
            out = y * F.silu(z)
            return self.out_proj(out)                      # (B, T, d_model)

    class MinimalMambaBackbone(nn.Module):
        def __init__(self, d_model, vocab_size, n_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([S6(d_model) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = x + layer(self.norm(x))
            return x

    return MinimalMambaBackbone(
        d_model=cfg.mamba_d_model,
        vocab_size=cfg.vocab_size,
        n_layers=cfg.mamba_n_layers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop (reused for both xLSTM and Mamba)
# ─────────────────────────────────────────────────────────────────────────────

def train_sora_recurrent(model, cfg, train_loader, val_loader,
                         device, arch_name, save_dir):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_mcc = -1.0
    avg_rank = 0.0
    eff_ranks = {}
    t0 = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            apply_proximal_updates(model, cfg.learning_rate)

        # ── evaluate ─────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        mcc = compute_mcc(all_preds, all_labels)

        eff_ranks_strict = get_effective_ranks(model, eps=1e-6)
        eff_ranks_effective = get_effective_ranks(model, eps=1e-3)
        avg_rank_strict = float(
            np.mean(list(eff_ranks_strict.values()))) if eff_ranks_strict else 0.0
        avg_rank_effective = float(
            np.mean(list(eff_ranks_effective.values()))) if eff_ranks_effective else 0.0
        eff_ranks = eff_ranks_effective
        avg_rank = avg_rank_effective

        print(f"  [{arch_name}] Epoch {epoch} | MCC={mcc:.4f} | "
              f"Exact rank(1e-6)={avg_rank_strict:.1f} | "
              f"Eff rank(1e-3)={avg_rank_effective:.1f}")

        if mcc > best_mcc:
            best_mcc = mcc

    elapsed = time.time() - t0
    return best_mcc, avg_rank, eff_ranks, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def train_sora_xlstm(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: SoRA-xLSTM")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    cfg.vocab_size = tokenizer.vocab_size
    train_ds, val_ds = load_cola(cfg, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collator)
    val_loader = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    backbone, d_model, target_modules = build_xlstm_backbone(cfg)
    model = SequenceClassifier(
        backbone, d_model, num_labels=cfg.num_labels).to(device)

    replaced = inject_sora(
        model, target_modules,
        r=cfg.sora_r, alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout, lora_lambda=cfg.sora_lambda,
    )
    model = model.to(device)
    print(
        f"  SoRA: replaced {replaced} linear layers (targets: {target_modules})")
    freeze_base(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_trainable:,}")

    best_mcc, avg_rank, eff_ranks, elapsed = train_sora_recurrent(
        model, cfg, train_loader, val_loader, device, "xLSTM", save_dir
    )

    metrics = {
        "method":                    "sora_xlstm",
        "mcc":                       best_mcc,
        "trainable_params":          n_trainable,
        "train_time_sec":            elapsed,
        "sora_r":                    cfg.sora_r,
        "effective_rank":            avg_rank,
        "effective_ranks_per_layer": eff_ranks,
        "target_modules":            target_modules,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/metrics_sora_xlstm.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  SoRA-xLSTM | MCC={best_mcc:.4f} | Params={n_trainable:,} | "
          f"Avg rank={avg_rank:.1f} | Time={elapsed:.1f}s")
    return metrics


def train_xlstm_baseline(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: xLSTM Baseline")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    cfg.vocab_size = tokenizer.vocab_size
    train_ds, val_ds = load_cola(cfg, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    backbone, d_model, _ = build_xlstm_backbone(cfg)

    model = SequenceClassifier(
        backbone,
        d_model,
        num_labels=cfg.num_labels,
    ).to(device)

    # Train EVERYTHING
    for p in model.parameters():
        p.requires_grad = True

    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"  Trainable parameters: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_mcc = -1.0
    t0 = time.time()

    for epoch in range(1, cfg.num_epochs + 1):

        model.train()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        mcc = compute_mcc(all_preds, all_labels)

        print(
            f"  [xLSTM Baseline] Epoch {epoch} | MCC={mcc:.4f}"
        )

        if mcc > best_mcc:
            best_mcc = mcc

    elapsed = time.time() - t0

    metrics = {
        "method": "xlstm_baseline",
        "mcc": best_mcc,
        "trainable_params": n_trainable,
        "train_time_sec": elapsed,
    }

    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/metrics_xlstm_baseline.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"\n  xLSTM Baseline | MCC={best_mcc:.4f} | "
        f"Params={n_trainable:,} | Time={elapsed:.1f}s"
    )

    return metrics


def train_sora_mamba(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: SoRA-Mamba")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    cfg.vocab_size = tokenizer.vocab_size
    train_ds, val_ds = load_cola(cfg, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collator)
    val_loader = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    backbone, d_model, target_modules = build_mamba_backbone(cfg)
    model = SequenceClassifier(
        backbone, d_model, num_labels=cfg.num_labels).to(device)

    replaced = inject_sora(
        model, target_modules,
        r=cfg.sora_r, alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout, lora_lambda=cfg.sora_lambda,
    )

    model = model.to(device)

    print(
        f"  SoRA: replaced {replaced} linear layers (targets: {target_modules})")
    freeze_base(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_trainable:,}")

    best_mcc, avg_rank, eff_ranks, elapsed = train_sora_recurrent(
        model, cfg, train_loader, val_loader, device, "Mamba", save_dir
    )

    metrics = {
        "method":                    "sora_mamba",
        "mcc":                       best_mcc,
        "trainable_params":          n_trainable,
        "train_time_sec":            elapsed,
        "sora_r":                    cfg.sora_r,
        "effective_rank":            avg_rank,
        "effective_ranks_per_layer": eff_ranks,
        "target_modules":            target_modules,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/metrics_sora_mamba.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  SoRA-Mamba | MCC={best_mcc:.4f} | Params={n_trainable:,} | "
          f"Avg rank={avg_rank:.1f} | Time={elapsed:.1f}s")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table + plot (mirrors Part 1 style)
# ─────────────────────────────────────────────────────────────────────────────

def print_part3_table(metrics_list):
    print("\n" + "=" * 90)
    print(f"{'Method':<16} {'MCC':>8} {'Params':>12} {'Eff.Rank':>10} {'Time(s)':>10}")
    print("-" * 90)
    for m in metrics_list:
        print(
            f"{m['method']:<16} "
            f"{m['mcc']:>8.4f} "
            f"{m['trainable_params']:>12,} "
            f"{m['effective_rank']:>10.1f} "
            f"{m['train_time_sec']:>10.1f}"
        )
    print("=" * 90)


def plot_part3(metrics_list, save_dir):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    methods = [m["method"] for m in metrics_list]
    mccs = [m["mcc"] for m in metrics_list]
    params = [m["trainable_params"] for m in metrics_list]
    ranks = [m["effective_rank"] for m in metrics_list]
    times = [m["train_time_sec"] for m in metrics_list]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(methods)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("SoRA on Transformer vs xLSTM vs Mamba — CoLA", fontsize=12)

    for ax, vals, title, ylabel in zip(
        axes,
        [mccs, [p/1e6 for p in params], ranks, times],
        ["MCC (↑ better)", "Trainable Params (↓ better)",
         "Effective Rank after Training", "Training Time (↓ better)"],
        ["MCC", "Parameters (M)", "Rank", "Seconds"],
    ):
        ax.bar(methods, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = f"{save_dir}/comparison_part3.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# run_part3  — called from main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_part3(cfg):
    save_dir = "experiments/part3"
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []

    # Load Part 1 SoRA result for comparison baseline
    try:
        with open("experiments/part1/metrics_sora.json") as f:
            m_sora_transformer = json.load(f)
            m_sora_transformer["method"] = "sora_transformer"
        all_metrics.append(m_sora_transformer)
        print("  Loaded Part 1 SoRA metrics for comparison")
    except FileNotFoundError:
        print("  Part 1 SoRA metrics not found — skipping baseline comparison")

    RUN_XLSTM_BASELINE = False

    RUN_XLSTM = True

    RUN_MAMBA = False

    if RUN_XLSTM_BASELINE:
        m_xlstm_base = train_xlstm_baseline(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_xlstm_baseline.json") as f:
            m_xlstm_base = json.load(f)
    all_metrics.append(m_xlstm_base)

    if RUN_XLSTM:
        m_xlstm = train_sora_xlstm(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_sora_xlstm.json") as f:
            m_xlstm = json.load(f)
    all_metrics.append(m_xlstm)

    if RUN_MAMBA:
        m_mamba = train_sora_mamba(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_sora_mamba.json") as f:
            m_mamba = json.load(f)
    all_metrics.append(m_mamba)

    print_part3_table(all_metrics)
    plot_part3(all_metrics, save_dir)

    with open(f"{save_dir}/all_metrics_part3.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\n  All Part 3 metrics saved → {save_dir}/all_metrics_part3.json")
    return all_metrics
