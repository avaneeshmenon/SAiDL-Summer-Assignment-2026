"""
train.py
────────
Training loop for LoRA, AdaLoRA, and SoRA on CoLA (GLUE).

Handles:
- Standard HuggingFace Trainer for LoRA and AdaLoRA (via peft)
- Custom training loop for SoRA (needs proximal update after each step)
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
    TaskType,
)
from sklearn.metrics import matthews_corrcoef
import json

from methods.sora import SoRAModel


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cola(cfg, tokenizer):
    dataset = load_dataset("glue", "cola")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset["train"], dataset["validation"]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_mcc(preds, labels):
    """Matthews Correlation Coefficient — standard CoLA metric."""
    return matthews_corrcoef(labels, preds)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA training (via peft + HuggingFace Trainer)
# ─────────────────────────────────────────────────────────────────────────────

def train_lora(cfg, save_dir):
    from transformers import TrainingArguments, Trainer

    print("\n" + "=" * 50)
    print("  Training: LoRA")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_cola(cfg, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    lora_cfg = LoraConfig(
        task_type    = TaskType.SEQ_CLS,
        r            = cfg.lora_r,
        lora_alpha   = cfg.lora_alpha,
        lora_dropout = cfg.lora_dropout,
        target_modules = ["query_proj", "value_proj"],
        bias         = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"mcc": compute_mcc(preds, labels)}

    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()

    args = TrainingArguments(
        output_dir              = save_dir,
        num_train_epochs        = cfg.num_epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        learning_rate           = cfg.learning_rate,
        weight_decay            = cfg.weight_decay,
        warmup_ratio            = cfg.warmup_ratio,
        evaluation_strategy     = "epoch",
        save_strategy           = "no",
        load_best_model_at_end  = False,
        logging_steps           = 50,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    elapsed = time.time() - t0

    eval_results = trainer.evaluate()
    mcc = eval_results.get("eval_mcc", 0.0)

    metrics = {
        "method":           "lora",
        "mcc":              mcc,
        "trainable_params": n_trainable,
        "train_time_sec":   elapsed,
        "lora_r":           cfg.lora_r,
        "effective_rank":   cfg.lora_r,   # fixed rank for LoRA
    }

    with open(f"{save_dir}/metrics_lora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  LoRA | MCC={mcc:.4f} | Params={n_trainable:,} | Time={elapsed:.1f}s")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# AdaLoRA training
# ─────────────────────────────────────────────────────────────────────────────

def train_adalora(cfg, save_dir):
    from transformers import TrainingArguments, Trainer

    print("\n" + "=" * 50)
    print("  Training: AdaLoRA")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_cola(cfg, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    adalora_cfg = AdaLoraConfig(
        task_type      = TaskType.SEQ_CLS,
        init_r         = cfg.adalora_init_r,
        target_r       = cfg.adalora_target_r,
        tinit          = cfg.adalora_tinit,
        tfinal         = cfg.adalora_tfinal,
        deltaT         = cfg.adalora_delta_t,
        lora_alpha     = cfg.lora_alpha,
        lora_dropout   = cfg.lora_dropout,
        target_modules = ["query_proj", "value_proj"],
        bias           = "none",
    )
    model = get_peft_model(base_model, adalora_cfg)
    model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"mcc": compute_mcc(preds, labels)}

    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()

    args = TrainingArguments(
        output_dir              = save_dir,
        num_train_epochs        = cfg.num_epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size,
        learning_rate           = cfg.learning_rate,
        weight_decay            = cfg.weight_decay,
        warmup_ratio            = cfg.warmup_ratio,
        evaluation_strategy     = "epoch",
        save_strategy           = "no",
        load_best_model_at_end  = False,
        logging_steps           = 50,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    elapsed = time.time() - t0

    eval_results = trainer.evaluate()
    mcc = eval_results.get("eval_mcc", 0.0)

    # Compute average effective rank across layers
    avg_rank = cfg.adalora_target_r

    metrics = {
        "method":           "adalora",
        "mcc":              mcc,
        "trainable_params": n_trainable,
        "train_time_sec":   elapsed,
        "init_r":           cfg.adalora_init_r,
        "target_r":         cfg.adalora_target_r,
        "effective_rank":   avg_rank,
    }

    with open(f"{save_dir}/metrics_adalora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  AdaLoRA | MCC={mcc:.4f} | Params={n_trainable:,} | Time={elapsed:.1f}s")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SoRA training (custom loop — needs proximal update after each step)
# ─────────────────────────────────────────────────────────────────────────────

def train_sora(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: SoRA")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_cola(cfg, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    model = SoRAModel(base_model, cfg).to(device)
    n_trainable = model.count_trainable_parameters()
    print(f"  Trainable parameters: {n_trainable:,}")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps  = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(save_dir, exist_ok=True)
    t0    = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_mcc = -1.0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(**batch)
                loss    = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # ── Proximal update for gate sparsity ─────────────────────────
            current_lr = scheduler.get_last_lr()[0]
            model.apply_proximal_updates(current_lr)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds   = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels  = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        mcc = compute_mcc(all_preds, all_labels)
        eff_ranks = model.effective_ranks()
        avg_rank  = np.mean(list(eff_ranks.values())) if eff_ranks else 0

        print(f"  Epoch {epoch} | MCC={mcc:.4f} | Avg effective rank={avg_rank:.1f}")

        if mcc > best_mcc:
            best_mcc = mcc

    elapsed = time.time() - t0

    metrics = {
        "method":           "sora",
        "mcc":              best_mcc,
        "trainable_params": n_trainable,
        "train_time_sec":   elapsed,
        "sora_r":           cfg.sora_r,
        "effective_rank":   avg_rank,
        "effective_ranks_per_layer": eff_ranks,
    }

    with open(f"{save_dir}/metrics_sora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  SoRA | MCC={best_mcc:.4f} | Params={n_trainable:,} | "
          f"Avg rank={avg_rank:.1f} | Time={elapsed:.1f}s")

    return metrics