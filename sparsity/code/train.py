"""
train.py
────────
Training loop for LoRA, AdaLoRA, and SoRA on CoLA (GLUE).
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


def load_cola(cfg, tokenizer):
    dataset = load_dataset("nyu-mll/glue", "cola")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=cfg.max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=[
                       "input_ids", "attention_mask", "labels"])
    return dataset["train"], dataset["validation"]


def compute_mcc(preds, labels):
    return matthews_corrcoef(labels, preds)


def _make_training_args(cfg, save_dir):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=100,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )


def _make_trainer(model, args, train_ds, val_ds, tokenizer, collator, compute_metrics):
    from transformers import Trainer
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )


def train_lora(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: LoRA")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_cola(cfg, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query_proj", "key_proj", "value_proj"],
        bias="none",
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
    args = _make_training_args(cfg, save_dir)
    trainer = _make_trainer(model, args, train_ds, val_ds,
                            tokenizer, collator, compute_metrics)

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
        "effective_rank":   cfg.lora_r,
    }

    with open(f"{save_dir}/metrics_lora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"\n  LoRA | MCC={mcc:.4f} | Params={n_trainable:,} | Time={elapsed:.1f}s")
    return metrics


def train_adalora(cfg, save_dir):
    print("\n" + "=" * 50)
    print("  Training: AdaLoRA")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds, val_ds = load_cola(cfg, tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=cfg.num_labels
    )

    adalora_cfg = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=cfg.adalora_init_r,
        target_r=cfg.adalora_target_r,
        tinit=cfg.adalora_tinit,
        tfinal=cfg.adalora_tfinal,
        deltaT=cfg.adalora_delta_t,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query_proj", "key_proj", "value_proj"],
        bias="none",
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
    args = _make_training_args(cfg, save_dir)
    trainer = _make_trainer(model, args, train_ds, val_ds,
                            tokenizer, collator, compute_metrics)

    trainer.train()
    elapsed = time.time() - t0

    eval_results = trainer.evaluate()
    mcc = eval_results.get("eval_mcc", 0.0)

    metrics = {
        "method":           "adalora",
        "mcc":              mcc,
        "trainable_params": n_trainable,
        "train_time_sec":   elapsed,
        "init_r":           cfg.adalora_init_r,
        "target_r":         cfg.adalora_target_r,
        "effective_rank":   float(cfg.adalora_target_r),
    }

    with open(f"{save_dir}/metrics_adalora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"\n  AdaLoRA | MCC={mcc:.4f} | Params={n_trainable:,} | Time={elapsed:.1f}s")
    return metrics


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

    # Separate gate parameters from other trainable parameters
    gate_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "gate" in name:
            gate_params.append(p)
        else:
            other_params.append(p)

    # Assign specific learning rates to each group
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay},
        {"params": gate_params,  "lr": cfg.sora_lr_gate,
            "weight_decay": 0.0},  # Gates rarely use weight decay
    ])

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_mcc = -1.0
    eff_ranks = {}
    avg_rank = 0.0

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
                cfg.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.apply_proximal_updates(cfg.learning_rate)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        mcc = compute_mcc(all_preds, all_labels)
        # Both strict (1e-6) and effective (1e-3) rank
        eff_ranks_strict = model.effective_ranks(eps=1e-6)
        eff_ranks_effective = model.effective_ranks(eps=1e-3)
        avg_rank_strict = float(
            np.mean(list(eff_ranks_strict.values()))) if eff_ranks_strict else 0.0
        avg_rank_effective = float(
            np.mean(list(eff_ranks_effective.values()))) if eff_ranks_effective else 0.0
        eff_ranks = eff_ranks_effective  # use for final metrics
        avg_rank = avg_rank_effective

        all_gates = []

        for module in model.model.modules():
            if hasattr(module, "gate"):
                all_gates.append(module.gate.detach().cpu())

        if len(all_gates) > 0:
            all_gates = torch.cat(all_gates)

            num_zero = (all_gates.abs() < 1e-4).sum().item()

            print(
                f" | zero={num_zero}/{all_gates.numel()}"
                f" min={all_gates.min():.4f}"
                f" mean={all_gates.mean():.4f}"
                f" max={all_gates.max():.4f}"
            )

        print(f"  Epoch {epoch} | MCC={mcc:.4f} | "
              f"Exact rank(1e-6)={avg_rank_strict:.1f} | "
              f"Effective rank(1e-3)={avg_rank_effective:.1f}")

        if mcc > best_mcc:
            best_mcc = mcc

    elapsed = time.time() - t0

    metrics = {
        "method":                    "sora",
        "mcc":                       best_mcc,
        "trainable_params":          n_trainable,
        "train_time_sec":            elapsed,
        "sora_r":                    cfg.sora_r,
        "effective_rank":            avg_rank,
        "effective_ranks_per_layer": eff_ranks,
    }

    with open(f"{save_dir}/metrics_sora.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  SoRA | MCC={best_mcc:.4f} | Params={n_trainable:,} | "
          f"Avg rank={avg_rank:.1f} | Time={elapsed:.1f}s")

    return metrics
