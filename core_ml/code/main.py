import torch

# ── Imports ────────────────────────────────────────────────────────────────
from utils.config import TransformerConfig

from model import TransformerLM

from training.train import train
from training.optimizer import build_optimizer
from training.generate import generate

from data.tokenizer import load_tokenizer, tokenize_wikitext2
from data.dataloader import build_loaders

from training.evaluate import evaluate
import matplotlib.pyplot as plt
import os
import json


def print_results_table(metrics_list):
    print("\n" + "="*90)
    print(f"{'Model':<25} {'Context':>8} {'Params':>12} {'Val Loss':>10} {'PPL':>10} {'Throughput':>12} {'Peak Mem':>10}")
    print("-"*90)

    for m in metrics_list:
        print(f"{m['model']:<25} "
              f"{m['context_length']:>8} "
              f"{m['params']:>12,} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>10.2f} "
              f"{m['throughput']:>12.0f} "
              f"{m['peak_mem_mb']:>10.1f} MB")

    print("="*90)


# ── Main runner ─────────────────────────────────────────────────────────────
def main():

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config
    cfg = TransformerConfig()
    print(cfg)

    # ── Tokenizer + Dataset ────────────────────────────────
    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)

    # ── DataLoaders ────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────
    model = TransformerLM(cfg).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer ──────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Training ───────────────────────────────────────────
    print("\nStarting training...\n")

    history = train(
        model,
        cfg,
        train_loader,
        val_loader,
        optimizer,
        device
    )

    # ── Create folder early ─────────────────────────────────
    os.makedirs("experiments/baseline", exist_ok=True)

    # ── Plotting ────────────────────────────────────────────
    steps = history["step"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, history["train_loss"], label="Train Loss")
    axes[0].plot(steps, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(steps, history["perplexity"])
    axes[1].set_title("Perplexity")

    axes[2].plot(steps, history["lr"])
    axes[2].set_title("LR")

    plt.tight_layout()
    plt.savefig("experiments/baseline/baseline_training_curves.png")
    plt.close()

    print("Saved plot")

    # ── Evaluation ─────────────────────────────────────────
    final_metrics = evaluate(model, val_loader, device,
                             max_iters=len(val_loader))

    # Ensure key exists (fix crash)
    if "peak_mem_mb" not in final_metrics:
        final_metrics["peak_mem_mb"] = 0.0

    # Add metadata
    final_metrics["model"] = "baseline"
    final_metrics["context_length"] = cfg.context_length
    final_metrics["params"] = model.count_parameters()

    # ── SAVE FIRST (critical) ──────────────────────────────
    with open("experiments/baseline/baseline_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    with open("experiments/baseline/results_table.txt", "w") as f:
        f.write(str(final_metrics))

    # ── Generation (move BEFORE saving file) ───────────────
    prompt = "The history of artificial intelligence began"

    generated = generate(
        model,
        enc,
        cfg,
        device,
        prompt,
        max_new_tokens=80
    )

    with open("experiments/baseline/sample_generation.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(generated)

    # ── Save model ─────────────────────────────────────────
    torch.save(
        model.state_dict(),
        "experiments/baseline/baseline_transformer.pt"
    )

    # ── Print table LAST (safe now) ─────────────────────────
    print_results_table([final_metrics])

    # ── Print sample ───────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Prompt   : {prompt}")
    print(f"Generated:\n{generated}")
    print("=" * 50)


# ── IMPORTANT (fixes Mac multiprocessing crash) ─────────────────────────────
if __name__ == "__main__":
    main()
