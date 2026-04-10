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

    # ── Tokenizer + Dataset ────────────────────────────────────────────────
    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = TransformerLM(cfg).to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Training ───────────────────────────────────────────────────────────
    print("\nStarting training...\n")

    history = train(
        model,
        cfg,
        train_loader,
        val_loader,
        optimizer,
        device
    )

    os.makedirs("experiments/baseline", exist_ok=True)

    steps = history["step"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Train vs Val loss
    axes[0].plot(steps, history["train_loss"], label="Train Loss")
    axes[0].plot(steps, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    # 2. Perplexity
    axes[1].plot(steps, history["perplexity"])
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Validation Perplexity")

    # 3. Learning rate
    axes[2].plot(steps, history["lr"])
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")

    plt.tight_layout()

    # 🔥 THIS is what you were missing
    plt.savefig("experiments/baseline/baseline_training_curves.png")

    plt.show()

    print("Saved plot → experiments/baseline/baseline_training_curves.png")
    # ── Final evaluation ─────────────────────────────
    final_metrics = evaluate(model, val_loader, device,
                             max_iters=len(val_loader))

    final_metrics["model"] = "baseline"
    final_metrics["context_length"] = cfg.context_length
    final_metrics["params"] = model.count_parameters()

    # ── print table ──
    print_results_table([final_metrics])

    # ── save table ──
    with open("experiments/baseline/results_table.txt", "w") as f:
        f.write(str(final_metrics))

    # ── save metrics ──
    os.makedirs("experiments/baseline", exist_ok=True)

    with open("experiments/baseline/baseline_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    with open("experiments/baseline/sample_generation.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(generated)

    torch.save(model.state_dict(),
               "experiments/baseline/baseline_transformer.pt")

    # ── Generation (qualitative check) ─────────────────────────────────────
    prompt = "The history of artificial intelligence began"

    generated = generate(
        model,
        enc,
        cfg,
        device,
        prompt,
        max_new_tokens=80
    )

    print("\n" + "=" * 50)
    print(f"Prompt   : {prompt}")
    print(f"Generated:\n{generated}")
    print("=" * 50)


# ── IMPORTANT (fixes Mac multiprocessing crash) ─────────────────────────────
if __name__ == "__main__":
    main()
