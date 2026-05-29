"""
main.py
───────
Entry point for Sparsity & Optimization experiments.

Parts:
    Part 1 — Compare LoRA, AdaLoRA, SoRA on CoLA
    Part 2 — SGD vs proximal update analysis (numpy + PyTorch)
    Part 3 — SoRA on xLSTM and Mamba (TBD)

Toggle which part to run at the bottom.
"""

from train import train_lora, train_adalora, train_sora
from config import SparsityConfig
import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(metrics_list):
    print("\n" + "=" * 90)
    print(f"{'Method':<12} {'MCC':>8} {'Params':>12} {'Eff.Rank':>10} {'Time(s)':>10}")
    print("-" * 90)
    for m in metrics_list:
        print(
            f"{m['method']:<12} "
            f"{m['mcc']:>8.4f} "
            f"{m['trainable_params']:>12,} "
            f"{m['effective_rank']:>10.1f} "
            f"{m['train_time_sec']:>10.1f}"
        )
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(metrics_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    methods = [m["method"] for m in metrics_list]
    mccs = [m["mcc"] for m in metrics_list]
    params = [m["trainable_params"] for m in metrics_list]
    ranks = [m["effective_rank"] for m in metrics_list]
    times = [m["train_time_sec"] for m in metrics_list]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "LoRA vs AdaLoRA vs SoRA on CoLA (DeBERTa-v3-base)", fontsize=12)

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # MCC
    axes[0].bar(methods, mccs, color=colors)
    axes[0].set_title("MCC (↑ better)")
    axes[0].set_ylabel("Matthews Correlation Coefficient")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(mccs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Trainable params
    axes[1].bar(methods, [p/1e6 for p in params], color=colors)
    axes[1].set_title("Trainable Parameters (↓ better)")
    axes[1].set_ylabel("Parameters (M)")
    for i, v in enumerate(params):
        axes[1].text(i, v/1e6 + 0.01,
                     f"{v/1e6:.2f}M", ha="center", fontsize=10)

    # Effective rank
    axes[2].bar(methods, ranks, color=colors)
    axes[2].set_title("Effective Rank after Training")
    axes[2].set_ylabel("Rank")
    for i, v in enumerate(ranks):
        axes[2].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)

    # Training time
    axes[3].bar(methods, times, color=colors)
    axes[3].set_title("Training Time (↓ better)")
    axes[3].set_ylabel("Seconds")
    for i, v in enumerate(times):
        axes[3].text(i, v + 1, f"{v:.0f}s", ha="center", fontsize=10)

    plt.tight_layout()
    out = f"{save_dir}/comparison_part1.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n  Saved comparison plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — LoRA vs AdaLoRA vs SoRA
# ─────────────────────────────────────────────────────────────────────────────

def run_part1():
    cfg = SparsityConfig()
    save_dir = "experiments/part1"
    os.makedirs(save_dir, exist_ok=True)

    all_metrics = []

    # ── Toggle which methods to run ──────────────────────────────────────
    RUN_LORA = True   # already done
    RUN_ADALORA = True   # already done
    RUN_SORA = True

    if RUN_LORA:
        m_lora = train_lora(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_lora.json") as f:
            m_lora = json.load(f)
    all_metrics.append(m_lora)

    if RUN_ADALORA:
        m_adalora = train_adalora(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_adalora.json") as f:
            m_adalora = json.load(f)
    all_metrics.append(m_adalora)

    if RUN_SORA:
        m_sora = train_sora(cfg, save_dir)
    else:
        with open(f"{save_dir}/metrics_sora.json") as f:
            m_sora = json.load(f)
    all_metrics.append(m_sora)

    print_comparison_table(all_metrics)
    plot_comparison(all_metrics, save_dir)

    with open(f"{save_dir}/all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\n  All Part 1 metrics saved → {save_dir}/all_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    RUN_PART1 = True
    RUN_PART2 = False
    RUN_PART3 = False

    if RUN_PART1:
        run_part1()
    elif RUN_PART2:
        pass  # added next
    elif RUN_PART3:
        pass  # added after part 2
