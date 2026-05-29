import torch
import os
import json
import matplotlib.pyplot as plt

from utils.config import TransformerConfig
from model import TransformerLM
from training.train import train
from training.optimizer import build_optimizer
from training.generate import generate
from data.tokenizer import load_tokenizer, tokenize_wikitext2
from data.dataloader import build_loaders
from training.evaluate import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Table printers
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(metrics_list):
    print("\n" + "=" * 110)
    print(f"{'Model':<20} {'Ctx':>6} {'Params':>10} {'ValLoss':>10} {'PPL':>8} {'Thrpt':>10} {'Mem(MB)':>10} {'Epoch(s)':>10}")
    print("-" * 110)
    for m in metrics_list:
        print(f"{m['model']:<20} "
              f"{m['context_length']:>6} "
              f"{m['params']:>10,} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>8.2f} "
              f"{m['throughput']:>10.0f} "
              f"{m['peak_mem_mb']:>10.1f} "
              f"{m['epoch_time_avg']:>10.1f}")
    print("=" * 110)


def print_positional_table(metrics_list):
    print("\n" + "=" * 120)
    print(f"{'PosEnc':<12} {'TrainCtx':>8} {'ValLoss':>10} {'PPL':>8} {'Thrpt':>10} {'Mem(MB)':>10} {'Epoch(s)':>10}")
    print("-" * 120)
    for m in metrics_list:
        print(f"{m['pos_type']:<12} "
              f"{m['train_ctx']:>8} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>8.2f} "
              f"{m['throughput']:>10.0f} "
              f"{m['peak_mem_mb']:>10.1f} "
              f"{m['epoch_time_avg']:>10.1f}")
    print("=" * 120)


def print_hybrid_table(metrics_list):
    print("\n" + "=" * 130)
    print(f"{'ConvType':<22} {'Attn':<12} {'Pos':<10} {'Ctx':>6} "
          f"{'Params':>10} {'ValLoss':>10} {'PPL':>8} "
          f"{'Thrpt':>10} {'Mem(MB)':>10} {'Epoch(s)':>10}")
    print("-" * 130)
    for m in metrics_list:
        print(f"{m['conv_type']:<22} "
              f"{m['attention_type']:<12} "
              f"{m['pos_type']:<10} "
              f"{m['context_length']:>6} "
              f"{m['params']:>10,} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>8.2f} "
              f"{m['throughput']:>10.0f} "
              f"{m['peak_mem_mb']:>10.1f} "
              f"{m['epoch_time_avg']:>10.1f}")
    print("=" * 130)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: saves all outputs for one run
# ─────────────────────────────────────────────────────────────────────────────

def save_run(model, enc, cfg, history, final_metrics, save_dir, run_tag, device):
    """
    Saves training curves, metrics JSON, generation sample, and model weights.
    run_tag  : short string used in filenames, e.g. "gqa_512" or "conv_before_attn"
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Training curves ───────────────────────────────────────────────────
    steps = history["step"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, history["train_loss"], label="Train")
    axes[0].plot(steps, history["val_loss"],   label="Val")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")

    axes[1].plot(steps, history["perplexity"])
    axes[1].set_title("PPL")
    axes[1].set_xlabel("Step")

    axes[2].plot(steps, history["lr"])
    axes[2].set_title("LR")
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves_{run_tag}.png", dpi=120)
    plt.close()

    # ── Metrics JSON ──────────────────────────────────────────────────────
    with open(f"{save_dir}/metrics_{run_tag}.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # ── Generation sample ─────────────────────────────────────────────────
    prompt = "The history of artificial intelligence began"
    generated = generate(model, enc, cfg, device, prompt,
                         max_new_tokens=80, temperature=0.7, top_k=40)
    with open(f"{save_dir}/generation_{run_tag}.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\n{generated}")

    # ── Model weights ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), f"{save_dir}/model_{run_tag}.pt")

    print(f"  ✅ Saved → {save_dir}/")
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 & 2 — Attention experiments  (one run at a time)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_attention():
    """
    Run one attention-type × context-length experiment.
    Change ATTN_TYPE and CTX below before each run.
    Saves to: experiments/attention/<attn>_ctx<ctx>/
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── CHANGE THESE EACH RUN ────────────────────────────────────────────
    ATTN_TYPE = "standard"   # standard | sliding_window | sparse_block |
    # linear   | gqa | mqa | softmax_free
    CTX = 512           # 512 | 1024 | 2048
    # ─────────────────────────────────────────────────────────────────────

    cfg = TransformerConfig()
    cfg.attention_type = ATTN_TYPE
    cfg.pos_encoding_type = "learned"
    cfg.context_length = CTX
    cfg.rope_scale = 1.0

    print(f"\nRunning attention: {ATTN_TYPE} | ctx={CTX}\n")

    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    model = TransformerLM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    history = train(model, cfg, train_loader, val_loader, optimizer, device)

    final = evaluate(model, val_loader, device, max_iters=len(val_loader))
    final["model"] = ATTN_TYPE
    final["context_length"] = CTX
    final["params"] = model.count_parameters()
    final["epoch_time_avg"] = sum(
        history["epoch_time"]) / len(history["epoch_time"])
    final["peak_mem_mb"] = max(
        history["peak_mem_mb"]) if history["peak_mem_mb"] else 0.0

    save_dir = f"experiments/attention/{ATTN_TYPE}_ctx{CTX}"
    run_tag = f"{ATTN_TYPE}_{CTX}"
    generated = save_run(model, enc, cfg, history, final,
                         save_dir, run_tag, device)

    print_results_table([final])
    print("\nSample:\n", generated)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Positional encoding experiments  (one run at a time)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_positional():
    """
    Run one positional-encoding experiment.
    Change POS_TYPE below before each run.
    Saves to: experiments/positional/<pos>_ctx512/
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── CHANGE THIS EACH RUN ─────────────────────────────────────────────
    POS_TYPE = "sinusoidal"   # learned | sinusoidal | rope | rope_interp |
    # alibi   | relative
    # ─────────────────────────────────────────────────────────────────────

    cfg = TransformerConfig()
    cfg.context_length = 512
    cfg.rope_scale = 1.0
    cfg.conv_type = "none"

    # Wire attention_type to match positional encoding
    if POS_TYPE in ["rope", "alibi", "relative"]:
        cfg.attention_type = POS_TYPE
        cfg.pos_encoding_type = POS_TYPE
    elif POS_TYPE == "rope_interp":
        cfg.attention_type = "rope"
        cfg.pos_encoding_type = "rope"
    else:
        cfg.attention_type = "standard"
        cfg.pos_encoding_type = POS_TYPE

    print(f"\nRunning positional: {POS_TYPE} | ctx=512\n")

    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    model = TransformerLM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    history = train(model, cfg, train_loader, val_loader, optimizer, device)

    final = evaluate(model, val_loader, device, max_iters=len(val_loader))
    final["model"] = POS_TYPE
    final["pos_type"] = POS_TYPE
    final["train_ctx"] = 512
    final["context_length"] = 512
    final["params"] = model.count_parameters()
    final["epoch_time_avg"] = sum(
        history["epoch_time"]) / len(history["epoch_time"])
    final["peak_mem_mb"] = max(
        history["peak_mem_mb"]) if history["peak_mem_mb"] else 0.0

    save_dir = f"experiments/positional/{POS_TYPE}_ctx512"
    run_tag = f"{POS_TYPE}_512"
    generated = save_run(model, enc, cfg, history, final,
                         save_dir, run_tag, device)

    print_positional_table([final])
    print("\nSample:\n", generated)


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 — Convolution + Attention Hybrid experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_hybrid_experiments():
    """
    Trains all four conv-hybrid designs plus a pure-Transformer baseline.
    Uses the best attention (GQA) and positional encoding (RoPE) from Parts 2/3.
    All five runs happen in one call — budget ~5x a single run on Colab.
    Saves to: experiments/hybrid/<conv_type>_<attn>_<pos>_ctx<ctx>/
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Set to your Part 2 / Part 3 winners ──────────────────────────────
    BEST_ATTN = "gqa"
    BEST_POS = "relative"   # ← change if different
    CTX = 512
    KERNEL = 3
    # ─────────────────────────────────────────────────────────────────────

    CONV_TYPES = [
        "none",              # baseline: pure Transformer, no conv
        "conv_before_attn",  # Design 1: DWSConv before every attn sublayer
        "interleaved",       # Design 2: even layers = ConvBlock, odd = AttnBlock
        "depthwise_subset",  # Design 3: first half of layers replace attn with DWSConv
        "gated_conv_ff",     # Design 4: replace FFN with Gated Conv FFN
    ]

    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)

    all_metrics = []

    for conv_type in CONV_TYPES:
        print(f"\n{'=' * 60}")
        print(f"  Hybrid run: conv_type={conv_type!r}")
        print(f"  attn={BEST_ATTN}  pos={BEST_POS}  ctx={CTX}")
        print(f"{'=' * 60}\n")

        cfg = TransformerConfig()
        cfg.context_length = CTX
        cfg.conv_type = conv_type
        cfg.conv_kernel_size = KERNEL
        cfg.rope_scale = 1.0

        cfg.attention_type = BEST_ATTN
        cfg.pos_encoding_type = BEST_POS

        train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

        model = TransformerLM(cfg).to(device)
        optimizer = build_optimizer(model, cfg)

        print(f"  Parameters: {model.count_parameters():,}")

        history = train(model, cfg, train_loader,
                        val_loader, optimizer, device)

        final = evaluate(model, val_loader, device, max_iters=len(val_loader))
        final["model"] = conv_type
        final["conv_type"] = conv_type
        final["attention_type"] = cfg.attention_type
        final["pos_type"] = cfg.pos_encoding_type
        final["context_length"] = CTX
        final["params"] = model.count_parameters()
        final["epoch_time_avg"] = sum(
            history["epoch_time"]) / max(len(history["epoch_time"]), 1)
        final["peak_mem_mb"] = max(
            history["peak_mem_mb"]) if history["peak_mem_mb"] else 0.0

        save_dir = f"experiments/hybrid/{conv_type}_{BEST_ATTN}_{BEST_POS}_ctx{CTX}"
        run_tag = conv_type
        generated = save_run(model, enc, cfg, history,
                             final, save_dir, run_tag, device)

        all_metrics.append(final)

        print(f"\n  [{conv_type}] PPL={final['perplexity']:.2f} | "
              f"Val Loss={final['val_loss']:.4f} | "
              f"Params={final['params']:,} | "
              f"Thrpt={final['throughput']:.0f} tok/s")

    # ── Comparative table ─────────────────────────────────────────────────
    print_hybrid_table(all_metrics)

    # ── PPL vs Compute scatter ─────────────────────────────────────────────
    _plot_ppl_vs_compute(all_metrics, BEST_ATTN, BEST_POS, CTX)

    return all_metrics


def _plot_ppl_vs_compute(metrics_list, attn, pos, ctx):
    os.makedirs("experiments/hybrid", exist_ok=True)

    labels = [m["conv_type"] for m in metrics_list]
    params = [m["params"] for m in metrics_list]
    ppls = [m["perplexity"] for m in metrics_list]
    thrpts = [m["throughput"] for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Hybrid designs | attn={attn} | pos={pos} | ctx={ctx}", fontsize=11)

    ax1.scatter(params, ppls, s=80, zorder=3)
    for label, x, y in zip(labels, params, ppls):
        ax1.annotate(label, (x, y), textcoords="offset points",
                     xytext=(5, 3), fontsize=8)
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Validation Perplexity")
    ax1.set_title("PPL vs Model Size")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(thrpts, ppls, s=80, zorder=3, color="orange")
    for label, x, y in zip(labels, thrpts, ppls):
        ax2.annotate(label, (x, y), textcoords="offset points",
                     xytext=(5, 3), fontsize=8)
    ax2.set_xlabel("Throughput (tok/s)")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("PPL vs Throughput")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "experiments/hybrid/ppl_vs_compute.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n  Saved PPL vs Compute scatter → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  —  toggle which experiment to run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Set exactly ONE of these to True ────────────────────────────────
    RUN_ATTENTION = False   # Part 1 & 2: single attention run
    RUN_POSITIONAL = False   # Part 3:     single positional run
    RUN_HYBRID = True    # Part 4:     all hybrid designs
    # ─────────────────────────────────────────────────────────────────────

    if RUN_ATTENTION:
        run_single_attention()
    elif RUN_POSITIONAL:
        run_single_positional()
    elif RUN_HYBRID:
        run_hybrid_experiments()
    else:
        print(
            "Nothing to run. Set one of RUN_ATTENTION / RUN_POSITIONAL / RUN_HYBRID = True")
