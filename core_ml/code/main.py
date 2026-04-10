# import torch

# # ── Imports ────────────────────────────────────────────────────────────────
# from utils.config import TransformerConfig

# from model import TransformerLM

# from training.train import train
# from training.optimizer import build_optimizer
# from training.generate import generate

# from data.tokenizer import load_tokenizer, tokenize_wikitext2
# from data.dataloader import build_loaders

# from training.evaluate import evaluate
# import matplotlib.pyplot as plt
# import os
# import json


# # ── Clean results table ─────────────────────────────────────
# def print_results_table(metrics_list):
#     print("\n" + "="*110)
#     print(f"{'Model':<20} {'Ctx':>6} {'Params':>10} {'ValLoss':>10} {'PPL':>8} {'Thrpt':>10} {'Mem(MB)':>10} {'Epoch(s)':>10}")
#     print("-"*110)

#     for m in metrics_list:
#         print(f"{m['model']:<20} "
#               f"{m['context_length']:>6} "
#               f"{m['params']:>10,} "
#               f"{m['val_loss']:>10.4f} "
#               f"{m['perplexity']:>8.2f} "
#               f"{m['throughput']:>10.0f} "
#               f"{m['peak_mem_mb']:>10.1f} "
#               f"{m['epoch_time_avg']:>10.1f}")

#     print("="*110)


# # ── Main runner ─────────────────────────────────────────────────────────────
# def main():

#     # ── Device ─────────────────────────────────────────────
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # ── Config ─────────────────────────────────────────────
#     cfg = TransformerConfig()

#     # ── Data ───────────────────────────────────────────────
#     enc = load_tokenizer()
#     train_tokens, val_tokens = tokenize_wikitext2(enc)
#     train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

#     # ── Model ──────────────────────────────────────────────
#     model = TransformerLM(cfg).to(device)
#     total_params = sum(p.numel()
#                        for p in model.parameters() if p.requires_grad)

#     print(f"Model params: {total_params:,}")
#     print("\nStarting training...\n")

#     # ── Optimizer ──────────────────────────────────────────
#     optimizer = build_optimizer(model, cfg)

#     # ── Training ───────────────────────────────────────────
#     print("\nStarting training...\n")

#     history = train(
#         model,
#         cfg,
#         train_loader,
#         val_loader,
#         optimizer,
#         device
#     )
#     # 🔥 IMPORTANT: dynamic folder
#     exp_name = f"{cfg.attention_type}_ctx{cfg.context_length}"
#     save_dir = f"experiments/attention/{exp_name}"
#     # ── Create folder early ─────────────────────────────────
#     os.makedirs("experiments/baseline", exist_ok=True)

#     # ── Plotting ────────────────────────────────────────────
#     steps = history["step"]

#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#     axes[0].plot(steps, history["train_loss"], label="Train Loss")
#     axes[0].plot(steps, history["val_loss"], label="Val Loss")
#     axes[0].set_title("Loss Curves")
#     axes[0].legend()

#     axes[1].plot(steps, history["perplexity"])
#     axes[1].set_title("Perplexity")

#     axes[2].plot(steps, history["lr"])
#     axes[2].set_title("LR")

#     plt.tight_layout()
#     plt.savefig("experiments/baseline/baseline_training_curves.png")
#     plt.close()

#     print("Saved training curves")

#     # ── Evaluation ─────────────────────────────────────────
#     final_metrics = evaluate(model, val_loader, device,
#                              max_iters=len(val_loader))

#     # Add metadata
#     final_metrics["model"] = "baseline"
#     final_metrics["context_length"] = cfg.context_length
#     final_metrics["params"] = model.count_parameters()
#     final_metrics["epoch_time_avg"] = sum(
#         history["epoch_time"]) / len(history["epoch_time"])

#     # ── SAVE FIRST (critical) ──────────────────────────────
#     with open("experiments/baseline/baseline_metrics.json", "w") as f:
#         json.dump(final_metrics, f, indent=4)

#     with open("experiments/baseline/results_table.txt", "w") as f:
#         f.write(str(final_metrics))

#     # ── Generation (move BEFORE saving file) ───────────────
#     prompt = "The history of artificial intelligence began"

#     generated = generate(
#         model,
#         enc,
#         cfg,
#         device,
#         prompt,
#         max_new_tokens=80,
#         temperature=0.7,
#         top_k=40
#     )

#     with open("experiments/baseline/sample_generation.txt", "w") as f:
#         f.write(f"Prompt: {prompt}\n\n{generated}")

#     # ── Save model ─────────────────────────────────────────
#     torch.save(
#         model.state_dict(),
#         "experiments/baseline/baseline_transformer.pt"
#     )

#     # ── Print table LAST (safe now) ─────────────────────────
#     print_results_table([final_metrics])

#    # ── Print final results (clean) ─────────────────────────
#     print_results_table([final_metrics])

#     print("\nSample Generation:\n")
#     print("="*50)
#     print(f"Prompt   : {prompt}")
#     print(f"Generated:\n{generated}")
#     print("="*50)


# # ── IMPORTANT (fixes Mac multiprocessing crash) ─────────────────────────────
# if __name__ == "__main__":
#     main()

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


# ─────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────
def print_results_table(metrics_list):
    print("\n" + "="*110)
    print(f"{'Model':<20} {'Ctx':>6} {'Params':>10} {'ValLoss':>10} {'PPL':>8} {'Thrpt':>10} {'Mem(MB)':>10} {'Epoch(s)':>10}")
    print("-"*110)

    for m in metrics_list:
        print(f"{m['model']:<20} "
              f"{m['context_length']:>6} "
              f"{m['params']:>10,} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>8.2f} "
              f"{m['throughput']:>10.0f} "
              f"{m['peak_mem_mb']:>10.1f} "
              f"{m['epoch_time_avg']:>10.1f}")

    print("="*110)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = TransformerConfig()

    # "standard", "sliding_window", "sparse_block", "linear", "gqa", "mqa", "softmax_free"
    cfg.attention_type = "mqa"
    cfg.context_length = 512

    print(f"\nRunning: {cfg.attention_type} | ctx={cfg.context_length}\n")

    # ── Data
    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    # ── Model
    model = TransformerLM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    # ── Train
    history = train(model, cfg, train_loader, val_loader, optimizer, device)

    # ── Folder (dynamic)
    exp_name = f"{cfg.attention_type}_ctx{cfg.context_length}"
    save_dir = f"experiments/attention/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ── Plot
    steps = history["step"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, history["train_loss"], label="Train")
    axes[0].plot(steps, history["val_loss"], label="Val")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(steps, history["perplexity"])
    axes[1].set_title("PPL")

    axes[2].plot(steps, history["lr"])
    axes[2].set_title("LR")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()

    # ── Evaluation
    final_metrics = evaluate(model, val_loader, device,
                             max_iters=len(val_loader))

    final_metrics["model"] = cfg.attention_type
    final_metrics["context_length"] = cfg.context_length
    final_metrics["params"] = model.count_parameters()
    final_metrics["epoch_time_avg"] = sum(
        history["epoch_time"]) / len(history["epoch_time"])

    # ── Save metrics
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # ── Generation
    prompt = "The history of artificial intelligence began"

    generated = generate(
        model, enc, cfg, device, prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40
    )

    with open(f"{save_dir}/generation.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\n{generated}")

    # ── Save model
    torch.save(model.state_dict(), f"{save_dir}/model.pt")

    # ─────────────────────────────────────────────
    # 🔥 GLOBAL COMPARISON TABLE (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    summary_path = "experiments/attention/summary.json"

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(final_metrics)

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Print full table
    print_results_table(all_results)

    print("\nSample:\n", generated)


if __name__ == "__main__":
    main()
