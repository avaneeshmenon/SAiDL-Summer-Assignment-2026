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


def print_positional_table(metrics_list):
    print("\n" + "="*120)
    print(f"{'PosEnc':<12} {'TrainCtx':>8} {'TestCtx':>8} {'ValLoss':>10} {'PPL':>8}")
    print("-"*120)

    for m in metrics_list:
        print(f"{m['pos_type']:<12} "
              f"{m['train_ctx']:>8} "
              f"{m['test_ctx']:>8} "
              f"{m['val_loss']:>10.4f} "
              f"{m['perplexity']:>8.2f}")

    print("="*120)
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = TransformerConfig()

    # "standard", "sliding_window", "sparse_block", "linear", "gqa", "mqa", "softmax_free"
    cfg.attention_type = "standard"
    # "learned", "sinusoidal", "rope", "alibi", "relative"
    cfg.pos_encoding_type = "learned"
    cfg.rope_scale = 1.0  # Only used for RoPE variants
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
    exp_name = f"{cfg.pos_encoding_type}_ctx{cfg.context_length}"
    save_dir = f"experiments/positional/{exp_name}"
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
    plt.savefig(
        f"{save_dir}/training_curves_{cfg.attention_type}_{cfg.context_length}.png")
    plt.close()

    # ── Evaluation
    final_metrics = evaluate(model, val_loader, device,
                             max_iters=len(val_loader))

    final_metrics["model"] = cfg.attention_type
    final_metrics["context_length"] = cfg.context_length
    final_metrics["params"] = model.count_parameters()
    final_metrics["epoch_time_avg"] = sum(
        history["epoch_time"]) / len(history["epoch_time"])
    final_metrics["peak_mem_mb"] = max(history["peak_mem_mb"])

    # ── Save metrics
    with open(f"{save_dir}/metrics_{cfg.attention_type}_{cfg.context_length}.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # ── Generation
    prompt = "The history of artificial intelligence began"

    generated = generate(
        model, enc, cfg, device, prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40
    )

    with open(f"{save_dir}/generation_{cfg.attention_type}_{cfg.context_length}.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\n{generated}")

    # ── Save model
    torch.save(model.state_dict(),
               f"{save_dir}/model_{cfg.attention_type}_{cfg.context_length}.pt")

    # Print table (single run)
    print_results_table([final_metrics])

    print("\nSample:\n", generated)


# def run_positional_experiments():

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     POS_TYPES = ["learned", "sinusoidal", "rope",
#                  "rope_interp", "alibi", "relative"]

#     # ✅ Load ONCE
#     enc = load_tokenizer()
#     train_tokens, val_tokens = tokenize_wikitext2(enc)

#     for pos_type in POS_TYPES:

#         print(f"\n===== Running {pos_type} =====\n")

#         cfg = TransformerConfig()

#         cfg.context_length = 512
#         cfg.rope_scale = 1.0

#         # ✅ Mapping
#         if pos_type in ["rope", "alibi", "relative"]:
#             cfg.attention_type = pos_type
#             cfg.pos_encoding_type = pos_type

#         elif pos_type == "rope_interp":
#             cfg.attention_type = "rope"
#             cfg.pos_encoding_type = "rope"

#         else:
#             cfg.attention_type = "standard"
#             cfg.pos_encoding_type = pos_type

#         # ─────────────────────────────────────
#         # Build loaders
#         # ─────────────────────────────────────
#         train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

#         # Model
#         model = TransformerLM(cfg).to(device)
#         optimizer = build_optimizer(model, cfg)

#         # Train
#         history = train(model, cfg, train_loader,
#                         val_loader, optimizer, device)

#         # ─────────────────────────────────────
#         # Save dir
#         # ─────────────────────────────────────
#         save_dir = f"experiments/positional/{pos_type}_ctx512"
#         os.makedirs(save_dir, exist_ok=True)

#         # ─────────────────────────────────────
#         # Final evaluation ONLY at 512 (safe)
#         # ─────────────────────────────────────
#         final_metrics = evaluate(
#             model, val_loader, device, max_iters=len(val_loader)
#         )

#         final_metrics["pos_type"] = pos_type
#         final_metrics["train_ctx"] = 512

#         # Save metrics
#         with open(f"{save_dir}/metrics.json", "w") as f:
#             json.dump(final_metrics, f, indent=4)

#         # Save model
#         torch.save(
#             model.state_dict(),
#             f"{save_dir}/model.pt"
#         )

#         print(
#             f"{pos_type} | train_ctx=512 | PPL={final_metrics['perplexity']:.2f}")

#     print("\n✅ Training complete for all positional encodings.")


def run_single_positional():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # "learned", "sinusoidal", "rope", "rope_interp", "alibi", "relative"
    POS_TYPE = "rope_interp"   # CHANGE THIS EACH TIME

    cfg = TransformerConfig()
    cfg.context_length = 512
    cfg.rope_scale = 1.0

    if POS_TYPE in ["rope", "alibi", "relative"]:
        cfg.attention_type = POS_TYPE
        cfg.pos_encoding_type = POS_TYPE

    elif POS_TYPE == "rope_interp":
        cfg.attention_type = "rope"
        cfg.pos_encoding_type = "rope"

    else:
        cfg.attention_type = "standard"
        cfg.pos_encoding_type = POS_TYPE

    # Data
    enc = load_tokenizer()
    train_tokens, val_tokens = tokenize_wikitext2(enc)
    train_loader, val_loader = build_loaders(cfg, train_tokens, val_tokens)

    # Model
    model = TransformerLM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)

    # Train
    history = train(model, cfg, train_loader, val_loader, optimizer, device)

    # Save
    save_dir = f"experiments/positional/{POS_TYPE}_ctx512"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        f"{save_dir}/model_{cfg.attention_type}_512.pt"
    )

    print(f"\n✅ Saved model for {POS_TYPE}")


if __name__ == "__main__":

    RUN_POSITIONAL = True   # 🔥 toggle this

    if RUN_POSITIONAL:
        # run_positional_experiments()
        run_single_positional()
    else:
        main()
