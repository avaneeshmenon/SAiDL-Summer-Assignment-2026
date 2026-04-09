import torch

# ── Imports ────────────────────────────────────────────────────────────────
from utils.config import TransformerConfig

from model import TransformerLM

from training.train import train
from training.optimizer import build_optimizer
from training.generate import generate

from data.tokenizer import load_tokenizer, tokenize_wikitext2
from data.dataloader import build_loaders


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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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