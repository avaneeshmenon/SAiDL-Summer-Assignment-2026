import torch
from torch.utils.data import DataLoader

from .dataset import TokenDataset


# ─────────────────────────────────────────────────────────────────────────────
# Build DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(cfg, train_tokens, val_tokens):
    """
    Creates PyTorch DataLoaders for training and validation.

    Args:
        cfg: configuration object (must contain batch_size, context_length)
        train_tokens: torch.Tensor of tokenized training data
        val_tokens: torch.Tensor of tokenized validation data

    Returns:
        train_loader, val_loader
    """

    train_ds = TokenDataset(train_tokens, cfg.context_length)
    val_ds = TokenDataset(val_tokens,   cfg.context_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Debug / sanity check helper
# ─────────────────────────────────────────────────────────────────────────────

def inspect_batch(loader):
    """
    Prints shape of one batch for debugging.
    """
    x, y = next(iter(loader))
    print(f"x shape: {x.shape}")  # (B, T)
    print(f"y shape: {y.shape}")  # (B, T)
