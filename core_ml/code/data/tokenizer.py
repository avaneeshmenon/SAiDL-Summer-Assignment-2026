import torch
from datasets import load_dataset
import tiktoken


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer():
    """
    Load GPT-2 tokenizer (BPE).
    """
    return tiktoken.get_encoding("gpt2")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_wikitext2(enc):
    """
    Loads WikiText-2 dataset and tokenizes it.

    Returns:
        train_tokens (torch.Tensor)
        val_tokens   (torch.Tensor)
    """

    print("Loading WikiText-2 dataset...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def encode_split(split):
        text = "\n".join(raw[split]["text"])
        tokens = enc.encode_ordinary(text)
        return torch.tensor(tokens, dtype=torch.long)

    train_tokens = encode_split("train")
    val_tokens = encode_split("validation")

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens  : {len(val_tokens):,}")

    return train_tokens, val_tokens
