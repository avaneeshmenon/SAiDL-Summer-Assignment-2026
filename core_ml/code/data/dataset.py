import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    Language modeling dataset using contiguous token chunks.

    Each sample:
        x = tokens[i : i+T]
        y = tokens[i+1 : i+T+1]

    So model learns: next-token prediction
    """

    def __init__(self, tokens: torch.Tensor, context_length: int):
        self.tokens = tokens
        self.ctx = context_length

        # Number of full chunks we can extract
        self.n = (len(tokens) - 1) // context_length

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.ctx

        x = self.tokens[start : start + self.ctx]
        y = self.tokens[start + 1 : start + self.ctx + 1]

        return x, y