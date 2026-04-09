import math
import time
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device, max_iters=50):
    """
    Evaluate model on validation set.

    Returns:
        dict with:
            - val_loss
            - perplexity
            - throughput (tokens/sec)
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    t0 = time.perf_counter()

    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break

        x, y = x.to(device), y.to(device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()
        total_tokens += x.numel()
        n_batches += 1

    elapsed = time.perf_counter() - t0

    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)
    throughput = total_tokens / elapsed

    model.train()

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "throughput": throughput,
    }
