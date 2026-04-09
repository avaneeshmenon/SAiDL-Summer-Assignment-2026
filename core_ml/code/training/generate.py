import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    enc,
    cfg,
    device,
    prompt,
    max_new_tokens=100,
    temperature=0.9,
    top_k=50,
):
    """
    Generate text using top-k sampling.

    Args:
        model: trained model
        enc: tokenizer (tiktoken)
        cfg: config (needs context_length)
        device: cpu/cuda
        prompt: input string
    """

    model.eval()

    tokens = torch.tensor(
        enc.encode_ordinary(prompt),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    for _ in range(max_new_tokens):

        # Crop to context window
        ctx = tokens[:, -cfg.context_length:]

        logits = model(ctx)[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return enc.decode(tokens[0].tolist())