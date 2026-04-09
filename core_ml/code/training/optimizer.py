import torch


def build_optimizer(model, cfg):
    """
    AdamW optimizer with proper weight decay handling:
    - Apply weight decay only to weights (not biases / norms)
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )

    return optimizer