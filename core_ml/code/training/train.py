import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.evaluate import evaluate
# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(step, warmup_steps, total_steps, min_lr, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)

    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, cfg, train_loader, val_loader, optimizer, device):

    total_steps = len(train_loader) * cfg.max_epochs
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history = {
        "step": [], "train_loss": [], "val_loss": [], "perplexity": [],
        "throughput": [], "peak_mem_mb": [], "lr": [],
        "grad_norm": [], "loss_spike": [], "epoch_time": []
    }

    step = 0
    recent_losses = []

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_tokens = 0

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # 🔥 Clean table header (printed once per epoch)
        print("\nStep | Train | Val | PPL | Thrpt | Mem")
        print("-" * 60)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # LR schedule
            lr = cosine_lr(step, cfg.warmup_steps, total_steps,
                           cfg.learning_rate * 0.1, cfg.learning_rate)

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Loss spike detection
            loss_val = loss.item()
            recent_losses.append(loss_val)
            if len(recent_losses) > 20:
                recent_losses.pop(0)

            baseline = sum(recent_losses[:-1]) / max(len(recent_losses) - 1, 1)
            loss_spike = (loss_val > 2.0 * baseline) and (step > 50)

            epoch_tokens += x.numel()
            step += 1

            # Periodic evaluation
            if step % cfg.eval_interval == 0:

                metrics = evaluate(model, val_loader, device, cfg.eval_iters)

                train_peak_mem = (
                    torch.cuda.max_memory_allocated() / 1e6
                    if device == "cuda" else 0.0
                )

                history["step"].append(step)
                history["train_loss"].append(loss_val)
                history["val_loss"].append(metrics["val_loss"])
                history["perplexity"].append(metrics["perplexity"])
                history["throughput"].append(metrics["throughput"])
                history["peak_mem_mb"].append(train_peak_mem)
                history["lr"].append(lr)
                history["grad_norm"].append(float(grad_norm))
                history["loss_spike"].append(loss_spike)

                print(
                    f"Step {step:>6} | "
                    f"Train {loss_val:.4f} | "
                    f"Val {metrics['val_loss']:.4f} | "
                    f"PPL {metrics['perplexity']:.2f} | "
                    f"GradNorm {float(grad_norm):.3f} | "
                    f"Eval thrpt {metrics['throughput']:.0f} tok/s | "
                    f"Mem {train_peak_mem:.0f} MB"
                )

        # ── End of epoch ─────────────────────────────
        metrics = evaluate(model, val_loader, device, cfg.eval_iters)

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)

        print(f"\n[Epoch {epoch}] Val Loss: {metrics['val_loss']:.4f}, "
              f"PPL: {metrics['perplexity']:.2f}")

        print(f"=== Epoch {epoch} | Time: {epoch_time:.1f}s | "
              f"Train thrpt: {epoch_tokens/epoch_time:.0f} tok/s ===\n")

    return history
