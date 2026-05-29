from dataclasses import dataclass


@dataclass
class SparsityConfig:
    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "microsoft/deberta-v3-base"
    task_name:  str = "cola"
    num_labels: int = 2

    # ── Training ─────────────────────────────────────────────────────────
    max_length:     int = 128
    batch_size:     int = 16
    num_epochs:     int = 10
    learning_rate:  float = 2e-4
    weight_decay:   float = 0.01
    warmup_ratio:   float = 0.06
    max_grad_norm:  float = 1.0

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_r:         int = 8
    lora_alpha:     int = 16
    lora_dropout:   float = 0.1

    # ── AdaLoRA ──────────────────────────────────────────────────────────
    adalora_init_r:     int = 12
    adalora_target_r:   int = 8
    adalora_tinit:      int = 200
    adalora_tfinal:     int = 1000
    adalora_delta_t:    int = 10

    # ── SoRA ─────────────────────────────────────────────────────────────
    sora_r:         int = 8
    sora_lambda:    float = 0.46
    sora_lr_gate:   float = 1e-4

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir:    str = "glue_data"
    output_dir:  str = "experiments"
