from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    # ── Vocabulary / sequence ────────────────────────────────────────────────
    vocab_size:      int   = 50257
    context_length:  int   = 1024

    # ── Architecture ─────────────────────────────────────────────────────────
    n_layers:  int   = 6
    n_heads:   int   = 8
    d_model:   int   = 512
    d_ff:      int   = 2048
    dropout:   float = 0.1

    # ── Attention variant ────────────────────────────────────────────────────
    # "standard" | "sliding_window" | "sparse_block" | "linear"
    # "gqa"      | "mqa"            | "softmax_free"
    # "rope"     | "rope_interp"    | "alibi"        | "relative"
    attention_type: str = "standard"

    # ── Positional encoding ──────────────────────────────────────────────────
    # "learned" | "sinusoidal" | "rope" | "rope_interp" | "alibi" | "relative"
    pos_encoding_type: str = "learned"

    # Only used for rope_interp; scale < 1.0 compresses positions for extension
    rope_scale: float = 1.0

    # ── Convolution hybrid (Part 4) ──────────────────────────────────────────
    #
    # conv_type controls which of the four designs is active:
    #
    #   "none"             – pure Transformer, no conv (default / baseline)
    #   "conv_before_attn" – Design 1: DWSConv prepended to every attn sublayer
    #   "interleaved"      – Design 2: even layers = ConvBlock, odd = AttnBlock
    #   "depthwise_subset" – Design 3: first (n_layers//2) layers replace attn
    #                                  with DWSConv
    #   "gated_conv_ff"    – Design 4: replace FFN with Gated Conv FFN in all
    #                                  layers; attention unchanged
    #
    # conv_kernel_size: kernel width for all Conv1D modules (3 or 5 recommended)
    #
    # Legacy flag kept for backward compatibility with any existing checkpoints:
    use_conv_hybrid: bool = False   # ignored when conv_type != "none"

    conv_type:        str = "none"
    conv_kernel_size: int = 3

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size:    int   = 8
    learning_rate: float = 3e-4
    weight_decay:  float = 0.1
    max_epochs:    int   = 10
    warmup_steps:  int   = 200
    grad_clip:     float = 1.0
    eval_interval: int   = 100
    eval_iters:    int   = 50