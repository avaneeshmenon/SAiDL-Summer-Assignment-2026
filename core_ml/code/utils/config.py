from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1

    attention_type: str = "standard"
    pos_encoding_type: str = "learned"
    use_conv_hybrid: bool = False

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 50