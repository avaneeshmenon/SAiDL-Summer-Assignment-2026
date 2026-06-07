# SAiDL Summer Induction Assignment 2026

**Avaneesh Nandakumar Menon — 2024B3A71063G**

---

## Tracks

- **Core ML** — Decoder-only Transformer language modelling on WikiText-2. Covers attention variants, positional encodings, convolution-attention hybrids, and AFT bonus.
- **Sparsity & Optimization** — Parameter-efficient fine-tuning (LoRA, AdaLoRA, SoRA) on CoLA/GLUE using DeBERTa-v3-base, extended to xLSTM and Mamba.

---

## Repo Structure

```
.
├── core_ml/
│   ├── code/
│   │   ├── data/
│   │   │   ├── dataloader.py         # DataLoader builder
│   │   │   ├── dataset.py            # TokenDataset
│   │   │   └── tokenizer.py          # GPT-2 BPE tokenizer + WikiText-2 loader
│   │   ├── training/
│   │   │   ├── evaluate.py           # Validation loop
│   │   │   ├── generate.py           # Top-k sampling generation
│   │   │   ├── optimizer.py          # AdamW with weight decay grouping
│   │   │   └── train.py              # Training loop with cosine LR schedule
│   │   ├── utils/
│   │   │   └── config.py             # TransformerConfig dataclass
│   │   ├── aft_attention.py          # AFT variants (self-register into ATTENTION_REGISTRY)
│   │   ├── attention.py              # All attention variants + ATTENTION_REGISTRY
│   │   ├── conv_blocks.py            # Convolutional block modules
│   │   ├── main.py                   # Entry point for Core ML experiments (toggle flags)
│   │   ├── model.py                  # TransformerLM, TransformerBlock, FeedForward
│   │   └── positional.py             # Positional encoding variants + POS_ENCODING_REGISTRY
│   └── experiments/
│       ├── aft/                      # AFT variant metrics & plots
│       ├── attention/                # Attention variant metrics & plots (ctx 512/1024/2048)
│       ├── baseline/                 # Baseline model metrics & plots
│       ├── hybrid/                   # Conv-attention hybrid metrics & plots
│       └── positional/               # Positional encoding extrapolation metrics & plots
│
└── sparsity/
    └── code/
        ├── methods/
        │   └── sora.py               # SoRALinear, SoRAModel
        ├── experiments/
        │   ├── part1/                # LoRA vs AdaLoRA vs SoRA metrics & plots
        │   ├── part2/                # SGD vs proximal gradient metrics & plots
        │   └── part3/                # SoRA on xLSTM and Mamba metrics & plots
        ├── config.py                 # SparsityConfig dataclass
        ├── main.py                   # Entry point for sparsity experiments (toggle flags)
        ├── part2.py                  # SGD vs proximal gradient analysis
        ├── part3.py                  # SoRA on xLSTM and Mamba
        └── train.py                  # LoRA, AdaLoRA, SoRA training loops
```

---

## Running

All training was done on Google Colab (T4 GPU). Experiment results are saved under `experiments/` and do not need to be rerun to read the report.

### Core ML

Toggle flags at the bottom of `core_ml/code/main.py`:

```python
RUN_POSITIONAL = True   # runs positional encoding experiments
# set False to run attention/hybrid experiments via main()
```

Switch attention type and positional encoding inside `main()` or `run_single_positional()`:

```python
cfg.attention_type     = "gqa"       # standard, sliding_window, sparse_block, mqa, gqa
cfg.pos_encoding_type  = "relative"  # learned, sinusoidal, rope, rope_interp, alibi, relative
cfg.context_length     = 512
```

AFT variants are registered in `aft_attention.py` and self-register into `ATTENTION_REGISTRY` on import.

### Sparsity & Optimization

Toggle flags at the bottom of `sparsity/code/main.py`:

```python
RUN_PART1 = False   # LoRA vs AdaLoRA vs SoRA
RUN_PART2 = False   # SGD vs proximal gradient analysis
RUN_PART3 = True    # SoRA on xLSTM and Mamba
```

Within each part, individual methods can be toggled independently (e.g. `RUN_LORA`, `RUN_ADALORA`, `RUN_SORA`) to skip rerunning completed experiments and load saved metrics from JSON instead.

### Experiment Inventory

**Core ML — Attention variants** (`cfg.attention_type`): `standard`, `sliding_window`, `sparse_block`, `mqa`, `gqa`; each run at context lengths 512 / 1024 / 2048.

**Core ML — Positional encodings** (`cfg.pos_encoding_type`): `learned`, `sinusoidal`, `rope`, `rope_interp`, `alibi`, `relative`; trained at ctx=512, evaluated at 512 / 1024 / 2048 to measure extrapolation.

**Core ML — Conv-attention hybrids** (`cfg.conv_type`): `none` (baseline), `conv_before_attn`, `interleaved`, `depthwise_subset`, `gated_conv_ff`.

**Core ML — AFT (bonus)** (`cfg.attention_type`): `aft_simple`, `aft_full`, `aft_local`, `aft_conv`, `aft_rope_simple`, `aft_decay`.

**Sparsity Part 1** — LoRA, AdaLoRA, SoRA on DeBERTa-v3-base / CoLA (MCC).

**Sparsity Part 2** — SGD vs. proximal gradient update analysis on the SoRA gate.

**Sparsity Part 3** — SoRA applied to xLSTM and Mamba.

### Dependencies

```
torch torchvision
transformers datasets peft
tiktoken
xlstm
scikit-learn
matplotlib
```

Mamba requires `mamba-ssm` (CUDA build). If unavailable, `part3.py` automatically falls back to a minimal pure-PyTorch S6 implementation.
