# gpt-from-scratch-pytorch

A decoder-only GPT built from scratch in PyTorch — every component hand-written, no
`nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace. Written to understand how
the pieces actually work, not to be fast.

## What's implemented

| Component | File | Notes |
|---|---|---|
| Char-level tokenizer | `src/my_tokenizer.py` | Fit-on-corpus vocab with `<unk>` fallback |
| FlashAttention | `src/flash_attention.py` | Tiled attention with online softmax — running max / numerator / denominator, causal masking by tile skipping |
| Multi-head attention | `src/flash_attention.py` | `num_heads` independent flash-attention heads, concatenated |
| RMSNorm | `src/rms_norm.py` | Root-mean-square normalization |
| Positional encoding | `src/pos_enc.py` | Sinusoidal |
| Feed-forward network | `src/ffn.py` | 4× expansion, ReLU |
| Transformer block | `src/trf.py` | Pre-norm, residual around both attention and FFN |
| GPT model | `src/gpt.py` | Embedding → pos-enc → N blocks → norm → LM head |
| Training loop | `src/training.py` | Cross-entropy on next-token prediction |
| Inference | `src/inference.py` | Greedy decoding |

The interesting file is `src/flash_attention.py`. Instead of materializing the full
`L × L` attention matrix, it walks query tiles against key tiles and maintains a running
softmax normalizer — the same trick that makes FlashAttention memory-efficient, written
out in plain PyTorch so the online-softmax recurrence is readable.

## Setup

```bash
uv sync
```

## Train

```bash
cd src
python training.py
```

Trains on `data/verdict.txt` (Edith Wharton, *The Verdict* — public domain).

## Configuration

All hyperparameters live in `src/settings.py` via `pydantic-settings`, so any of them can
be overridden by environment variable:

```bash
CONTEXT_LENGTH=16 EMBEDDING_DIM=256 python training.py
```

| Setting | Default |
|---|---|
| `context_length` | 8 |
| `embedding_dim` | 128 |
| `num_heads` | 4 |
| `tile_size` | 2 |
| `num_trf_blocks` | 12 |
| `epochs` | 20 |

Device is auto-selected: Apple MPS if available, otherwise CPU.

## Status

Work in progress — this is a learning project. Known rough edges:

- Attention hard-codes `context_length`, so sequences must be exactly that long
- Positional encodings are multiplied into the embeddings rather than added
- FFN applies ReLU to its output projection
- Training is full-batch with no train/validation split
- Inference is single-shot argmax, not an autoregressive generation loop

## Roadmap

- [ ] RoPE (rotary positional embeddings) to replace sinusoidal
- [ ] Mixture-of-Experts FFN layer
- [ ] KV cache for generation
- [ ] Mini-batching and a validation split

## Tooling

Formatting and linting with [ruff](https://docs.astral.sh/ruff/):

```bash
uv run ruff format src/
uv run ruff check --fix src/
```
