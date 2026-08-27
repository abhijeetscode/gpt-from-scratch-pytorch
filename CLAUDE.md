# CLAUDE.md

## How to work with me

**I am learning. Guide me — do not write my code.**

- **Never edit, create, or delete files unless I explicitly say "make the change", "apply it", "do it", or similar.** Answering "can you fix it?" with a diff is not what I want. Answering with *where* the bug is and *why* is.
- Point at the exact location (`file.py:line`) and explain the root cause. Show the minimal corrected snippet inline in your reply as an illustration — that is fine, and useful. Writing it into the file is not.
- When you find several problems, say which order to fix them in and why. Flag any fix that would make another bug *silent* instead of loud — that ordering matters more to me than the fixes themselves.
- Prove claims by running code, not by reading it. Diagnose empirically: run it, capture the real error, then explain.
- Give me a test that **discriminates** — one that fails before the fix and passes after. Tell me which cases are traps that pass even with broken code.
- If I propose a fix, check my reasoning and say plainly whether I am right or wrong. Do not soften a "wrong", and do not pad a "right" with an unrequested rewrite.
- Reading, searching, and running things to diagnose: always allowed, no need to ask.
- Scratch/throwaway test scripts: fine, put them in the scratchpad dir, not in the repo.

## What this repo is

GPT from scratch, hand-rolled, for learning. Char-level tokenizer, RoPE/MoE are the eventual goal. Nothing here should be replaced with a `torch.nn` builtin — writing it by hand *is* the point.

## Running it

**Run from `src/`.** Modules use flat imports (`from gpt import AbbyGPT`) and `training.py` opens `../data/verdict.txt`.

```bash
cd src && ../.venv/bin/python training.py
```

`uv` manages the env (`.venv/`, Python 3.14). Debug configs live in `.vscode/launch.json` (F5) — they set `cwd` to `src/` for the reasons above.

All hyperparameters are in `src/settings.py` (pydantic-settings). Device is `mps` when available.

## Shape conventions

`B` batch, `L` sequence length, `E` embedding dim, `D` head dim, `T` tile size.
Tensors flow as `(B, L, E)`. `AbbyGPT.forward` unsqueezes 1-D token input to `(1, L)`.

Sequence length must come from the tensor (`x.shape[-2]`), never from `settings.context_length` — variable-length input is a requirement, and `L` is often not a multiple of `T`, so the last tile is partial.

## Verifying attention

The trustworthy check for `flash_attention.py` is comparing against plain reference causal softmax attention across lengths that exercise partial tiles (L = 1, 2, 3, 5, 7, 15, 16). Expect ~1e-7 in float32. `L=1` and any `L` divisible by `T` pass even with broken tiling — they are not evidence.

`training.py` first-step loss is ~4.16 at `L=16`; a change there means something in the `L=16` path moved.

## Known open issues (not fixed unless I ask)

- `inference.py:15` — `num_token_to_predict` is unused; it argmaxes every position instead of generating autoregressively.
- One shared `RMSNorm` instance is reused at every norm site in `gpt.py` / `trf.py`. Harmless while it has no parameters; becomes accidental weight sharing the moment a learnable gain is added.
- `MultiHeadAttention` has no output projection (`W_o`).
- `PositionalEncoding` table is `context_length` long and silently truncates if `L` exceeds it.
