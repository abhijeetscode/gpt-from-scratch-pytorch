# Making AbbyGPT SFT-ready

Task list to get from "pretrained-ish" to "can be supervised fine-tuned."
All numbers below were **measured** in this repo, not estimated. Re-measure after each change.

## Where we are (measured 2026-09-02)

| Quantity | Value |
| --- | --- |
| Corpus | `verdict.txt`, 5,251 BPE tokens |
| Vocab | 1,605 (1,383 actually used — 222 embedding rows never get a gradient) |
| Compression | 3.9 chars/token |
| Params | 2.18M |
| `context_length` | 16 tokens ≈ 60 characters |
| `batch_size` | 32 → 328 chunks → 10 batches, 8 chunks dropped |
| `epochs` | 1 (raised to 5 for the run below) |
| Optimizer steps | 10 per epoch |

### Loss landmarks — use these as tripwires

| Level | Loss | Perplexity | Meaning |
| --- | --- | --- | --- |
| Uniform `ln(1605)` | 7.3809 | 1605 | untrained |
| **Unigram entropy** | **6.8165** | **913** | knows token frequencies, **zero context** |
| Ours, 50 steps | 6.3581 | 577 | first real context learning, on *training* data |
| Target for SFT | single-digit ppl on **held-out** data | | not yet measurable — no val split |

Observed run (5 epochs = 50 steps): `7.3582 → 6.9679 → 6.8144 → 6.6875 → 6.3581`.
Epoch 3 sat almost exactly on the unigram floor. Everything below 6.82 is attention earning its keep.

---

## Progress tracker

`[x]` done and verified · `[~]` partially done · `[ ]` not started

| # | Task | Done | Blocks | Silent if wrong |
| --- | --- | :---: | --- | :---: |
| 1 | Fix model input contract | **[x]** | everything | — |
| 2 | Save fine-tunable checkpoint | **[~]** | 3, 7 | — |
| 3 | Validation split | [ ] | 4 (interpretability) | — |
| 4 | Scale window + training run | [ ] | 6, 7 | — |
| 5 | Shuffle batches | [ ] | — | **yes** |
| 6 | `[EOS]` in token stream | [ ] | 7 | — |
| 7 | Label masking in loss | [ ] | — | **yes, worst** |
| 8 | Grow the corpus | [ ] | meaning of 3-7 | — |

**Gate: SFT can begin when 1-7 are checked and val loss is single-digit perplexity.**
Task 8 is not strictly a gate, but 3-7 tell you little without it.

### ⚠ Do task 3 before task 4

`training.py:68-70` now checkpoints on **best training loss**. Training loss falls
monotonically past the point of generalisation, so "best" converges on the most
overfit checkpoint ever produced. Harmless at 3 epochs; actively harmful the moment
task 4 raises `epochs` to 100+. Gate the save on `val_loss` instead.

This failure is silent — the printed number improves while the saved weights get worse.

---

## Task order

Ordering is deliberate. Each task either unblocks the next or would be **hidden** by doing a later one first.

### 1. Fix the model input contract — **DONE** (verified 2026-09-03)

**Files:** `gpt.py:26`, `training.py:57`

- [x] Delete `x = x.flatten(0, 1)` from `gpt.py:26`
- [x] Delete `.unsqueeze(0)` from the `training.py:57` forward call
- [x] Add the `assert x.ndim == 2` contract line
- [x] Remove the now-dead `if x.ndim == 2: unsqueeze` branch
- [x] Update `inference.py` to send `(1, n)`
- [x] Discriminating test passes

Verified bit-identical output (maxdiff `0.0`, same loss) without the flatten/unsqueeze pair.

Measured after the fix:

```
(32, 16)    -> (32, 16, 1605)      (8,)        -> rejected: AbbyGPT expects 2D input matrix
(1, 16)     -> (1, 16, 1605)       (1, 32, 16) -> rejected: AbbyGPT expects 2D input matrix
(1, 8)      -> (1, 8, 1605)
(4, 5)      -> (4, 5, 1605)
```

Caveat: `inference.py` now only does a **single** next-token step — the `Inference`
class and its sliding-window generation were removed. When generation is rebuilt,
re-test the traps: window must be `tokens[max(0, cur - context_length) : cur]`, read
the real last position (`n - 1` once padding enters, never `-1`), and use a prompt
where `prompt + generated > context_length`. Anything shorter passes with broken slicing.

Settle on one contract — `(B, L)` always, single input is `(1, L)`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, f"AbbyGPT expects (B, L), got {tuple(x.shape)}"
    x = self.embedding_layer(x)
    x = self.posn_encodings(x)
    ...
```

This also removes the `if x.ndim == 2: unsqueeze` branch — `forward` becomes straight-line code, which is what `torch.export` needs.

**Why first:** every task below runs through `forward`. Export *bakes in* whatever contract exists at trace time, so exporting a broken contract makes it permanent and 40 frames deep to debug (`size of tensor a (16*s77)`).

**Discriminating test**

```python
m(torch.randint(0, V, (32, 16)))   # -> (32, 16, V)
m(torch.randint(0, V, (1, 8)))     # -> (1, 8, V)
m(torch.randint(0, V, (8,)))       # -> AssertionError
```

Trap: `model(xs[i].unsqueeze(0))` passes both before *and* after. Not evidence.

Callers to update: `inference.py` window becomes `tokens[start:cur].unsqueeze(0)`.

---

### 2. Save a fine-tunable checkpoint — **PARTIAL**

**File:** `training.py:68-70`

- [x] Add `torch.save(model.state_dict(), "AbbyGPT.pt")`
- [ ] Save the tokenizer beside it, in the same code path
- [ ] Assert `tokenizer.get_vocab_size()` matches the checkpoint's embedding rows on load
- [x] Discriminating test passes
- [ ] Gate the save on **val** loss, not training loss (see task 3)

Measured:

```
state_dict loads into fresh AbbyGPT: OK, 171 tensors
backward() works: 171/171 params got grads
embedding rows: (1605, 128)   vocab: 1605
10 MB (vs 63 MB for the exported program)
```

`bpe_tokenizer.json` and `AbbyGPT.pt` both live in `src/` but are written by different
code at different times, and nothing enforces that they agree. A regenerated tokenizer
with a *different* `vocab_size` makes `load_state_dict` throw — loud, fine. Same size
with a different merge order loads clean and emits garbage — silent. Hence the two
open boxes above.

`torch.export.save` produces an `ExportedProgram` — a frozen inference graph. No `train()`, no gradients, not loadable into `AbbyGPT`. **SFT literally cannot start from it.**

```python
torch.save(model.state_dict(), "AbbyGPT.pt")
```

Keep the export too if you want, but `state_dict` is the SFT input. Also ~9 MB vs 63 MB.

Tokenizer and weights must ship as a **pair**. `fit`/BPE training assigns ids by first-appearance order, so a re-derived tokenizer can have the same `vocab_size` and a different mapping — the model then runs and emits garbage, with no error.

**Why second:** without it there is no artifact to fine-tune, so nothing below can be tested end to end.

**Discriminating test:** load the `state_dict` into a fresh `AbbyGPT`, run one `loss.backward()`, confirm gradients are non-`None`.

---

### 3. Add a validation split

**File:** `training.py`, chunk-building loop

- [ ] Hold out the last ~10% of chunks as val
- [ ] Report train **and** val loss every epoch
- [ ] Wrap val in `torch.no_grad()` and `model.eval()`, restore `model.train()` after
- [x] Divide epoch loss by `num_batches` — done, `training.py:66`
- [ ] Change the checkpoint gate at `training.py:68` from `average_loss_epoch` to `val_loss`
- [ ] Discriminating test passes (seen the divergence)

Hold out the last ~10% of chunks. Report train and val loss each epoch.

**Why third — before any more training.** With 5,251 tokens you will overfit within a handful of steps. Do task 4 first and you will train longer, watch train loss fall, and have **no way to know** whether the model is learning or memorising. This is the one task whose absence makes every later result uninterpretable.

**Discriminating test:** train long enough that val loss turns upward while train loss keeps falling. If you never see that divergence, the split is not wired in.

---

### 4. Scale the window and the training run

**Files:** `settings.py`

- [ ] `context_length: 16 → 128` (or more). 16 tokens ≈ 60 characters. An instruction plus a response does not fit in 60 characters — hard architectural blocker, not a tuning knob.
- [ ] `epochs: 1 → 100+`. Watch val loss (task 3) to pick the stopping point.
- [ ] `batch_size`: keep at 32 or below; at 512 only 2 batches exist and 20% of chunks get dropped.
- [ ] Add `torch.manual_seed(0)` and re-record the loss landmarks in this file
- [ ] Attention-vs-reference check re-run at the new `context_length`
- [ ] Val loss below the 6.8165 unigram floor on **held-out** data

Constraints to respect:
- `flash_attention.py` asserts `context_length % tile_size == 0`. `tile_size = 2`, so any even value works.
- `pos_enc.py` builds its table from `settings.context_length`, so it grows automatically. Any `L` beyond the table crashes loudly on broadcast — good, leave it loud.
- Longer `L` means more tile iterations in the Python loop at `flash_attention.py:42`. Expect the run to get noticeably slower.

**Discriminating test:** re-run the attention-vs-reference check at the new `context_length` across lengths `1, 2, 3, 5, 7, 15, 16` and a few partial tiles near the new max. Expect ~1e-7 in float32. `L=1` and any `L` divisible by `tile_size` pass even with broken tiling — not evidence.

---

### 5. Shuffle the batches

**File:** `training.py`, batch-building loop

- [ ] Build flat `(n_chunks, L)` once instead of a frozen `(n_batches, B, L)`
- [ ] Draw a fresh `torch.randperm` each epoch
- [ ] Discriminating test passes (epoch 1 batch 0 ≠ epoch 2 batch 0)

Currently batch 0 is literally the first 128 contiguous characters of the book, split 8 ways, and the order is identical every epoch. Effective batch size is far below the nominal one.

Build flat `(n_chunks, L)` once, then draw a fresh permutation each epoch:

```python
perm = torch.randperm(x.shape[0], device=x.device)
for b in range(0, x.shape[0] - batch_size + 1, batch_size):
    idx = perm[b : b + batch_size]
    xb, yb = x[idx], y[idx]
```

**Silent-failure warning:** shuffling wrong produces no error. `xs.shape` is unchanged and loss still falls. The only discriminating check is on content:

```python
assert not torch.equal(epoch1_first_batch, epoch2_first_batch)
```

---

### 6. Put `[EOS]` into the token stream

**File:** whatever script builds `bpe_tokenizer.json`

- [ ] Add a `TemplateProcessing` post-processor wrapping each example
- [ ] Confirm `[EOS]` (id 1) actually appears in the encoded stream
- [ ] Update the lossless assert to the new invariant — do **not** loosen it
- [ ] Re-save `bpe_tokenizer.json` and re-run the round-trip check
- [ ] Discriminating test passes

Currently `post_processor: None`, and `[BOS]`/`[EOS]`/`[PAD]` (ids 0/1/2) **never appear in the data** — verified. They are dead vocab rows.

Without `[EOS]` the model cannot learn where a response ends. Generation runs to the token budget and stops mid-word. This is the single clearest difference between a base LM and something SFT-able.

Add a `TemplateProcessing` post-processor that wraps each training example.

**Expect this to break the lossless assert — that is correct.** Injecting specials means `decode(encode(x)) != x` by design. Update the assert to the new invariant. Do **not** loosen it to `.lstrip()` or `skip_special_tokens` hand-waving; that hides real corruption. Verified: a `lstrip`-based assert passes even on doubled-space input.

**Discriminating test:** `assert tokenizer.encode(example).ids[-1] == EOS_ID` on a sample of formatted examples.

---

### 7. Label masking in the loss

**File:** `training.py:49`

- [ ] `CrossEntropyLoss(ignore_index=-100)`
- [ ] Set `y = -100` at every **prompt** position
- [ ] Set `y = -100` at every **pad** position
- [ ] Right-pad (never left-pad — shifts positions and lets pads be attended)
- [ ] Read logits at `n - 1`, never `-1`
- [ ] Discriminating test passes

Do all six in **one** change — see the warning below.

SFT trains on **response tokens only** — no gradient on the instruction you fed in.

```python
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
```

Then set `y = -100` at every prompt position and every pad position.

**Silent-failure warning — this is the worst one in the list.** If you right-pad with `[PAD]` (id 2) and forget the `-100`, the loss happily trains the model to predict padding. No error, no crash, loss even looks *lower* because padding is trivially predictable. Set `ignore_index` and the masking together, in one change.

You do **not** need a separate attention pad-mask. Measured in this repo: with right-padding, row `n-1` of a padded batch matches the unpadded model to 1e-6, because the causal mask already prevents pads from reaching real positions. Right-pad + `ignore_index` is sufficient for decoder-only SFT.

**Discriminating test:** build a batch where the response is one token. Loss must equal the cross-entropy of that single position. If it is lower, masking is not applied.

Trap: a batch with no padding and no prompt passes with masking completely broken.

---

### 8. Grow the corpus

- [ ] Base corpus grown beyond `verdict.txt`
- [ ] Loss landmarks in this file re-measured on the new corpus (vocab and unigram entropy both change)

`verdict.txt` is 5,251 tokens. For 2.18M params, a Chinchilla-ish rule of thumb wants ~43M — four orders of magnitude more. Treat as an order-of-magnitude illustration, not a law, but the conclusion holds: **this corpus cannot produce a language-capable base model.**

Fine for learning the mechanics. If SFT is a real goal, the base corpus has to grow before tasks 3-7 mean anything.

---

## Deliberately excluded

On the known-issues list and **not** SFT blockers — they affect pretraining and SFT equally:

- `MultiHeadAttention` has no output projection (`W_o`).
- One shared `RMSNorm` instance reused at every norm site.
- `inference.py` empty-prompt `IndexError`.
- Debug prints in `inference.py`.
- `torch.empty` → `zeros` for the generation buffer.

## Also note

- `torch.export` **cannot** make the sequence dim dynamic on this model. Verified: the Python tile loop at `flash_attention.py:42` unrolls at trace time and welds `L` to a constant. Swapping in a vectorised attention makes it work — but the hand-rolled tile loop is the point of this repo. So: export with dynamic batch only, pad to `context_length`, and read logits at `n - 1` (**not** `-1`). Measured exact to 1e-6, including mixed-length batches.
- `settings.py` sets no random seed. The old CLAUDE.md landmark "first-step loss ~4.16" is not reproducible without one — measured 4.16–4.36 across seeds. Add `torch.manual_seed(0)` and re-record before relying on any tripwire in this document.
