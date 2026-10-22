# Evaluation

`dipai.evaluate` scores a model with **perplexity** (the canonical intrinsic
language-model metric) plus optional generation smoke samples, and can compare
a candidate against a base model with a delta table. Implementation:
[`dipai/evaluate.py`](../dipai/evaluate.py); tests: `tests/test_evaluate.py`.

## What perplexity means here

Perplexity is `exp(mean cross-entropy loss)` over a held-out token set — the
geometric-mean "surprisal per token" of the model. Lower is better.

Hard comparability rules (the evaluator enforces/warns on them):

- Compare only across checkpoints evaluated on the **same JSONL file, the same
  tokenizer, and the same settings** (`max_length`, chat-template flag).
- Different tokenizer → different vocabulary → losses are on different scales.
  Different eval set → different text difficulty. `compare()` records both
  situations in a `warnings` list, the CLI logs them as warnings, and the
  Markdown report prints them under **Caveats**.
- Numbers are token-weighted: each batch's mean loss is multiplied by its
  count of real (non-pad, non-ignored) target tokens before the final
  division, so batching never changes the result.

What it is **not**: a benchmark of downstream tasks, and not comparable to a
perplexity printed by some other tool unless you reproduce its exact
tokenization and eval text.

## Quick start

```bat
:: scratch checkpoint (Mode A / native D output)
python -m dipai.evaluate --checkpoint checkpoints\scratch ^
    --dataset data\processed\finetune.jsonl ^
    --out-json reports\eval.json --out-md reports\eval.md

:: compare a candidate against a base, on the same eval set
python -m dipai.evaluate --checkpoint checkpoints\sft50m_v2 ^
    --dataset data\processed\finetune.jsonl ^
    --base-checkpoint checkpoints\50m ^
    --generate-samples 3 --generate-prompt "The meaning of life is" ^
    --out-json reports\sft_vs_base.json --out-md reports\sft_vs_base.md

:: HF model (Mode B adapter base or Mode C dir)
python -m dipai.evaluate --hf-model Qwen/Qwen2.5-0.5B-Instruct ^
    --dataset data\processed\finetune.jsonl --limit 50
```

The dataset is **JSONL with a `"text"` field**, one document per line
(`data/processed/finetune.jsonl` is one). Conversation files with a
`"messages"` field (e.g. `sft_val.jsonl`) must be flattened to plain text
first — the evaluator deliberately does not guess schemas.

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | — | DIPAI scratch checkpoint dir (backend `scratch`) |
| `--hf-model` | — | HF model name/path (backend `hf`); one of the two is required |
| `--dataset` | **required** | JSONL eval file with a `text` field |
| `--device` | `auto` | `auto` \| `cuda` \| `cpu` |
| `--max-length` | `512` | truncate each document to this many tokens |
| `--batch-size` | `1` | texts per forward pass (result is token-weighted, batch-invariant) |
| `--limit` | `0` | evaluate only the first N examples (0 = all) |
| `--apply-chat-template` | off | wrap each text in the HF tokenizer's chat template before scoring (for SFT models scored as they are used) |
| `--load-in-4bit` | off | load the HF model in 4-bit (saves VRAM; HF backend only) |
| `--generate-samples` | `0` | also produce N generation smoke samples |
| `--generate-prompt` | `The meaning of life is` | prompt for those samples |
| `--generate-max-tokens` | `40` | length of those samples |
| `--base-checkpoint` | — | DIPAI base to compare against |
| `--base-hf-model` | — | HF base to compare against |
| `--out-json` | `reports/eval.json` | JSON report path |
| `--out-md` | — | optional Markdown report path |

Exit codes: `0` success, `1` evaluation failed (bad path, OOM, unreadable
checkpoint), `2` neither `--checkpoint` nor `--hf-model` was given. If the
**base** side fails, the run still completes and reports the candidate, with a
logged warning that the comparison was skipped.

## The two backends

| | `scratch` | `hf` |
|---|---|---|
| Loads | `DipaiGPT` from a `dipai.checkpoints` directory | `AutoModelForCausalLM` (+ `--apply-chat-template`, `--load-in-4bit`) |
| Loss | the model's own `forward(input_ids, targets)` | the model's own forward on shifted labels |
| Tokenizer | `tokenizer.model` beside the checkpoint | the model's tokenizer |
| Typical use | Mode A, Mode D checkpoints | Qwen base, Mode B/C dirs, external models |

Both paths: pad-aware targets (`-100` on pads and the final position),
`torch.no_grad()`, model in eval mode.

## Reading the output

**JSON** (`--out-json`) contains: `label`, `backend`, `num_examples`,
`num_tokens`, `mean_loss`, `perplexity`, `per_example_loss[]`,
`tokenizer_vocab`, `dataset`, `generation_samples[]`, `notes`, and — when a
base was supplied — `comparison`:

```json
"comparison": {
  "dataset_match": true,
  "loss_delta": -0.18, "perplexity_delta": -3.4, "perplexity_ratio": 0.87,
  "improved": true,
  "warnings": []
}
```

Negative delta = improvement. `warnings` is non-empty whenever the two sides
were scored on different datasets or tokenizers — treat any such comparison
as informational only.

**Markdown** (`--out-md`) is the same numbers plus the comparison table and a
**Caveats** section echoing the warnings — the file to attach to a report, so
the caveat travels with the number.

**Stdout** gets the full JSON (script-friendly) plus a one-line log summary
and, with a base, the `IMPROVED`/`regressed` verdict.

## Generation smoke samples

`--generate-samples N` runs the model on `--generate-prompt` and records raw
continuations. Use them to catch degenerate output (repetition, garbage) that
a perplexity number hides. They are *smoke* samples, not a scored benchmark.

## Building a good eval set

- **Hold out before packing.** `dipai.data.prepare` hash-splits documents into
  `train.bin` / `val.bin` (no duplicate crosses the split). Keep a JSONL of
  held-out raw text *before* packing so `--dataset` never sees training text.
- **Match the domain of interest.** Perplexity is only informative relative to
  what you care about: score general text for a general model, conversational
  text for an SFT model.
- **Keep it fixed.** Freeze one `eval.jsonl` per project and quote every
  number against it. Mixing eval sets between runs is how teams end up
  comparing 12.3 against 11.8 and believing it.
- **Size.** Tens to hundreds of documents are enough for a stable relative
  signal on small models; use `--limit` for quick checks and the full file for
  the numbers you quote.

## Suggested workflow

```bat
:: 0. once: create the frozen eval set from held-out raw text (never trained on)
::    e.g. copy the val-side documents of your corpus into data\eval\heldout.jsonl
::    (one {"text": ...} record per line)

:: 1. baseline on the frozen eval set
python -m dipai.evaluate --checkpoint checkpoints\50m --dataset data\eval\heldout.jsonl --out-json reports\base.json

:: 2. after training, evaluate the candidate against that same base
python -m dipai.evaluate --checkpoint checkpoints\sft50m_v2 --dataset data\eval\heldout.jsonl ^
    --base-checkpoint checkpoints\50m --generate-samples 3 ^
    --out-json reports\sft_vs_base.json --out-md reports\sft_vs_base.md

:: 3. read the caveats before quoting the number
type reports\sft_vs_base.md
```

If `warnings` is non-empty or `dataset_match` is false, fix the eval setup
before drawing conclusions — that is exactly what the warning is for.

## Related

- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — how the checkpoints being
  evaluated were produced.
- [`CHECKPOINTS_AND_RESUMING.md`](CHECKPOINTS_AND_RESUMING.md) — what a
  checkpoint directory contains and how `--checkpoint` resolves it.
- [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) — what was built and
  the test counts backing these claims.
