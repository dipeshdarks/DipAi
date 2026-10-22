# Implementation report

What was built, in what order, and the evidence behind each claim. Phases 1–8
were executed sequentially on the target machine (**RTX 2050 4 GB, 6 GB RAM,
Windows 11, Python 3.10, torch 2.14.1+cu126, transformers 4.57.6**); each
phase ran its focused tests before the next began. Final verification: the
full suite reports **200 passed, 1 skipped** (baseline before this work:
126 passed, 1 skipped — so 74 tests were added, none removed).

The single skip is the real-GPU vision smoke test, which runs only under
`DIPAI_VISION_SMOKE=1`.

## Phase 1 — audit and baseline

- Established the starting state: library, webui, vision, and the existing
  test suite all runnable; no lint config in the repo, so `ruff` (355
  pre-existing findings) is advisory, not a gate.
- Baseline recorded and re-verified: **126 passed, 1 skipped**.
- Environment constraints noted and respected throughout: CPU full-param
  training of 0.5B models is impractical here; complex `python -c` one-liners
  break in this PowerShell (temp script files instead); checkpoints under
  `checkpoints/` (3.8 GB, 7 runs) are never modified or deleted.

## Phase 2 — licensed, bounded data collection

Files: `dipai/data/registry.py`, `manifest.py`, `collect.py`,
`configs/data.yaml`.

- **Source registry** — six sources (`fineweb`, `fineweb-2`, `fineweb-edu`,
  `oasst1`, `wikipedia-20231101-en`, `wikimedia-dump-en-stub`) each carrying
  SPDX licence metadata, attribution/share-alike flags, HF id or dump URL,
  and pilot flags. Pure metadata: registering never downloads.
- **Collector** — streaming HF reader (`streaming=True`, never
  materialises), bounded Wikimedia range-request reader (hard
  `max_dump_bytes` ceiling), retry/backoff, `.progress.json` resume, and a
  `CollectManifest` that records licence, revision, counts, and a rolling
  sha1 digest per source.
- **Caps are deliberate** (`configs/data.yaml`): 500 docs / 20 MB per
  source, pilot mode on. Nothing in this project downloads a multi-TB
  corpus.
- **Verified offline**: `--list` prints the registry; `--dry-run` plans a
  fineweb-edu pilot without network; unknown source names fail with the
  known-name list. Live network collection was not exercised during
  development (dry-run + unit-level verification only).

## Phase 3 — deterministic cleaning

File: `dipai/data/clean.py` (CLI: `python -m dipai.data.clean`).

Stages, in fixed order, each counting its drops: NFKC normalisation →
empty/encoding/length filters → boilerplate/spam → stopword language ID
(emails/URLs stripped first so `example.com` does not pollute tokens) →
exact sha1 dedup → optional banded-MinHash near-dup → heuristic quality
score → PII detect/redact (emails, phones, IPv4, card-like numbers, SSNs) →
contamination n-gram check → `CleanReport` (JSON + optional Markdown).

Fixed during development: a set-mutation-during-iteration bug in exact dedup
(drops collected in `to_drop` first), import placement.

Evidence: `tests/test_clean.py` — **26 passed** (every stage plus report
contents and CLI paths).

## Phase 4 — prepare (JSONL → train/val bins)

File: `dipai/data/prepare.py` (CLI: `python -m dipai.data.prepare`).

Thin glue over the existing, tested `dipai.dataset.prepare_dataset`:
clean → hash-split (`sha1(seed + text)`, so duplicates never cross the
split) → SentencePiece tokenize → pack `train.bin` / `val.bin`, plus
`dataset_stats.json` and a `manifest.json` recording inputs, tokenizer,
block size, seed, counts, and the cleaning report — reproducibility lives in
the manifest.

Evidence: `tests/test_prepare.py` — **4 passed**, including an end-to-end
run with a real (tiny) SentencePiece tokenizer and the empty-input refusal
(`InvalidDatasetError`).

## Phase 5 — training modes A/B/C/D

Files: `dipai/train.py`, `finetune.py`, `config.py`, `inference.py`,
`chat.py`.

- **Reproducible resume** — torch/numpy/python RNG states are captured at
  checkpoint time and restored on resume, so a resumed run continues the
  interrupted run's random stream.
- **Mode switch in `finetune.py`** — `use_lora: true` (default) keeps the
  PEFT adapter path; `use_lora: false` trains *all* parameters and saves
  `out_dir/model`; config validation refuses full-param + 4-bit and
  full-param + `learning_rate > 1e-4` before any weight loads.
- **VRAM estimator** branches for full-param (~12 bytes/param optimizer
  cost) vs LoRA, so the dry run tells the truth on a 4 GB card.
- **Loader + CLI** — `DIPAIEngine.from_model()` (HF full model, backend
  `hf-full`) and `chat.py --model`, giving chat the same three-backend
  surface the webui has.
- Fixed: a `quant_config = None` initialisation bug on the non-4-bit path.

Evidence: `tests/test_phase5_modes.py` — **7 passed**: both config rules,
the estimator's full-param > LoRA ordering, a mocked-`Trainer` full-param
save → load → generate round-trip on a tiny llama config, the LoRA branch
still wrapping PEFT, and the RNG save/restore determinism round-trip.
(Honesty note: full-parameter training of real-size models was never run
here — the path is proven with tiny models and a mocked trainer, not with a
multi-hour 0.5B run.)

## Phase 6 — evaluation

File: `dipai/evaluate.py` (CLI: `python -m dipai.evaluate`).

- Two backends: **scratch** (`DipaiGPT.forward` loss, pad-aware targets with
  `-100`, token-weighted mean NLL across batches) and **hf**
  (`AutoModelForCausalLM`, optional `--apply-chat-template` so SFT models
  are scored as they are used, optional `--load-in-4bit`).
- Perplexity = `exp(mean loss)` (capped at `exp(20)` against overflow).
  Documents are truncated to `--max-length`; this is stated in the docs, not
  hidden.
- **Base-vs-candidate comparison** with `loss_delta`, `perplexity_delta`,
  `perplexity_ratio`, `improved`, plus a `warnings` list that fires when the
  two sides used different datasets or tokenizers — logged by the CLI and
  printed as **Caveats** in the Markdown report (added after the docstring's
  "warn loudly" promise was found unimplemented in code — code now matches
  the promise).
- Reports: JSON (with per-example losses + generation samples) and
  Markdown. Exit codes: 0 ok, 1 evaluation failed, 2 no model given; a
  failed *base* run degrades to candidate-only with a warning.

Evidence: `tests/test_evaluate.py` — **13 passed**: JSONL reading, scratch
perplexity sanity (finite; random model ⇒ high ppl ≈ vocab; token counts
scale), end-to-end checkpoint evaluation with generation samples, compare
improvement/regression, dataset-mismatch and tokenizer-mismatch warnings,
JSON/Markdown reports including the Caveats section, HF backend on a tiny
model, and the no-chat-template HF path.

## Phase 7 — model registry and CLI/webui wiring

Files: `dipai/model_registry.py`, `dipai/chat.py`, `webui/app.py`,
`dipai/inference.py`.

- **Registry** mirroring the data registry's shape: `RegisteredModel`
  (name, backend ∈ {`scratch`, `hf`, `hf-full`}, path, provenance fields,
  `exists()`), built-in `scratch` entry as default, `resolve`/`available`/
  `summary_json`.
- **`--list-models`** in both CLIs prints name / backend / path / on-disk.
- **Backend-aware loading** — `DIPAIEngine.from_named()` resolves a registry
  name and dispatches to *its* backend's loader; unknown values fall back to
  the legacy `--model <HF dir>` behaviour (raw path → `from_model`).

**Bug found and fixed during this phase:** the first wiring resolved a
registry name to its path but always called `from_model`, ignoring the
entry's backend — `--model scratch` would have failed with "no config.json".
The `from_named` dispatch, the pure `resolve_for_loading` helper, and
regression tests (`from_named` routes `scratch` → `from_checkpoint`, raw
paths stay `hf-full`, `hf` → `from_adapter`) now pin the correct behaviour.

Evidence: `tests/test_model_registry.py` — **18 passed** (registration,
default flag, validation, existence filtering, JSON round-trip,
`resolve_for_loading`); `tests/test_chat_registry.py` — **6 passed**
(`--list-models` parsing and CLI run, `from_named` dispatch, empty-value
refusal). Live end-to-end check: `python -m dipai.chat --model scratch
--once "the quick brown fox"` loaded the real checkpoint (21.11M params,
step 6000, CUDA) and generated text.

## Phase 8 — docs, notebook, README, final verification

- **Six required documents**, each written against the code rather than from
  memory, with claims re-checked against the source while writing (this is
  how the missing `--model` dispatch bug and several stale README commands
  were caught):
  - `DATA_SOURCES_AND_LICENSES.md` — per-source licence table,
    attribution/share-alike implications, bounded-collection policy.
  - `TRAINING_GUIDE.md` — modes A–D, exact commands, config-validation
    rules, registry usage, resume pointers.
  - `CLOUD_GPU_TRAINING.md` — provisioning, credentials hygiene, data
    strategy, tmux, checkpoints home, cost discipline (vendor-neutral, no
    prices).
  - `CHECKPOINTS_AND_RESUMING.md` — atomic writes, pointer semantics,
    retention, RNG restore, per-mode resume, interrupt safety, exit codes.
  - `EVALUATION.md` — metric semantics, comparability rules, full CLI
    reference, frozen-eval-set workflow.
  - `IMPLEMENTATION_REPORT.md` — this file.
- **Stale README claims fixed** while updating: the old `--adapter
  checkpoints\finetune\latest` path (real adapter dir is
  `checkpoints\finetune\adapter`), a nonexistent `--vision-load-on-startup`
  flag (eager loading is the `load_on_startup` config key), and the old
  test count.
- **Notebook** `notebooks/quickstart.ipynb` (12 cells): hardware probe,
  licence registry listing, dataset stats inspection, a real 2-example
  evaluation of `checkpoints/smoke`, model-registry listing — network- and
  training-free by design. Because Jupyter is not a dependency, it ships
  with `notebooks/verify_notebook.py`, which parses the notebook JSON and
  executes every code cell in-process. **Verified: all 6 code cells executed
  in 24.4 s** (the eval cell wrote its report and printed
  `ppl=2764.82` on 2 documents of the tiny smoke model).
- `.gitignore` extended with `reports/` (regenerable eval outputs).
- **Final full-suite run: 200 passed, 1 skipped.**

## Test evidence summary

| Area | File | Tests |
|---|---|---|
| Data registry + collection planning | CLI-verified (`--list`, `--dry-run`) | — |
| Cleaning stages | `tests/test_clean.py` | 26 |
| Prepare end-to-end | `tests/test_prepare.py` | 4 |
| Training modes (config, estimator, save/load, RNG) | `tests/test_phase5_modes.py` | 7 |
| Evaluation (backends, reports, warnings) | `tests/test_evaluate.py` | 13 |
| Model registry | `tests/test_model_registry.py` | 18 |
| CLI wiring + backend dispatch | `tests/test_chat_registry.py` | 6 |
| Pre-existing suite (model, tokenizer, dataset, train recovery, checkpoints, sft, generate, inference, hardware, config, webui, vision, …) | 15 files | 126 |
| **Total** | | **200 passed, 1 skipped** |

## What was deliberately *not* done

- **No multi-terabyte downloads.** Collection caps stay at pilot size; live
  network collection was not run during development (offline `--list` /
  `--dry-run` verification only).
- **No long training runs.** Mode A/B/C/D code paths are proven with tiny
  models, mocked trainers, and config validation — not by burning days of
  GPU time. Existing checkpoints were only *read* (eval/chat), never
  modified or overwritten.
- **No credentials anywhere.** No tokens in configs, commits, or docs.
- **No lint gate.** The repo has no lint config; `ruff` reports 355
  pre-existing findings that predate this work, so it was not used as a pass
  criterion (it was consulted on new code for obvious errors only).
- **No architecture changes.** The model, its configs, and its checkpoint
  format are untouched; new features wrap the existing training/eval stack.

## How to re-verify everything yourself

```bat
.venv\Scripts\pytest.exe                      :: 200 passed, 1 skipped (~1-6 min)
python notebooks\verify_notebook.py           :: executes every notebook cell
python -m dipai.chat --list-models            :: registry resolves
python -m dipai.data.collect --config configs\data.yaml --dry-run   :: no network
python -m dipai.chat --model scratch --once "hello"                 :: loads + generates
```
