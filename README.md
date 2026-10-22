# DIPAI

A local, memory-efficient language-model toolkit for consumer GPUs — ingest
licensed public web data (bounded and auditable), train a decoder-only
Transformer from scratch, fine-tune it (LoRA/QLoRA or full-parameter),
evaluate it, chat with it in the terminal or a web UI, and ask questions
about images.

Built and measured on a laptop with an **NVIDIA RTX 2050 (4 GB VRAM), 6 GB RAM,
Windows 11, Python 3.10**. Every number in this README was measured on that
machine.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Data pipeline — licensed, bounded, reproducible](#data-pipeline--licensed-bounded-reproducible)
- [Mode A — train from scratch](#mode-a--train-from-scratch)
- [Mode B — fine-tune (LoRA / QLoRA)](#mode-b--fine-tune-lora--qlora)
- [Modes C and D — full-parameter runs](#modes-c-and-d--full-parameter-runs)
- [Evaluate](#evaluate)
- [Chat](#chat)
- [Web UI](#web-ui)
- [Image understanding (vision)](#image-understanding-vision)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **Licensed, bounded data ingestion** — a registry of public corpora
  (FineWeb/FineWeb-Edu, OpenAssistant, Wikipedia) with SPDX licence metadata
  attached to every source and recorded in every collection manifest.
  Collection streams with hard per-source caps (500 docs / 20 MB pilots — no
  multi-TB downloads), retries with backoff, and resumes from a progress file.
- **Deterministic cleaning** — NFKC normalisation, encoding/spam/boilerplate
  filters, stopword language ID, exact + optional MinHash near-dup removal,
  heuristic quality scoring, PII redaction, and a JSON/Markdown audit report
  of every drop. No neural models, no hidden randomness.
- **Train from scratch** — a standard GPT-style decoder-only Transformer
  (~21 M parameters in the default config) with a SentencePiece tokenizer,
  packed binary dataset, cosine LR schedule, gradient accumulation, periodic
  checkpoints (atomic, with RNG state), and automatic resume. Pure PyTorch;
  no training framework.
- **Four training modes** — Mode A from-scratch pretraining; Mode B
  LoRA/QLoRA adapter fine-tuning of a HF base; Mode C full-parameter
  continued-pretraining (`use_lora: false`, config-validated); Mode D
  full-parameter supervised fine-tuning of a local checkpoint with loss
  masked on user turns. See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).
- **Evaluate** — perplexity on held-out JSONL with a scratch or Hugging Face
  backend, base-vs-candidate delta reports (JSON + Markdown), and loud
  warnings when the two sides used different data or tokenizers.
- **Model registry** — logical names for model directories, resolved to the
  right inference backend; `--list-models` in both CLIs.
- **Chat** — streaming terminal chat with multi-turn memory, sampling
  controls, and optional LoRA adapter.
- **Web UI** — Gradio app with a streaming chat tab and an image-understanding
  tab.
- **Image understanding** — SmolVLM-256M-Instruct (Apache-2.0) loaded lazily;
  upload a JPEG/PNG/WebP and ask a question. The text chat never pays the
  memory cost of the vision model.
- **Memory-aware** — hardware probing, VRAM budgeting, `low_cpu_mem_usage`,
  and CUDA-OOM fallback to CPU are built in, because the target machine is a
  4 GB laptop GPU.

## Quick start

```bat
:: 1. Environment (Python 3.10, Windows)
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

:: 2. Check the hardware is seen correctly
python -m dipai.hardware

:: 3. Get data, prepare it, train, chat
python -m dipai.data.collect --config configs\data.yaml     :: bounded, licensed pilot
python -m dipai.data.clean   --input data\raw\web\*.jsonl --output data\clean\clean.jsonl ^
    --report data\clean\report.json                         :: deterministic cleaning
python -m dipai.data.prepare  --inputs data\clean\clean.jsonl --output-dir data\processed ^
    --tokenizer data\processed\tokenizer.model --block-size 256
python -m dipai.train --config configs\scratch.yaml
python -m dipai.chat --ckpt checkpoints\scratch
```

Linux/macOS: use `python`/`pip` directly and drop the `.bat`-style paths.

## Data pipeline — licensed, bounded, reproducible

```bat
python -m dipai.data.collect  --config configs\data.yaml --list     :: sources + licences
python -m dipai.data.collect  --config configs\data.yaml --dry-run  :: plan, no network
python -m dipai.data.collect  --config configs\data.yaml            :: collect (pilot caps)

python -m dipai.data.clean    --input data\raw\web\*.jsonl --output data\clean\clean.jsonl ^
    --report data\clean\report.json --markdown                      :: deterministic cleaning

python -m dipai.data.prepare  --inputs data\clean\clean.jsonl --output-dir data\processed ^
    --tokenizer data\processed\tokenizer.model --block-size 256 --seed 1337
                                                                      :: train.bin / val.bin + manifest
```

Every source carries SPDX licence metadata (attribution / share-alike flags)
that lands in each collection's manifest; caps are tiny on purpose
(`configs/data.yaml`: 500 docs, 20 MB per source, pilot mode on). Details:
[`docs/DATA_SOURCES_AND_LICENSES.md`](docs/DATA_SOURCES_AND_LICENSES.md).

A guided tour without running anything heavy:
[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)
(verify with `python notebooks/verify_notebook.py`).

## Mode A — train from scratch

DIPAI trains a ~21 M-parameter decoder-only Transformer (6 layers, 8 heads,
512 embedding width, tied embeddings, 256-token context) on a packed token
dataset.

```bat
:: Prepare the dataset (cleans, trains a SentencePiece tokenizer, packs .bin)
python scripts\download_training_data.py     :: fetch source corpora
python scripts\prepare_data.py               :: -> data/processed/train.bin, val.bin

:: Train (auto-resumes from the latest checkpoint if one exists)
python -m dipai.train --config configs\scratch.yaml

:: Options
python -m dipai.train --max-iters 2000       :: shorter run
python -m dipai.train --no-resume            :: start fresh
python -m dipai.train --data-dir data\processed --ckpt-dir checkpoints\scratch
```

Configs live in `configs/`:

| Config | Purpose |
|---|---|
| `scratch.yaml` | default ~21 M from-scratch run |
| `dipai_50m.yaml` | larger ~50 M variant |
| `smoke_4m.yaml` | tiny 4 M config for fast smoke tests |
| `sft_oasst.yaml` | supervised fine-tune on OpenAssistant |
| `finetune.yaml` | LoRA/QLoRA fine-tune |
| `vision.yaml` | vision model / image limits |

Estimate the parameter count of a config before committing to a run:

```bat
python scripts\estimate_model.py --config configs\scratch.yaml
```

## Mode B — fine-tune (LoRA / QLoRA)

Fine-tune a chat adapter on top of a base checkpoint with Hugging Face PEFT.
QLoRA (4-bit, `bitsandbytes`) keeps the footprint small enough for a 4 GB GPU;
a dry-run VRAM check runs before training unless you pass `--train`.

```bat
python scripts\prepare_finetune_data.py      :: build chat-format JSONL
python -m dipai.finetune --config configs\finetune.yaml            :: dry run: VRAM estimate
python -m dipai.finetune --config configs\finetune.yaml --train    :: train -> <output_dir>\adapter
```

Useful flags: `--max-steps`, `--max-samples`, `--output-dir`, `--resume`.

## Modes C and D — full-parameter runs

- **Mode C** — continued-pretraining of a Hugging Face base with *all*
  parameters trainable: copy `configs/finetune.yaml`, set `use_lora: false`
  and `load_in_4bit: false`. Config validation refuses the 4-bit combination
  and any learning rate above 1e-4 (full-param updates need a gentler LR).
  Output: `<output_dir>\model`, loadable with `--model`.
- **Mode D** — full-parameter supervised fine-tuning of a *local* DIPAI
  checkpoint on conversations, with loss masked on user turns:

  ```bat
  python scripts\prepare_finetune_data.py --oasst-threads   :: -> data\processed\sft_*.jsonl
  python -m dipai.finetune --config configs\sft_oasst.yaml            :: dry run
  python -m dipai.finetune --config configs\sft_oasst.yaml --train    :: train
  ```

Both branches, their resume behaviour, and their VRAM math:
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).

## Evaluate

Perplexity on held-out JSONL, with optional base comparison and generation
smoke samples:

```bat
python -m dipai.evaluate --checkpoint checkpoints\scratch ^
    --dataset data\processed\finetune.jsonl --generate-samples 3 ^
    --base-checkpoint checkpoints\smoke ^
    --out-json reports\eval.json --out-md reports\eval.md
```

The report carries loud caveats when candidate and base used different data
or tokenizers — perplexity is only comparable within one eval setup. Details:
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## Chat

```bat
:: interactive multi-turn chat (streams tokens)
python -m dipai.chat --ckpt checkpoints\scratch

:: with a fine-tuned adapter on top
python -m dipai.chat --ckpt checkpoints\scratch --adapter checkpoints\finetune\adapter

:: single question, no REPL
python -m dipai.chat --ckpt checkpoints\scratch --once "your question"

:: sampling controls
python -m dipai.chat --temperature 0.8 --top-k 40 --top-p 0.95 --max-new-tokens 256

:: model registry (logical names instead of paths)
python -m dipai.chat --list-models
python -m dipai.chat --model scratch --once "hello"
```

Precedence: `--model` > `--adapter` > `--ckpt`. Slash commands in the REPL:
`/new` (reset conversation), `/save`, `/exit`.

## Web UI

```bat
python -m webui.app                      # http://127.0.0.1:7860
python -m webui.app --vision-config configs\vision.yaml
python -m webui.app --model scratch      # any registry name or model dir
python -m webui.app --list-models
```

Eager vision loading is a config key (`load_on_startup` in
`configs/vision.yaml`), not a CLI flag.

Two tabs:

- **Chat** — streaming text chat with per-request sampling controls and a
  stop button.
- **Image understanding** — upload an image, type a question, get an answer.

The vision model loads lazily on first use (or eagerly with the flag above);
the text tab works whether or not it is ever loaded.

## Image understanding (vision)

Powered by [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
(Apache-2.0, 256 M parameters) — a small, commercially usable vision-language
model that runs in ~0.6 GB of VRAM on the target GPU. Native in
`transformers>=4.49`; no custom code, no `trust_remote_code`.

```bat
:: one-shot question about an image
python -m dipai.chat --image data\vision_samples\red_square.png --prompt "what color is this?"

:: the vision module directly (also has its own CLI)
python -m dipai.vision --image photo.jpg --prompt "describe this"
python -m dipai.vision --image scan.png --prompt "read the text" --max-new-tokens 128
```

Accepted inputs: JPEG, PNG, WebP — file path, PIL image, or raw bytes.
Uploads are capped by `configs/vision.yaml` (`max_upload_mb`, `max_image_side`,
`max_image_pixels`); remote URLs are rejected and image contents are never
written to logs.

Measured on the RTX 2050: first load ≈ 15 s (downloads ~1 GB to the Hugging
Face cache), inference 1–17 s per image, ~0.6 GB VRAM while loaded, CPU
fallback on CUDA OOM.

Details, troubleshooting, and limitations: [`docs/VISION.md`](docs/VISION.md).

## Testing

```bat
.venv\Scripts\pytest.exe tests\ -q                                        :: full suite
.venv\Scripts\pytest.exe tests\ -q --ignore=tests\test_vision.py          :: text-only
.venv\Scripts\pytest.exe tests\test_vision.py -q                          :: vision (mocked)

:: real-GPU vision smoke test (downloads/loads the model, ~1 min)
set DIPAI_VISION_SMOKE=1
.venv\Scripts\pytest.exe tests\test_vision.py -q -k gpu_smoke
```

Current status: **200 passed, 1 skipped** (the GPU smoke test is skipped
unless `DIPAI_VISION_SMOKE=1`). The suite covers model correctness (causal
masking, parameter counts), tokenizer round-trips, dataset packing, training
recovery/resume (including RNG-state determinism), generation, inference,
hardware probing, the web UI handlers, the vision pipeline, the data pipeline
(source-licence registry, bounded collection planning, every cleaning stage,
prepare end-to-end), the four training modes (config rules, VRAM estimator,
full-param save/load round-trip), evaluation (perplexity backends, delta
reports, comparability warnings), and both registries (24 registry tests,
backend-aware loading dispatch).

## Project layout

```
DIPAI/
├── dipai/                  # the library
│   ├── model.py            # decoder-only Transformer (from scratch)
│   ├── tokenizer.py        # SentencePiece wrapper
│   ├── dataset.py          # cleaning, packing, .bin mmap dataset
│   ├── data/               # licensed ingestion pipeline
│   │   ├── registry.py     #   sources + SPDX licence metadata
│   │   ├── collect.py      #   bounded, resumable, streaming collectors
│   │   ├── clean.py        #   deterministic cleaning + PII + dedup
│   │   ├── manifest.py     #   provenance manifests
│   │   └── prepare.py      #   JSONL -> train.bin / val.bin + manifest.json
│   ├── train.py            # Mode A training loop (RNG-preserving resume)
│   ├── finetune.py         # Modes B/C/D (LoRA / full-param / native SFT)
│   ├── evaluate.py         # perplexity + base comparison reports
│   ├── model_registry.py   # logical model name -> (backend, path)
│   ├── generate.py         # sampling (top-k, top-p, temperature, repetition)
│   ├── inference.py        # streaming chat engine (from_* loaders)
│   ├── vision.py           # image understanding (SmolVLM)
│   ├── chat.py             # terminal chat CLI
│   ├── config.py           # YAML config dataclasses
│   ├── checkpoints.py      # atomic save / load / resume
│   ├── hardware.py         # GPU / VRAM probing
│   └── utils.py            # logging, seeding, OOM guard
├── webui/app.py            # Gradio app (chat + image tabs)
├── configs/                # YAML configs (see table above; + data.yaml)
├── scripts/                # data download, prep, helpers (Python + PowerShell)
├── notebooks/              # quickstart.ipynb + verify_notebook.py
├── tests/                  # pytest suite (200 tests)
├── docs/                   # six workflow docs + VISION.md
└── data/                   # raw/processed data + small vision samples
```

Model weights, datasets, evaluation reports, and logs are git-ignored —
everything regenerable is regenerated by the scripts, not committed.

## Documentation

- [`docs/DATA_SOURCES_AND_LICENSES.md`](docs/DATA_SOURCES_AND_LICENSES.md) —
  every registered corpus: licence (SPDX), attribution/share-alike flags,
  why collection is bounded, how provenance is recorded.
- [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) — the four training
  modes (A/B/C/D) end-to-end: data → train → fine-tune → registry → evaluate,
  with the exact commands and config-validation rules.
- [`docs/CLOUD_GPU_TRAINING.md`](docs/CLOUD_GPU_TRAINING.md) — when the
  laptop is not enough: provisioning checklist, credentials hygiene, data
  strategy, tmux runs, getting checkpoints home.
- [`docs/CHECKPOINTS_AND_RESUMING.md`](docs/CHECKPOINTS_AND_RESUMING.md) —
  atomic saves, the `latest.json` pointer, retention, per-mode resume,
  interrupt safety, what is inside a checkpoint.
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — perplexity semantics,
  comparability rules, CLI reference, building a frozen eval set.
- [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md) — what was
  built phase by phase and the test evidence behind each claim.
- [`docs/VISION.md`](docs/VISION.md) — image understanding: model choice,
  install, measured memory, config reference, troubleshooting, limitations.
- [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — guided tour
  (registry, dataset stats, a fast eval); verify with
  `python notebooks/verify_notebook.py`.
- `configs/*.yaml` — every training/inference knob, commented inline.

## License

MIT — see [`pyproject.toml`](pyproject.toml).

The SmolVLM vision model is Apache-2.0 (Hugging Face TB). Training corpora
retain their own licenses — see
[`docs/DATA_SOURCES_AND_LICENSES.md`](docs/DATA_SOURCES_AND_LICENSES.md);
every collection manifest records the licence that applied to the data it
collected.
