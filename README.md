# DIPAI

A local, memory-efficient language-model toolkit for consumer GPUs — train a
decoder-only Transformer from scratch, fine-tune it with LoRA/QLoRA, chat with
it in the terminal or a web UI, and ask questions about images.

Built and measured on a laptop with an **NVIDIA RTX 2050 (4 GB VRAM), 6 GB RAM,
Windows 11, Python 3.10**. Every number in this README was measured on that
machine.

---

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Mode A — train from scratch](#mode-a--train-from-scratch)
- [Mode B — fine-tune (LoRA / QLoRA)](#mode-b--fine-tune-lora--qlora)
- [Chat](#chat)
- [Web UI](#web-ui)
- [Image understanding (vision)](#image-understanding-vision)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **Train from scratch** — a standard GPT-style decoder-only Transformer
  (~21 M parameters in the default config) with a SentencePiece tokenizer,
  packed binary dataset, cosine LR schedule, gradient accumulation, periodic
  checkpoints, and automatic resume. Pure PyTorch; no training framework.
- **Fine-tune** — Hugging Face PEFT LoRA/QLoRA on top of the scratch model or
  any compatible checkpoint, with chat-template conversation packing and VRAM
  pre-flight checks.
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
python scripts\download_training_data.py
python scripts\prepare_data.py
python -m dipai.train --config configs\scratch.yaml
python -m dipai.chat --ckpt checkpoints\scratch
```

Linux/macOS: use `python`/`pip` directly and drop the `.bat`-style paths.

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
python -m dipai.finetune --config configs\finetune.yaml --train
python -m dipai.finetune --config configs\sft_oasst.yaml --train --max-steps 200
```

Useful flags: `--max-steps`, `--max-samples`, `--output-dir`, `--resume`.

## Chat

```bat
:: interactive multi-turn chat (streams tokens)
python -m dipai.chat --ckpt checkpoints\scratch

:: with a fine-tuned adapter on top
python -m dipai.chat --ckpt checkpoints\scratch --adapter checkpoints\finetune\latest

:: single question, no REPL
python -m dipai.chat --ckpt checkpoints\scratch --once "your question"

:: sampling controls
python -m dipai.chat --temperature 0.8 --top-k 40 --top-p 0.95 --max-new-tokens 256
```

Slash commands in the REPL: `/new` (reset conversation), `/save`, `/exit`.

## Web UI

```bat
python -m webui.app                      # http://127.0.0.1:7860
python -m webui.app --vision-config configs\vision.yaml
python -m webui.app --vision-load-on-startup   # eager-load the vision model
```

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

Current status: **126 passed, 1 skipped** (the GPU smoke test is skipped
unless `DIPAI_VISION_SMOKE=1`). The suite covers model correctness (causal
masking, parameter counts), tokenizer round-trips, dataset packing, training
recovery/resume, generation, inference, hardware probing, the web UI handlers,
and the vision pipeline.

## Project layout

```
DIPAI/
├── dipai/                  # the library
│   ├── model.py            # decoder-only Transformer (from scratch)
│   ├── tokenizer.py        # SentencePiece wrapper
│   ├── dataset.py          # cleaning, packing, .bin mmap dataset
│   ├── train.py            # Mode A training loop
│   ├── finetune.py         # Mode B LoRA/QLoRA
│   ├── generate.py         # sampling (top-k, top-p, temperature, repetition)
│   ├── inference.py        # streaming chat engine
│   ├── vision.py           # image understanding (SmolVLM)
│   ├── chat.py             # terminal chat CLI
│   ├── config.py           # YAML config dataclasses
│   ├── checkpoints.py      # save / load / resume
│   ├── hardware.py         # GPU / VRAM probing
│   └── utils.py            # logging, seeding, OOM guard
├── webui/app.py            # Gradio app (chat + image tabs)
├── configs/                # YAML configs (see table above)
├── scripts/                # data download, prep, helpers (Python + PowerShell)
├── tests/                  # pytest suite
├── docs/                   # VISION.md and other docs
└── data/                   # raw/processed data + small vision samples
```

Model weights, datasets, and logs are git-ignored — everything regenerable is
regenerated by the scripts, not committed.

## Documentation

- [`docs/VISION.md`](docs/VISION.md) — image understanding: model choice,
  install, measured memory, config reference, troubleshooting, limitations.
- `configs/*.yaml` — every training/inference knob, commented inline.

## License

MIT — see [`pyproject.toml`](pyproject.toml).

The SmolVLM vision model is Apache-2.0 (Hugging Face TB). Training corpora
retain their own licenses; check the download scripts for sources.
