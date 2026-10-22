# DIPAI Image Understanding (Vision)

This document covers the image-understanding capability added to DIPAI:
upload an image and ask questions about it through the web UI or the CLI.

**Honest summary:** image understanding runs on a *separate, pretrained*
vision-language model (VLM). The DIPAI scratch Transformer (DIPAI-20M/50M)
is a text-only model — it has no visual encoder and cannot see images. The
two models coexist; nothing about text-only training, checkpoints, or chat
is modified.

---

## 1. Model choice and licensing

| Item | Value |
|---|---|
| Model | [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| License | **Apache-2.0** (verified from the model card) |
| Architecture | `Idefics3ForConditionalGeneration` (SmolVLM family) |
| Components | SmolLM2-135M-Instruct (language) + SigLIP-base-patch16-512 (vision) |
| Parameters | 256,484,928 (reported by the repo's safetensors index) |
| Transformers support | Native since v4.49; **verified working with the installed 4.57.6** |
| Pipeline | `image-text-to-text`, official `AutoProcessor` preprocessing |

Why this model: it is one of the smallest capable open VLMs, it is fully
supported by the installed Transformers version (no `trust_remote_code`
needed), and its bf16 weights (~513 MB) fit the 4 GB RTX 2050 with room for
the text model.

Upgrade option: `HuggingFaceTB/SmolVLM-500M-Instruct` (same license and
architecture family, 507M params, ~1.0 GB weights) — better answers, slower
on this GPU. Set `model_id` in `configs/vision.yaml` to switch.

The model is loaded **separately** from `dipai.inference.DIPAIEngine`
(`dipai/vision.py` has its own module state). The scratch model's
vocabulary and checkpoint format are untouched.

## 2. Installation requirements

```bash
# from the DIPAI project root, inside the venv
pip install -r requirements.txt        # includes Pillow + transformers
# or just the vision extras:
pip install "dipai[vision]"            # transformers>=4.49, accelerate, Pillow
```

Already satisfied on this machine: `torch 2.14.1+cu126`, `transformers 4.57.6`,
`accelerate`, `Pillow 11.3.0`. No new packages had to be installed for the
feature to work here.

## 3. Disk and RAM requirements (measured on this machine)

| Resource | Measured value |
|---|---|
| Download (one-time) | **1,036 MB** of cache on Windows (weights are 513 MB bf16; the HF hub keeps a second copy instead of symlinking on Windows) |
| Disk check before download | Automatic — `dipai.vision` queries the Hub for the safetensors size and refuses the download if free space is insufficient (was 14.1 GB free before download; 19.4 GB free after) |
| System RAM during load+inference | process peak **1,259 MB** RSS (idle 506 MB); system total is 5.9 GB |
| VRAM after load | model occupies **0.58 GB** of the 4.29 GB card (free went 3.46 → 2.89 GB) |
| VRAM after unload | back to 3.41 GB free |
| Load time (cached weights) | 14.7 s (first-ever run additionally downloads the weights) |
| Inference time | 1.1–17.1 s per question depending on answer length (default cap 256 tokens) |

`unload_vision_model()` releases all VRAM. PyTorch may keep a small amount
of process RAM reserved after unload (allocator caching); this is normal.

## 4. Setup and launch commands

```bash
cd C:\Users\Lenovo\Downloads\Dipai\DIPAI

# Web UI (chat tab + "Image understanding" tab)
python webui\app.py --ckpt checkpoints\scratch
#   optional: --vision-config configs\vision.yaml --port 7860

# Chat CLI, single-shot vision (the text model is not loaded at all)
python -m dipai.chat --image data\vision_samples\red_square.png --prompt "What is in this picture?"

# Dedicated vision CLI
python -m dipai.vision --image data\vision_samples\text_hello.webp --prompt "Read the visible text"

# Text-only chat (unchanged behaviour)
python -m dipai.chat --ckpt checkpoints\scratch
```

`--help` on both CLIs documents every flag.

## 5. Image upload instructions (web UI)

1. Start the web UI (command above) and open `http://127.0.0.1:7860`.
2. Open the **Image understanding** tab (next to **Chat**).
3. Drop or pick a **JPEG, PNG, or WebP** file in *Upload image* — a preview
   appears.
4. Type a question (e.g. "What is in this picture?", "Describe this image.",
   "Read visible text.", "Identify the objects.") and press **ASK**.
5. The first ask loads the VLM (status line shows *loading → ready*); later
   asks reuse it. **CLEAR IMAGE** resets image, question, and answer.

Example (real output from the verification run):

```
image: data/vision_samples/green_circle.jpg  (green circle on blue)
Q: What shape and color is the main object in this image?
A: The image depicts a solid green circle in the center of a blue background...
```

## 6. Configuration (`configs/vision.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `HuggingFaceTB/SmolVLM-256M-Instruct` | HF id or local directory |
| `device` | `auto` | `auto` \| `cuda` \| `cpu` |
| `dtype` | `auto` | `auto` = bfloat16/float16 on CUDA, float32 on CPU |
| `cpu_fallback` | `true` | retry on CPU if CUDA loading runs out of memory |
| `load_on_startup` | `false` | `false` = load on first image question |
| `low_cpu_mem_usage` | `true` | stream weights from disk (low-RAM machines) |
| `local_files_only` | `false` | `true` = never contact huggingface.co |
| `max_upload_mb` | `10.0` | reject larger files |
| `max_image_side` | `4096` | reject wider/taller images (px) |
| `max_image_pixels` | `4000000` | reject images above total pixel count |
| `max_new_tokens` | `256` | hard cap on generated text |
| `temperature` | `0.2` | `0` = greedy/deterministic |
| `top_p` | `0.95` | nucleus sampling threshold |

A missing config file is fine — validated built-in defaults are used, so
text-only setups keep working without vision being configured.

## 7. Troubleshooting

**CUDA out of memory (loading or answering)**
- The message is explicit; text chat keeps working regardless.
- Lower `max_new_tokens` (each generated token allocates KV-cache memory).
- Close other GPU apps (browser video, games): the model needs only ~0.6 GB,
  but the 4 GB card is shared.
- Set `device: cpu` in `configs/vision.yaml` (works, but ~3-5x slower).
- `cpu_fallback: true` retries automatically when *loading* OOMs.

**Download failures / offline use**
- First load needs network once; afterwards run with `local_files_only: true`.
- The disk pre-check refuses the download when space is insufficient.
- `model_id` may point to a local folder containing the saved weights.

**"not a recognized image file" / "corrupt or truncated"**
- Only decoded JPEG/PNG/WebP content is accepted (checked from the bytes,
  not the file extension). Re-export the image or use another file.

**Answers look vague or wrong**
- A 256M model is small: it identifies colors, shapes, objects and short
  text reliably, but can ramble or hallucinate details on complex photos.
  Do not treat its output as authoritative.

**`torch_dtype is deprecated` warning**
- Cosmetic warning from transformers 4.57 (the argument name is still
  `torch_dtype` for compatibility with 4.49-4.56). Safe to ignore.

## 8. Known limitations and privacy

- **Text-only scratch model unchanged**: DIPAI-20M/50M still cannot see
  images; only the separate VLM can.
- **Formats**: JPEG/PNG/WebP only; GIF/BMP/TIFF are rejected with a clear
  error.
- **Size limits**: 10 MB / 4096 px per side / 4M pixels (configurable).
- **No multi-image comparison yet**: two-image "spot the difference" style
  prompts are not wired into the UI/CLI (single image per request). The
  model itself accepts one image per turn here.
- **Charts**: summarizing charts works only as well as this small model
  allows; not guaranteed.
- **Privacy**: images and prompts are processed 100% locally. They are never
  uploaded, never sent to third-party services, and never written to logs
  (only sizes/devices/status are logged). Remote access happens *only* to
  download model weights from Hugging Face — disable it with
  `local_files_only: true`.
- **No execution**: image files are decoded by Pillow; nothing embedded in
  an image is executed, and remote URLs are never fetched.
- **Never fabricated**: if inference fails or produces empty output, the UI
  and CLI show an explicit `[vision error: ...]` instead of an invented
  description.

## 9. Test results (run on this machine)

Commands:

```bash
.venv\Scripts\pytest.exe tests\ -q                 # full suite
DIPAI_VISION_SMOKE=1 pytest tests\test_vision.py::test_gpu_smoke_real_model
```

Results (2026-10-22, Python 3.10.1 / torch 2.14.1+cu126 / RTX 2050):

| Run | Result |
|---|---|
| Pre-existing tests (suite excluding `test_vision.py`, run against the modified code) | **87 passed, 0 failed, 0 skipped** |
| Full suite after adding `tests/test_vision.py` | **126 passed, 1 skipped** |
| Vision file alone (`tests/test_vision.py`) | 39 passed, 1 skipped |
| Skipped test | `test_gpu_smoke_real_model` — skipped by default (it downloads real weights); **passes when run with `DIPAI_VISION_SMOKE=1`** |
| Gradio server smoke | HTTP 200 on `http://127.0.0.1:7861` with all 8 UI components present |
| Real CLI inference | `python -m dipai.chat --image ... --prompt ...` → "A red background." (exit 0) |
| Text chat regression | `python -m dipai.chat --once "once upon a time"` → normal generation (exit 0) |

Unit tests mock model artifacts, so they never download weights; the one
env-gated test is the only place real inference runs in the suite.
