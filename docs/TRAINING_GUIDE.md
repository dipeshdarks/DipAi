# Training guide

End-to-end walkthrough of every training mode DIPAI supports, with the exact
commands and configs that exist in this repository. Everything here was run on
the target machine: **NVIDIA RTX 2050 (4 GB VRAM), 6 GB RAM, Windows 11,
Python 3.10**.

## The four modes at a glance

| Mode | What it does | Entry point | Config |
|---|---|---|---|
| **A** | Train a decoder-only LM from scratch on packed tokens | `python -m dipai.train` | `configs/scratch.yaml`, `dipai_50m.yaml`, `smoke_4m.yaml` |
| **B** | LoRA / QLoRA fine-tune a HF base into a small adapter | `python -m dipai.finetune` | `configs/finetune.yaml` (`use_lora: true`) |
| **C** | Full-parameter continued-pretraining of a HF base | `python -m dipai.finetune` | your own config (`use_lora: false`, `load_in_4bit: false`) |
| **D** | Full-parameter supervised fine-tune of a local DIPAI checkpoint on conversations | `python -m dipai.finetune` | `configs/sft_oasst.yaml` (`dipai_checkpoint:` set) |

Note on naming: `dipai/finetune.py` labels *all* of its paths "Mode B" in code
comments as an umbrella term. This guide splits it by branch so each letter
means one workflow — B is the LoRA branch, C is full-parameter
continued-pretraining on a HF base, D is the native local-checkpoint SFT
branch. The code paths are exactly as the table shows.

Which branch runs is decided by the config, not by CLI flags:

```
dipai_checkpoint set?          -> Mode D (native branch; HF LoRA/4-bit keys ignored)
  no -> use_lora: true?        -> Mode B (PEFT adapter -> out_dir/adapter)
  no -> use_lora: false        -> Mode C (all params -> out_dir/model)
```

---

## 0. Environment

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m dipai.hardware          :: confirm the GPU is seen
```

Linux/macOS: same commands with `python`/`pip`, drop `.bat`-style paths.

## 1. Get data (licensed, bounded, resumable)

```bat
python -m dipai.data.collect --config configs\data.yaml --list      :: what's registered
python -m dipai.data.collect --config configs\data.yaml --dry-run   :: no network, shows plan
python -m dipai.data.collect --config configs\data.yaml             :: bounded pilot (fineweb-edu)
```

Caps are deliberately tiny in `configs/data.yaml` (500 docs / 20 MB per
source, 50 MB dump ceiling, pilot mode on). Sources, licences, and the
attribution rules: [`docs/DATA_SOURCES_AND_LICENSES.md`](DATA_SOURCES_AND_LICENSES.md).

Clean, then tokenize and pack:

```bat
python -m dipai.data.clean --input data\raw\web\*.jsonl --output data\clean\clean.jsonl --report data\clean\report.json --markdown

python -m dipai.data.prepare --inputs data\clean\clean.jsonl --output-dir data\processed ^
    --tokenizer data\processed\tokenizer.model --block-size 256 --seed 1337
```

`data/prepare` writes `train.bin`, `val.bin`, `dataset_stats.json`, and a
`manifest.json` recording inputs, tokenizer, seed, counts, and the cleaning
report — so any training run is reproducible from its manifest.

Alternatively the legacy scripts still work:

```bat
python scripts\download_training_data.py
python scripts\prepare_data.py
```

## 2. Mode A — train from scratch

```bat
:: smoke config first (4M params, minutes not hours)
python -m dipai.train --config configs\smoke_4m.yaml

:: the real run (~21M)
python -m dipai.train --config configs\scratch.yaml

:: ~50M variant
python -m dipai.train --config configs\dipai_50m.yaml
```

Behaviour you can rely on:

- Model `vocab_size` is synced to the trained tokenizer (logged, never silent).
- `param_target` in the config is enforced: training refuses (exit 5) if the
  measured parameter count falls outside the configured range.
- Peak memory is estimated and printed **before** the model is allocated.
- Checkpoints are atomic (`save` → fsync → `os.replace`), a `latest.json`
  pointer names the newest complete one, and `keep_last_n` retention prunes
  old ones with the pointer staying valid.
- Training auto-resumes from `latest.json`; `--no-resume` starts fresh
  (existing checkpoints are never deleted or overwritten — a fresh run needs
  a new `--ckpt-dir`).
- On CUDA OOM mid-accumulation: partial gradients are zeroed, cache cleared,
  one automatic retry is attempted; if that fails the loop stops cleanly with
  the last checkpoint intact.

Useful overrides: `--max-iters N`, `--data-dir`, `--ckpt-dir`, `--resume`,
`--no-resume`.

Before committing to a run, estimate its size:

```bat
python scripts\estimate_model.py --config configs\scratch.yaml
```

## 3. Mode B — LoRA / QLoRA fine-tune (adapter)

QLoRA (4-bit, `bitsandbytes`) keeps a 0.5B base trainable on the 4 GB GPU.
Default target is `Qwen/Qwen2.5-0.5B-Instruct`.

```bat
python scripts\prepare_finetune_data.py                  :: build chat-format JSONL
python -m dipai.finetune --config configs\finetune.yaml   :: dry run: VRAM estimate only
python -m dipai.finetune --config configs\finetune.yaml --train
```

Useful flags: `--max-steps`, `--max-samples`, `--output-dir`, `--resume`.
Output: a PEFT adapter directory (`<output_dir>/adapter`).

Rules enforced by config validation (they raise `ConfigError` before any
weights load):

- `use_lora: false` cannot be combined with `load_in_4bit: true`
  (full params cannot live in 4-bit).
- Full-parameter runs (`use_lora: false`) require `learning_rate <= 1e-4` —
  adapter LRs like 2e-4 are rejected as too high for full-param updates.
- `fp16` and `bf16` cannot both be true.

Chat with the adapter:

```bat
python -m dipai.chat --ckpt checkpoints\scratch --adapter checkpoints\finetune\latest
```

## 4. Mode C — full-parameter continued-pretraining (HF base)

Same module, different config: set `use_lora: false` and `load_in_4bit: false`
and every parameter of the HF base is trained. There is no committed Mode C
config — make one by copying `configs/finetune.yaml` and flipping those two
keys (plus a gentler `learning_rate`, ≤1e-4, which validation enforces).
Checkpoints are saved (`out_dir/model`) and can be loaded directly:

```bat
python -m dipai.finetune --config configs\cpt_qwen.yaml --train    :: your config

:: load it anywhere a model dir is accepted
python -m dipai.chat --model checkpoints\cpt_qwen\model --once "hello"
python -m webui.app --model checkpoints\cpt_qwen\model
```

VRAM estimate: the pre-flight check knows full-param runs cost ~12 bytes per
parameter (AdamW states) instead of LoRA's small delta, and warns/refuses
accordingly on a 4 GB GPU — a real full-param run of a 0.5B model will not
fit, which is why this mode is meant for smaller bases or cloud GPUs
([`docs/CLOUD_GPU_TRAINING.md`](CLOUD_GPU_TRAINING.md)).

## 5. Mode D — supervised fine-tune a local DIPAI checkpoint (SFT)

The native branch of `dipai.finetune`, selected by `dipai_checkpoint:` in the
config. Full-parameter SFT of a Mode A output, with **loss masked on user
turns** — only assistant tokens (and the separators the assistant must
produce) contribute to training.

```bat
python scripts\prepare_finetune_data.py --oasst-threads   :: conversations -> {"messages": [...]}
python -m dipai.finetune --config configs\sft_oasst.yaml          :: dry run
python -m dipai.finetune --config configs\sft_oasst.yaml --train  :: train
```

Details that matter:

- Dataset schema is `{"messages": [{"role": "user"|"assistant", "content": ...}]}`
  and train/val are split by conversation *tree* upstream, so no thread leaks
  across the split.
- The prompt layout (`dipai.conversations.format_dialog`) is the **same
  layout chat inference uses**, so what you train is what you serve.
- HF `model_name_or_path` / LoRA / 4-bit keys are ignored in this branch and
  the CLI logs that they were.
- Output: a full model directory (e.g. `checkpoints/sft50m_v2`).

Chat with it — the native branch saves a **DIPAI checkpoint**
(`ckpt_*.pt` + `config.yaml` + `latest.json` + tokenizer), so load it with
`--ckpt` (the `scratch` backend), not `--model`:

```bat
python -m dipai.chat --ckpt checkpoints\sft50m_v2
```

## 6. Naming and loading models (the model registry)

Model directories can always be passed as paths. The registry adds logical
names so you don't have to remember them:

```bat
python -m dipai.chat --list-models              :: name, backend, path
python -m webui.app --list-models
```

Register your own in code:

```python
from dipai.model_registry import ModelRegistry, RegisteredModel

r = ModelRegistry()
r.add(RegisteredModel(name="sft-50m", backend="scratch",
                      path="checkpoints/sft50m_v2", trained_from="sft"))
```

Backends map 1:1 to the real loaders in `dipai.inference`:

| backend | loader | what it reads |
|---|---|---|
| `scratch` | `DIPAIEngine.from_checkpoint` | DIPAI checkpoint dir (Mode A/D output) |
| `hf` | `DIPAIEngine.from_adapter` | PEFT adapter dir (Mode B output) |
| `hf-full` | `DIPAIEngine.from_model` | full HF model dir (Mode C output) |

Precedence in both CLIs: `--model` > `--adapter` > `--ckpt`.

## 7. Evaluate

```bat
python -m dipai.evaluate --checkpoint checkpoints\scratch ^
    --dataset data\processed\finetune.jsonl --generate-samples 3 ^
    --base-checkpoint checkpoints\smoke --out-json reports\eval.json --out-md reports\eval.md
```

The dataset must be JSONL with a `text` field (`data/processed/finetune.jsonl`
is one such file). Conversation-format files (`sft_val.jsonl`) must be
flattened to plain text first.

Perplexity is only comparable across checkpoints evaluated on the **same
tokenized data and tokenizer** — the evaluator warns if you try otherwise.
Full reference: [`docs/EVALUATION.md`](EVALUATION.md).

## 8. Resume, checkpoints, and safety rules

```bat
python -m dipai.train --config configs\scratch.yaml              :: auto-resumes
python -m dipai.train --config configs\scratch.yaml --no-resume  :: fresh (needs new ckpt-dir)
python -m dipai.finetune --config configs\sft_oasst.yaml --train --resume
```

Hard rules this project follows (they are enforced in code, not just policy):

- Existing checkpoints are **never** deleted or overwritten by a run.
- The model architecture is never silently replaced; config changes that
  would change it require a new config.
- Checkpoint dirs, raw/processed data, and `*.safetensors` are git-ignored —
  nothing multi-hundred-MB is ever committed.

Details: [`docs/CHECKPOINTS_AND_RESUMING.md`](CHECKPOINTS_AND_RESUMING.md).

## 9. Tests

```bat
.venv\Scripts\pytest.exe            :: full suite (~2-6 min)
.venv\Scripts\pytest.exe -m "not slow"   :: skip slow markers
```

Current status: **200 passed, 1 skipped** (the skip is the real-GPU vision
smoke test unless `DIPAI_VISION_SMOKE=1`). See
[`docs/IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) for the phase-by-phase
account of what was built and verified.
