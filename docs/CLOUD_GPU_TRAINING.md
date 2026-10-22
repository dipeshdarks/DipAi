# Cloud GPU training

DIPAI is designed to run on a laptop (RTX 2050, 4 GB VRAM, 6 GB RAM). That is
enough for smoke runs, Mode A of the small configs, and Mode B (QLoRA). It is
**not** enough for:

- Mode C (full-parameter continued-pretraining) — AdamW states alone cost
  ~12 bytes/parameter, so a 0.5 B model needs ≈6 GB of optimizer memory on
  top of weights and activations;
- long Mode A runs on the 50M config with a real corpus;
- any run you want finished today rather than this week.

That is what a cloud GPU is for. This guide is cloud-agnostic on purpose: no
vendor, no price (they change monthly), no committed spend.

## 1. Decide what you actually need

| Workload | On the laptop | Cloud when |
|---|---|---|
| Mode A smoke (`smoke_4m.yaml`) | minutes, fits | never |
| Mode A `scratch.yaml` (~21M) | hours, fits | you want it faster or longer |
| Mode A `dipai_50m.yaml` | slow; RAM is the binding constraint for data | you want a real corpus run |
| Mode B QLoRA 0.5B | fits in 4-bit | you want fp16/bf16 instead of 4-bit |
| Mode C full-param | **does not fit** | always |
| Mode D SFT (full-param, native) | borderline (fits small models with grad checkpointing) | larger model or longer schedule |

Rule of thumb for Mode C: budget **≥ 16 bytes/parameter** for training in
bf16 (16-bit weights + grads + AdamW), plus activations — a 0.5 B model wants
~10–12 GB; a 1–2 B model wants 24 GB+.

## 2. Provisioning checklist

Pick any CUDA GPU instance (A10G/L4-class is a good price/performance point;
consumer 3090/4090-class also works). What matters:

- **Persistent disk / volume** — checkpoints are hundreds of MB each; an
  ephemeral disk means re-downloading models and losing resume state on every
  stop. Attach a volume and put `checkpoints/` on it.
- **CUDA driver ≥ your torch build** — the repo pins the `cu126` wheels.
- **Enough disk for the HF cache** — Mode B/C download a base model once
  (~1–2 GB for 0.5B). Put `HF_HOME` on the persistent volume so it survives
  restarts.
- **SSH** (or Jupyter terminal) plus `tmux`/`screen` — training must survive
  you closing the laptop.

## 3. Get the code and environment there

```bash
git clone https://github.com/dipeshdarks/Dipai.git
cd Dipai

python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt    # covers finetune, webui, vision, and pytest

python -m dipai.hardware     # verify the GPU is visible
```

Sanity check before anything expensive:

```bash
.venv/bin/pytest -m "not slow"      # text-only suites must be green here too
```

## 4. Credentials — never in the repo

- If a base model is gated, export a read token in the shell:
  `export HF_TOKEN=hf_xxx` — never write it into a config, a YAML, or a commit.
- The same goes for cloud API keys: environment variables or your instance's
  secret store only. `.gitignore` already excludes checkpoints and data; it
  does not excuse committing secrets.

## 5. Data strategy

Two honest options:

1. **Collect in the cloud (preferred for big runs).** The collection CLI is
   bounded and resumable, so a pilot run in the cloud writes the same manifest
   it would locally. Point `output_dir` at the persistent volume:

   ```bash
   python -m dipai.data.collect --config configs/data.yaml --output-dir data/raw/web
   python -m dipai.data.clean   --input data/raw/web/*.jsonl --output data/clean/clean.jsonl --report data/clean/report.json --markdown
   python -m dipai.data.prepare --inputs data/clean/clean.jsonl --output-dir data/processed \
       --tokenizer data/processed/tokenizer.model --block-size 256 --seed 1337
   ```

2. **Upload prepared data.** Prepare locally, then `rsync`/`scp` `data/`
   (bins + tokenizer + `manifest.json`) up. Bins are small (packed uint16);
   raw JSONL is not — upload what you prepared, not what you scraped.

Either way the `manifest.json` that lands in `data/processed/` travels with
the data, so the run stays reproducible on someone else's machine.

## 6. Run training under tmux

```bash
tmux new -s train
python -m dipai.train --config configs/dipai_50m.yaml
# Ctrl-b d to detach; tmux attach -t train to return
```

- Auto-resume is on: if the instance restarts, re-run the same command and it
  continues from `latest.json`.
- For Mode C, copy `configs/finetune.yaml`, set `use_lora: false`,
  `load_in_4bit: false`, `learning_rate: 1.0e-4` (validation rejects higher),
  and raise batch/accumulation to suit the card:

  ```bash
  python -m dipai.finetune --config configs/cpt_qwen.yaml --train --resume
  ```

- Mode D is unchanged: `python -m dipai.finetune --config configs/sft_oasst.yaml --train`.
  (`sft_oasst.yaml`'s `max_steps: 1055` was sized for the local pilot; raise
  it for a full run.)

## 7. Keep the checkpoints (and bring them home)

Checkpoints are written atomically and pruned to `keep_last_n` — increase it
in the config if the GPU is expensive and you want more resume points.

```bash
# from your laptop
rsync -av user@instance:/path/Dipai/checkpoints/run/ ./checkpoints/run/
```

A `scratch`/native checkpoint (`ckpt_*.pt` + `config.yaml` + `tokenizer.model`)
loads with `python -m dipai.chat --ckpt checkpoints/run`. A Mode B adapter
loads with `--adapter`; a Mode C dir with `--model`.

## 8. Evaluate, then compare honestly

Run the evaluator on the **same eval JSONL and tokenizer** on both machines —
perplexity numbers from different tokenizers or datasets are not comparable,
and `dipai.evaluate` warns when the base and candidate sides disagree on
dataset:

```bash
python -m dipai.evaluate --checkpoint checkpoints/run \
    --dataset data/processed/finetune.jsonl \
    --base-checkpoint checkpoints/baseline \
    --out-json reports/eval.json --out-md reports/eval.md
```

See [`EVALUATION.md`](EVALUATION.md).

## 9. Stop early, check the bill

- Verify runs with a short `--max-iters`/`--max-steps` before launching the
  long one; watch the first few loss steps instead of assuming.
- `early_stop_val_loss` / `early_stop_patience` in Mode A configs stop the run
  and save a checkpoint automatically.
- Stop the instance when the run is done; a forgotten training box is the
  classic cloud bill.

## Known limits to keep in mind

- The repo has been *developed and tested* on Windows/CUDA 12.6. Linux is the
  normal cloud OS and the code is OS-portable (paths go through `pathlib`),
  but the first cloud run should be treated as a verification run: test suite
  first, smoke config second, real run third.
- `bitsandbytes` (Mode B 4-bit) needs a CUDA build matching the driver; on a
  cloud Linux image install it explicitly if `import bitsandbytes` fails.
- Never point a run at a directory containing checkpoints you cannot afford to
  lose — Mode A starts fresh only with an explicit new `--ckpt-dir`, but
  `--no-resume` plus a reused dir is still your mistake to avoid.
