# Checkpoints and resuming

How DIPAI saves training state, what a checkpoint directory contains, and how
each mode resumes. The implementation is
[`dipai/checkpoints.py`](../dipai/checkpoints.py); its guarantees are covered
by `tests/test_checkpoints.py`.

## Guarantees

1. **Atomic writes.** Every checkpoint is written as `ckpt_XXXXXXXX.pt.tmp`,
   flushed and `fsync`ed, then `os.replace()`d into place (atomic on Windows
   within a volume). A crash mid-save never corrupts an existing checkpoint —
   at worst you get a leftover `.tmp`, which is cleaned up, or a missing new
   step, never a half-written one.
2. **A pointer, not a symlink.** `latest.json` names the newest *complete*
   checkpoint (`{"step", "file", "saved_at"}`). It is itself written atomically
   (tmp + replace), so it always points at a finished file. Symlinks are not
   used — Windows.
3. **Pointer fallback.** If `latest.json` is missing or points at a deleted
   file, `latest_checkpoint()` logs a warning and scans for the newest numbered
   file instead. A stale pointer degrades gracefully, it does not fail the run.
4. **Self-describing directories.** Each checkpoint dir carries a snapshot of
   the run config (`config.yaml`) and the tokenizer files
   (`tokenizer.model`, `.vocab`, `.model.meta.json`), so the directory can be
   reloaded or exported on its own, without the original config files.
5. **Retention.** `keep_last_n` (default 3) newest numbered checkpoints are
   kept; older ones are pruned *after* each successful save, and the pointer is
   re-validated by construction (pruning never removes the file the pointer
   references — the newest is always kept).

## What's inside a checkpoint file

Saved by `save_checkpoint()` as one `torch.save` payload:

| Key | Contents |
|---|---|
| `model` | model `state_dict` |
| `optimizer` | optimizer state (present when an optimizer is passed) |
| `scaler` | autocast scaler state (only if enabled) |
| `model_config` | the model's own config dict (rebuild the arch from this) |
| `config` | the full run config dict |
| `step` | global step at save time (resume starts at `step + 1`) |
| `extra` | caller extras — **RNG state**, loss history |
| `saved_at`, `format_version`, `num_params` | metadata |

RNG state (`torch`, `numpy`, `python` random) is captured at save time and
restored on resume, so a resumed run continues the same random stream the
interrupted one was on.
`tests/test_phase5_modes.py::test_rng_state_save_restore_roundtrip_is_deterministic`
proves the saved states reproduce, after arbitrary perturbation, the exact
next draws a never-stopped run would have emitted.

## Layout of a checkpoint directory

```
checkpoints/scratch/
├── ckpt_00005000.pt        # numbered checkpoints (keep_last_n kept)
├── ckpt_00005500.pt
├── ckpt_00006000.pt
├── latest.json             # pointer: newest complete checkpoint
├── config.yaml             # snapshot of the run config
├── tokenizer.model         # tokenizer copied in (self-describing)
├── tokenizer.vocab
├── tokenizer.model.meta.json
└── loss_log.csv            # written by the training loop
```

`dipai.load_checkpoint(path)` accepts either a file or a directory (resolves
via the pointer) and validates the payload shape, raising `CheckpointError`
with a clear message on corruption — including the advice to fall back to an
older step if an interrupted save ever did land.

## Resuming per mode

### Mode A — `dipai.train`

```bat
python -m dipai.train --config configs\scratch.yaml              :: auto-resume
python -m dipai.train --config configs\scratch.yaml --resume     :: force resume (error if none)
python -m dipai.train --config configs\scratch.yaml --no-resume  :: start fresh
```

- **Auto-resume is the default**: if `--ckpt-dir` already contains a
  checkpoint, training continues from it (model + optimizer + scaler + RNG).
- `--no-resume` starts a fresh run **without deleting anything**. The first
  save then lands beside the old files — to keep runs cleanly separated, use
  `--ckpt-dir` to give the new run its own directory.
- `--resume` forces a resume and fails loudly (`CheckpointError`) if there is
  nothing to resume from.
- Early stopping (`early_stop_val_loss` / `early_stop_patience`) saves a
  checkpoint on stop, so a stopped run is resumable too.

### Mode B/C — `dipai.finetune` (Hugging Face Trainer)

```bat
python -m dipai.finetune --config configs\finetune.yaml --train --resume
```

`--resume` picks the newest Trainer `checkpoint-*/` in `output_dir`
(`get_last_checkpoint`); the config key `resume_from_checkpoint` does the same
at config level (it may also name an explicit path). Trainer checkpoints hold
model + optimizer + scheduler + trainer state.

### Mode D — `dipai.finetune` (native local-checkpoint branch)

Same `--resume` flag: it resolves to `"auto"` and finds the newest `ckpt_*.pt`
in `output_dir` via `latest_checkpoint()`, then continues with weights,
optimizer, scaler, and RNG state restored.

### Interrupt safety in `dipai.finetune`

The **native (Mode D) branch** catches `KeyboardInterrupt` and saves an
emergency checkpoint (`extra: {"interrupted": true}`) before exiting, so a
Ctrl+C mid-SFT loses at most the step in flight. The HF Trainer branch relies
on its own periodic `save_steps` checkpoints instead. Defined exit codes let
scripts branch on the outcome:

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | runtime / RAM problem |
| 2 | input or config error |
| 3 | resume failed (nothing to resume from, or unreadable) |
| 4 | CUDA OOM |
| 130 | interrupted — emergency checkpoint saved (native branch), resume with `--resume` |

## Moving or copying checkpoints

- A checkpoint directory is portable: copy the whole directory (`.pt` files +
  `latest.json` + `config.yaml` + tokenizer files). Keep `latest.json` — but if
  you lose it, loaders fall back to scanning.
- `ckpt_*.pt` files are large (a ~593 MB file for a 50M model includes
  optimizer state). They are **git-ignored** — move them with `rsync`/drive
  copies, never `git add`.
- When copying between machines, copy to the same volume before rename-based
  operations; `os.replace()` is only atomic within a volume.

## Safety rules (enforced, not aspirational)

- **Existing checkpoints are never overwritten by a training run**: step
  numbers are part of the filename, so a new run at a different step cannot
  collide, and no code path deletes a checkpoint the user did not prune.
- **Never modify or delete `checkpoints/` in this repository** — it holds
  trained weights for existing runs (`50m`, `scratch`, `sft50m`, `sft50m_v2`,
  `finetune`, `smoke`, `smoke4m`). Adding a run means adding files, not
  touching these.
- Explicit pruning is available as a library call (`prune(ckpt_dir, keep_last_n)`)
  and happens automatically within a dir you chose to train into (retention is
  a feature: it is what keeps a long run from filling the disk). Set
  `keep_last_n` higher if you want more history per run.
- Corrupt-file recovery: `load_checkpoint` refuses a bad file with a message
  telling you to delete it or resume from an older step — it never silently
  trains from garbage.

## Verifying a checkpoint after a crash

```python
from dipai.checkpoints import latest_checkpoint, load_checkpoint
ckpt = latest_checkpoint("checkpoints/scratch")     # -> path or None
payload = load_checkpoint("checkpoints/scratch")    # raises CheckpointError if bad
print(payload["step"])
```

If `load_checkpoint` raises, read the message: it distinguishes
"nothing trained yet" from "file unreadable/corrupt" from "not a DIPAI
checkpoint", and tells you the recovery action for each.
