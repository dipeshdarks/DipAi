# DIPAI Training Data

Put raw training files here (`data/raw/`), then run `scripts/prepare_data.py`
to clean, tokenize, and pack them into `data/processed/`.

## Supported input formats

| Format  | Notes                                                                 |
|---------|-----------------------------------------------------------------------|
| `.txt`  | Paragraphs separated by blank lines become individual records.        |
| `.jsonl`| One JSON object per line; the `text` field is read (configurable).    |
| `.csv`  | Must have a `text` column (header row required).                     |

All files are read as UTF-8; invalid bytes are replaced rather than fatal.

## What preparation does

1. Validates each file exists, is non-empty, and has a supported extension.
2. Drops empty records, over-long records (>200k chars), and exact duplicates.
3. Counts invalid-JSON and missing-field lines in the statistics report.
4. Splits into train/validation (default 5% val, seeded shuffle).
5. Tokenizes with SentencePiece BPE and packs into fixed `block_size` windows.
6. Writes `train.bin` / `val.bin` (uint16) + `dataset_stats.json`.

## Sample data

`data/examples/sample.txt` is a **tiny synthetic corpus written for smoke tests**.
It is deliberately limited: a model trained on it will echo its topics and
style only. It cannot produce a capable general-purpose assistant. Do not
mistake a good smoke-test loss for a useful model.

## Using your own data

- Only use files you have the right to use (your own writing, permissively
  licensed corpora, public-domain text, etc.).
- Document the source and license of every dataset you add, in this file.
- Never include copyrighted books, private conversations, or personal data
  you are not allowed to process.
- Bigger is better, but even a few MB of clean text will train the smoke
  model to reproduce its style.

## License log (add your datasets here)

| Dataset              | Source | License | Added      |
|----------------------|--------|---------|------------|
| `examples/sample.txt`| written for DIPAI (synthetic) | CC0 / MIT (project license) | 2026-10-19 |
| `raw/tinystories_50k.txt` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), 50,000-story streamed subset via `scripts/download_training_data.py` | CDLA-Sharing-1.0 - review before redistribution/commercial use | 2026-10-19 |
| `raw/wikipedia_en_20231101.jsonl` | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.en`, 59,983-article spread subset streamed via `scripts/download_wikipedia.py` (URLs + row-group coverage in `raw/wikipedia_en_20231101_manifest.json`) | **GFDL-1.3 OR CC-BY-SA-3.0** (NOT public domain; share-alike/attribution apply) | 2026-10-20 |
| `processed_50m/` | pretraining corpus packed from the Wikipedia subset by `scripts/prepare_data.py` (sha1 document split, seed 1337, 2% val) | inherits GFDL-1.3 / CC-BY-SA-3.0 from its source | 2026-10-20 |
| `raw/oasst1_english.jsonl` | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1), English split via `scripts/download_oasst.py` | Apache-2.0 | 2026-10-20 |
| `raw/oasst1_full.jsonl` | same export with `--with-metadata` (message_tree_id/role/review fields needed for thread reconstruction) | Apache-2.0 | 2026-10-20 |
| `processed/sft_train.jsonl`, `processed/sft_val.jsonl` | user/assistant threads rebuilt from `oasst1_full.jsonl` by `scripts/prepare_finetune_data.py --oasst-threads`; split by message TREE (leak-free) | Apache-2.0 | 2026-10-20 |
| `processed/tokenizer.model` | SentencePiece trained on `raw/oasst1_english.jsonl` (vocab 8192 + specials = 8200) for the flat-message corpus in `processed/` | Apache-2.0 (tokenizer follows its corpus) | 2026-10-20 |

### Provenance statements

- Wikipedia content is **not** public domain: GFDL-1.3 / CC-BY-SA-3.0 require
  attribution and share-alike when redistributing text derived from it.
- No Google, OpenAI, or ChatGPT private/proprietary training data was
  downloaded or used anywhere in this project. DIPAI is trained only on the
  rows above.
- Raw Wikipedia articles are used as **pretraining text (Mode A)**, never
  presented as conversational answers; conversation behavior comes solely
  from the OASST1 SFT files (Mode B, native branch).
- `raw/wikipedia_en_20231101_manifest.json` records the dataset URL, resolved
  parquet URLs, snapshot date, license, limits, and per-shard coverage used
  for this run (verifiable provenance).

### About the TinyStories subset

- Short, synthetically generated English stories (50,000 of them, ~42.7 MiB).
- Good for testing whether a small model learns coherent simple text.
- It does **not** teach broad knowledge or coding skill. A model trained on it
  produces simple children's-story text only.
- Full dataset is ~7.62 GB - deliberately not downloaded (RAM/disk limits).
