# Data sources and licences

Every corpus DIPAI can ingest is registered in
[`dipai/data/registry.py`](../dipai/data/registry.py) with its licence and
provenance attached. Nothing in the registry downloads anything — it is pure
metadata. The collector ([`dipai/data/collect.py`](../dipai/data/collect.py))
reads it, and every collection run writes the resolved licence block into its
manifest, so a training run's data provenance is auditable after the fact.

Run `python -m dipai.data.collect --config configs/data.yaml --list` to see
this table live, or `--dry-run` to see what *would* be collected without any
network access.

## Registered sources

| Name | Kind | Hugging Face id / URL | Licence (SPDX) | Attribution | Share-alike | Pilot default |
|---|---|---|---|---|---|---|
| `fineweb-edu` | hf_dataset | `HuggingFaceFW/fineweb-edu` | ODC-By-1.0 | yes | no | **yes** |
| `fineweb` | hf_dataset | `HuggingFaceFW/fineweb` | ODC-By-1.0 | yes | no | no |
| `fineweb-2` | hf_dataset | `HuggingFaceFW/fineweb-2` | ODC-By-1.0 | yes | no | no |
| `oasst1` | hf_dataset | `OpenAssistant/oasst1` | Apache-2.0 | no | no | no |
| `wikipedia-20231101-en` | hf_dataset | `wikimedia/wikipedia` (20231101 en) | CC-BY-SA-3.0 OR GFDL-1.3 | yes | **yes** | no |
| `wikimedia-dump-en-stub` | wikimedia_dump | `dumps.wikimedia.org/enwiki/latest/*stub-articles*` | CC-BY-SA-3.0 OR GFDL-1.3 | yes | **yes** | no |

`kind` values map to collector implementations:

- `hf_dataset` — Hugging Face `datasets` streaming (`streaming=True`, never
  materialised to disk in full).
- `wikimedia_dump` — direct HTTP range-request download from
  `dumps.wikimedia.org` with a hard byte ceiling (`max_dump_bytes`).
- `local_files` — already-present `.txt` / `.jsonl` / `.csv` on disk; no
  network at all.

## Licence notes per source

### fineweb / fineweb-2 / fineweb-edu — ODC-By-1.0

FineWeb and FineWeb-Edu are CommonCrawl derivatives released under the
**Open Data Commons Attribution License**. Training on them is permitted
provided you attribute the source. The FineWeb-Edu model card explicitly
states there are no additional restrictions on AI use. No share-alike
obligation attaches to your model weights.

Attribution is recorded in each collection manifest (`license.url`,
`license.name`, `license.training_use`); keep the manifest alongside the
dataset.

### oasst1 — Apache-2.0

OpenAssistant Conversations is a human-generated assistant dataset released
under **Apache-2.0**: permissive, attribution required only via the licence
text, no copyleft on derivative works or model outputs. This is the cleanest
source here for instruction tuning (`dipai.finetune` Mode B/C).

### wikipedia (snapshot and stub dumps) — CC-BY-SA-3.0 OR GFDL-1.3

Wikipedia text is dual-licensed **CC-BY-SA-3.0 or GFDL-1.3**. Both licences
are share-alike: derivative *text* (e.g. redistributed corpus extracts)
must carry the same licence, and attribution is required.

Two different shapes are registered:

- `wikipedia-20231101-en` — the pinned HF snapshot
  (`wikimedia/wikipedia`, `20231101` config). A snapshot date is recorded in
  the manifest so a run is reproducible.
- `wikimedia-dump-en-stub` — the live *stub articles* dump from
  `dumps.wikimedia.org`. Stubs carry page titles and opening snippets only,
  **not full articles**; use them for smoke tests, not for real training text.

**Modelling note:** share-alike obligations on training data are an open
question across jurisdictions. DIPAI takes the conservative position of
recording the licence, requiring attribution, and flagging `share_alike: true`
so you can decide — deliberately — whether a given run's outputs need to be
treated as derivatives. Nothing here is legal advice.

## Bounded collection (never a multi-TB download)

Every source stops when **either** cap in `configs/data.yaml` is hit:

```yaml
max_docs_per_source: 500        # pilot cap
max_bytes_per_source: 20000000  # 20 MB of text per source
max_dump_bytes: 50000000        # 50 MB hard ceiling for wikimedia_dump
```

Pilot mode (`pilot: true`, the default) only collects sources flagged
`pilot_default` — right now just `fineweb-edu`. Raise the caps or pass
`--no-pilot` only when you deliberately want more data and have the disk for
it. To date this project has run bounded pilots; **no multi-terabyte corpus
has ever been downloaded.**

Collection is resumable: a `.progress.json` beside the output lets an
interrupted run skip documents it already wrote. Provenance (licence,
HF revision, document counts, rolling sha1 digest) lands in
`manifest.json` / `CollectManifest`.

## Local files

`kind: local_files` accepts paths you already hold — run them through the
same clean → prepare pipeline. You are responsible for confirming you have
the right to use data you supply yourself; the registry cannot verify that.

## Where this fits in the pipeline

```
python -m dipai.data.collect  --config configs/data.yaml   # registry-driven, bounded, resumable
python -m dipai.data.clean    --input data/raw/web/*.jsonl  # deterministic cleaning + PII + dedup
python -m dipai.data.prepare  --inputs data/clean/*.jsonl   # tokenize + pack -> train.bin / val.bin
python -m dipai.train         --config configs/scratch.yaml # Mode A
```

See [`docs/TRAINING_GUIDE.md`](TRAINING_GUIDE.md) for the full walkthrough and
[`docs/IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) for what was built
and measured.
