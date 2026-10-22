"""Permitted public web-data registry for DIPAI.

Every source carries licence/provenance metadata so collection decisions are
auditable. Nothing here downloads anything: it is pure metadata + resolution.
See docs/DATA_SOURCES_AND_LICENSES.md for the full licence discussion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class RegistryError(Exception):
    """Unknown source, bad config, or refused enable."""


# Kinds of collector. Each maps to one implementation in dipai.data.collect.
#   hf_dataset        - Hugging Face `datasets` streaming (fineweb-edu, oasst1, wikipedia snapshot)
#   wikimedia_dump    - direct file from dumps.wikimedia.org with a hard byte cap
#   local_files       - already-present .txt/.jsonl/.csv (no network)
KINDS = ("hf_dataset", "wikimedia_dump", "local_files")


@dataclass
class LicenseInfo:
    """Licence + provenance for one source. Kept beside every manifest."""
    spdx: str                     # SPDX id or "see-terms"
    name: str                     # human name
    url: str                      # licence text URL
    training_use: str             # one-line note on AI-training suitability
    attribution_required: bool = False
    share_alike: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSource:
    """One configured ingestion source."""
    name: str                     # short id used in configs and manifests
    kind: str                     # one of KINDS
    description: str = ""
    # --- hf_dataset ---
    hf_id: str = ""               # e.g. "HuggingFaceFW/fineweb-edu"
    hf_config: str = ""           # e.g. "sample-10BT" ("" = dataset default)
    hf_split: str = "train"
    hf_revision: str = ""         # pin a commit/branch; "" = latest (recorded in manifest)
    text_field: str = "text"      # field holding the document text
    # --- wikimedia_dump ---
    dump_url: str = ""            # full URL to one dump file
    # --- local_files ---
    paths: List[str] = field(default_factory=list)
    # --- shared limits/selection ---
    language_field: str = ""      # field used for language filtering ("" = no filter)
    languages: List[str] = field(default_factory=list)   # keep only these (""/empty = all)
    min_chars: int = 200          # drop short docs at collect time
    max_chars: int = 100_000      # drop absurdly long docs at collect time
    license: LicenseInfo = field(default_factory=lambda: LicenseInfo(
        spdx="unknown", name="unknown", url="", training_use="unreviewed"))
    enabled: bool = True
    pilot_default: bool = False   # included when a config says `pilot: true`

    def validate(self) -> None:
        if not self.name:
            raise RegistryError("source name is required")
        if self.kind not in KINDS:
            raise RegistryError(f"{self.name}: unknown kind {self.kind!r} (expected one of {KINDS})")
        if self.kind == "hf_dataset" and not self.hf_id:
            raise RegistryError(f"{self.name}: hf_dataset requires hf_id")
        if self.kind == "wikimedia_dump" and not self.dump_url:
            raise RegistryError(f"{self.name}: wikimedia_dump requires dump_url")
        if self.kind == "local_files" and not self.paths:
            raise RegistryError(f"{self.name}: local_files requires paths")
        if self.min_chars < 0 or self.max_chars <= 0 or self.min_chars > self.max_chars:
            raise RegistryError(f"{self.name}: bad min_chars/max_chars")
        if not self.license.spdx or not self.license.name:
            raise RegistryError(f"{self.name}: licence metadata incomplete")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Built-in catalogue. Only sources whose licence/terms we have actually read
# are listed with a concrete SPDX id; everything else is `unknown` and must be
# reviewed before enabling. Add sources here, not ad-hoc in collectors.
# ---------------------------------------------------------------------------

def _builtin_sources() -> Dict[str, DataSource]:
    fineweb_edu = DataSource(
        name="fineweb-edu",
        kind="hf_dataset",
        description="FineWeb-Edu: educational-quality filtered subset of FineWeb (Common Crawl).",
        hf_id="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        hf_split="train",
        text_field="text",
        language_field="language",
        languages=["en"],
        min_chars=200,
        max_chars=100_000,
        pilot_default=True,
        license=LicenseInfo(
            spdx="ODC-By-1.0",
            name="Open Data Commons Attribution License v1.0",
            url="https://opendatacommons.org/licenses/by/1-0/",
            training_use="Permitted with attribution; card states no additional restrictions on AI use.",
            attribution_required=True,
            notes="Card: 'AI usage: There are no restrictions on usage of the dataset for training AI models.'",
        ),
    )
    fineweb = DataSource(
        name="fineweb",
        kind="hf_dataset",
        description="FineWeb: 15T-token English web-text pipeline over Common Crawl (sample-10BT slice).",
        hf_id="HuggingFaceFW/fineweb",
        hf_config="sample-10BT",
        hf_split="train",
        text_field="text",
        min_chars=200,
        max_chars=100_000,
        license=LicenseInfo(
            spdx="ODC-By-1.0",
            name="Open Data Commons Attribution License v1.0",
            url="https://opendatacommons.org/licenses/by/1-0/",
            training_use="Permitted with attribution.",
            attribution_required=True,
        ),
    )
    fineweb2 = DataSource(
        name="fineweb-2",
        kind="hf_dataset",
        description="FineWeb2: multilingual web-text pipeline (sample-10BT slice, default English).",
        hf_id="HuggingFaceFW/fineweb-2",
        hf_config="default",
        hf_split="train",
        text_field="text",
        min_chars=200,
        max_chars=100_000,
        license=LicenseInfo(
            spdx="ODC-By-1.0",
            name="Open Data Commons Attribution License v1.0",
            url="https://opendatacommons.org/licenses/by/1-0/",
            training_use="Permitted with attribution.",
            attribution_required=True,
        ),
    )
    wikipedia_snapshot = DataSource(
        name="wikipedia-20231101-en",
        kind="hf_dataset",
        description="English Wikipedia snapshot 2023-11-01 (hosted parquet on HF; mirrors dumps.wikimedia.org).",
        hf_id="wikimedia/wikipedia",
        hf_config="20231101.en",
        hf_split="train",
        text_field="text",
        min_chars=200,
        max_chars=100_000,
        license=LicenseInfo(
            spdx="CC-BY-SA-3.0 OR GFDL-1.3",
            name="Creative Commons Attribution-ShareAlike 3.0 / GFDL 1.3",
            url="https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License",
            training_use="Permitted with attribution + share-alike on derivatives.",
            attribution_required=True,
            share_alike=True,
        ),
    )
    oasst1 = DataSource(
        name="oasst1",
        kind="hf_dataset",
        description="OpenAssistant Conversational Language Model / OASST1 human-generated tree data.",
        hf_id="OpenAssistant/oasst1",
        hf_config="2023-05-01",
        hf_split="train+validation",
        text_field="text",
        min_chars=1,
        max_chars=100_000,
        license=LicenseInfo(
            spdx="Apache-2.0",
            name="Apache License 2.0",
            url="https://www.apache.org/licenses/LICENSE-2.0",
            training_use="Permitted; permissive, no copyleft on model outputs.",
        ),
    )
    wiki_dump_en = DataSource(
        name="wikimedia-dump-en-stub",
        kind="wikimedia_dump",
        description=(
            "Direct file from dumps.wikimedia.org. Default points at a STUB dump "
            "(page metadata only) because full enwiki articles dumps are ~20 GB; "
            "set dump_url to an articles dump and raise max_dump_bytes deliberately."
        ),
        dump_url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-stub-articles0.xml.gz",
        min_chars=200,
        max_chars=100_000,
        license=LicenseInfo(
            spdx="CC-BY-SA-3.0 OR GFDL-1.3",
            name="Creative Commons Attribution-ShareAlike 3.0 / GFDL 1.3",
            url="https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License",
            training_use="Permitted with attribution + share-alike on derivatives.",
            attribution_required=True,
            share_alike=True,
            notes="Stub dumps carry page titles/snippets only; use articles dumps for real text.",
        ),
    )
    return {
        s.name: s
        for s in (fineweb_edu, fineweb, fineweb2, wikipedia_snapshot, oasst1, wiki_dump_en)
    }


_BUILTIN = _builtin_sources()


class SourceRegistry:
    """Lookup + validation over named sources. Starts from the built-ins."""

    def __init__(self, extra: Optional[List[DataSource]] = None):
        self._sources: Dict[str, DataSource] = dict(_BUILTIN)
        for s in extra or []:
            s.validate()
            self._sources[s.name] = s

    def get(self, name: str) -> DataSource:
        try:
            return self._sources[name]
        except KeyError:
            raise RegistryError(
                f"unknown source {name!r}; known: {sorted(self._sources)}"
            ) from None

    def names(self) -> List[str]:
        return sorted(self._sources)

    def enabled(self) -> List[DataSource]:
        return [s for s in self._sources.values() if s.enabled]

    def pilot(self) -> List[DataSource]:
        return [s for s in self.enabled() if s.pilot_default]

    def add(self, source: DataSource) -> None:
        source.validate()
        self._sources[source.name] = source

    def to_dict(self) -> Dict[str, Any]:
        return {name: s.to_dict() for name, s in sorted(self._sources.items())}

    def licences_json(self, path: str | Path) -> str:
        """Dump every source's licence block (for docs/manifests)."""
        return json.dumps(
            {n: s.license.to_dict() for n, s in sorted(self._sources.items())},
            indent=2, ensure_ascii=False,
        )


def default_registry() -> SourceRegistry:
    return SourceRegistry()
