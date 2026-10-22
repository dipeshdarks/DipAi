"""DIPAI data pipeline: licensed collection, cleaning, and preparation.

Submodules:
    registry  - permitted-source catalogue with licence metadata
    collect   - bounded, resumable streaming collectors
    manifest  - provenance manifests
    clean     - deterministic cleaning / dedup / PII / quality stages
    prepare   - clean -> tokenize -> pack into train.bin/val.bin
"""
from dipai.data.registry import DataSource, LicenseInfo, RegistryError, SourceRegistry, default_registry
from dipai.data.manifest import CollectManifest, SourceResult, load_manifest, new_run_id

__all__ = [
    "DataSource", "LicenseInfo", "RegistryError", "SourceRegistry", "default_registry",
    "CollectManifest", "SourceResult", "load_manifest", "new_run_id",
]
