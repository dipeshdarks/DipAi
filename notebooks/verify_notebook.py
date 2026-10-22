"""Execute every code cell of notebooks/quickstart.ipynb in-process.

Jupyter is not a dependency of this repo, so the notebook is verified by
parsing its JSON and running each code cell's source in one shared
namespace, with the working directory pinned to the repo root. Markdown
cells are skipped (they are prose and command references).

Run from anywhere:
    python notebooks/verify_notebook.py

Exit code 0 = every code cell executed; non-zero = first failure (printed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parent / "quickstart.ipynb"


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4, f"unexpected nbformat {nb['nbformat']}"

    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    print(f"{NOTEBOOK.name}: {len(cells)} cells, {len(code_cells)} code")

    import os

    os.chdir(REPO_ROOT)
    namespace: dict = {"__name__": "__notebook__"}
    started = time.time()

    for idx, cell in enumerate(cells, 1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        t0 = time.time()
        try:
            exec(compile(source, f"{NOTEBOOK.name}:cell{idx}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report and stop
            print(f"  cell {idx}: FAIL {type(exc).__name__}: {exc}")
            return 1
        print(f"  cell {idx}: ok ({time.time() - t0:.1f}s)")

    print(f"all code cells executed in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
