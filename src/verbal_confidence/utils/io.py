"""JSON and NumPy I/O helpers, with metadata wrapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Plain JSON / NPZ
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict | list | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_json(obj: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz(path: str | Path) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=False))


def save_npz(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Metadata-wrapped saves
# ---------------------------------------------------------------------------

def save_with_meta(
    results: list | dict,
    path: str | Path,
    meta: dict,
) -> None:
    """
    Save results to JSON with an embedded _meta block.

    The file layout is always:
        {
          "_meta": { ... },          # reproducibility info
          "results": [ ... ]         # list, or dict if results is a dict
        }

    Args:
        results: The experiment output (list of records, or summary dict).
        path:    Destination file path (created with parents).
        meta:    Dict returned by config.build_meta().
    """
    payload: dict[str, Any] = {"_meta": meta}
    if isinstance(results, list):
        payload["results"] = results
    else:
        # For summary dicts (e.g. variance partitioning), merge at top level
        payload.update(results)
    save_json(payload, path)


def load_results(path: str | Path) -> tuple[list | dict | None, dict | None]:
    """
    Load a file saved by save_with_meta.

    Returns:
        (results, meta)  — both None if file doesn't exist.
        results is the list/dict under "results" (or the full payload minus _meta).
        meta is the _meta dict.
    """
    raw = load_json(path)
    if raw is None:
        return None, None
    meta = raw.pop("_meta", {})
    if "results" in raw:
        return raw["results"], meta
    return raw, meta
