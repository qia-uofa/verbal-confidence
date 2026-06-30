"""JSON and NumPy I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
    data = np.load(path, allow_pickle=False)
    return dict(data)


def save_npz(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
