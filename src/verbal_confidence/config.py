"""
Configuration loader.

Loads config/default.yaml and optionally merges an override file.
Resolves ${var} placeholders (including nested like ${paths.scratch_root}).
Also exports all HuggingFace cache env-vars so every script just calls
`cfg = load_config()` at the top and the environment is configured.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dot-separated keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _resolve(value: str, flat: dict[str, str]) -> str:
    """Replace ${key} and ${section.key} placeholders in a string."""

    def _sub(m: re.Match) -> str:
        ref = m.group(1)
        # Try config dict first, then environment
        if ref in flat:
            return str(flat[ref])
        env_val = os.environ.get(ref.upper(), os.environ.get(ref, ""))
        return env_val or m.group(0)  # leave unresolved if not found

    for _ in range(10):  # up to 10 rounds of substitution
        new = _PLACEHOLDER_RE.sub(_sub, value)
        if new == value:
            break
        value = new
    return value


def _resolve_all(d: dict) -> dict:
    """Recursively resolve all string values in a nested dict."""
    flat = _flatten(d)
    # Keep resolving the flat dict itself to handle cross-references
    for _ in range(5):
        for k, v in flat.items():
            if isinstance(v, str):
                flat[k] = _resolve(v, flat)

    def _rebuild(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            return {k: _rebuild(v, f"{prefix}.{k}" if prefix else k) for k, v in node.items()}
        if isinstance(node, list):
            return [_rebuild(item, prefix) for item in node]
        if isinstance(node, str):
            return _resolve(node, flat)
        return node

    return _rebuild(d, "")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# DotDict — attribute-style access to config
# ---------------------------------------------------------------------------

class DotDict:
    """Recursive dot-access wrapper around a dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, DotDict(v) if isinstance(v, dict) else v)

    def __repr__(self) -> str:
        return f"DotDict({vars(self)})"

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        out = {}
        for k, v in vars(self).items():
            out[k] = v.to_dict() if isinstance(v, DotDict) else v
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "default.yaml"


def load_config(override_path: str | Path | None = None) -> DotDict:
    """
    Load and return the merged configuration.

    Args:
        override_path: Optional YAML file whose values take precedence.

    Returns:
        DotDict with fully resolved values.

    Side effects:
        Sets HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE, TOKENIZERS_PARALLELISM
        environment variables so HuggingFace libraries respect the configured paths.
    """
    with open(_DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))

    cfg = _resolve_all(cfg)
    dot = DotDict(cfg)

    # ---------- Set HF environment variables ----------
    hf_home = dot.paths.hf_home
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = dot.paths.hf_datasets
    os.environ["TRANSFORMERS_CACHE"] = dot.paths.model_cache
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Make sure dirs exist
    for attr in ("hf_home", "hf_datasets", "model_cache", "results_root", "logs_root"):
        path = getattr(dot.paths, attr, None)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    return dot


def results_dir(cfg: DotDict) -> Path:
    """Return (and create) the run-specific results directory."""
    p = Path(cfg.paths.results_root) / cfg.paths.run_name
    p.mkdir(parents=True, exist_ok=True)
    return p
