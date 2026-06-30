"""
Configuration loader.

Load order (later sources win):
  1. config/default.yaml        — baseline defaults
  2. .env (project root)        — two required paths + HF_TOKEN
  3. override YAML (optional)   — per-run overrides passed at runtime

Two required .env variables
───────────────────────────
  EPHEMERAL_ROOT   Path for large, re-downloadable data: HF model weights,
                   HF dataset cache, pip packages, conda envs.
                   Can live on fast scratch (will survive wipes).
                   Example: /scratch/mygroup/myuser

  PERMANENT_ROOT   Path for irreplaceable outputs: experiment results, logs.
                   Should be backed up / not wiped.
                   Example: /home/myuser  or  /work/mygroup/myuser

Optional .env variables
───────────────────────
  HF_TOKEN         HuggingFace read token (needed for gated models like Gemma)
  RUN_NAME         Override the run subdirectory name
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    """
    Parse a .env file and inject variables into os.environ.
    - Skips blank lines and # comments.
    - Strips surrounding quotes from values.
    - Does NOT overwrite variables already set in the shell environment
      (shell always wins — lets SLURM job scripts override if needed).
    """
    if not env_path.exists():
        return
    with open(env_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # shell wins
                os.environ[key] = value


def _find_dotenv() -> Path:
    """
    Walk up from this file until we find .env (project root) or hit /.
    Falls back to cwd/.env.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return Path.cwd() / ".env"


# ---------------------------------------------------------------------------
# Placeholder resolver
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _resolve(value: str, flat: dict[str, str]) -> str:
    """Replace ${config.key} or ${ENV_VAR} placeholders."""

    def _sub(m: re.Match) -> str:
        ref = m.group(1)
        if ref in flat:
            return str(flat[ref])
        # Try environment (exact case, then upper)
        return os.environ.get(ref, os.environ.get(ref.upper(), m.group(0)))

    for _ in range(10):
        new = _PLACEHOLDER_RE.sub(_sub, value)
        if new == value:
            break
        value = new
    return value


def _resolve_all(d: dict) -> dict:
    flat = _flatten(d)
    for _ in range(5):
        for k, v in flat.items():
            if isinstance(v, str):
                flat[k] = _resolve(v, flat)

    def _rebuild(node: Any, prefix: str = "") -> Any:
        if isinstance(node, dict):
            return {k: _rebuild(v, f"{prefix}.{k}" if prefix else k)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [_rebuild(item, prefix) for item in node]
        if isinstance(node, str):
            return _resolve(node, flat)
        return node

    return _rebuild(d)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# DotDict — attribute-style access
# ---------------------------------------------------------------------------

class DotDict:
    """Recursive dot-access wrapper around a plain dict."""

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
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_ENV = ("EPHEMERAL_ROOT", "PERMANENT_ROOT")


def _validate_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            "Set them in your .env file (copy .env.example) or in the shell."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "default.yaml"


def load_config(override_path: str | Path | None = None) -> DotDict:
    """
    Load, merge, and return the fully resolved configuration.

    Load order (later wins):
      1. config/default.yaml
      2. .env  (EPHEMERAL_ROOT, PERMANENT_ROOT, HF_TOKEN, …)
      3. override_path YAML (optional)

    Side effects:
      Sets HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE, TOKENIZERS_PARALLELISM,
      and optionally HUGGING_FACE_HUB_TOKEN from .env.
    """
    # 1. Load .env first so its values are available as ${VAR} placeholders
    _load_dotenv(_find_dotenv())
    _validate_env()

    # 2. Load base YAML
    with open(_DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # 3. Merge optional override
    if override_path is not None:
        with open(override_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f))

    # 4. Resolve all ${...} placeholders (config keys + env vars)
    cfg = _resolve_all(cfg)
    dot = DotDict(cfg)

    # 5. Set HuggingFace env vars
    os.environ["HF_HOME"]             = dot.paths.hf_home
    os.environ["HF_DATASETS_CACHE"]   = dot.paths.hf_datasets
    os.environ["TRANSFORMERS_CACHE"]  = dot.paths.model_cache
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Propagate HF token if present in .env
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    # 6. Create directories
    ephemeral_dirs = ("hf_home", "hf_datasets", "model_cache")
    permanent_dirs = ("results_root", "logs_root")
    for attr in ephemeral_dirs + permanent_dirs:
        path = getattr(dot.paths, attr, None)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

    return dot


def results_dir(cfg: DotDict) -> Path:
    """Return (and create) the permanent per-run results directory."""
    p = Path(cfg.paths.results_root) / cfg.paths.run_name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Reproducibility snapshot
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_meta(cfg: DotDict, experiment: str, **extra_params) -> dict:
    """
    Build a _meta dict that fully describes how an output was produced.
    Embed this in every saved result file.

    Args:
        cfg:        Full config DotDict.
        experiment: Name of the experiment (e.g. "steering").
        **extra_params: Experiment-specific hyperparameters to record.

    Returns:
        Dict suitable for embedding as {"_meta": build_meta(...), "results": ...}
    """
    import datetime

    model_key  = cfg.active_model
    model_name = getattr(getattr(cfg.models, model_key, None), "name", "unknown")

    return {
        # When
        "created_at":   datetime.datetime.utcnow().isoformat() + "Z",
        # Where / which code
        "hostname":     socket.gethostname(),
        "git_hash":     _git_hash(),
        # What
        "experiment":   experiment,
        "run_name":     cfg.paths.run_name,
        # Model
        "model_key":    model_key,
        "model_name":   model_name,
        # Data
        "dataset":      cfg.active_dataset,
        "n_questions":  cfg.n_questions,
        "seed":         cfg.seed,
        # Prompt
        "prompt_variant": getattr(cfg.phase1, "prompt_variant", None),
        # Experiment-specific params (e.g. layers, alphas, positions)
        **extra_params,
    }
