#!/usr/bin/env python3
"""
Print resolved environment variables and config values.
Useful for verifying setup before submitting a SLURM job.

Usage:
    python scripts/print_env.py [--config config/custom.yaml]
"""

import os
import sys
from pathlib import Path

# ── Set HF_HOME before huggingface_hub is imported ───────────────────────────
_dotenv = Path(__file__).parents[1] / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()
if "EPHEMERAL_ROOT" in os.environ:
    os.environ.setdefault("HF_HOME", os.environ["EPHEMERAL_ROOT"] + "/hf_cache")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from verbal_confidence.config import load_config

SECTION = "=" * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    print(SECTION)
    print("ENVIRONMENT VARIABLES")
    print(SECTION)

    env_vars = [
        "EPHEMERAL_ROOT",
        "PERMANENT_ROOT",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_PARTITION",
        "SLURM_TEST_PARTITION",
        "ENV_PATH",
    ]
    for key in env_vars:
        val = os.environ.get(key, "<not set>")
        if "TOKEN" in key and val != "<not set>":
            val = val[:8] + "..."
        print(f"  {key:<30} {val}")

    print()
    print(SECTION)
    print("RESOLVED CONFIG")
    print(SECTION)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"  ERROR loading config: {e}")
        sys.exit(1)

    model_key = cfg.active_model
    model_cfg = getattr(cfg.models, model_key, None)

    print(f"  {'active_model':<30} {model_key}")
    if model_cfg:
        print(f"  {'  name':<30} {getattr(model_cfg, 'name', '?')}")
        print(f"  {'  dtype':<30} {getattr(model_cfg, 'dtype', '?')}")
        print(f"  {'  load_in_4bit':<30} {getattr(model_cfg, 'load_in_4bit', False)}")
        print(f"  {'  load_in_8bit':<30} {getattr(model_cfg, 'load_in_8bit', False)}")
        print(f"  {'  attn_implementation':<30} {getattr(model_cfg, 'attn_implementation', '?')}")
    print(f"  {'active_dataset':<30} {getattr(cfg, 'active_dataset', '?')}")
    print(f"  {'n_questions':<30} {getattr(cfg, 'n_questions', '?')}")
    print(f"  {'seed':<30} {getattr(cfg, 'seed', '?')}")

    print()
    print("  PATHS")
    print(f"  {'  ephemeral_root':<30} {cfg.paths.ephemeral_root}")
    print(f"  {'  permanent_root':<30} {cfg.paths.permanent_root}")
    print(f"  {'  hf_home':<30} {cfg.paths.hf_home}")
    print(f"  {'  hf_datasets':<30} {cfg.paths.hf_datasets}")
    print(f"  {'  results_root':<30} {cfg.paths.results_root}")
    print(f"  {'  logs_root':<30} {cfg.paths.logs_root}")
    print(f"  {'  run_name':<30} {cfg.paths.run_name}")

    print()
    print("  PATH EXISTS CHECK")
    paths_to_check = {
        "ephemeral_root": cfg.paths.ephemeral_root,
        "permanent_root": cfg.paths.permanent_root,
        "hf_home":        cfg.paths.hf_home,
        "results_root":   cfg.paths.results_root,
        "logs_root":      cfg.paths.logs_root,
    }
    for label, path in paths_to_check.items():
        exists = "✓ exists" if Path(path).exists() else "✗ missing"
        print(f"  {'  ' + label:<30} {exists}  ({path})")

    print(SECTION)


if __name__ == "__main__":
    main()
