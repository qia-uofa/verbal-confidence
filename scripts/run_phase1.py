#!/usr/bin/env python3
"""
Phase 1 — Confidence elicitation.

Depends on phase0 output. Run run_phase0.py first.

Usage:
    python scripts/run_phase1.py [--config config/custom.yaml] \
        [--variant categorical|numerical|minimal]
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
from verbal_confidence.models.loader import load_model_and_tokenizer
from verbal_confidence.experiments.phase0 import run_phase0
from verbal_confidence.experiments.phase1 import run_phase1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default=None)
    ap.add_argument("--model",   default=None)
    ap.add_argument("--variant", default=None,
                    choices=["categorical", "numerical", "minimal"])
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.active_model = args.model
    if args.run_name:
        cfg.paths.run_name = args.run_name
    if args.variant:
        cfg.phase1.prompt_variant = args.variant

    model, tokenizer = load_model_and_tokenizer(cfg)
    p0 = run_phase0(cfg, model, tokenizer)
    p1 = run_phase1(cfg, model, tokenizer, p0)
    print(f"Phase 1 complete: {len(p1)} records")


if __name__ == "__main__":
    main()
