#!/usr/bin/env python3
"""
Run the full experiment pipeline end-to-end.

Experiments run in order:
  Phase 0 → Phase 1 → Steering → Patching → Noising → Swap
  → Probing → Variance Partitioning → Attention Blocking → Generalization

Each experiment caches its output; re-running skips completed stages.

Usage:
    python scripts/run_all_experiments.py \
        [--config config/custom.yaml] \
        [--model primary|qwen|magistral] \
        [--run-name my_run] \
        [--skip steering patching]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from verbal_confidence.config import load_config
from verbal_confidence.models.loader import load_model_and_tokenizer
from verbal_confidence.experiments import (
    run_phase0, run_phase1,
    run_steering, run_patching, run_noising, run_swap,
    run_probing, run_variance_partitioning,
    run_attention_blocking, run_generalization,
)
from verbal_confidence.utils.logging import get_logger

log = get_logger("run_all")

ALL_EXPERIMENTS = [
    "steering", "patching", "noising", "swap",
    "probing", "variance_partitioning", "attention_blocking",
    "generalization",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default=None)
    ap.add_argument("--model",    default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--skip",     nargs="*", default=[], choices=ALL_EXPERIMENTS,
                    metavar="EXP", help="Experiments to skip")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.active_model = args.model
    if args.run_name:
        cfg.paths.run_name = args.run_name

    skip = set(args.skip)

    # ── Print environment and config on startup ──────────────────────────────
    import os, yaml as _yaml
    log.info("=" * 60)
    log.info("ENVIRONMENT")
    for key in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
                "HUGGING_FACE_HUB_TOKEN", "EPHEMERAL_ROOT", "PERMANENT_ROOT",
                "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        val = os.environ.get(key, "<not set>")
        if "TOKEN" in key and val != "<not set>":
            val = val[:8] + "..."   # redact token
        log.info("  %-30s %s", key, val)
    log.info("CONFIG")
    log.info("  active_model   = %s (%s)", cfg.active_model,
             getattr(getattr(cfg.models, cfg.active_model, {}), "name", "?"))
    log.info("  active_dataset = %s", getattr(cfg, "active_dataset", "?"))
    log.info("  n_questions    = %s", getattr(cfg, "n_questions", "?"))
    log.info("  seed           = %s", getattr(cfg, "seed", "?"))
    log.info("  ephemeral_root = %s", cfg.paths.ephemeral_root)
    log.info("  permanent_root = %s", cfg.paths.permanent_root)
    log.info("  hf_home        = %s", cfg.paths.hf_home)
    log.info("  results_root   = %s", cfg.paths.results_root)
    log.info("  run_name       = %s", cfg.paths.run_name)
    log.info("=" * 60)
    # ─────────────────────────────────────────────────────────────────────────

    log.info("Starting full pipeline. Skip: %s", skip or "none")

    model, tokenizer = load_model_and_tokenizer(cfg)

    log.info("=== Phase 0: Answer Generation ===")
    p0 = run_phase0(cfg, model, tokenizer)

    log.info("=== Phase 1: Confidence Elicitation ===")
    p1 = run_phase1(cfg, model, tokenizer, p0)

    if "steering" not in skip:
        log.info("=== Exp 1: Activation Steering ===")
        run_steering(cfg, model, tokenizer, p1)

    if "patching" not in skip:
        log.info("=== Exp 2: Activation Patching ===")
        run_patching(cfg, model, tokenizer, p1)

    if "noising" not in skip:
        log.info("=== Exp 3: Activation Noising ===")
        run_noising(cfg, model, tokenizer, p1)

    if "swap" not in skip:
        log.info("=== Exp 4: Activation Swap ===")
        run_swap(cfg, model, tokenizer, p1)

    if "probing" not in skip:
        log.info("=== Exp 5: Linear Probing ===")
        run_probing(cfg, model, tokenizer, p1)

    if "variance_partitioning" not in skip:
        log.info("=== Exp 6: Variance Partitioning ===")
        run_variance_partitioning(cfg, model, tokenizer, p1)

    if "attention_blocking" not in skip:
        log.info("=== Exp 7: Attention Blocking ===")
        run_attention_blocking(cfg, model, tokenizer, p1)

    if "generalization" not in skip:
        log.info("=== Generalization ===")
        run_generalization(cfg, model, tokenizer)

    log.info("Pipeline complete. Results in: %s/%s",
             cfg.paths.results_root, cfg.paths.run_name)


if __name__ == "__main__":
    main()
