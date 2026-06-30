"""
Experiment 2 — Activation Patching.

Patch activations from a "source" prompt into a "target" prompt at
specific layers and positions, then measure confidence change.
Source: high-confidence Q/A pair. Target: low-confidence Q/A pair.
"""

from __future__ import annotations

import random
from itertools import product
from tqdm import tqdm
import numpy as np
import torch

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.models.inference import ActCollector, forward_logits, make_patch_hook
from verbal_confidence.utils.io import load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, find_positions

log = get_logger(__name__)

HIGH_CLS = [7, 8, 9]
LOW_CLS  = [0, 1, 2]


def run_patching(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.patching.output_file
    cached, _ = load_results(out_path)
    if cached is not None:
        log.info("Patching: loaded cached results")
        return cached

    layers     = cfg.patching.layers
    positions  = cfg.patching.positions   # list of position key strings
    class_tids = CLASS_TIDS(tokenizer)

    high_items = [r for r in phase1_results if r["pred_class"] in HIGH_CLS]
    low_items  = [r for r in phase1_results if r["pred_class"] in LOW_CLS]

    rng = random.Random(cfg.seed)
    n_pairs = min(50, len(high_items), len(low_items))
    sources  = rng.sample(high_items, n_pairs)
    targets  = rng.sample(low_items,  n_pairs)

    results = []
    for src, tgt in tqdm(zip(sources, targets), total=n_pairs, desc="Patching"):
        src_prompt = src["prompt"]
        tgt_prompt = tgt["prompt"]

        src_inputs = tokenizer(src_prompt, return_tensors="pt").to(model.device)
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt").to(model.device)

        src_pos = find_positions(src_inputs["input_ids"][0], tokenizer)
        tgt_pos = find_positions(tgt_inputs["input_ids"][0], tokenizer)

        # Collect source activations
        with ActCollector(model, layers=layers) as src_collector:
            with torch.no_grad():
                model(**src_inputs)

        baseline = forward_logits(model, tokenizer, tgt_prompt, class_tids)

        for layer, pos_key in product(layers, positions):
            src_p = src_pos.get(pos_key)
            tgt_p = tgt_pos.get(pos_key)
            if src_p is None or tgt_p is None:
                continue
            if layer not in src_collector.activations:
                continue

            patch_vec = torch.tensor(
                src_collector.activations[layer][src_p], dtype=torch.float32
            )
            decoder = model.model.layers[layer]
            hook = decoder.register_forward_hook(make_patch_hook(patch_vec, tgt_p))
            try:
                patched = forward_logits(model, tokenizer, tgt_prompt, class_tids)
            finally:
                hook.remove()

            results.append({
                "layer":            layer,
                "position":         pos_key,
                "baseline_class":   baseline["pred_class"],
                "patched_class":    patched["pred_class"],
                "delta_class":      patched["pred_class"] - baseline["pred_class"],
                "src_class":        src["pred_class"],
                "tgt_class":        tgt["pred_class"],
            })

    meta = build_meta(cfg, "patching", layers=cfg.patching.layers, positions=cfg.patching.positions)
    save_with_meta(results, out_path, meta)
    log.info("Patching: saved %d records to %s", len(results), out_path)
    return results
