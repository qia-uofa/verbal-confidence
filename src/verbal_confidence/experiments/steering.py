"""
Experiment 1 — Activation Steering.

Compute steering vectors (mean diff between high- and low-confidence
activations) and apply them at specified layers/positions. Measure
change in predicted confidence class.
"""

from __future__ import annotations

import contextlib
from itertools import product
from tqdm import tqdm
import numpy as np
import torch

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.models.inference import ActCollector, forward_logits, make_patch_hook
from verbal_confidence.utils.io import load_npz, load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, find_positions

log = get_logger(__name__)

# High-confidence classes: indices 7-9; low-confidence: indices 0-2
HIGH_CLS = [7, 8, 9]
LOW_CLS  = [0, 1, 2]


def _compute_steering_vectors(
    model, tokenizer, phase1_results: list[dict], layers: list[int]
) -> dict[str, dict[int, np.ndarray]]:
    """
    Collect activations at PANL and CC positions for high/low confidence items,
    then compute high_mean - low_mean per layer.

    Returns: {position_key: {layer: steering_vector}}
    """
    high_acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    low_acts:  dict[int, list[np.ndarray]] = {l: [] for l in layers}

    for rec in tqdm(phase1_results, desc="Collecting acts for steering"):
        pred_cls = rec["pred_class"]
        if pred_cls not in HIGH_CLS and pred_cls not in LOW_CLS:
            continue

        prompt = rec["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos_map = find_positions(inputs["input_ids"][0], tokenizer)
        panl_pos = pos_map["panl"]
        if panl_pos is None:
            continue

        with ActCollector(model, layers=layers) as collector:
            with torch.no_grad():
                model(**inputs)

        bucket = high_acts if pred_cls in HIGH_CLS else low_acts
        for l in layers:
            if l in collector.activations:
                bucket[l].append(collector.activations[l][panl_pos])

    vectors: dict[str, dict[int, np.ndarray]] = {"panl": {}}
    for l in layers:
        if high_acts[l] and low_acts[l]:
            high_mean = np.mean(high_acts[l], axis=0)
            low_mean  = np.mean(low_acts[l], axis=0)
            vectors["panl"][l] = high_mean - low_mean

    return vectors


def run_steering(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.steering.output_file
    cached, _ = load_results(out_path)
    if cached is not None:
        log.info("Steering: loaded cached results from %s", out_path)
        return cached

    s_cfg   = cfg.steering
    layers  = s_cfg.layers
    alphas  = s_cfg.alphas
    class_tids = CLASS_TIDS(tokenizer)

    log.info("Steering: computing steering vectors on %d items", len(phase1_results))
    vectors = _compute_steering_vectors(model, tokenizer, phase1_results, layers)

    results = []
    for rec in tqdm(phase1_results[:50], desc="Steering eval"):  # subset for speed
        prompt  = rec["prompt"]
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos_map = find_positions(inputs["input_ids"][0], tokenizer)
        panl_pos = pos_map["panl"]
        if panl_pos is None:
            continue

        baseline = forward_logits(model, tokenizer, prompt, class_tids)

        for layer, alpha in product(layers, alphas):
            if layer not in vectors.get("panl", {}):
                continue
            sv = torch.tensor(vectors["panl"][layer] * alpha, dtype=torch.float32)

            # Register patch hook
            decoder = model.model.layers[layer]
            hook_handle = decoder.register_forward_hook(
                make_patch_hook(sv, panl_pos)
            )
            try:
                steered = forward_logits(model, tokenizer, prompt, class_tids)
            finally:
                hook_handle.remove()

            delta_class = steered["pred_class"] - baseline["pred_class"]
            results.append({
                "question":      rec["question"],
                "baseline_class": baseline["pred_class"],
                "steered_class":  steered["pred_class"],
                "delta_class":    delta_class,
                "layer":          layer,
                "alpha":          alpha,
                "position":       "panl",
            })

    meta = build_meta(cfg, "steering", layers=cfg.steering.layers, positions=cfg.steering.positions, alphas=cfg.steering.alphas)
    save_with_meta(results, out_path, meta)
    log.info("Steering: saved %d records to %s", len(results), out_path)
    return results
