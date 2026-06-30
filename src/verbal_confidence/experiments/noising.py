"""
Experiment 3 — Activation Noising (mean ablation).

Replace the activation at a specific layer/position with the mean
activation computed over a reference set, then measure confidence change.
"""

from __future__ import annotations

from itertools import product
from tqdm import tqdm
import numpy as np
import torch

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.models.inference import ActCollector, forward_logits, make_noise_hook
from verbal_confidence.utils.io import load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, find_positions

log = get_logger(__name__)


def _compute_mean_activations(
    model, tokenizer, reference_items: list[dict], layers: list[int], positions: list[str]
) -> dict[tuple[int, str], np.ndarray]:
    """Compute mean activation per (layer, position) over reference items."""
    buckets: dict[tuple[int, str], list[np.ndarray]] = {}

    for rec in tqdm(reference_items, desc="Computing mean activations"):
        prompt = rec["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos_map = find_positions(inputs["input_ids"][0], tokenizer)

        with ActCollector(model, layers=layers) as collector:
            with torch.no_grad():
                model(**inputs)

        for layer, pos_key in product(layers, positions):
            p = pos_map.get(pos_key)
            if p is None or layer not in collector.activations:
                continue
            key = (layer, pos_key)
            buckets.setdefault(key, []).append(collector.activations[layer][p])

    return {k: np.mean(v, axis=0) for k, v in buckets.items()}


def run_noising(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.noising.output_file
    cached, _ = load_results(out_path)
    if cached is not None:
        log.info("Noising: loaded cached results")
        return cached

    layers     = cfg.noising.layers
    positions  = cfg.noising.positions
    n_mean     = cfg.noising.n_mean_samples
    class_tids = CLASS_TIDS(tokenizer)

    reference = phase1_results[:n_mean]
    log.info("Noising: computing mean activations over %d reference items", len(reference))
    mean_acts = _compute_mean_activations(model, tokenizer, reference, layers, positions)

    results = []
    for rec in tqdm(phase1_results, desc="Noising eval"):
        prompt = rec["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos_map = find_positions(inputs["input_ids"][0], tokenizer)

        baseline = forward_logits(model, tokenizer, prompt, class_tids)

        for layer, pos_key in product(layers, positions):
            p = pos_map.get(pos_key)
            key = (layer, pos_key)
            if p is None or key not in mean_acts:
                continue

            mean_vec = torch.tensor(mean_acts[key], dtype=torch.float32)
            decoder  = model.model.layers[layer]
            hook     = decoder.register_forward_hook(make_noise_hook(mean_vec, p))
            try:
                noised = forward_logits(model, tokenizer, prompt, class_tids)
            finally:
                hook.remove()

            results.append({
                "layer":           layer,
                "position":        pos_key,
                "baseline_class":  baseline["pred_class"],
                "noised_class":    noised["pred_class"],
                "delta_class":     noised["pred_class"] - baseline["pred_class"],
                "logit_delta":     float(
                    np.mean(np.abs(
                        np.array(noised["logits_cls"]) - np.array(baseline["logits_cls"])
                    ))
                ),
            })

    meta = build_meta(cfg, "noising",  layers=cfg.noising.layers,  positions=cfg.noising.positions, n_mean_samples=cfg.noising.n_mean_samples)
    save_with_meta(results, out_path, meta)
    log.info("Noising: saved %d records to %s", len(results), out_path)
    return results
