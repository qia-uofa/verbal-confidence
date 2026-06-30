"""
Experiment 4 — Activation Swap (2×2 factorial).

For a pair of questions Q1/Q2, swap the activations at PANL (and/or CC)
between the two prompts. Measures whether the confidence "follows"
the activation or the text.
"""

from __future__ import annotations

import random
from itertools import product
from tqdm import tqdm
import numpy as np
import torch

from verbal_confidence.config import DotDict, results_dir
from verbal_confidence.models.inference import ActCollector, forward_logits, make_patch_hook
from verbal_confidence.utils.io import load_json, save_json
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, find_positions

log = get_logger(__name__)


def run_swap(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.swap.output_file
    cached = load_json(out_path)
    if cached is not None:
        log.info("Swap: loaded cached results")
        return cached

    layers     = cfg.swap.layers
    positions  = cfg.swap.positions
    n_pairs    = min(cfg.swap.n_pairs, len(phase1_results) // 2)
    class_tids = CLASS_TIDS(tokenizer)

    rng = random.Random(cfg.seed)
    shuffled = rng.sample(phase1_results, len(phase1_results))
    pairs = list(zip(shuffled[:n_pairs], shuffled[n_pairs:2*n_pairs]))

    results = []
    for (rec_a, rec_b) in tqdm(pairs, desc="Swap"):
        pa = rec_a["prompt"]
        pb = rec_b["prompt"]

        in_a = tokenizer(pa, return_tensors="pt").to(model.device)
        in_b = tokenizer(pb, return_tensors="pt").to(model.device)
        pos_a = find_positions(in_a["input_ids"][0], tokenizer)
        pos_b = find_positions(in_b["input_ids"][0], tokenizer)

        with ActCollector(model, layers=layers) as ca:
            with torch.no_grad():
                model(**in_a)
        with ActCollector(model, layers=layers) as cb:
            with torch.no_grad():
                model(**in_b)

        base_a = forward_logits(model, tokenizer, pa, class_tids)
        base_b = forward_logits(model, tokenizer, pb, class_tids)

        for layer, pos_key in product(layers, positions):
            idx_a = pos_a.get(pos_key)
            idx_b = pos_b.get(pos_key)
            if idx_a is None or idx_b is None:
                continue
            if layer not in ca.activations or layer not in cb.activations:
                continue

            vec_a = torch.tensor(ca.activations[layer][idx_a], dtype=torch.float32)
            vec_b = torch.tensor(cb.activations[layer][idx_b], dtype=torch.float32)

            # A with B's activation
            dec = model.model.layers[layer]
            h = dec.register_forward_hook(make_patch_hook(vec_b, idx_a))
            try:
                a_with_b = forward_logits(model, tokenizer, pa, class_tids)
            finally:
                h.remove()

            # B with A's activation
            h = dec.register_forward_hook(make_patch_hook(vec_a, idx_b))
            try:
                b_with_a = forward_logits(model, tokenizer, pb, class_tids)
            finally:
                h.remove()

            results.append({
                "layer":       layer,
                "position":    pos_key,
                "base_a":      base_a["pred_class"],
                "base_b":      base_b["pred_class"],
                "a_with_b":    a_with_b["pred_class"],
                "b_with_a":    b_with_a["pred_class"],
                # Did A's confidence shift toward B's?
                "a_toward_b":  abs(a_with_b["pred_class"] - base_b["pred_class"])
                               < abs(base_a["pred_class"] - base_b["pred_class"]),
                "b_toward_a":  abs(b_with_a["pred_class"] - base_a["pred_class"])
                               < abs(base_b["pred_class"] - base_a["pred_class"]),
            })

    save_json(results, out_path)
    log.info("Swap: saved %d records to %s", len(results), out_path)
    return results
