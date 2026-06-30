"""
Experiment 5 — Linear Probing.

Train Ridge/Logistic probes on layer activations at key positions to
predict the verbal confidence class or correctness. Reports R² / AUROC.
"""

from __future__ import annotations

from itertools import product
from tqdm import tqdm
import numpy as np
import torch

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from verbal_confidence.config import DotDict, results_dir
from verbal_confidence.models.inference import ActCollector
from verbal_confidence.utils.io import load_json, save_json
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import find_positions

log = get_logger(__name__)


def _collect_representations(
    model, tokenizer, phase1_results: list[dict],
    layers: list[int], positions: list[str]
) -> dict[tuple[int, str], np.ndarray]:
    """Collect activation matrix X[n_items, hidden] per (layer, position)."""
    n = len(phase1_results)
    buckets: dict[tuple[int, str], list[np.ndarray]] = {}

    for rec in tqdm(phase1_results, desc="Collecting reps for probing"):
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

    return {k: np.stack(v) for k, v in buckets.items()}


def run_probing(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.probing.output_file
    cached = load_json(out_path)
    if cached is not None:
        log.info("Probing: loaded cached results")
        return cached

    p_cfg    = cfg.probing
    n_layers = model.config.num_hidden_layers
    layers   = list(range(n_layers)) if p_cfg.layers == "all" else p_cfg.layers
    positions = p_cfg.positions
    cv_folds  = p_cfg.cv_folds
    task      = p_cfg.task

    y_cls     = np.array([r["pred_class"] for r in phase1_results])
    y_correct = np.array([int(r["is_correct"]) for r in phase1_results])

    reps = _collect_representations(model, tokenizer, phase1_results, layers, positions)

    results = []
    for (layer, pos_key), X in tqdm(reps.items(), desc="Fitting probes"):
        if task == "regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", Ridge())])
            scores = cross_val_score(pipe, X, y_cls.astype(float),
                                     cv=cv_folds, scoring="r2")
            metric, metric_name = float(np.mean(scores)), "r2"
        else:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=1000))])
            scores = cross_val_score(pipe, X, y_correct,
                                     cv=cv_folds, scoring="roc_auc")
            metric, metric_name = float(np.mean(scores)), "auroc"

        results.append({
            "layer":       layer,
            "position":    pos_key,
            metric_name:   metric,
            "scores":      scores.tolist(),
        })
        log.debug("Layer %d / %s: %s=%.4f", layer, pos_key, metric_name, metric)

    save_json(results, out_path)
    log.info("Probing: saved %d records to %s", len(results), out_path)
    return results
