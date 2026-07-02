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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.models.inference import ActCollector
from verbal_confidence.utils.io import load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import find_positions

log = get_logger(__name__)


def _collect_representations(
    model, tokenizer, phase1_results: list[dict],
    layers: list[int], positions: list[str]
) -> tuple[dict[tuple[int, str], np.ndarray], dict[tuple[int, str], list[int]]]:
    """
    Collect activation matrix X[n_valid, hidden] per (layer, position).

    Returns (reps, valid_indices) where valid_indices[(layer, pos)] is the
    list of phase1_results indices that had a valid position token — used
    to align y_cls / y_correct with X during probe fitting.
    """
    buckets:      dict[tuple[int, str], list[np.ndarray]] = {}
    valid_indices: dict[tuple[int, str], list[int]]        = {}

    for i, rec in enumerate(tqdm(phase1_results, desc="Collecting reps for probing")):
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
            arr = collector.activations[layer][p]
            # float16 can overflow → NaN; replace before stacking
            if np.isnan(arr).any() or np.isinf(arr).any():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            buckets.setdefault(key, []).append(arr)
            valid_indices.setdefault(key, []).append(i)

    reps = {k: np.stack(v) for k, v in buckets.items()}
    return reps, valid_indices


def run_probing(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> list[dict]:
    out_path = results_dir(cfg) / cfg.probing.output_file
    cached, _ = load_results(out_path)
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

    reps, valid_indices = _collect_representations(
        model, tokenizer, phase1_results, layers, positions
    )

    results = []
    for (layer, pos_key), X in tqdm(reps.items(), desc="Fitting probes"):
        idx = valid_indices[(layer, pos_key)]
        n_valid = len(idx)

        # Need at least cv_folds samples to run cross-validation
        if n_valid < cv_folds:
            log.debug("Layer %d / %s: only %d valid samples, skipping", layer, pos_key, n_valid)
            continue

        # Align labels to the samples that had a valid position token
        y_cls_sub     = y_cls[idx].astype(float)
        y_correct_sub = y_correct[idx]

        # Safety net: replace any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if task == "regression":
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", Ridge()),
            ])
            scores = cross_val_score(pipe, X, y_cls_sub, cv=cv_folds, scoring="r2")
            metric, metric_name = float(np.mean(scores)), "r2"
        else:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
            scores = cross_val_score(pipe, X, y_correct_sub, cv=cv_folds, scoring="roc_auc")
            metric, metric_name = float(np.mean(scores)), "auroc"

        results.append({
            "layer":       layer,
            "position":    pos_key,
            metric_name:   metric,
            "scores":      scores.tolist(),
        })
        log.debug("Layer %d / %s: %s=%.4f", layer, pos_key, metric_name, metric)

    meta = build_meta(cfg, "probing",   layers=p_cfg.layers,        positions=p_cfg.positions,       cv_folds=p_cfg.cv_folds, task=p_cfg.task)
    save_with_meta(results, out_path, meta)
    log.info("Probing: saved %d records to %s", len(results), out_path)
    return results
