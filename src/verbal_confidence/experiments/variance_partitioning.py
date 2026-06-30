"""
Experiment 6 — Variance Partitioning.

Fit Ridge regressors to predict log P(confidence class) from different
baselines (question embedding, answer embedding, Q+A, layer reprs).
Report R² unique to each predictor set.
"""

from __future__ import annotations

from itertools import combinations
from tqdm import tqdm
import numpy as np
import torch

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.models.inference import ActCollector, forward_logits
from verbal_confidence.utils.io import load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, find_positions

log = get_logger(__name__)


def _fit_r2(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    """Cross-validated R² of a Ridge regression pipeline."""
    from sklearn.model_selection import cross_val_score
    pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge())])
    scores = cross_val_score(pipe, X, y, cv=n_folds, scoring="r2")
    return float(np.mean(scores))


def run_variance_partitioning(
    cfg: DotDict,
    model,
    tokenizer,
    phase1_results: list[dict],
) -> dict:
    vp_path = results_dir(cfg) / cfg.variance_partitioning.output_file
    lp_path = results_dir(cfg) / cfg.variance_partitioning.logprobs_file

    cached_vp, _ = load_results(vp_path)
    if cached_vp is not None:
        log.info("VP: loaded cached results")
        return cached_vp

    vp_cfg = cfg.variance_partitioning
    positions = vp_cfg.positions
    class_tids = CLASS_TIDS(tokenizer)

    # ---------- Collect log-probs and representations ----------
    log.info("VP: collecting log-probs and representations")
    y_log = []      # log P(predicted class) — regression target

    # Representations: question-only, answer-only, Q+A at each position
    from verbal_confidence.data.prompts import phase0_prompt, phase1_categorical
    from verbal_confidence.utils.tokens import CONFIDENCE_CLASSES

    X_q, X_a, X_qa = [], [], []  # [n, hidden]
    layer_reps: dict[str, list[np.ndarray]] = {p: [] for p in positions}
    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2

    for rec in tqdm(phase1_results, desc="VP collection"):
        # Log-prob of predicted class
        probs = np.array(rec["probs_cls"])
        y_log.append(float(np.log(probs[rec["pred_class"]] + 1e-8)))

        prompt_full = rec["prompt"]
        prompt_q    = phase0_prompt(rec["question"])
        prompt_a    = f"Answer: {rec['model_answer']}\nConfidence:"

        in_q  = tokenizer(prompt_q,    return_tensors="pt").to(model.device)
        in_a  = tokenizer(prompt_a,    return_tensors="pt").to(model.device)
        in_qa = tokenizer(prompt_full, return_tensors="pt").to(model.device)
        pos_map = find_positions(in_qa["input_ids"][0], tokenizer)

        # Question repr at last token
        with ActCollector(model, layers=[mid_layer]) as c:
            with torch.no_grad():
                model(**in_q)
        X_q.append(c.activations[mid_layer][-1] if mid_layer in c.activations else np.zeros(model.config.hidden_size))

        # Answer repr
        with ActCollector(model, layers=[mid_layer]) as c:
            with torch.no_grad():
                model(**in_a)
        X_a.append(c.activations[mid_layer][-1] if mid_layer in c.activations else np.zeros(model.config.hidden_size))

        # Q+A repr at key positions
        with ActCollector(model, layers=[mid_layer]) as c:
            with torch.no_grad():
                model(**in_qa)
        X_qa.append(c.activations[mid_layer][-1] if mid_layer in c.activations else np.zeros(model.config.hidden_size))

        for pos_key in positions:
            p = pos_map.get(pos_key)
            if p is not None and mid_layer in c.activations:
                layer_reps[pos_key].append(c.activations[mid_layer][p])
            else:
                layer_reps[pos_key].append(np.zeros(model.config.hidden_size))

    y = np.array(y_log)
    X_q  = np.stack(X_q)
    X_a  = np.stack(X_a)
    X_qa = np.stack(X_qa)

    # ---------- Variance partitioning ----------
    vp_results: dict[str, float] = {}
    baselines = {"question": X_q, "answer": X_a, "question_answer": X_qa}
    for pos_key in positions:
        baselines[f"layer_{pos_key}"] = np.stack(layer_reps[pos_key])

    # R² for each individual predictor
    for name, X in baselines.items():
        vp_results[f"r2_{name}"] = _fit_r2(X, y)

    # R²_unique via semi-partial R²
    predictor_names = list(baselines.keys())
    for name in predictor_names:
        others = [n for n in predictor_names if n != name]
        X_others = np.hstack([baselines[n] for n in others])
        X_all    = np.hstack([baselines[n] for n in predictor_names])
        r2_all    = _fit_r2(X_all,    y)
        r2_others = _fit_r2(X_others, y)
        vp_results[f"r2_unique_{name}"] = max(0.0, r2_all - r2_others)

    meta = build_meta(cfg, "variance_partitioning", positions=vp_cfg.positions, baselines=vp_cfg.baselines)
    save_with_meta(vp_results, vp_path, meta)
    log.info("VP: saved to %s", vp_path)
    return vp_results
