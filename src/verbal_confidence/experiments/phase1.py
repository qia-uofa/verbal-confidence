"""
Phase 1 — Elicit verbal confidence and calibration metrics.

For each (question, model_answer) pair from phase0, run the confidence
elicitation prompt and record class probabilities, predicted class, etc.
"""

from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
import numpy as np

from verbal_confidence.config import DotDict, results_dir, build_meta
from verbal_confidence.data.prompts import get_phase1_prompt
from verbal_confidence.models.inference import forward_logits
from verbal_confidence.utils.io import load_results, save_with_meta
from verbal_confidence.utils.logging import get_logger
from verbal_confidence.utils.tokens import CLASS_TIDS, CONFIDENCE_CLASSES

log = get_logger(__name__)


def _correctness(model_answer: str, gold_answers: list[str]) -> bool:
    """Loose string-match correctness check."""
    ma = model_answer.strip().lower()
    return any(ma in gold.lower() or gold.lower() in ma for gold in gold_answers)


def run_phase1(
    cfg: DotDict,
    model,
    tokenizer,
    phase0_results: list[dict],
    variant: str | None = None,
) -> list[dict]:
    variant = variant or cfg.phase1.prompt_variant
    out_path = results_dir(cfg) / cfg.phase1.output_file
    cached, _ = load_results(out_path)
    if cached is not None:
        log.info("Phase 1: loaded %d cached records from %s", len(cached), out_path)
        return cached

    class_tids = CLASS_TIDS(tokenizer)
    log.info("Phase 1: eliciting confidence (%s) for %d items", variant, len(phase0_results))

    results = []
    for rec in tqdm(phase0_results, desc="Phase 1"):
        prompt = get_phase1_prompt(variant, rec["question"], rec["model_answer"])
        metrics = forward_logits(model, tokenizer, prompt, class_tids)
        is_correct = _correctness(rec["model_answer"], rec["gold_answers"])

        results.append({
            **rec,
            "prompt":      prompt,
            "pred_class":  metrics["pred_class"],
            "pred_label":  metrics["pred_label"],
            "probs_cls":   metrics["probs_cls"].tolist(),
            "logits_cls":  metrics["logits_cls"].tolist(),
            "is_correct":  is_correct,
            "variant":     variant,
        })

    # Calibration summary
    confidences = [r["probs_cls"][r["pred_class"]] for r in results]
    accuracies  = [float(r["is_correct"]) for r in results]
    log.info(
        "Phase 1: mean confidence=%.3f, mean accuracy=%.3f",
        np.mean(confidences), np.mean(accuracies),
    )

    meta = build_meta(cfg, "phase1", prompt_variant=variant)
    save_with_meta(results, out_path, meta)
    log.info("Phase 1: saved to %s", out_path)
    return results
