"""
Generalization experiments.

Repeats phase0+phase1 across:
  - Different prompt variants (categorical / numerical / minimal)
  - Different models (Qwen 2.5 7B, Magistral Small)
  - Different datasets (BigMath, MMLU)
"""

from __future__ import annotations

from verbal_confidence.config import DotDict, results_dir
from verbal_confidence.data.loader import load_dataset_split, sample_questions
from verbal_confidence.models.loader import switch_model
from verbal_confidence.utils.io import load_json, save_json
from verbal_confidence.utils.logging import get_logger
from .phase0 import run_phase0
from .phase1 import run_phase1

log = get_logger(__name__)


def run_generalization(
    cfg: DotDict,
    model,
    tokenizer,
) -> dict:
    out_path = results_dir(cfg) / cfg.generalization.output_file
    cached = load_json(out_path)
    if cached is not None:
        log.info("Generalization: loaded cached results")
        return cached

    g_cfg   = cfg.generalization
    results = {}

    # ---- 1. Prompt variant sweep (primary model, primary dataset) ----
    for variant in g_cfg.prompt_variants:
        log.info("Generalization: variant=%s", variant)
        p0 = run_phase0(cfg, model, tokenizer)
        p1 = run_phase1(cfg, model, tokenizer, p0, variant=variant)
        results[f"variant_{variant}"] = {
            "mean_pred_class": sum(r["pred_class"] for r in p1) / len(p1),
            "n": len(p1),
        }

    # ---- 2. Extra models ----
    for model_key in g_cfg.extra_models:
        log.info("Generalization: model=%s", model_key)
        m, tok = switch_model(cfg, model_key)
        p0 = run_phase0(cfg, m, tok)
        p1 = run_phase1(cfg, m, tok, p0)
        results[f"model_{model_key}"] = {
            "mean_pred_class": sum(r["pred_class"] for r in p1) / len(p1),
            "n": len(p1),
        }

    # ---- 3. Extra datasets (primary model) ----
    for ds_key in g_cfg.extra_datasets:
        log.info("Generalization: dataset=%s", ds_key)
        records = load_dataset_split(cfg, dataset_key=ds_key)
        records = sample_questions(records, cfg.n_questions, cfg.seed)
        # Temporarily override active_dataset — use a copy of phase0 records
        p0_fake = [
            {"question": r["question"], "gold_answers": r["answers"],
             "model_answer": r["answer"], "source": r["source"]}
            for r in records
        ]
        p1 = run_phase1(cfg, model, tokenizer, p0_fake)
        results[f"dataset_{ds_key}"] = {
            "mean_pred_class": sum(r["pred_class"] for r in p1) / len(p1),
            "n": len(p1),
        }

    save_json(results, out_path)
    log.info("Generalization: saved to %s", out_path)
    return results
