"""
Phase 0 — Generate model answers for each question.

Output: list of {question, gold_answer, model_answer} dicts
"""

from __future__ import annotations

from pathlib import Path
from tqdm import tqdm

from verbal_confidence.config import DotDict, results_dir
from verbal_confidence.data.loader import load_dataset_split, sample_questions
from verbal_confidence.data.prompts import phase0_prompt
from verbal_confidence.models.inference import generate_answer
from verbal_confidence.utils.io import load_json, save_json
from verbal_confidence.utils.logging import get_logger

log = get_logger(__name__)


def run_phase0(cfg: DotDict, model, tokenizer) -> list[dict]:
    out_path = results_dir(cfg) / cfg.phase0.output_file
    cached = load_json(out_path)
    if cached is not None:
        log.info("Phase 0: loaded %d cached answers from %s", len(cached), out_path)
        return cached

    records = load_dataset_split(cfg)
    records = sample_questions(records, cfg.n_questions, cfg.seed)
    log.info("Phase 0: generating answers for %d questions", len(records))

    results = []
    for rec in tqdm(records, desc="Phase 0"):
        prompt = phase0_prompt(rec["question"])
        answer = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=cfg.phase0.max_new_tokens,
            temperature=cfg.phase0.temperature,
        )
        results.append({
            "question":     rec["question"],
            "gold_answers": rec["answers"],
            "model_answer": answer,
            "source":       rec["source"],
        })

    save_json(results, out_path)
    log.info("Phase 0: saved to %s", out_path)
    return results
