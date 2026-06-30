"""
Dataset loading utilities.

Supports: TriviaQA, BigMath, MMLU.
All datasets are loaded through HuggingFace datasets; the cache directory is
controlled by HF_DATASETS_CACHE (set by config.load_config()).
"""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from verbal_confidence.config import DotDict
from verbal_confidence.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Normalised record format
# Every dataset returns a list of dicts with keys:
#   question : str
#   answer   : str   (primary answer / gold label)
#   answers  : list[str]  (all acceptable answers)
#   source   : str   (dataset name)
# ---------------------------------------------------------------------------

def _load_trivia_qa(cfg: DotDict) -> list[dict[str, Any]]:
    ds_cfg = cfg.datasets.trivia_qa
    log.info("Loading TriviaQA (%s, %s, %s)", ds_cfg.repo, ds_cfg.config, ds_cfg.split)
    ds = load_dataset(ds_cfg.repo, ds_cfg.config, split=ds_cfg.split)
    records = []
    for ex in ds:
        answers = ex["answer"]["aliases"]
        records.append({
            "question": ex["question"],
            "answer":   ex["answer"]["value"],
            "answers":  answers,
            "source":   "trivia_qa",
        })
    return records


def _load_bigmath(cfg: DotDict) -> list[dict[str, Any]]:
    ds_cfg = cfg.datasets.bigmath
    log.info("Loading BigMath (%s, %s)", ds_cfg.repo, ds_cfg.split)
    ds = load_dataset(ds_cfg.repo, split=ds_cfg.split)
    records = []
    for ex in ds:
        records.append({
            "question": ex["problem"],
            "answer":   str(ex["answer"]),
            "answers":  [str(ex["answer"])],
            "source":   "bigmath",
        })
    return records


def _load_mmlu(cfg: DotDict) -> list[dict[str, Any]]:
    ds_cfg = cfg.datasets.mmlu
    log.info("Loading MMLU (%s, %s, %s)", ds_cfg.repo, ds_cfg.config, ds_cfg.split)
    ds = load_dataset(ds_cfg.repo, ds_cfg.config, split=ds_cfg.split)
    choice_labels = ["A", "B", "C", "D"]
    records = []
    for ex in ds:
        choices = ex["choices"]
        ans_idx = ex["answer"]
        ans_text = choices[ans_idx]
        records.append({
            "question": ex["question"] + "\n" + "\n".join(
                f"{choice_labels[i]}. {choices[i]}" for i in range(len(choices))
            ),
            "answer":   f"{choice_labels[ans_idx]}. {ans_text}",
            "answers":  [f"{choice_labels[ans_idx]}. {ans_text}", choice_labels[ans_idx]],
            "source":   "mmlu",
        })
    return records


_LOADERS = {
    "trivia_qa": _load_trivia_qa,
    "bigmath":   _load_bigmath,
    "mmlu":      _load_mmlu,
}


def load_dataset_split(cfg: DotDict, dataset_key: str | None = None) -> list[dict[str, Any]]:
    """
    Load the specified dataset (default: cfg.active_dataset).

    Returns a flat list of normalised records.
    """
    key = dataset_key or cfg.active_dataset
    if key not in _LOADERS:
        raise ValueError(f"Unknown dataset '{key}'. Choose from: {list(_LOADERS)}")
    return _LOADERS[key](cfg)


def sample_questions(
    records: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample n questions."""
    rng = random.Random(seed)
    if n >= len(records):
        return records
    return rng.sample(records, n)
