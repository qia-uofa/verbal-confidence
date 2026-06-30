"""
Token-position utilities.

Key positions used in the paper:
  AC    — Answer Colon       (the ':' after "Answer")
  PANL  — Post-Answer Newline (the '\n' immediately after the answer text)
  PANL1 — PANL + 1           (the token right after PANL)
  CC    — Confidence Colon   (the ':' after "Confidence")
  FCC   — First Confidence Colon (in two-step prompts)
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Confidence class → first token mapping
# Computed lazily per tokenizer to handle different vocabularies.
# ---------------------------------------------------------------------------

CONFIDENCE_CLASSES = [
    "No chance",
    "Really unlikely",
    "Chances are slight",
    "Unlikely",
    "About even",
    "Likely",
    "Good chance",
    "Very good chance",
    "Highly likely",
    "Almost certain",
]


def CLASS_TIDS(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Return the first-token ID for each confidence class."""
    ids = []
    for cls in CONFIDENCE_CLASSES:
        toks = tokenizer.encode(" " + cls, add_special_tokens=False)
        ids.append(toks[0])
    return ids


# ---------------------------------------------------------------------------
# Position finding
# ---------------------------------------------------------------------------

def find_positions(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int | None]:
    """
    Locate key token positions in a single tokenized sequence.

    Returns a dict with keys: ac, panl, panl1, cc, fcc
    Values are integer indices into input_ids (or None if not found).
    """
    ids = input_ids.tolist()
    n = len(ids)

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    colon_id   = tokenizer.encode(":", add_special_tokens=False)[-1]

    # Find all positions of ':' and '\n'
    colons   = [i for i, t in enumerate(ids) if t == colon_id]
    newlines = [i for i, t in enumerate(ids) if t == newline_id]

    # AC  — last colon before the first newline (separates "Answer:" from answer)
    panl_idx = newlines[0] if newlines else None
    ac_idx   = max((c for c in colons if panl_idx is None or c < panl_idx), default=None)

    # PANL — first newline after the answer
    panl_idx = newlines[0] if newlines else None

    # PANL+1
    panl1_idx = (panl_idx + 1) if (panl_idx is not None and panl_idx + 1 < n) else None

    # CC — last colon in sequence (after "Confidence:")
    cc_idx = colons[-1] if colons else None

    # FCC — second-to-last colon (first confidence colon in two-step prompts)
    fcc_idx = colons[-2] if len(colons) >= 2 else None

    return {
        "ac":    ac_idx,
        "panl":  panl_idx,
        "panl1": panl1_idx,
        "cc":    cc_idx,
        "fcc":   fcc_idx,
    }
