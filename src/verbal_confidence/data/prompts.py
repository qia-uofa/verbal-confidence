"""
Prompt templates (exactly as in paper).

Phase 0 — generate answer.
Phase 1 — elicit verbal confidence (categorical / numerical / minimal).
Magistral uses its own system/user split.
"""

from __future__ import annotations

from verbal_confidence.config import DotDict

# ---------------------------------------------------------------------------
# Confidence class list (for categorical prompts)
# ---------------------------------------------------------------------------
_CLASS_STR = (
    '"No chance", "Really unlikely", "Chances are slight", "Unlikely", '
    '"About even", "Likely", "Good chance", "Very good chance", '
    '"Highly likely", "Almost certain"'
)

# ---------------------------------------------------------------------------
# Phase 0 — Answer generation
# ---------------------------------------------------------------------------

PHASE0_TEMPLATE = (
    "Question: {question}\n"
    "Answer:"
)


def phase0_prompt(question: str) -> str:
    return PHASE0_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Phase 1 — Confidence elicitation (categorical)
# ---------------------------------------------------------------------------

PHASE1_CAT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "On a scale from {classes}, how confident are you that the answer is correct?\n"
    "Confidence:"
)


def phase1_categorical(question: str, answer: str) -> str:
    return PHASE1_CAT_TEMPLATE.format(
        question=question, answer=answer, classes=_CLASS_STR
    )


# ---------------------------------------------------------------------------
# Phase 1 — Confidence elicitation (numerical, 0–100)
# ---------------------------------------------------------------------------

PHASE1_NUM_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "How confident are you that the answer is correct? "
    "Give a number between 0 and 100.\n"
    "Confidence:"
)


def phase1_numerical(question: str, answer: str) -> str:
    return PHASE1_NUM_TEMPLATE.format(question=question, answer=answer)


# ---------------------------------------------------------------------------
# Phase 1 — Minimal prompt (for attention blocking experiment)
# ---------------------------------------------------------------------------

PHASE1_MINIMAL_TEMPLATE = (
    "Q: {question}\nA: {answer}\nC:"
)


def phase1_minimal(question: str, answer: str) -> str:
    return PHASE1_MINIMAL_TEMPLATE.format(question=question, answer=answer)


# ---------------------------------------------------------------------------
# Magistral-style prompts (system + user)
# ---------------------------------------------------------------------------

MAGISTRAL_SYSTEM = (
    "You are a helpful assistant. When asked for your confidence, "
    "choose exactly one of these expressions: " + _CLASS_STR + "."
)

MAGISTRAL_PHASE1_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "How confident are you? Reply with exactly one expression from the list."
)


def magistral_phase1(question: str, answer: str) -> list[dict[str, str]]:
    """Return a list of chat messages for Magistral."""
    return [
        {"role": "system",    "content": MAGISTRAL_SYSTEM},
        {"role": "user",      "content": MAGISTRAL_PHASE1_TEMPLATE.format(
            question=question, answer=answer)},
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def get_phase1_prompt(variant: str, question: str, answer: str) -> str:
    """Return phase-1 prompt string for the given variant."""
    dispatch = {
        "categorical": phase1_categorical,
        "numerical":   phase1_numerical,
        "minimal":     phase1_minimal,
    }
    if variant not in dispatch:
        raise ValueError(f"Unknown prompt variant '{variant}'. Choose from {list(dispatch)}")
    return dispatch[variant](question, answer)
