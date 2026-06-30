"""
Model + tokenizer loading.

Respects HF_HOME / TRANSFORMERS_CACHE set by config.load_config().
Supports bfloat16, optional 8-bit, and ROCm-safe attn_implementation.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verbal_confidence.config import DotDict
from verbal_confidence.utils.logging import get_logger

log = get_logger(__name__)


def _model_cfg(cfg: DotDict, model_key: str | None = None) -> DotDict:
    key = model_key or cfg.active_model
    return getattr(cfg.models, key)


def load_model_and_tokenizer(
    cfg: DotDict,
    model_key: str | None = None,
    device_map: str = "auto",
) -> tuple:
    """
    Load model and tokenizer according to config.

    Args:
        cfg:        Full config DotDict.
        model_key:  One of "primary", "qwen", "magistral". Defaults to cfg.active_model.
        device_map: Passed to from_pretrained. "auto" splits across available GPUs.

    Returns:
        (model, tokenizer)
    """
    mcfg = _model_cfg(cfg, model_key)
    name = mcfg.name
    log.info("Loading model: %s", name)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(getattr(mcfg, "dtype", "bfloat16"), torch.bfloat16)

    load_kwargs: dict = {
        "torch_dtype":          dtype,
        "device_map":           device_map,
        "attn_implementation":  getattr(mcfg, "attn_implementation", "eager"),
    }

    if getattr(mcfg, "load_in_8bit", False):
        load_kwargs["load_in_8bit"] = True
        load_kwargs.pop("torch_dtype", None)  # incompatible with 8-bit

    model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Model loaded on %s", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hot-swap helper (used in generalisation experiments)
# ---------------------------------------------------------------------------

_current: dict = {}


def switch_model(
    cfg: DotDict,
    model_key: str,
    device_map: str = "auto",
) -> tuple:
    """
    Free the previously loaded model from GPU memory and load a new one.

    Returns (model, tokenizer).
    """
    global _current
    if _current:
        log.info("Freeing model: %s", _current.get("key"))
        old_model = _current.get("model")
        if old_model is not None:
            del old_model
        _current.clear()
        torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(cfg, model_key, device_map)
    _current = {"key": model_key, "model": model}
    return model, tokenizer
