"""
Forward-pass helpers: generation, logit extraction, activation collection.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from verbal_confidence.utils.tokens import CLASS_TIDS


# ---------------------------------------------------------------------------
# Greedy generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    """Run greedy decoding and return only the generated tokens as a string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Logit metrics for confidence tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    class_tids: list[int] | None = None,
) -> dict:
    """
    Run a single forward pass and return logit-derived metrics over
    the 10 confidence-class first tokens.

    Returns dict with:
        logits_all  : np.ndarray [vocab]
        logits_cls  : np.ndarray [10]  — raw logits for class tokens
        probs_cls   : np.ndarray [10]  — softmax over class tokens only
        pred_class  : int              — argmax class index
        pred_label  : str              — confidence class label
    """
    if class_tids is None:
        class_tids = CLASS_TIDS(tokenizer)

    from verbal_confidence.utils.tokens import CONFIDENCE_CLASSES

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    last_logits = out.logits[0, -1, :].float().cpu()

    logits_cls = last_logits[class_tids].numpy()
    probs_cls  = torch.softmax(torch.tensor(logits_cls), dim=0).numpy()
    pred_idx   = int(np.argmax(probs_cls))

    return {
        "logits_cls": logits_cls,
        "probs_cls":  probs_cls,
        "pred_class": pred_idx,
        "pred_label": CONFIDENCE_CLASSES[pred_idx],
    }


# ---------------------------------------------------------------------------
# Activation collection via forward hooks
# ---------------------------------------------------------------------------

class ActCollector:
    """
    Context manager that registers forward hooks on specified layers
    and collects residual-stream activations at every forward pass.

    Usage:
        with ActCollector(model, layers=[10, 20]) as collector:
            model(**inputs)
        acts = collector.activations  # {layer: np.ndarray [seq, hidden]}
    """

    def __init__(self, model: PreTrainedModel, layers: list[int] | str = "all"):
        self.model = model
        n_layers = model.config.num_hidden_layers
        if layers == "all":
            self.layers = list(range(n_layers))
        else:
            self.layers = [l for l in layers if 0 <= l < n_layers]
        self.activations: dict[int, np.ndarray] = {}
        self._hooks: list = []

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: nn.Module, input: tuple, output) -> None:
            # Decoder layer output is typically (hidden_state, ...) or a tuple
            hs = output[0] if isinstance(output, (tuple, list)) else output
            self.activations[layer_idx] = hs[0].float().cpu().detach().numpy()
        return hook

    def __enter__(self) -> "ActCollector":
        self.activations.clear()
        decoder_layers = _get_decoder_layers(self.model)
        for idx in self.layers:
            h = decoder_layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)
        return self

    def __exit__(self, *_) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the list of transformer decoder blocks for supported architectures."""
    # Gemma / Llama style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Mistral style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Qwen style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Cannot identify decoder layers. "
        "Extend _get_decoder_layers() for this architecture."
    )


# ---------------------------------------------------------------------------
# Patch / noise / swap hooks
# ---------------------------------------------------------------------------

def make_patch_hook(patch_vector: torch.Tensor, position: int):
    """
    Return a forward hook that replaces the hidden state at `position`
    with `patch_vector` (a 1-D tensor of shape [hidden]).
    """
    def hook(module, input, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        hs[0, position, :] = patch_vector.to(hs.device)
        if isinstance(output, (tuple, list)):
            return (hs,) + output[1:]
        return hs
    return hook


def make_noise_hook(mean_vector: torch.Tensor, position: int):
    """
    Return a forward hook that replaces the hidden state at `position`
    with the mean activation (mean ablation / noising).
    """
    return make_patch_hook(mean_vector, position)
