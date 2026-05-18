"""Identify and manage the last-N vision encoder layers in a Gemma 4 model.

Background
----------
Gemma 4 E2B's `Gemma4VisionEncoder` (see the HF Gemma 4 source,
``modeling_gemma4.py`` around line 983) contains
an ``nn.ModuleList`` named ``layers`` of length ``num_hidden_layers``
(default 16). Each entry is a ``Gemma4VisionEncoderLayer`` (attention +
MLP). The full module path under the HF Gemma4Model is
``vision_tower.encoder.layers.{i}``.

This helper identifies the **last N** of those layers so we can unfreeze
them as full parameters via PEFT's ``modules_to_save`` — mirroring the
projector-tuning pattern from ``src/projector.py`` and
``feature/lora-plus-projector``.

Used by
-------
``finetune.py`` calls these helpers when
``cfg.lora.tune_last_n_vision_layers > 0`` to:

1. Discover the actual layer count (introspection, no hardcoding).
2. Produce the ``modules_to_save`` suffix strings for PEFT.
3. Build the freeze-pass allowlist and the 3rd optimizer param group.

Design notes
------------
* The module-name strings we return for ``modules_to_save`` are
  **disambiguating suffixes** (``vision_tower.encoder.layers.{i}``), not
  bare leaf names like ``"layers.14"`` — the bare leaf would collide
  with the text decoder's ``model.layers.14``.

* Parameter-name matching uses substring + boundary-dot semantics
  identical to ``freeze.py`` / ``projector.py``, so it is robust to
  arbitrary PEFT/HF wrapping depth.

* PEFT's ``modules_to_save`` mechanism inserts an ``.original_module.``
  frozen copy alongside the trainable ``.modules_to_save.{adapter}.``
  copy. We explicitly exclude ``.original_module.`` so only the
  trainable copy is flagged.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Sequence

log = logging.getLogger(__name__)


# Token that identifies a vision encoder layer at a path-component
# boundary. Matched as substring with leading dot (or string start)
# anchoring, identical to the matchers in projector.py and freeze.py.
VISION_ENCODER_LAYERS_TOKEN: str = "vision_tower.encoder.layers."


# Regex that captures the layer index immediately after the token.
# Anchored with a path boundary to reject look-alikes such as
# ``vision_tower_descriptor.encoder.layers.14`` and to ensure we look at
# a path-component start (either string start or a preceding dot).
_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)vision_tower\.encoder\.layers\.(\d+)(?:\.|$)"
)


def _extract_layer_index(name: str) -> int | None:
    """Return the integer layer index from a param/module name, or None.

    Only matches when ``vision_tower.encoder.layers.{i}`` appears at a
    path-component boundary (start-of-string or after a dot).
    """
    m = _LAYER_INDEX_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


def find_vision_encoder_layer_count(model) -> int:
    """Return the number of vision encoder layers in `model`.

    Walks ``model.named_modules()`` (or ``named_parameters()`` as a
    fallback) for entries matching the vision-encoder-layer token and
    returns the highest layer index + 1.

    Raises
    ------
    RuntimeError
        If no vision encoder layers are found. The caller (finetune.py)
        treats this as a fatal config / model-layout error.
    """
    highest = -1
    for source in (model.named_modules(), model.named_parameters()):
        for name, _ in source:
            idx = _extract_layer_index(name)
            if idx is not None and idx > highest:
                highest = idx
    if highest < 0:
        raise RuntimeError(
            "found no vision_tower.encoder.layers.{i} modules in model. "
            "If this is Gemma 4 E2B, expect 16 such layers. Check the "
            "model layout via model.named_modules()."
        )
    return highest + 1


def find_last_n_vision_layer_module_names(model, n: int) -> List[str]:
    """Return PEFT-compatible ``modules_to_save`` suffix strings for the
    last `n` vision encoder layers.

    The returned strings have NO wrapping prefix
    (``base_model.model.``, ``model.``, etc.) — PEFT's
    ``modules_to_save`` does suffix matching and tolerates any wrapping
    depth.

    Parameters
    ----------
    model
        Any module supporting ``named_modules()``.
    n : int
        Number of trailing layers to return. Must be 1 <= n <= total.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``n > total_layer_count``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    total = find_vision_encoder_layer_count(model)
    if n > total:
        raise ValueError(
            f"n={n} exceeds vision encoder layer count total={total}"
        )
    indices = range(total - n, total)
    return [f"{VISION_ENCODER_LAYERS_TOKEN}{i}" for i in indices]


def is_tuned_vision_layer_param(
    name: str,
    tuned_layer_indices: Iterable[int],
) -> bool:
    """Return True iff `name` refers to a parameter inside one of the
    tuned vision encoder layers (and is NOT a PEFT frozen copy).

    Matching rules (in order):

    1. PEFT's frozen reference copy: any name containing
       ``.original_module.`` returns False (never trainable).
    2. The name must contain ``vision_tower.encoder.layers.{i}.`` at a
       path-component boundary, with `i` in ``tuned_layer_indices``.

    Used by:
    * ``finetune.py`` optimizer param-group split (vision group).
    * ``freeze.py`` allowlist when freezing everything except the
      projector and the tuned vision layers.
    * ``assert_frozen`` tripwire.
    """
    # PEFT's frozen reference copy — never trainable.
    if ".original_module." in name:
        return False
    idx = _extract_layer_index(name)
    if idx is None:
        return False
    return idx in set(tuned_layer_indices)


def find_vision_layer_param_names(model, n: int) -> List[str]:
    """Return parameter names belonging to the last `n` vision encoder layers.

    Excludes PEFT's ``.original_module.`` frozen copies — only the
    trainable ``.modules_to_save.{adapter}.`` copies are returned (when
    present).

    Raises
    ------
    ValueError
        If ``n < 1`` or ``n > total_layer_count``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    total = find_vision_encoder_layer_count(model)
    if n > total:
        raise ValueError(
            f"n={n} exceeds vision encoder layer count total={total}"
        )
    tuned_indices = set(range(total - n, total))
    matched: List[str] = []
    for name, _param in model.named_parameters():
        if is_tuned_vision_layer_param(name, tuned_indices):
            matched.append(name)
    return matched


def ensure_vision_layers_trainable(
    model,
    tuned_layer_indices: Iterable[int],
) -> int:
    """Belt-and-braces: flip ``requires_grad`` to True on every param under
    the tuned vision encoder layers.

    Called after PEFT wrapping as a fallback for the case where
    ``FastModel.get_peft_model(modules_to_save=...)`` silently drops the
    kwarg (unsloth regression). The caller logs a WARNING when this
    function returns > 0.

    Parameters
    ----------
    tuned_layer_indices
        Indices (0-based) of vision encoder layers that should be
        trainable. Typically ``range(total - n, total)``.

    Returns
    -------
    int
        Count of parameters flipped from ``requires_grad=False`` to True.
    """
    tuned = set(tuned_layer_indices)
    if not tuned:
        return 0
    flipped = 0
    for name, param in model.named_parameters():
        if not is_tuned_vision_layer_param(name, tuned):
            continue
        if not param.requires_grad:
            param.requires_grad = True
            flipped += 1
    return flipped
