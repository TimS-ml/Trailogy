"""Identify the vision-language projector in a Gemma 4 multimodal model.

Background
----------
Gemma 4 E2B's `Gemma4Model` carries a `Gemma4MultimodalEmbedder` named
``embed_vision`` (see the HF Gemma 4 source,
``modeling_gemma4.py`` around line 2147). That embedder is the
**vision-language projector**: it RMSNorm-then-Linear-projects the
vision tower's hidden states into the language model's hidden space.

A non-public Gemma 4 layout may instead place the projector inside
``vision_tower.*`` (e.g. ``vision_tower.embedding_projection``). To
absorb either possibility — and to handle PEFT/HF wrapping prefixes
(``base_model.model.``, ``model.``) — this helper does **substring +
trailing-dot matching** against a candidate-token list, mirroring the
matcher in ``freeze.py``.

Encoder sub-modules (``vision_tower.{patch_embedder, encoder, pooler}``)
must NEVER be flagged as projector. We guard against this with an
explicit exclude-token list that is checked **before** the candidate
match.

Used by
-------
``finetune.py`` calls these helpers to decide which params to keep
trainable when ``cfg.lora.tune_projector`` is True, and to wire the
short module names into ``FastModel.get_peft_model(modules_to_save=...)``.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

log = logging.getLogger(__name__)


# Tokens that, when found at a path-component boundary in a parameter
# name, identify projector params. Order matters only for diagnostic
# logging — the matcher returns on first hit.
PROJECTOR_CANDIDATE_TOKENS: tuple[str, ...] = (
    # HF reference layout: top-level Gemma4MultimodalEmbedder
    "embed_vision.",
    # Hypothetical alternate layout (internal note): projector inside vision_tower
    "vision_tower.embedding_projection.",
    "vision_tower.projection.",
    # Some HF VLM variants nest under multi_modal_projector
    "multi_modal_projector.vision.",
    "multi_modal_projector.image.",
)


# Encoder paths that share the ``vision_tower.`` prefix but are NOT the
# projector. These are checked BEFORE candidate matching so that a model
# with ``vision_tower.embedding_projection.*`` can still match without
# accidentally pulling in encoder siblings.
PROJECTOR_EXCLUDE_TOKENS: tuple[str, ...] = (
    "vision_tower.patch_embedder.",
    "vision_tower.encoder.",
    "vision_tower.pooler.",
    # Audio-side embedder uses the same Gemma4MultimodalEmbedder class
    # under a different name; explicitly exclude.
    "embed_audio.",
    "audio_tower.",
)


def _matches_token(name: str, tokens: Iterable[str]) -> bool:
    """Substring + boundary-dot match (same semantics as freeze._first_matching_token)."""
    for tok in tokens:
        if name.startswith(tok):
            return True
        if ("." + tok) in name:
            return True
    return False


def find_projector_param_names(model) -> List[str]:
    """Return parameter names that belong to the vision-language projector.

    Walks ``model.named_parameters()`` and keeps names that match a token
    in ``PROJECTOR_CANDIDATE_TOKENS`` while not matching any token in
    ``PROJECTOR_EXCLUDE_TOKENS``.

    Raises
    ------
    RuntimeError
        If no projector params are found. We fail loud rather than
        silently train an empty set — a downstream
        ``modules_to_save=[]`` would degrade projector tuning to a
        no-op, which is the exact silent regression we want to prevent.
    """
    matched: List[str] = []
    for name, _param in model.named_parameters():
        if _matches_token(name, PROJECTOR_EXCLUDE_TOKENS):
            continue
        if _matches_token(name, PROJECTOR_CANDIDATE_TOKENS):
            matched.append(name)
    if not matched:
        raise RuntimeError(
            "found no projector params in model (looked for "
            f"{list(PROJECTOR_CANDIDATE_TOKENS)} excluding "
            f"{list(PROJECTOR_EXCLUDE_TOKENS)}). "
            "If the model is Gemma 4 E2B, the projector should appear at "
            "embed_vision.* or vision_tower.embedding_projection.*. Use "
            "model.named_parameters() to inspect the actual layout."
        )
    log.info(
        "Identified %d projector parameter(s): %s",
        len(matched),
        ", ".join(matched[:5]) + (" ..." if len(matched) > 5 else ""),
    )
    return matched


def find_projector_module_names(model) -> List[str]:
    """Return short module names suitable for PEFT's ``modules_to_save``.

    PEFT matches ``modules_to_save`` entries as **suffixes** of the full
    module path (any wrapping prefix is stripped). We therefore return
    the leaf component (e.g. ``"embed_vision"``) of each candidate
    module path, deduplicated.

    **Important**: we only return top-level projector modules, not their
    children. Children like ``embedding_projection`` are shared names
    between ``embed_vision`` and ``embed_audio`` — passing them to
    ``modules_to_save`` would also wrap the audio embedder, which we
    want frozen. By wrapping only ``embed_vision``, its children are
    automatically included.

    Raises
    ------
    RuntimeError
        If no projector modules are found. Same rationale as
        ``find_projector_param_names``.
    """
    # Collect all matching module names (full paths, not leaf names).
    all_matches: List[str] = []
    for name, _module in model.named_modules():
        if not name:
            continue
        if _matches_token(name + ".", PROJECTOR_EXCLUDE_TOKENS):
            continue
        if _matches_token(name + ".", PROJECTOR_CANDIDATE_TOKENS):
            all_matches.append(name)

    if not all_matches:
        raise RuntimeError(
            "found no projector modules in model. Looked for module names "
            f"matching {list(PROJECTOR_CANDIDATE_TOKENS)}. If the model is "
            "Gemma 4 E2B, expect 'embed_vision' as a top-level submodule."
        )

    # Filter to only top-level projector modules: remove any name that
    # is a child of another matched name. This prevents generic child
    # names (e.g. "embedding_projection") from leaking into
    # modules_to_save and accidentally wrapping the audio embedder.
    top_level: List[str] = []
    for name in all_matches:
        is_child = any(
            name != other and name.startswith(other + ".")
            for other in all_matches
        )
        if not is_child:
            top_level.append(name)

    # Return the leaf component of each top-level match.
    seen: List[str] = []
    for name in top_level:
        short = name.rsplit(".", 1)[-1]
        if short not in seen:
            seen.append(short)
    return seen


def ensure_projector_trainable(
    model,
    projector_param_names: Sequence[str],
) -> int:
    """Belt-and-braces: set ``requires_grad=True`` on every projector param.

    Used as a fallback when ``FastModel.get_peft_model(modules_to_save=...)``
    silently drops the kwarg (an unsloth regression we want to detect
    rather than ship around). The caller logs a WARNING when this
    function returns > 0 so we notice the regression.

    Returns
    -------
    int
        Number of parameters that were flipped from ``requires_grad=False``
        to ``True`` by this call. Names not present in
        ``model.named_parameters()`` are silently skipped (the model
        legitimately doesn't have that exact name under this wrapping).
    """
    # projector_param_names is accepted for API compat but NOT used for
    # matching. After PEFT wrapping with modules_to_save, param names
    # gain infixes like "original_module." / "modules_to_save.default."
    # that break substring containment against the pre-PEFT name. We use
    # token-based matching via _matches_token + candidate/exclude lists.
    #
    # Critically, we skip PEFT's "original_module" frozen reference copy.
    # Only "modules_to_save.{adapter}" copies should be trainable.
    _ = projector_param_names  # consumed by signature, matching is token-based
    flipped = 0
    for name, param in model.named_parameters():
        if ".original_module." in name:
            continue
        if _matches_token(name, PROJECTOR_EXCLUDE_TOKENS):
            continue
        if _matches_token(name, PROJECTOR_CANDIDATE_TOKENS) and not param.requires_grad:
            param.requires_grad = True
            flipped += 1
    return flipped
