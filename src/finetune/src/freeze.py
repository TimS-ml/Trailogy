"""Freeze the vision + audio towers of a Gemma 4 multimodal model.

Background
----------
Unsloth's `FastModel.get_peft_model(..., finetune_vision_layers=False)`
already prevents LoRA adapters from being inserted into the vision tower,
but the unsloth API does **not** yet expose an equivalent flag for audio
(see notebook commentary: "the audio part can also be finetuned — we're
working to make it selectable as well!").

For the hikeCompanion finetune we must keep both vision *and* audio
frozen — the iOS runtime only uses the language tower (text + image),
the audio tower is stripped at deploy time, and we don't want stray
gradients flowing into either.

This helper walks the module tree and sets `requires_grad = False` on any
parameter that contains one of the freeze tokens (e.g. ``vision_tower.``,
``embed_audio.``) at a module-path boundary. We use **substring matching
with a trailing dot** rather than `startswith(prefix)` because PEFT and
HuggingFace wrap the underlying model differently:

* raw HF model:            ``vision_tower.encoder...``
* HF Gemma4 wrapper:        ``model.vision_tower.encoder...``
* unsloth + PEFT (single):  ``base_model.model.vision_tower.encoder...``
* unsloth + PEFT (double):  ``base_model.model.model.vision_tower.encoder...``

A finite prefix list misses any wrapping depth we forget; substring +
trailing-dot anchoring catches all of them while still rejecting
look-alikes such as ``vision_tower_descriptor.weight``.

The helper is intentionally name-based so it works on a real Gemma 4
model AND on a synthetic torch module tree built by the unit tests,
without any unsloth-specific imports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

log = logging.getLogger(__name__)


# Tokens that, when found as a module-path component (i.e. with a trailing
# dot), identify a parameter that MUST NOT receive gradients.
#
# We deliberately store the trailing dot so that ``vision_tower_descriptor.``
# (a hypothetical look-alike) does NOT match ``vision_tower.``. The match
# is "anywhere in the parameter name", not "at the start", to absorb
# arbitrary wrapping depth from PEFT / HF / unsloth.
DEFAULT_FROZEN_TOKENS: tuple[str, ...] = (
    # Audio tower (Gemma 4 audio encoder)
    "audio_tower.",
    "embed_audio.",
    # Vision tower (defensive — unsloth already freezes via LoRA gating,
    # but we want the post-LoRA tripwire to catch any leak)
    "vision_tower.",
    "embed_vision.",
    # Some HF VLM variants nest audio under multi_modal_projector
    "multi_modal_projector.audio.",
)

# Backwards-compatible alias — the previous public name was
# DEFAULT_FROZEN_PREFIXES. External callers and tests should keep working.
DEFAULT_FROZEN_PREFIXES: tuple[str, ...] = DEFAULT_FROZEN_TOKENS


@dataclass
class FreezeReport:
    """Summary of what `freeze_vision_audio_towers` did."""

    total_params: int
    frozen_params: int
    trainable_params: int
    frozen_param_count_by_prefix: dict[str, int]
    trainable_lora_params: int  # subset of `trainable_params` that look like LoRA

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"FreezeReport(total={self.total_params:,}, "
            f"frozen={self.frozen_params:,}, "
            f"trainable={self.trainable_params:,}, "
            f"trainable_lora={self.trainable_lora_params:,})",
        ]
        for prefix, count in sorted(self.frozen_param_count_by_prefix.items()):
            if count:
                lines.append(f"  - {prefix:<32} froze {count} params")
        return "\n".join(lines)


def freeze_vision_audio_towers(
    model,
    extra_prefixes: Sequence[str] = (),
    prefixes: Sequence[str] = DEFAULT_FROZEN_TOKENS,
    *,
    log_summary: bool = True,
) -> FreezeReport:
    """Walk the parameter tree and disable gradients on vision/audio towers.

    Matching is **substring with trailing-dot anchoring**, not prefix
    matching. This is required because PEFT and HF wrap the underlying
    model with extra prefixes (``base_model.model.``, ``model.``) and a
    finite prefix list silently misses any wrapping depth we forget.

    The ``extra_prefixes`` / ``prefixes`` parameter names are kept for
    backwards compatibility with the original prefix-based API. They
    behave as freeze *tokens* now (substring + trailing-dot match).

    Parameters
    ----------
    model : torch.nn.Module
        Any module whose `named_parameters()` yields the standard tree.
        Works on real Gemma 4 from unsloth as well as synthetic test modules.
    extra_prefixes : Sequence[str], optional
        Additional freeze tokens (with trailing dot) on top of the defaults.
    prefixes : Sequence[str]
        Override the default token list entirely. Use `extra_prefixes`
        if you only want to add to it.
    log_summary : bool
        If True (default), log a one-line summary at INFO.

    Returns
    -------
    FreezeReport
        Counts of frozen vs trainable parameters.
    """
    all_tokens: tuple[str, ...] = tuple(prefixes) + tuple(extra_prefixes)
    counts_by_token: dict[str, int] = {p: 0 for p in all_tokens}

    total = 0
    frozen = 0
    trainable_lora = 0
    for name, param in model.named_parameters():
        total += param.numel()
        matched_token = _first_matching_token(name, all_tokens)
        if matched_token is not None:
            param.requires_grad = False
            frozen += param.numel()
            counts_by_token[matched_token] += param.numel()
        else:
            if param.requires_grad and _looks_like_lora(name):
                trainable_lora += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = FreezeReport(
        total_params=total,
        frozen_params=frozen,
        trainable_params=trainable,
        frozen_param_count_by_prefix=counts_by_token,
        trainable_lora_params=trainable_lora,
    )
    if log_summary:
        log.info(
            "Froze %d params (%.2f%% of model). Trainable: %d (LoRA: %d)",
            frozen,
            100.0 * frozen / max(total, 1),
            trainable,
            trainable_lora,
        )
    return report


def _first_matching_token(name: str, tokens: Iterable[str]) -> str | None:
    """Return the first freeze token that appears as a path component in `name`.

    Tokens are expected to end with a dot (e.g. ``vision_tower.``). The
    match is satisfied when the token appears either at the start of the
    name, or immediately after a path-boundary dot. This rejects
    look-alike substrings (e.g. ``vision_tower_descriptor.``) while
    accepting any wrapping depth (``base_model.model.vision_tower.``,
    ``base_model.model.model.vision_tower.``, etc.).
    """
    for tok in tokens:
        if name.startswith(tok):
            return tok
        # Boundary match: the token starts after a "." in `name`.
        idx = name.find("." + tok)
        if idx >= 0:
            return tok
    return None


# Backwards-compat alias for any external callers / older tests.
def _first_matching_prefix(name: str, prefixes: Iterable[str]) -> str | None:
    return _first_matching_token(name, prefixes)


def _looks_like_lora(name: str) -> bool:
    """Cheap heuristic: PEFT/unsloth LoRA params have 'lora_' in their name."""
    return "lora_" in name


# Projector candidate/exclude tokens — mirrored from projector.py.
# Duplicated here to keep freeze.py free of cross-module imports (the
# module is tested independently with synthetic models).
_PROJECTOR_CANDIDATE_TOKENS: tuple[str, ...] = (
    "embed_vision.",
    "vision_tower.embedding_projection.",
    "vision_tower.projection.",
    "multi_modal_projector.vision.",
    "multi_modal_projector.image.",
)
_PROJECTOR_EXCLUDE_TOKENS: tuple[str, ...] = (
    "vision_tower.patch_embedder.",
    "vision_tower.encoder.",
    "vision_tower.pooler.",
    "embed_audio.",
    "audio_tower.",
)


# Vision-encoder-layer token + regex — mirrored from vision_layers.py.
# Duplicated here to keep freeze.py free of cross-module imports (the
# module is tested independently with synthetic models).
_VISION_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)vision_tower\.encoder\.layers\.(\d+)(?:\.|$)"
)


def _is_tuned_vision_layer_param(
    name: str, tuned_layer_indices: "set[int]"
) -> bool:
    """Check if `name` belongs to one of the tuned vision encoder layers.

    Used by ``freeze_vision_audio_towers_keeping_projector_and_vision_layers``
    and ``assert_frozen`` to allowlist the last-N vision encoder layers
    when the vision-tower last-N tuning path is enabled
    (``feature/lora-plus-projector-plus-vision-tower``).

    Excludes PEFT's ``.original_module.`` frozen reference copy — only
    ``.modules_to_save.{adapter}.`` copies are flagged as trainable.
    """
    # PEFT's frozen reference copy — never a trainable layer param.
    if ".original_module." in name:
        return False
    if not tuned_layer_indices:
        return False
    m = _VISION_LAYER_INDEX_RE.search(name)
    if m is None:
        return False
    return int(m.group(1)) in tuned_layer_indices


def _is_projector_param(name: str) -> bool:
    """Check if a parameter name belongs to the **trainable** projector.

    Works on both raw HF names and PEFT-wrapped names. Uses token-based
    substring matching, NOT full pre-PEFT param-name matching.

    Critically, this excludes PEFT's ``original_module`` copy. When PEFT
    wraps a module via ``modules_to_save``, it creates two paths:

    - ``embed_vision.original_module.embedding_projection.weight``
      — frozen reference copy, must NOT be trainable or in optimizer
    - ``embed_vision.modules_to_save.default.embedding_projection.weight``
      — trainable copy, should be in optimizer

    Without the ``original_module`` exclusion, both copies get flagged
    as projector params, leading to duplicate optimizer entries and the
    frozen copy being incorrectly re-enabled.
    """
    # PEFT's frozen reference copy — never a trainable projector param.
    if ".original_module." in name:
        return False
    # Exclude tokens checked next to avoid false positives.
    for ex in _PROJECTOR_EXCLUDE_TOKENS:
        if ex in name:
            return False
    for tok in _PROJECTOR_CANDIDATE_TOKENS:
        if tok in name:
            return True
    return False


def assert_frozen(
    model,
    prefixes: Sequence[str] = DEFAULT_FROZEN_TOKENS,
    *,
    allowlist: Sequence[str] = (),
    tuned_vision_layer_indices: Sequence[int] = (),
) -> None:
    """Raise if any parameter matching a frozen token still has `requires_grad`.

    Useful as a tripwire after applying LoRA but before training kicks off.

    Parameters
    ----------
    allowlist
        Parameter names that are allowed to remain trainable even though
        they match a frozen token. Used by the projector-tuning path
        (``feature/lora-plus-projector``): when ``cfg.lora.tune_projector``
        is True, the vision-language projector params are intentionally
        kept trainable, so the caller passes them here to silence the
        tripwire for those specific names. The check still fires for any
        OTHER frozen-token-matching param that slipped through (e.g. an
        accidental audio_tower or vision encoder unfreeze).
    tuned_vision_layer_indices
        Indices of vision encoder layers (under
        ``vision_tower.encoder.layers.{i}``) that are intentionally
        trainable. Used by the vision-tower last-N tuning path
        (``feature/lora-plus-projector-plus-vision-tower``). Params
        whose name matches one of these indices at a path boundary are
        skipped by the tripwire. Other vision encoder layers (and the
        rest of the frozen tokens) are still checked.
    """
    # When allowlist is non-empty, projector params are intentionally
    # trainable. Use token-based detection (same as the freeze pass)
    # instead of exact/substring name matching — PEFT wrapping inserts
    # path components that break contiguity against pre-PEFT names.
    use_projector_allowlist = len(allowlist) > 0
    tuned_vision_indices_set = set(tuned_vision_layer_indices)
    offenders: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_projector_allowlist and _is_projector_param(name):
            continue
        if tuned_vision_indices_set and _is_tuned_vision_layer_param(
            name, tuned_vision_indices_set
        ):
            continue
        if _first_matching_token(name, prefixes) is not None:
            offenders.append(name)
    if offenders:
        sample = ", ".join(offenders[:5])
        raise RuntimeError(
            f"{len(offenders)} param(s) under frozen tokens still trainable: {sample}"
            + (" ..." if len(offenders) > 5 else "")
        )


def freeze_vision_audio_towers_keeping_projector(
    model,
    projector_param_names: Sequence[str],
    extra_prefixes: Sequence[str] = (),
    prefixes: Sequence[str] = DEFAULT_FROZEN_TOKENS,
    *,
    log_summary: bool = True,
) -> FreezeReport:
    """Like ``freeze_vision_audio_towers``, but skips ``projector_param_names``.

    Used by the projector-tuning path (``feature/lora-plus-projector``).
    Identical matching semantics — the only difference is that any param
    whose name appears in ``projector_param_names`` keeps its current
    ``requires_grad`` (typically True after PEFT's ``modules_to_save``
    has done its work).

    The vision ENCODER (``vision_tower.encoder.*`` etc.), audio tower,
    and audio embedder are still frozen.

    Parameters
    ----------
    projector_param_names
        Exact parameter names (matching ``model.named_parameters()`` keys)
        that should NOT be frozen. Get this list from
        ``src.projector.find_projector_param_names(model)``.

    Returns
    -------
    FreezeReport
        Same shape as ``freeze_vision_audio_towers``.
    """
    # projector_param_names is accepted for API compat but NOT used for
    # matching. After PEFT wrapping with modules_to_save, param names
    # gain infixes like "original_module." / "modules_to_save.default."
    # that break substring containment against the pre-PEFT name. We use
    # token-based matching via _is_projector_param() instead.
    _ = projector_param_names  # consumed by signature, matching is token-based
    all_tokens: tuple[str, ...] = tuple(prefixes) + tuple(extra_prefixes)
    counts_by_token: dict[str, int] = {p: 0 for p in all_tokens}

    total = 0
    frozen = 0
    trainable_lora = 0
    for name, param in model.named_parameters():
        total += param.numel()
        matched_token = _first_matching_token(name, all_tokens)
        # Use token-based projector detection — robust to PEFT wrapping.
        is_kept = _is_projector_param(name)
        if matched_token is not None and not is_kept:
            param.requires_grad = False
            frozen += param.numel()
            counts_by_token[matched_token] += param.numel()
        else:
            if param.requires_grad and _looks_like_lora(name):
                trainable_lora += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = FreezeReport(
        total_params=total,
        frozen_params=frozen,
        trainable_params=trainable,
        frozen_param_count_by_prefix=counts_by_token,
        trainable_lora_params=trainable_lora,
    )
    if log_summary:
        log.info(
            "Froze %d params (%.2f%% of model). Trainable: %d (LoRA: %d, "
            "projector kept: %d)",
            frozen,
            100.0 * frozen / max(total, 1),
            trainable,
            trainable_lora,
            len(projector_param_names),
        )
    return report


def freeze_vision_audio_towers_keeping_projector_and_vision_layers(
    model,
    projector_param_names: Sequence[str],
    tuned_vision_layer_indices: Sequence[int] = (),
    extra_prefixes: Sequence[str] = (),
    prefixes: Sequence[str] = DEFAULT_FROZEN_TOKENS,
    *,
    log_summary: bool = True,
) -> FreezeReport:
    """Like ``freeze_vision_audio_towers_keeping_projector``, but also
    skips freezing the last-N vision encoder layers identified by
    ``tuned_vision_layer_indices``.

    Used by the vision-tower last-N tuning path
    (``feature/lora-plus-projector-plus-vision-tower``). Identical
    matching semantics to the other variants — the additions are:

    * Any param whose name resolves to
      ``vision_tower.encoder.layers.{i}.`` for `i` in
      ``tuned_vision_layer_indices`` keeps its current
      ``requires_grad`` (typically True after PEFT's
      ``modules_to_save`` has done its work).
    * The rest of ``vision_tower.encoder.*`` (other layers,
      ``patch_embedder``, ``pooler``) is still frozen, as is the audio
      side.

    Parameters
    ----------
    projector_param_names
        Same as the existing ``_keeping_projector`` variant. Accepted
        for API compatibility; matching is token-based via
        ``_is_projector_param``.
    tuned_vision_layer_indices
        Indices (0-based) of vision encoder layers under
        ``vision_tower.encoder.layers`` that should remain trainable.
        Pass ``()`` to get behavior identical to
        ``freeze_vision_audio_towers_keeping_projector``.

    Returns
    -------
    FreezeReport
        Same shape as the other variants.
    """
    _ = projector_param_names  # consumed by signature, matching is token-based
    tuned_indices_set: set[int] = set(tuned_vision_layer_indices)
    all_tokens: tuple[str, ...] = tuple(prefixes) + tuple(extra_prefixes)
    counts_by_token: dict[str, int] = {p: 0 for p in all_tokens}

    total = 0
    frozen = 0
    trainable_lora = 0
    for name, param in model.named_parameters():
        total += param.numel()
        matched_token = _first_matching_token(name, all_tokens)
        # Projector params are kept trainable.
        is_kept_projector = _is_projector_param(name)
        # Tuned vision encoder layers are kept trainable.
        is_kept_vision_layer = _is_tuned_vision_layer_param(
            name, tuned_indices_set
        )
        is_kept = is_kept_projector or is_kept_vision_layer
        if matched_token is not None and not is_kept:
            param.requires_grad = False
            frozen += param.numel()
            counts_by_token[matched_token] += param.numel()
        else:
            if param.requires_grad and _looks_like_lora(name):
                trainable_lora += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = FreezeReport(
        total_params=total,
        frozen_params=frozen,
        trainable_params=trainable,
        frozen_param_count_by_prefix=counts_by_token,
        trainable_lora_params=trainable_lora,
    )
    if log_summary:
        log.info(
            "Froze %d params (%.2f%% of model). Trainable: %d "
            "(LoRA: %d, projector kept: %d, vision layers kept: %d at indices %s)",
            frozen,
            100.0 * frozen / max(total, 1),
            trainable,
            trainable_lora,
            len(projector_param_names),
            len(tuned_indices_set),
            sorted(tuned_indices_set),
        )
    return report
