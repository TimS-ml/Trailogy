"""Tests for src.projector — vision-language projector identification.

These tests use the same FakeParam/FakeModel duck-types as test_freeze.py
(they only need named_parameters() / named_modules() shape, no torch).

The projector is what `Gemma4MultimodalEmbedder` is in the HF reference
(``modeling_gemma4.py`` around lines 2023-2047): a tiny module
sitting between the vision encoder output and the language-model hidden
space. In the public layout it lives at top-level `embed_vision`. An
internal/non-public layout might place it inside `vision_tower.*`. The
identification helper accepts either, by candidate-token list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import pytest

from src.projector import (
    PROJECTOR_CANDIDATE_TOKENS,
    PROJECTOR_EXCLUDE_TOKENS,
    ensure_projector_trainable,
    find_projector_module_names,
    find_projector_param_names,
)


# ---------------------------------------------------------------------------
# Fakes (duck-typed for named_parameters / named_modules)
# ---------------------------------------------------------------------------


@dataclass
class FakeParam:
    name: str
    shape: Tuple[int, ...] = (10,)
    requires_grad: bool = True

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass
class FakeModule:
    """Minimal module duck-type. Only used so named_modules() can return tuples."""

    label: str = ""


@dataclass
class FakeModel:
    params: List[Tuple[str, FakeParam]] = field(default_factory=list)
    modules: List[Tuple[str, FakeModule]] = field(default_factory=list)

    def named_parameters(self) -> Iterator[Tuple[str, FakeParam]]:
        for name, p in self.params:
            yield name, p

    def named_modules(self) -> Iterator[Tuple[str, FakeModule]]:
        for name, m in self.modules:
            yield name, m


def _model_with_top_level_embed_vision(prefix: str = "") -> FakeModel:
    """HF reference layout: projector at top-level embed_vision."""
    params = [
        # Vision ENCODER (must NOT be matched as projector)
        (f"{prefix}vision_tower.patch_embedder.weight", FakeParam("patch", shape=(768, 768))),
        (f"{prefix}vision_tower.encoder.layer.0.attn.q_proj.weight", FakeParam("v_q", shape=(768, 768))),
        (f"{prefix}vision_tower.encoder.layer.1.mlp.fc1.weight", FakeParam("v_fc", shape=(768, 3072))),
        (f"{prefix}vision_tower.pooler.weight", FakeParam("pool", shape=(768, 768))),
        # Vision PROJECTOR (Gemma4MultimodalEmbedder)
        (f"{prefix}embed_vision.embedding_projection.weight", FakeParam("ev_proj", shape=(2048, 768))),
        # Audio (frozen, NOT projector)
        (f"{prefix}audio_tower.encoder.layer.0.linear.weight", FakeParam("a", shape=(512, 512))),
        (f"{prefix}embed_audio.embedding_projection.weight", FakeParam("ea", shape=(2048, 512))),
        # Language
        (f"{prefix}language_model.layers.0.self_attn.q_proj.weight", FakeParam("lq", shape=(2048, 2048))),
    ]
    modules = [
        (f"{prefix}vision_tower", FakeModule("vt")),
        (f"{prefix}vision_tower.encoder", FakeModule("enc")),
        (f"{prefix}embed_vision", FakeModule("ev")),
        (f"{prefix}embed_vision.embedding_projection", FakeModule("ev_proj")),
        (f"{prefix}embed_audio", FakeModule("ea")),
        (f"{prefix}audio_tower", FakeModule("at")),
        (f"{prefix}language_model", FakeModule("lm")),
    ]
    return FakeModel(params=params, modules=modules)


def _model_with_projector_under_vision_tower(prefix: str = "") -> FakeModel:
    """Hypothetical alternate layout: projector lives inside vision_tower."""
    params = [
        # Encoder (must NOT match)
        (f"{prefix}vision_tower.patch_embedder.weight", FakeParam("patch", shape=(768, 768))),
        (f"{prefix}vision_tower.encoder.layer.0.attn.q_proj.weight", FakeParam("v_q", shape=(768, 768))),
        (f"{prefix}vision_tower.pooler.weight", FakeParam("pool", shape=(768, 768))),
        # Projector inside vision_tower (alt layout)
        (f"{prefix}vision_tower.embedding_projection.weight", FakeParam("vt_proj", shape=(2048, 768))),
        # Audio + language
        (f"{prefix}audio_tower.encoder.weight", FakeParam("a")),
        (f"{prefix}language_model.layers.0.self_attn.q_proj.weight", FakeParam("lq")),
    ]
    modules = [
        (f"{prefix}vision_tower", FakeModule("vt")),
        (f"{prefix}vision_tower.encoder", FakeModule("enc")),
        (f"{prefix}vision_tower.embedding_projection", FakeModule("vt_proj")),
    ]
    return FakeModel(params=params, modules=modules)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_candidate_tokens_cover_known_layouts() -> None:
    assert "embed_vision." in PROJECTOR_CANDIDATE_TOKENS
    assert any("vision_tower" in t and "projection" in t for t in PROJECTOR_CANDIDATE_TOKENS)


def test_exclude_tokens_cover_encoder_components() -> None:
    """Whatever PROJECTOR_EXCLUDE_TOKENS are, they must shield the encoder
    sub-modules so that matching `vision_tower.embedding_projection.` does
    not accidentally pull in `vision_tower.encoder.*`."""
    for needle in ("patch_embedder", "encoder", "pooler"):
        assert any(needle in t for t in PROJECTOR_EXCLUDE_TOKENS), (
            f"PROJECTOR_EXCLUDE_TOKENS missing protection for {needle}"
        )


# ---------------------------------------------------------------------------
# find_projector_param_names — top-level embed_vision layout (HF reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_find_projector_top_level_embed_vision(wrapper: str) -> None:
    model = _model_with_top_level_embed_vision(prefix=wrapper)
    names = find_projector_param_names(model)
    assert names == [f"{wrapper}embed_vision.embedding_projection.weight"]


# ---------------------------------------------------------------------------
# find_projector_param_names — alt layout (projector under vision_tower)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_find_projector_under_vision_tower(wrapper: str) -> None:
    model = _model_with_projector_under_vision_tower(prefix=wrapper)
    names = find_projector_param_names(model)
    assert names == [f"{wrapper}vision_tower.embedding_projection.weight"]


# ---------------------------------------------------------------------------
# Encoder exclusion — the most important guarantee
# ---------------------------------------------------------------------------


def test_find_projector_excludes_encoder_components() -> None:
    """Encoder parameters (patch_embedder, encoder, pooler) must NEVER be
    flagged as projector, regardless of which candidate token matched first.
    A regression here would silently unfreeze the SigLIP encoder."""
    model = _model_with_top_level_embed_vision()
    names = find_projector_param_names(model)
    forbidden = (
        "vision_tower.patch_embedder",
        "vision_tower.encoder",
        "vision_tower.pooler",
    )
    for n in names:
        for f in forbidden:
            assert f not in n, f"projector incorrectly matched encoder param {n}"


def test_find_projector_excludes_encoder_under_alt_layout() -> None:
    """Same guarantee under the alt layout where projector is also under
    vision_tower. The embedding_projection must match, but the encoder
    siblings must not."""
    model = _model_with_projector_under_vision_tower()
    names = find_projector_param_names(model)
    assert any("embedding_projection" in n for n in names)
    for n in names:
        assert ".encoder." not in n
        assert ".patch_embedder." not in n
        assert ".pooler." not in n


# ---------------------------------------------------------------------------
# Look-alike rejection (boundary-dot anchoring, like freeze.py)
# ---------------------------------------------------------------------------


def test_find_projector_rejects_lookalikes() -> None:
    """`embed_vision_descriptor.*` must not match `embed_vision.`."""
    model = FakeModel(
        params=[
            ("embed_vision_descriptor.weight", FakeParam("fp")),
            ("language_model.embed_vision_aux.weight", FakeParam("fp2")),
            # One real match so the helper doesn't raise on empty.
            ("embed_vision.embedding_projection.weight", FakeParam("real")),
        ],
        modules=[("embed_vision", FakeModule("real"))],
    )
    names = find_projector_param_names(model)
    assert names == ["embed_vision.embedding_projection.weight"]


# ---------------------------------------------------------------------------
# Audio embedder must NOT be flagged (`embed_audio.*` is symmetric naming)
# ---------------------------------------------------------------------------


def test_find_projector_does_not_match_audio_embedder() -> None:
    model = _model_with_top_level_embed_vision()
    names = find_projector_param_names(model)
    for n in names:
        assert "embed_audio" not in n
        assert "audio_tower" not in n


# ---------------------------------------------------------------------------
# Fail-loud when no projector found
# ---------------------------------------------------------------------------


def test_find_projector_raises_when_no_match() -> None:
    """Model with no projector params → RuntimeError. This guards against a
    silent no-op where modules_to_save would be empty and projector tuning
    would silently degrade to LoRA-only."""
    model = FakeModel(
        params=[
            ("language_model.layers.0.q_proj.weight", FakeParam("lq")),
            ("vision_tower.encoder.layer.0.weight", FakeParam("v")),  # encoder, not projector
        ]
    )
    with pytest.raises(RuntimeError, match="no projector"):
        find_projector_param_names(model)


# ---------------------------------------------------------------------------
# find_projector_module_names — short names suitable for PEFT modules_to_save
# ---------------------------------------------------------------------------


def test_find_projector_module_names_returns_short_names() -> None:
    """PEFT's modules_to_save accepts short names that are matched as
    suffixes of the full module path. We return the short final segment
    (e.g. "embed_vision") so PEFT can match it across PEFT/HF wrapping."""
    model = _model_with_top_level_embed_vision()
    short_names = find_projector_module_names(model)
    # We expect at least the top-level Gemma4MultimodalEmbedder module name.
    assert "embed_vision" in short_names


def test_find_projector_module_names_under_vision_tower() -> None:
    model = _model_with_projector_under_vision_tower()
    short_names = find_projector_module_names(model)
    # The module is at vision_tower.embedding_projection — short name is
    # "embedding_projection".
    assert "embedding_projection" in short_names


def test_find_projector_module_names_raises_when_no_match() -> None:
    model = FakeModel(
        params=[("language_model.layers.0.q_proj.weight", FakeParam("lq"))],
        modules=[("language_model", FakeModule("lm"))],
    )
    with pytest.raises(RuntimeError, match="no projector"):
        find_projector_module_names(model)


# ---------------------------------------------------------------------------
# ensure_projector_trainable
# ---------------------------------------------------------------------------


def test_ensure_projector_trainable_flips_only_named_params() -> None:
    """Synthetic model with frozen projector + frozen non-projector. Only
    the projector requires_grad flips; return value matches count flipped."""
    model = FakeModel(params=[
        ("base_model.model.embed_vision.embedding_projection.weight",
         FakeParam("ev", requires_grad=False)),
        ("base_model.model.embed_vision.embedding_pre_projection_norm.weight",
         FakeParam("evn", requires_grad=False)),
        ("base_model.model.vision_tower.encoder.weight",
         FakeParam("vte", requires_grad=False)),
    ])
    targets = [
        "base_model.model.embed_vision.embedding_projection.weight",
        "base_model.model.embed_vision.embedding_pre_projection_norm.weight",
    ]
    flipped = ensure_projector_trainable(model, targets)
    assert flipped == 2

    by_name = dict(model.named_parameters())
    assert by_name[targets[0]].requires_grad is True
    assert by_name[targets[1]].requires_grad is True
    # Encoder must remain untouched (frozen).
    assert by_name["base_model.model.vision_tower.encoder.weight"].requires_grad is False


def test_ensure_projector_trainable_no_op_when_already_trainable() -> None:
    model = FakeModel(params=[
        ("embed_vision.embedding_projection.weight", FakeParam("ev", requires_grad=True)),
    ])
    flipped = ensure_projector_trainable(model, ["embed_vision.embedding_projection.weight"])
    assert flipped == 0


def test_ensure_projector_trainable_ignores_unlisted_names() -> None:
    """If a target name doesn't appear in named_parameters, it's silently
    skipped (the model genuinely doesn't have that param). The function
    only flips params it actually finds."""
    model = FakeModel(params=[
        ("embed_vision.embedding_projection.weight", FakeParam("ev", requires_grad=False)),
    ])
    flipped = ensure_projector_trainable(
        model,
        [
            "embed_vision.embedding_projection.weight",
            "embed_vision.nonexistent.weight",
        ],
    )
    assert flipped == 1


def test_ensure_projector_trainable_skips_original_module() -> None:
    """PEFT's original_module is a frozen reference copy. Even though it
    matches projector candidate tokens (embed_vision.), it must NOT be
    flipped to requires_grad=True."""
    prefix = "base_model.model.model."
    model = FakeModel(params=[
        # PEFT frozen reference — must stay frozen
        (f"{prefix}embed_vision.original_module.embedding_projection.weight",
         FakeParam("ev_orig", requires_grad=False)),
        # PEFT trainable copy — must be flipped
        (f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight",
         FakeParam("ev_mts", requires_grad=False)),
        # Audio — must stay frozen
        (f"{prefix}embed_audio.embedding_projection.weight",
         FakeParam("ea", requires_grad=False)),
    ])
    flipped = ensure_projector_trainable(
        model,
        [f"{prefix}embed_vision.embedding_projection.weight"],  # pre-PEFT name
    )
    # Only modules_to_save copy should be flipped — NOT original_module
    assert flipped == 1
    by_name = dict(model.named_parameters())
    assert by_name[
        f"{prefix}embed_vision.original_module.embedding_projection.weight"
    ].requires_grad is False
    assert by_name[
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight"
    ].requires_grad is True
    assert by_name[
        f"{prefix}embed_audio.embedding_projection.weight"
    ].requires_grad is False
