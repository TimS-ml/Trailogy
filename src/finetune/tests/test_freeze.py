"""Tests for src.freeze — vision/audio tower freeze logic.

These tests use a duck-typed fake model so they run without torch on CPU.
The real `freeze_vision_audio_towers` only touches `named_parameters()`,
`param.requires_grad`, and `param.numel()` — same surface as torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import pytest

from src.freeze import (
    DEFAULT_FROZEN_PREFIXES,
    assert_frozen,
    freeze_vision_audio_towers,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeParam:
    """Stand-in for torch.nn.Parameter for unit tests."""

    name: str  # for debugging only
    shape: Tuple[int, ...] = (10,)
    requires_grad: bool = True

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass
class FakeModel:
    """Mimics nn.Module.named_parameters() iteration."""

    params: List[Tuple[str, FakeParam]] = field(default_factory=list)

    def named_parameters(self) -> Iterator[Tuple[str, FakeParam]]:
        for name, p in self.params:
            yield name, p

    def parameters(self) -> Iterator[FakeParam]:
        for _, p in self.params:
            yield p


def _make_gemma4_like_model() -> FakeModel:
    """Build a fake module tree shaped like Gemma 4 multimodal params."""
    entries = []

    # Vision tower (16 layers in real model — represent two)
    for i in range(2):
        entries.append((f"vision_tower.encoder.layer.{i}.attn.q_proj.weight",
                        FakeParam("v_q", shape=(1024, 1024))))
    entries.append(("embed_vision.weight", FakeParam("ev", shape=(1024, 256))))

    # Audio tower
    for i in range(2):
        entries.append((f"audio_tower.encoder.layer.{i}.linear.weight",
                        FakeParam("a", shape=(512, 512))))
    entries.append(("embed_audio.weight", FakeParam("ea", shape=(512, 256))))

    # Language layers (with LoRA-ish names)
    for i in range(3):
        entries.append((f"language_model.layers.{i}.self_attn.q_proj.lora_A.weight",
                        FakeParam("lA", shape=(8, 1024))))
        entries.append((f"language_model.layers.{i}.self_attn.q_proj.lora_B.weight",
                        FakeParam("lB", shape=(1024, 8))))
        # The base linear weight stays trainable in QLoRA but no grad in PEFT;
        # we model that as requires_grad=True here so we can confirm the
        # freeze pass leaves language unaffected.
        entries.append((f"language_model.layers.{i}.self_attn.q_proj.weight",
                        FakeParam("ql", shape=(1024, 1024))))

    return FakeModel(params=entries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_prefixes_cover_audio_and_vision() -> None:
    # Sanity: the prefix list mentions both towers.
    assert any("audio_tower" in p for p in DEFAULT_FROZEN_PREFIXES)
    assert any("vision_tower" in p for p in DEFAULT_FROZEN_PREFIXES)
    assert any("embed_audio" in p for p in DEFAULT_FROZEN_PREFIXES)
    assert any("embed_vision" in p for p in DEFAULT_FROZEN_PREFIXES)


def test_freeze_disables_grad_on_audio_and_vision_only() -> None:
    model = _make_gemma4_like_model()
    report = freeze_vision_audio_towers(model)

    for name, param in model.named_parameters():
        if name.startswith("audio_tower.") or name.startswith("embed_audio.") \
                or name.startswith("vision_tower.") or name.startswith("embed_vision."):
            assert param.requires_grad is False, f"{name} should be frozen"
        else:
            # Language params untouched
            assert param.requires_grad is True, f"{name} should remain trainable"

    assert report.frozen_params > 0
    assert report.trainable_params > 0
    assert report.total_params == report.frozen_params + report.trainable_params


def test_freeze_counts_lora_params() -> None:
    model = _make_gemma4_like_model()
    report = freeze_vision_audio_towers(model)
    # 3 layers × 2 LoRA matrices per layer = 6 lora_* tensors
    # lora_A: 8 * 1024 = 8192 ; lora_B: 1024 * 8 = 8192 ; total per pair = 16384
    # 3 pairs = 49152
    assert report.trainable_lora_params == 3 * (8 * 1024 + 1024 * 8)


def test_assert_frozen_passes_after_freeze() -> None:
    model = _make_gemma4_like_model()
    freeze_vision_audio_towers(model)
    # Should not raise
    assert_frozen(model)


def test_assert_frozen_raises_when_audio_still_trainable() -> None:
    model = _make_gemma4_like_model()
    freeze_vision_audio_towers(model)
    # Manually re-enable a single audio param to simulate a leak.
    for name, param in model.named_parameters():
        if name.startswith("audio_tower."):
            param.requires_grad = True
            break
    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(model)


def test_extra_prefixes_are_honored() -> None:
    model = FakeModel(
        params=[
            ("custom_extra.weight", FakeParam("ce")),
            ("language_model.layer.weight", FakeParam("lm")),
        ]
    )
    freeze_vision_audio_towers(model, extra_prefixes=("custom_extra.",))
    params = dict(model.named_parameters())
    assert params["custom_extra.weight"].requires_grad is False
    assert params["language_model.layer.weight"].requires_grad is True


def test_idempotent() -> None:
    """Running the freeze pass twice should not change anything."""
    model = _make_gemma4_like_model()
    r1 = freeze_vision_audio_towers(model)
    r2 = freeze_vision_audio_towers(model)
    assert r1.frozen_params == r2.frozen_params
    assert r1.trainable_params == r2.trainable_params


# ---------------------------------------------------------------------------
# PEFT-wrapped names — what unsloth actually produces in production.
#
# unsloth's FastModel.get_peft_model wraps the HF Gemma4 model in PEFT, which
# prepends "base_model.model." to every parameter name. HF's
# Gemma4ForConditionalGeneration also wraps the inner sub-modules under "model.",
# so vision_tower can show up as any of:
#
#   vision_tower.encoder...                       (raw HF model, top-level)
#   model.vision_tower.encoder...                 (HF wrapping)
#   base_model.model.vision_tower.encoder...      (PEFT + HF, single model.)
#   base_model.model.model.vision_tower.encoder...(PEFT + HF, double model.)
#
# An earlier version of DEFAULT_FROZEN_PREFIXES used `name.startswith(prefix)`
# with a finite list, which silently missed the PEFT-wrapped variants — both
# the freeze (no-op) AND the assert_frozen tripwire would skip them entirely.
#
# These tests pin the PEFT-aware behavior so it can't regress.
# ---------------------------------------------------------------------------


def _make_peft_wrapped_gemma4_model(extra_prefix: str = "base_model.model.") -> FakeModel:
    """Same shape as the Gemma 4 fake but with a PEFT wrapper prefix."""
    entries = []

    # Vision tower (frozen — wrapped name)
    for i in range(2):
        entries.append((
            f"{extra_prefix}vision_tower.encoder.layer.{i}.attn.q_proj.weight",
            FakeParam("v_q", shape=(1024, 1024)),
        ))
    entries.append((
        f"{extra_prefix}embed_vision.weight",
        FakeParam("ev", shape=(1024, 256)),
    ))

    # Audio tower (frozen — wrapped name)
    for i in range(2):
        entries.append((
            f"{extra_prefix}audio_tower.encoder.layer.{i}.linear.weight",
            FakeParam("a", shape=(512, 512)),
        ))
    entries.append((
        f"{extra_prefix}embed_audio.weight",
        FakeParam("ea", shape=(512, 256)),
    ))

    # Language layers (trainable, with LoRA)
    for i in range(3):
        entries.append((
            f"{extra_prefix}language_model.layers.{i}.self_attn.q_proj.lora_A.weight",
            FakeParam("lA", shape=(8, 1024)),
        ))
        entries.append((
            f"{extra_prefix}language_model.layers.{i}.self_attn.q_proj.lora_B.weight",
            FakeParam("lB", shape=(1024, 8)),
        ))

    return FakeModel(params=entries)


@pytest.mark.parametrize(
    "wrapper",
    [
        "",                               # raw HF model
        "model.",                         # HF wrapping only
        "base_model.model.",              # PEFT + single HF wrap
        "base_model.model.model.",        # PEFT + double HF wrap
    ],
)
def test_freeze_handles_peft_wrapping(wrapper: str) -> None:
    """Vision/audio params must be frozen regardless of wrapper depth."""
    model = _make_peft_wrapped_gemma4_model(extra_prefix=wrapper)
    freeze_vision_audio_towers(model)

    for name, param in model.named_parameters():
        is_vision_or_audio = any(
            tower in name
            for tower in ("vision_tower.", "audio_tower.", "embed_vision.", "embed_audio.")
        )
        if is_vision_or_audio:
            assert param.requires_grad is False, (
                f"{name} (wrapper={wrapper!r}) should be frozen but requires_grad=True"
            )
        else:
            assert param.requires_grad is True, (
                f"{name} (wrapper={wrapper!r}) should be trainable but requires_grad=False"
            )


@pytest.mark.parametrize(
    "wrapper",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_assert_frozen_catches_leak_under_peft_wrapping(wrapper: str) -> None:
    """assert_frozen must raise even when the leaked param has a PEFT prefix."""
    model = _make_peft_wrapped_gemma4_model(extra_prefix=wrapper)
    freeze_vision_audio_towers(model)
    # Simulate a leak: re-enable a vision param.
    for name, param in model.named_parameters():
        if "vision_tower." in name:
            param.requires_grad = True
            break
    else:
        pytest.fail(f"No vision_tower param found in fixture (wrapper={wrapper!r})")

    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(model)


# ---------------------------------------------------------------------------
# Projector-keep variant (feature/lora-plus-projector)
#
# When training the vision-language projector as full params (PEFT's
# modules_to_save), the freeze pass must SKIP projector params so they
# remain trainable. The vision ENCODER, audio tower, and embed_audio must
# still be frozen — only the projector is exempted.
# ---------------------------------------------------------------------------


from src.freeze import freeze_vision_audio_towers_keeping_projector  # noqa: E402


def _make_gemma4_with_projector(extra_prefix: str = "") -> tuple[FakeModel, list[str]]:
    """Like _make_peft_wrapped_gemma4_model but with explicit projector
    params so we can test the keep-projector variant. Returns (model,
    projector_param_names)."""
    entries = []
    projector_names = []

    # Vision encoder (frozen — must NOT be in projector_names)
    for i in range(2):
        entries.append((
            f"{extra_prefix}vision_tower.encoder.layer.{i}.attn.q_proj.weight",
            FakeParam("v_q", shape=(768, 768)),
        ))
    entries.append((
        f"{extra_prefix}vision_tower.patch_embedder.weight",
        FakeParam("patch", shape=(768, 768)),
    ))
    entries.append((
        f"{extra_prefix}vision_tower.pooler.weight",
        FakeParam("pool", shape=(768, 768)),
    ))

    # Vision PROJECTOR (Gemma4MultimodalEmbedder) — TRAINABLE under keep-variant
    proj_a = f"{extra_prefix}embed_vision.embedding_projection.weight"
    entries.append((proj_a, FakeParam("ev_proj", shape=(2048, 768))))
    projector_names.append(proj_a)

    # Audio (frozen)
    entries.append((
        f"{extra_prefix}audio_tower.encoder.layer.0.weight",
        FakeParam("a", shape=(512, 512)),
    ))
    entries.append((
        f"{extra_prefix}embed_audio.embedding_projection.weight",
        FakeParam("ea", shape=(2048, 512)),
    ))

    # Language (LoRA — trainable, untouched by freeze)
    for i in range(2):
        entries.append((
            f"{extra_prefix}language_model.layers.{i}.self_attn.q_proj.lora_A.weight",
            FakeParam("lA", shape=(8, 2048)),
        ))
        entries.append((
            f"{extra_prefix}language_model.layers.{i}.self_attn.q_proj.lora_B.weight",
            FakeParam("lB", shape=(2048, 8)),
        ))

    return FakeModel(params=entries), projector_names


@pytest.mark.parametrize(
    "wrapper",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_freeze_keeping_projector_unfreezes_only_projector(wrapper: str) -> None:
    model, projector_names = _make_gemma4_with_projector(extra_prefix=wrapper)
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    by_name = dict(model.named_parameters())

    # Projector(s) trainable
    for n in projector_names:
        assert by_name[n].requires_grad is True, f"{n} should remain trainable"

    # Vision ENCODER still frozen
    for name, param in model.named_parameters():
        if any(
            t in name
            for t in (
                "vision_tower.encoder.",
                "vision_tower.patch_embedder.",
                "vision_tower.pooler.",
            )
        ):
            assert param.requires_grad is False, f"{name} should be frozen (encoder)"

    # Audio fully frozen
    for name, param in model.named_parameters():
        if "audio_tower." in name or "embed_audio." in name:
            assert param.requires_grad is False, f"{name} should be frozen (audio)"

    # Language untouched
    for name, param in model.named_parameters():
        if "language_model." in name:
            assert param.requires_grad is True, f"{name} should remain trainable"


def test_assert_frozen_with_allowlist_passes_when_only_projector_trainable() -> None:
    model, projector_names = _make_gemma4_with_projector(extra_prefix="base_model.model.")
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    # Should NOT raise: projector is allowlisted.
    assert_frozen(model, allowlist=projector_names)


def test_assert_frozen_with_allowlist_still_catches_audio_leak() -> None:
    """Allowlist exempts only the named projector params. An accidental
    audio_tower unfreeze must still trigger the tripwire."""
    model, projector_names = _make_gemma4_with_projector(extra_prefix="base_model.model.")
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    # Simulate an audio leak.
    for name, param in model.named_parameters():
        if "audio_tower." in name:
            param.requires_grad = True
            break
    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(model, allowlist=projector_names)


def test_assert_frozen_with_allowlist_still_catches_encoder_leak() -> None:
    """A vision_tower.encoder leak must still trigger even when the
    projector is allowlisted."""
    model, projector_names = _make_gemma4_with_projector(extra_prefix="base_model.model.")
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    for name, param in model.named_parameters():
        if "vision_tower.encoder." in name:
            param.requires_grad = True
            break
    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(model, allowlist=projector_names)


def test_freeze_keeping_projector_idempotent() -> None:
    model, projector_names = _make_gemma4_with_projector(extra_prefix="base_model.model.")
    r1 = freeze_vision_audio_towers_keeping_projector(model, projector_param_names=projector_names)
    r2 = freeze_vision_audio_towers_keeping_projector(model, projector_param_names=projector_names)
    assert r1.frozen_params == r2.frozen_params
    assert r1.trainable_params == r2.trainable_params


def test_freeze_does_not_match_unrelated_substrings() -> None:
    """Substring matching must not over-match arbitrary names containing the
    tower words. We only want exact tower / embed module paths."""
    model = FakeModel(
        params=[
            # Plausible false-positive shapes:
            ("language_model.layers.0.vision_tower_descriptor.weight",
             FakeParam("fp1")),  # "vision_tower" appears mid-name without the dot anchor
            ("some.other.audio_tower_aux.weight", FakeParam("fp2")),
            # Real ones:
            ("base_model.model.vision_tower.encoder.weight", FakeParam("real_v")),
            ("base_model.model.audio_tower.encoder.weight", FakeParam("real_a")),
        ]
    )
    freeze_vision_audio_towers(model)
    params = dict(model.named_parameters())
    # Real towers MUST be frozen.
    assert params["base_model.model.vision_tower.encoder.weight"].requires_grad is False
    assert params["base_model.model.audio_tower.encoder.weight"].requires_grad is False
    # Bogus look-alikes must stay trainable. The "." after the tower name
    # (e.g. "vision_tower." vs "vision_tower_") is the discriminator.
    assert params["language_model.layers.0.vision_tower_descriptor.weight"].requires_grad is True
    assert params["some.other.audio_tower_aux.weight"].requires_grad is True


# ---------------------------------------------------------------------------
# Regression: PEFT dual-path (original_module + modules_to_save.default)
# ---------------------------------------------------------------------------


def _make_peft_wrapped_projector_model() -> tuple[FakeModel, list[str]]:
    """Simulate the PEFT modules_to_save wrapping for embed_vision.

    After ``FastModel.get_peft_model(modules_to_save=['embed_vision'])``,
    the parameter tree has TWO copies of each projector param:

    - ``embed_vision.original_module.embedding_projection.weight``
      (frozen reference — PEFT keeps this to compute deltas at save time)
    - ``embed_vision.modules_to_save.default.embedding_projection.weight``
      (trainable copy — this is the one that should be in the optimizer)

    The ``original_module`` copy must NEVER be unfrozen or placed in the
    optimizer's projector param group.
    """
    prefix = "base_model.model.model."
    entries = []

    # Vision encoder (frozen)
    entries.append((
        f"{prefix}vision_tower.encoder.layer.0.weight",
        FakeParam("v_enc", shape=(768, 768)),
    ))

    # Projector — PEFT original_module (frozen reference, requires_grad=False)
    entries.append((
        f"{prefix}embed_vision.original_module.embedding_projection.weight",
        FakeParam("ev_orig", shape=(1536, 768), requires_grad=False),
    ))
    entries.append((
        f"{prefix}embed_vision.original_module.embedding_pre_projection_norm.weight",
        FakeParam("ev_norm_orig", shape=(768,), requires_grad=False),
    ))

    # Projector — PEFT modules_to_save.default (trainable copy)
    entries.append((
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight",
        FakeParam("ev_mts", shape=(1536, 768), requires_grad=True),
    ))
    entries.append((
        f"{prefix}embed_vision.modules_to_save.default.embedding_pre_projection_norm.weight",
        FakeParam("ev_norm_mts", shape=(768,), requires_grad=True),
    ))

    # Audio (frozen)
    entries.append((
        f"{prefix}audio_tower.encoder.layer.0.weight",
        FakeParam("a_enc", shape=(512, 512)),
    ))
    entries.append((
        f"{prefix}embed_audio.embedding_projection.weight",
        FakeParam("ea", shape=(1536, 512)),
    ))

    # Language LoRA (trainable)
    entries.append((
        f"{prefix}language_model.layers.0.self_attn.q_proj.lora_A.weight",
        FakeParam("lA", shape=(8, 1536)),
    ))

    # Pre-PEFT projector names (for API compat — not used for matching)
    projector_names = [f"{prefix}embed_vision.embedding_projection.weight"]

    return FakeModel(params=entries), projector_names


def test_freeze_keeping_projector_skips_original_module() -> None:
    """The freeze-with-projector pass must NOT unfreeze original_module copies.

    Only modules_to_save.default.* should remain trainable. The
    original_module.* copy is PEFT's frozen reference and must stay frozen.
    """
    model, projector_names = _make_peft_wrapped_projector_model()
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    by_name = dict(model.named_parameters())
    prefix = "base_model.model.model."

    # modules_to_save.default copies MUST be trainable
    assert by_name[
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight"
    ].requires_grad is True
    assert by_name[
        f"{prefix}embed_vision.modules_to_save.default.embedding_pre_projection_norm.weight"
    ].requires_grad is True

    # original_module copies MUST be frozen
    assert by_name[
        f"{prefix}embed_vision.original_module.embedding_projection.weight"
    ].requires_grad is False
    assert by_name[
        f"{prefix}embed_vision.original_module.embedding_pre_projection_norm.weight"
    ].requires_grad is False

    # Audio still frozen
    assert by_name[f"{prefix}audio_tower.encoder.layer.0.weight"].requires_grad is False
    assert by_name[f"{prefix}embed_audio.embedding_projection.weight"].requires_grad is False

    # Vision encoder still frozen
    assert by_name[f"{prefix}vision_tower.encoder.layer.0.weight"].requires_grad is False


def test_assert_frozen_allows_only_modules_to_save_projector() -> None:
    """assert_frozen with allowlist must pass when only modules_to_save.*
    projector params are trainable, and original_module.* stay frozen."""
    model, projector_names = _make_peft_wrapped_projector_model()
    freeze_vision_audio_towers_keeping_projector(
        model, projector_param_names=projector_names
    )
    # Should NOT raise — only modules_to_save copies are trainable.
    assert_frozen(model, allowlist=projector_names)


# ---------------------------------------------------------------------------
# freeze_vision_audio_towers_keeping_projector_and_vision_layers
# (feature/lora-plus-projector-plus-vision-tower)
#
# Like _keeping_projector, but ALSO unfreezes the last-N vision encoder
# layers via the tuned_vision_layer_indices arg. Everything else under
# the frozen tokens (audio, vision encoder layers 0..(total-N-1),
# patch_embedder, pooler) stays frozen.
# ---------------------------------------------------------------------------


from src.freeze import (  # noqa: E402
    freeze_vision_audio_towers_keeping_projector_and_vision_layers,
)


def _make_gemma4_with_projector_and_vision_layers(
    extra_prefix: str = "", total_vision_layers: int = 16,
) -> tuple[FakeModel, list[str], list[int]]:
    """Build a synthetic model with full vision encoder layer set so we
    can test selectively unfreezing the last N.

    Returns (model, projector_param_names, all_layer_indices).
    """
    entries = []
    projector_names = []

    # patch_embedder + pooler (frozen)
    entries.append((
        f"{extra_prefix}vision_tower.patch_embedder.weight",
        FakeParam("patch", shape=(768, 768)),
    ))
    entries.append((
        f"{extra_prefix}vision_tower.pooler.weight",
        FakeParam("pool", shape=(768, 768)),
    ))

    # Vision encoder layers
    for i in range(total_vision_layers):
        entries.append((
            f"{extra_prefix}vision_tower.encoder.layers.{i}.attn.q_proj.weight",
            FakeParam(f"v_q_{i}", shape=(768, 768)),
        ))
        entries.append((
            f"{extra_prefix}vision_tower.encoder.layers.{i}.mlp.fc1.weight",
            FakeParam(f"v_fc_{i}", shape=(3072, 768)),
        ))

    # Projector
    proj = f"{extra_prefix}embed_vision.embedding_projection.weight"
    entries.append((proj, FakeParam("ev_proj", shape=(2048, 768))))
    projector_names.append(proj)

    # Audio (frozen)
    entries.append((
        f"{extra_prefix}audio_tower.encoder.weight",
        FakeParam("a", shape=(512, 512)),
    ))
    entries.append((
        f"{extra_prefix}embed_audio.embedding_projection.weight",
        FakeParam("ea", shape=(2048, 512)),
    ))

    # Language (LoRA, trainable)
    for i in range(2):
        entries.append((
            f"{extra_prefix}language_model.layers.{i}.self_attn.q_proj.lora_A.weight",
            FakeParam("lA"),
        ))

    return FakeModel(params=entries), projector_names, list(range(total_vision_layers))


@pytest.mark.parametrize(
    "wrapper",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_freeze_keeps_last_n_vision_layers_trainable(wrapper: str) -> None:
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        extra_prefix=wrapper, total_vision_layers=16,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[14, 15],
    )
    by_name = dict(model.named_parameters())

    # Projector trainable
    for n in projector_names:
        assert by_name[n].requires_grad is True, n

    # Layers 14, 15 trainable
    for i in (14, 15):
        for suffix in ("attn.q_proj.weight", "mlp.fc1.weight"):
            key = f"{wrapper}vision_tower.encoder.layers.{i}.{suffix}"
            assert by_name[key].requires_grad is True, key

    # Layers 0..13 FROZEN
    for i in range(14):
        for suffix in ("attn.q_proj.weight", "mlp.fc1.weight"):
            key = f"{wrapper}vision_tower.encoder.layers.{i}.{suffix}"
            assert by_name[key].requires_grad is False, key

    # patch_embedder, pooler frozen
    assert by_name[f"{wrapper}vision_tower.patch_embedder.weight"].requires_grad is False
    assert by_name[f"{wrapper}vision_tower.pooler.weight"].requires_grad is False

    # Audio frozen
    assert by_name[f"{wrapper}audio_tower.encoder.weight"].requires_grad is False
    assert by_name[f"{wrapper}embed_audio.embedding_projection.weight"].requires_grad is False

    # Language untouched
    for name, param in model.named_parameters():
        if "language_model." in name:
            assert param.requires_grad is True


def test_freeze_keeps_only_specified_indices() -> None:
    """tuned_vision_layer_indices=[15] keeps ONLY layer 15 trainable;
    layer 14 stays frozen."""
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        total_vision_layers=16,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[15],
    )
    by_name = dict(model.named_parameters())
    assert by_name["vision_tower.encoder.layers.15.attn.q_proj.weight"].requires_grad is True
    assert by_name["vision_tower.encoder.layers.14.attn.q_proj.weight"].requires_grad is False


def test_freeze_empty_indices_matches_keeping_projector() -> None:
    """With tuned_vision_layer_indices=(), behavior must match the
    existing _keeping_projector variant exactly: all vision encoder
    layers frozen."""
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        total_vision_layers=4,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=(),
    )
    by_name = dict(model.named_parameters())
    for i in range(4):
        assert by_name[f"vision_tower.encoder.layers.{i}.attn.q_proj.weight"].requires_grad is False
    # Projector still trainable
    for n in projector_names:
        assert by_name[n].requires_grad is True


def test_assert_frozen_with_vision_layer_allowlist_passes() -> None:
    """assert_frozen with both projector allowlist + vision_layer_indices
    must pass when only those are trainable."""
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        extra_prefix="base_model.model.", total_vision_layers=16,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[14, 15],
    )
    # Should NOT raise.
    assert_frozen(
        model,
        allowlist=projector_names,
        tuned_vision_layer_indices=[14, 15],
    )


def test_assert_frozen_with_vision_layer_allowlist_catches_audio_leak() -> None:
    """Even with vision-layer allowlist, audio_tower leak must still raise."""
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        total_vision_layers=16,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[14, 15],
    )
    # Simulate audio leak.
    for name, param in model.named_parameters():
        if "audio_tower." in name:
            param.requires_grad = True
            break
    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(
            model,
            allowlist=projector_names,
            tuned_vision_layer_indices=[14, 15],
        )


def test_assert_frozen_catches_unauthorized_vision_layer_leak() -> None:
    """If layer 13 is unfrozen but only 14,15 are in the allowlist, raise."""
    model, projector_names, _ = _make_gemma4_with_projector_and_vision_layers(
        total_vision_layers=16,
    )
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[14, 15],
    )
    # Simulate layer-13 leak.
    by_name = dict(model.named_parameters())
    by_name["vision_tower.encoder.layers.13.attn.q_proj.weight"].requires_grad = True
    with pytest.raises(RuntimeError, match="param"):
        assert_frozen(
            model,
            allowlist=projector_names,
            tuned_vision_layer_indices=[14, 15],
        )


def test_freeze_keeping_vision_layers_skips_peft_original_module() -> None:
    """PEFT modules_to_save creates an .original_module. frozen copy that
    must remain frozen; only the .modules_to_save.{adapter}. copy is the
    trainable one."""
    entries = [
        # PEFT wrapped vision layer 15
        ("base_model.model.model.vision_tower.encoder.layers.15"
         ".original_module.attn.q_proj.weight",
         FakeParam("orig", shape=(768, 768))),
        ("base_model.model.model.vision_tower.encoder.layers.15"
         ".modules_to_save.default.attn.q_proj.weight",
         FakeParam("mts", shape=(768, 768))),
        # Layer 0 — should be frozen
        ("base_model.model.model.vision_tower.encoder.layers.0"
         ".attn.q_proj.weight",
         FakeParam("v0", shape=(768, 768))),
        # Projector
        ("base_model.model.model.embed_vision.original_module."
         "embedding_projection.weight", FakeParam("ev_orig", shape=(2048, 768))),
        ("base_model.model.model.embed_vision.modules_to_save.default."
         "embedding_projection.weight", FakeParam("ev_mts", shape=(2048, 768))),
        # Audio
        ("base_model.model.model.audio_tower.encoder.weight",
         FakeParam("a", shape=(512, 512))),
    ]
    model = FakeModel(params=entries)
    projector_names = [
        "base_model.model.model.embed_vision.modules_to_save.default."
        "embedding_projection.weight",
    ]
    freeze_vision_audio_towers_keeping_projector_and_vision_layers(
        model,
        projector_param_names=projector_names,
        tuned_vision_layer_indices=[15],
    )
    by_name = dict(model.named_parameters())
    # modules_to_save copy trainable
    assert by_name[
        "base_model.model.model.vision_tower.encoder.layers.15"
        ".modules_to_save.default.attn.q_proj.weight"
    ].requires_grad is True
    # original_module copy frozen
    assert by_name[
        "base_model.model.model.vision_tower.encoder.layers.15"
        ".original_module.attn.q_proj.weight"
    ].requires_grad is False
    # Layer 0 frozen
    assert by_name[
        "base_model.model.model.vision_tower.encoder.layers.0.attn.q_proj.weight"
    ].requires_grad is False
    # Audio frozen
    assert by_name[
        "base_model.model.model.audio_tower.encoder.weight"
    ].requires_grad is False
