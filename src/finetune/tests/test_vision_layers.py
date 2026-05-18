"""Tests for src.vision_layers — last-N vision encoder layer tuning.

Uses the same FakeParam/FakeModule duck-types as test_projector.py — no
torch needed. Covers raw HF layout AND PEFT-wrapped layouts (single + double
``model.`` infix) to ensure introspection works regardless of wrapping
depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import pytest

from src.vision_layers import (
    VISION_ENCODER_LAYERS_TOKEN,
    ensure_vision_layers_trainable,
    find_last_n_vision_layer_module_names,
    find_vision_encoder_layer_count,
    find_vision_layer_param_names,
    is_tuned_vision_layer_param,
)


# ---------------------------------------------------------------------------
# Fakes
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


def _build_model(
    total_layers: int = 16,
    prefix: str = "",
    include_peft_original_module: bool = False,
) -> FakeModel:
    """Build a synthetic Gemma 4 layout with `total_layers` vision encoder layers.

    `prefix` simulates HF/PEFT wrapping (e.g. 'base_model.model.model.').
    """
    params: List[Tuple[str, FakeParam]] = []
    modules: List[Tuple[str, FakeModule]] = []

    # Text decoder layers (collision-test for the suffix matcher).
    for i in range(4):
        params.append(
            (f"{prefix}language_model.layers.{i}.self_attn.q_proj.weight",
             FakeParam(f"lm_q_{i}", shape=(2048, 2048))),
        )
        modules.append(
            (f"{prefix}language_model.layers.{i}", FakeModule(f"lm_layer_{i}")),
        )

    # Vision encoder layers.
    modules.append((f"{prefix}vision_tower", FakeModule("vt")))
    modules.append((f"{prefix}vision_tower.encoder", FakeModule("enc")))
    modules.append((f"{prefix}vision_tower.patch_embedder", FakeModule("patch")))
    modules.append((f"{prefix}vision_tower.pooler", FakeModule("pooler")))
    params.append(
        (f"{prefix}vision_tower.patch_embedder.weight",
         FakeParam("patch", shape=(768, 768))),
    )
    params.append(
        (f"{prefix}vision_tower.pooler.weight",
         FakeParam("pool", shape=(768, 768))),
    )
    for i in range(total_layers):
        modules.append(
            (f"{prefix}vision_tower.encoder.layers.{i}",
             FakeModule(f"vt_layer_{i}")),
        )
        params.append(
            (f"{prefix}vision_tower.encoder.layers.{i}.attn.q_proj.weight",
             FakeParam(f"vt_q_{i}", shape=(768, 768))),
        )
        params.append(
            (f"{prefix}vision_tower.encoder.layers.{i}.mlp.fc1.weight",
             FakeParam(f"vt_fc_{i}", shape=(3072, 768))),
        )

    # Projector
    modules.append((f"{prefix}embed_vision", FakeModule("ev")))
    params.append(
        (f"{prefix}embed_vision.embedding_projection.weight",
         FakeParam("ev_proj", shape=(2048, 768))),
    )

    # Audio (frozen)
    modules.append((f"{prefix}audio_tower", FakeModule("at")))
    params.append(
        (f"{prefix}audio_tower.encoder.weight", FakeParam("au", shape=(512, 512))),
    )

    if include_peft_original_module:
        # PEFT modules_to_save creates an .original_module. frozen-copy
        # alongside the .modules_to_save.{adapter}. trainable copy.
        # Only the modules_to_save copy should be detected as tunable.
        params.append(
            (f"{prefix}vision_tower.encoder.layers.{total_layers - 1}"
             ".original_module.attn.q_proj.weight",
             FakeParam("orig", shape=(768, 768), requires_grad=False)),
        )
        params.append(
            (f"{prefix}vision_tower.encoder.layers.{total_layers - 1}"
             ".modules_to_save.default.attn.q_proj.weight",
             FakeParam("mts", shape=(768, 768))),
        )

    return FakeModel(params=params, modules=modules)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_vision_encoder_layers_token_has_trailing_dot() -> None:
    assert VISION_ENCODER_LAYERS_TOKEN.endswith(".")
    assert "vision_tower.encoder.layers" in VISION_ENCODER_LAYERS_TOKEN


# ---------------------------------------------------------------------------
# find_vision_encoder_layer_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    ["", "model.", "base_model.model.", "base_model.model.model."],
)
def test_layer_count_works_across_wrapping_depths(prefix: str) -> None:
    model = _build_model(total_layers=16, prefix=prefix)
    assert find_vision_encoder_layer_count(model) == 16


def test_layer_count_small_model() -> None:
    model = _build_model(total_layers=4)
    assert find_vision_encoder_layer_count(model) == 4


def test_layer_count_raises_when_no_encoder_layers() -> None:
    model = FakeModel(
        params=[("language_model.layers.0.weight", FakeParam("lm"))],
        modules=[("language_model.layers.0", FakeModule("lm"))],
    )
    with pytest.raises(RuntimeError, match="vision_tower"):
        find_vision_encoder_layer_count(model)


# ---------------------------------------------------------------------------
# find_last_n_vision_layer_module_names
# ---------------------------------------------------------------------------


def test_find_last_n_returns_correct_suffixes() -> None:
    model = _build_model(total_layers=16)
    names = find_last_n_vision_layer_module_names(model, n=2)
    assert names == [
        "vision_tower.encoder.layers.14",
        "vision_tower.encoder.layers.15",
    ]


def test_find_last_n_does_not_carry_wrapping_prefix() -> None:
    """PEFT matches modules_to_save as suffix — we strip any wrapping prefix
    so the same string works under any wrapping depth."""
    model = _build_model(total_layers=16, prefix="base_model.model.model.")
    names = find_last_n_vision_layer_module_names(model, n=2)
    assert names == [
        "vision_tower.encoder.layers.14",
        "vision_tower.encoder.layers.15",
    ]


def test_find_last_n_with_n_one() -> None:
    model = _build_model(total_layers=16)
    names = find_last_n_vision_layer_module_names(model, n=1)
    assert names == ["vision_tower.encoder.layers.15"]


def test_find_last_n_raises_on_zero() -> None:
    model = _build_model(total_layers=16)
    with pytest.raises(ValueError, match="n must be >= 1"):
        find_last_n_vision_layer_module_names(model, n=0)


def test_find_last_n_raises_on_negative() -> None:
    model = _build_model(total_layers=16)
    with pytest.raises(ValueError, match="n must be >= 1"):
        find_last_n_vision_layer_module_names(model, n=-2)


def test_find_last_n_raises_when_n_exceeds_total() -> None:
    model = _build_model(total_layers=16)
    with pytest.raises(ValueError, match="n=20.*total=16"):
        find_last_n_vision_layer_module_names(model, n=20)


def test_find_last_n_accepts_n_equal_to_total() -> None:
    model = _build_model(total_layers=4)
    names = find_last_n_vision_layer_module_names(model, n=4)
    assert names == [
        "vision_tower.encoder.layers.0",
        "vision_tower.encoder.layers.1",
        "vision_tower.encoder.layers.2",
        "vision_tower.encoder.layers.3",
    ]


# ---------------------------------------------------------------------------
# is_tuned_vision_layer_param
# ---------------------------------------------------------------------------


def test_is_tuned_matches_last_two_indices() -> None:
    tuned = {14, 15}
    assert is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.14.attn.q_proj.weight", tuned
    )
    assert is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.15.mlp.fc1.weight", tuned
    )


def test_is_tuned_rejects_earlier_layers() -> None:
    tuned = {14, 15}
    assert not is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.0.attn.q_proj.weight", tuned
    )
    assert not is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.13.attn.q_proj.weight", tuned
    )


def test_is_tuned_rejects_text_decoder_layers_at_same_index() -> None:
    """Critical collision case: model.layers.14 (text decoder) MUST NOT match."""
    tuned = {14, 15}
    assert not is_tuned_vision_layer_param(
        "language_model.layers.14.self_attn.q_proj.weight", tuned
    )
    assert not is_tuned_vision_layer_param(
        "model.layers.14.self_attn.q_proj.weight", tuned
    )


def test_is_tuned_works_under_peft_wrapping() -> None:
    tuned = {14, 15}
    assert is_tuned_vision_layer_param(
        "base_model.model.model.vision_tower.encoder.layers.14.attn.q_proj.weight",
        tuned,
    )


def test_is_tuned_rejects_peft_original_module_copy() -> None:
    """PEFT modules_to_save creates a frozen `.original_module.` copy.
    Only the `.modules_to_save.{adapter}.` copy is the trainable one."""
    tuned = {14, 15}
    assert not is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.14.original_module.attn.q_proj.weight",
        tuned,
    )
    assert is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.14.modules_to_save.default.attn.q_proj.weight",
        tuned,
    )


def test_is_tuned_with_empty_indices_returns_false() -> None:
    assert not is_tuned_vision_layer_param(
        "vision_tower.encoder.layers.14.attn.q_proj.weight", set()
    )


def test_is_tuned_rejects_lookalike_paths() -> None:
    """Avoid false positives on look-alikes."""
    tuned = {14, 15}
    # vision_tower_descriptor (a lookalike, hypothetical)
    assert not is_tuned_vision_layer_param(
        "vision_tower_descriptor.encoder.layers.14.weight", tuned
    )
    # layers without the encoder. prefix
    assert not is_tuned_vision_layer_param(
        "vision_tower.layers.14.weight", tuned
    )


# ---------------------------------------------------------------------------
# find_vision_layer_param_names
# ---------------------------------------------------------------------------


def test_find_vision_layer_param_names_last_two() -> None:
    model = _build_model(total_layers=16)
    names = find_vision_layer_param_names(model, n=2)
    # Each of layers.14, layers.15 contributes 2 params (q_proj, fc1).
    assert len(names) == 4
    for name in names:
        assert "vision_tower.encoder.layers.14." in name or \
               "vision_tower.encoder.layers.15." in name


def test_find_vision_layer_param_names_excludes_original_module() -> None:
    model = _build_model(total_layers=16, include_peft_original_module=True)
    names = find_vision_layer_param_names(model, n=2)
    for name in names:
        assert ".original_module." not in name
    # The modules_to_save.default copy IS included.
    assert any(".modules_to_save.default." in n for n in names)


def test_find_vision_layer_param_names_excludes_other_layers() -> None:
    model = _build_model(total_layers=16)
    names = find_vision_layer_param_names(model, n=2)
    for name in names:
        assert "layers.0." not in name
        assert "layers.13." not in name
        assert "language_model.layers." not in name


def test_find_vision_layer_param_names_works_under_peft_wrapping() -> None:
    model = _build_model(total_layers=16, prefix="base_model.model.model.")
    names = find_vision_layer_param_names(model, n=2)
    assert len(names) == 4
    for name in names:
        assert name.startswith("base_model.model.model.")


# ---------------------------------------------------------------------------
# ensure_vision_layers_trainable
# ---------------------------------------------------------------------------


def test_ensure_vision_layers_flips_frozen_to_trainable() -> None:
    model = _build_model(total_layers=16)
    # Freeze everything first.
    for _name, param in model.named_parameters():
        param.requires_grad = False
    flipped = ensure_vision_layers_trainable(model, {14, 15})
    # 2 layers x 2 params each = 4.
    assert flipped == 4
    # Verify the right params got flipped.
    for name, param in model.named_parameters():
        if "vision_tower.encoder.layers.14." in name or \
           "vision_tower.encoder.layers.15." in name:
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def test_ensure_vision_layers_returns_zero_when_already_trainable() -> None:
    model = _build_model(total_layers=16)
    # Default state has requires_grad=True on all.
    flipped = ensure_vision_layers_trainable(model, {14, 15})
    assert flipped == 0


def test_ensure_vision_layers_skips_original_module() -> None:
    """The PEFT frozen-copy must stay frozen even if it shares the layer index."""
    model = _build_model(total_layers=16, include_peft_original_module=True)
    # Freeze everything.
    for _name, param in model.named_parameters():
        param.requires_grad = False
    ensure_vision_layers_trainable(model, {15})
    # The .original_module. copy must remain frozen.
    for name, param in model.named_parameters():
        if ".original_module." in name and "layers.15." in name:
            assert not param.requires_grad, name
        if ".modules_to_save.default." in name and "layers.15." in name:
            assert param.requires_grad, name


def test_ensure_vision_layers_works_under_peft_wrapping() -> None:
    model = _build_model(total_layers=16, prefix="base_model.model.model.")
    for _name, param in model.named_parameters():
        param.requires_grad = False
    flipped = ensure_vision_layers_trainable(model, {14, 15})
    assert flipped == 4


def test_ensure_vision_layers_empty_set_no_ops() -> None:
    model = _build_model(total_layers=16)
    for _name, param in model.named_parameters():
        param.requires_grad = False
    flipped = ensure_vision_layers_trainable(model, set())
    assert flipped == 0
    for _name, param in model.named_parameters():
        assert not param.requires_grad
