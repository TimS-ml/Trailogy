"""Tests for the pre/post-train trainable-param consistency check.

Defense-in-depth for resume-from-checkpoint: if PEFT silently drops
modules_to_save on resume, projector / vision-layer params become
frozen mid-run. The save-side tripwires catch this at save time;
this snapshot check catches it the moment trainer.train() returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import pytest

from src.finetune import (
    _assert_trainable_set_unchanged_post_train,
    _snapshot_trainable_params,
)


@dataclass
class FakeParam:
    name: str
    requires_grad: bool = True


@dataclass
class FakeModel:
    params: List[Tuple[str, FakeParam]] = field(default_factory=list)

    def named_parameters(self) -> Iterator[Tuple[str, FakeParam]]:
        for name, p in self.params:
            yield name, p


def _make_three_group_model() -> FakeModel:
    """LoRA + projector + vision-layer trainable param mix under PEFT wrap."""
    prefix = "base_model.model.model."
    return FakeModel(params=[
        # Tuned vision layers
        (f"{prefix}vision_tower.encoder.layers.14"
         ".modules_to_save.default.attn.q_proj.weight", FakeParam("v14_q")),
        (f"{prefix}vision_tower.encoder.layers.14"
         ".modules_to_save.default.mlp.fc1.weight", FakeParam("v14_fc")),
        (f"{prefix}vision_tower.encoder.layers.15"
         ".modules_to_save.default.attn.q_proj.weight", FakeParam("v15_q")),
        # PEFT frozen reference copies (must not count as trainable)
        (f"{prefix}vision_tower.encoder.layers.14"
         ".original_module.attn.q_proj.weight",
         FakeParam("v14_orig", requires_grad=False)),
        # Projector
        (f"{prefix}embed_vision.modules_to_save.default."
         "embedding_projection.weight", FakeParam("ev_proj")),
        # LoRA
        (f"{prefix}language_model.layers.0.q_proj.lora_A.default.weight",
         FakeParam("lA")),
        (f"{prefix}language_model.layers.0.q_proj.lora_B.default.weight",
         FakeParam("lB")),
        # Frozen base linear weight (not trainable; not counted)
        (f"{prefix}language_model.layers.0.q_proj.weight",
         FakeParam("base_q", requires_grad=False)),
    ])


# ---------------------------------------------------------------------------
# _snapshot_trainable_params
# ---------------------------------------------------------------------------


def test_snapshot_groups_three_classes_correctly() -> None:
    model = _make_three_group_model()
    snap = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    assert len(snap["vision"]) == 3
    assert len(snap["projector"]) == 1
    assert len(snap["lora_other"]) == 2
    # Frozen params not in any group
    all_names = snap["vision"] | snap["projector"] | snap["lora_other"]
    assert all(".original_module." not in n for n in all_names)


def test_snapshot_with_no_vision_layers_collapses_to_two_groups() -> None:
    """tune_last_n_vision_layers=0 → vision group is empty; projector
    and lora_other still populate normally."""
    model = _make_three_group_model()
    snap = _snapshot_trainable_params(model, tuned_vision_layer_indices=[])
    assert snap["vision"] == set()
    assert len(snap["projector"]) == 1
    # Vision-layer params now classify as lora_other since the matcher
    # is gated on tuned_idx_set.
    assert len(snap["lora_other"]) == 5


def test_snapshot_ignores_frozen_params() -> None:
    model = FakeModel(params=[
        ("vision_tower.encoder.layers.15.modules_to_save.default.attn.q_proj.weight",
         FakeParam("v15", requires_grad=False)),
        ("embed_vision.embedding_projection.weight",
         FakeParam("ev", requires_grad=False)),
    ])
    snap = _snapshot_trainable_params(model, tuned_vision_layer_indices=[15])
    assert snap["vision"] == set()
    assert snap["projector"] == set()
    assert snap["lora_other"] == set()


# ---------------------------------------------------------------------------
# _assert_trainable_set_unchanged_post_train
# ---------------------------------------------------------------------------


def test_post_train_check_passes_when_sets_identical() -> None:
    model = _make_three_group_model()
    snap = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    # No raise expected.
    _assert_trainable_set_unchanged_post_train(snap, snap)


def test_post_train_check_fires_when_vision_layer_silently_frozen() -> None:
    """Simulate the resume-from-checkpoint regression: vision-layer
    params become requires_grad=False after training starts."""
    model = _make_three_group_model()
    pre = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    # Simulate: PEFT regression freezes one vision layer.
    by_name = dict(model.params)
    by_name[
        "base_model.model.model.vision_tower.encoder.layers.14"
        ".modules_to_save.default.attn.q_proj.weight"
    ].requires_grad = False
    post = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    with pytest.raises(RuntimeError, match="vision"):
        _assert_trainable_set_unchanged_post_train(pre, post)


def test_post_train_check_fires_when_projector_silently_frozen() -> None:
    model = _make_three_group_model()
    pre = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    by_name = dict(model.params)
    by_name[
        "base_model.model.model.embed_vision.modules_to_save.default."
        "embedding_projection.weight"
    ].requires_grad = False
    post = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    with pytest.raises(RuntimeError, match="projector"):
        _assert_trainable_set_unchanged_post_train(pre, post)


def test_post_train_check_fires_when_lora_silently_frozen() -> None:
    model = _make_three_group_model()
    pre = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    by_name = dict(model.params)
    by_name[
        "base_model.model.model.language_model.layers.0.q_proj.lora_A.default.weight"
    ].requires_grad = False
    post = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    with pytest.raises(RuntimeError, match="lora_other"):
        _assert_trainable_set_unchanged_post_train(pre, post)


def test_post_train_check_fires_when_new_param_becomes_trainable() -> None:
    """If a frozen-by-design param suddenly becomes trainable mid-run,
    that's also a regression."""
    model = _make_three_group_model()
    pre = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    by_name = dict(model.params)
    by_name["base_model.model.model.language_model.layers.0.q_proj.weight"].requires_grad = True
    post = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    with pytest.raises(RuntimeError, match="lora_other"):
        _assert_trainable_set_unchanged_post_train(pre, post)


def test_post_train_check_message_lists_lost_params() -> None:
    """Error message must include enough detail to diagnose which params
    silently flipped."""
    model = _make_three_group_model()
    pre = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    by_name = dict(model.params)
    by_name[
        "base_model.model.model.vision_tower.encoder.layers.15"
        ".modules_to_save.default.attn.q_proj.weight"
    ].requires_grad = False
    post = _snapshot_trainable_params(model, tuned_vision_layer_indices=[14, 15])
    with pytest.raises(RuntimeError) as ei:
        _assert_trainable_set_unchanged_post_train(pre, post)
    msg = str(ei.value)
    assert "vision" in msg
    assert "Lost" in msg
    assert "layers.15" in msg
