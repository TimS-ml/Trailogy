"""Tests for KL-penalty + L2-weight-anchor regularizers (v3).

Two regularizers, one module:

  1. ``KLPenalty`` — KL(student logits ‖ teacher logits) computed on the
     SAME training batch, where teacher = base model (PEFT
     ``disable_adapter()`` context). Anti-drift; mirrors RLHF.
  2. ``WeightL2Anchor`` — L2 penalty toward each trainable parameter's
     value at trainer init (= pretrained / zero for LoRA). Anchors all
     trainable params, not just modules_to_save.

The tests here are intentionally numerics-light (we don't need a full GPU
model to verify the math). Pure-torch fixtures + a fake "PEFT-like"
context-manager double let us exercise:

  - Loss sign + scale (must be ≥ 0; must shrink as student → teacher).
  - Label masking (KL is computed only on supervised target tokens).
  - Snapshot detach (the L2 anchor must NOT be a view onto the live
    param; otherwise the penalty is always zero).
  - Disable-adapter context dispatch (KL teacher forward must run inside
    the supplied context manager, no manual swap).
  - Aggregation across param groups (L2 sums correctly).
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level imports — these live in src/regularization.py (not yet
# implemented; this file drives the implementation).
# ---------------------------------------------------------------------------
from src.regularization import (
    KLPenalty,
    WeightL2Anchor,
    RegularizationState,
    build_regularizers,
)


# =========================================================================
# WeightL2Anchor
# =========================================================================


def test_weight_l2_anchor_zero_at_init() -> None:
    """At init, param == snapshot, so L2 penalty is exactly zero."""
    m = nn.Linear(8, 4)
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    loss = anchor.compute()
    assert torch.is_tensor(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_weight_l2_anchor_grows_with_drift() -> None:
    """After mutating the live param, the anchor penalty is the squared
    Frobenius norm of (param - snapshot), summed over params."""
    m = nn.Linear(4, 4, bias=False)
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    with torch.no_grad():
        m.weight.add_(1.0)  # uniform +1 perturbation
    expected = (m.weight - (m.weight - 1.0)).pow(2).sum().item()
    loss = anchor.compute()
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_weight_l2_anchor_snapshot_is_detached_copy() -> None:
    """Regression guard: snapshot must be a detached *copy*, not a view
    onto the live param. If it were a view, every L2 penalty would be
    zero because the buffer would track the param.
    """
    m = nn.Linear(4, 4)
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    with torch.no_grad():
        m.weight.fill_(99.0)
    loss = anchor.compute()
    # If the snapshot were a view, loss would be exactly 0.
    assert loss.item() > 0.0


def test_weight_l2_anchor_sums_across_params() -> None:
    """Multiple trainable tensors should contribute additively."""
    m = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    with torch.no_grad():
        for _, p in m.named_parameters():
            p.add_(2.0)
    # Each param has 4 elements, each perturbed by 2 → contributes 4*4=16.
    # Two params → 32.
    loss = anchor.compute()
    assert loss.item() == pytest.approx(32.0, rel=1e-6)


def test_weight_l2_anchor_only_tracks_listed_params() -> None:
    """Anchor must NOT snapshot params the caller didn't pass in
    (otherwise it would penalize drift on frozen base-model weights at
    every step — huge memory + wrong semantics).
    """
    base = nn.Linear(4, 4)
    extra = nn.Linear(4, 4)
    # Only register `base` with the anchor.
    anchor = WeightL2Anchor(named_params=list(base.named_parameters()))
    # Perturb extra; anchor must report 0 loss because extra wasn't snapshotted.
    with torch.no_grad():
        extra.weight.add_(5.0)
    assert anchor.compute().item() == pytest.approx(0.0, abs=1e-12)


def test_weight_l2_anchor_grad_flows_to_param() -> None:
    """The L2 penalty must produce grads on the live param so the
    optimizer can pull it back toward the snapshot.
    """
    m = nn.Linear(4, 4, bias=False)
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    with torch.no_grad():
        m.weight.add_(0.5)
    loss = anchor.compute()
    loss.backward()
    assert m.weight.grad is not None
    # Grad should be 2*(weight - snapshot) = 2 * 0.5 = 1.0 everywhere.
    assert torch.allclose(m.weight.grad, torch.full_like(m.weight, 1.0))


def test_weight_l2_anchor_skips_non_requires_grad() -> None:
    """If a param has requires_grad=False at snapshot time, the anchor
    should silently skip it. This keeps the anchor cheap on the frozen
    majority of the base model.
    """
    m = nn.Linear(4, 4, bias=False)
    m.weight.requires_grad_(False)
    anchor = WeightL2Anchor(named_params=list(m.named_parameters()))
    # Should snapshot nothing.
    assert anchor.num_anchored_params() == 0
    assert anchor.compute().item() == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# KLPenalty
# =========================================================================


class _ToyModel(nn.Module):
    """Tiny LM-shaped student with a 'disable_adapter' context manager.

    Mimics PEFT-wrapped behaviour: ``forward`` uses the in-place
    ``delta`` term; ``disable_adapter()`` temporarily zeros it so the
    forward returns the *base* logits. This lets us drive
    ``KLPenalty.compute`` without a real PEFT model.
    """

    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, vocab, bias=False)
        self.delta = nn.Parameter(torch.zeros_like(self.base.weight))
        self._adapter_enabled = True

    def forward(self, x: torch.Tensor) -> Any:
        w = self.base.weight + (self.delta if self._adapter_enabled else 0.0)
        logits = x @ w.T
        return type("Out", (), {"logits": logits})()

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = prev


def test_kl_penalty_zero_when_student_equals_teacher() -> None:
    """At init, student adapter is 0 → student logits == teacher logits
    → KL exactly 0."""
    torch.manual_seed(0)
    model = _ToyModel()
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(2, 3, 4)  # [B, T, H]
    labels = torch.zeros(2, 3, dtype=torch.long)
    student_out = model(x)
    loss = kl.compute(
        model=model,
        student_logits=student_out.logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_penalty_positive_when_student_diverges() -> None:
    """After perturbing the student adapter, the teacher (adapter-off)
    produces different logits → KL must be > 0.

    Note: the perturbation must be **non-uniform** across the vocab
    dimension. A uniform additive shift to all vocab dims is invariant
    under softmax (a property the implementation relies on for
    numerical stability), so a uniform 0.5 delta would give KL=0.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    with torch.no_grad():
        # Add a *non-uniform* perturbation so the student softmax
        # actually shifts relative to the teacher's.
        model.delta.add_(torch.randn_like(model.delta) * 0.5)
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    student_out = model(x)
    loss = kl.compute(
        model=model,
        student_logits=student_out.logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    assert loss.item() > 1e-3


def test_kl_penalty_respects_label_mask() -> None:
    """KL is computed only on positions where labels != -100.

    Set all but one position to -100 and verify the value matches a
    manual one-position KL computation.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    with torch.no_grad():
        model.delta.add_(torch.randn_like(model.delta) * 0.3)
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(1, 4, 4)
    labels = torch.tensor([[-100, -100, 5, -100]])
    student_logits = model(x).logits
    loss = kl.compute(
        model=model,
        student_logits=student_logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    # Manual: only position [0, 2] contributes. KL(student ‖ teacher) is
    # the production convention (matches RLHF/DPO: penalize policy
    # divergence from reference). torch.nn.functional.kl_div has signature
    # kl_div(input=log_q, target=log_p) → KL(p ‖ q), so to compute
    # KL(student ‖ teacher) we pass input=log_teacher, target=log_student.
    with torch.no_grad():
        with model.disable_adapter():
            teacher_logits = model(x).logits
    s = torch.log_softmax(student_logits[0, 2].to(torch.float32), dim=-1)
    t = torch.log_softmax(teacher_logits[0, 2].to(torch.float32), dim=-1)
    expected = torch.nn.functional.kl_div(t, s, reduction="sum", log_target=True).item()
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_kl_penalty_teacher_forward_runs_in_no_grad() -> None:
    """The teacher forward must run under ``torch.no_grad()`` so the
    teacher graph doesn't bloat memory or accidentally update params.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(1, 2, 4)
    labels = torch.zeros(1, 2, dtype=torch.long)
    student_logits = model(x).logits

    grad_status: dict = {"was_enabled": None}

    def teacher_forward(**kw) -> torch.Tensor:
        grad_status["was_enabled"] = torch.is_grad_enabled()
        return model(kw["inputs_embeds"]).logits

    kl.compute(
        model=model,
        student_logits=student_logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=teacher_forward,
    )
    assert grad_status["was_enabled"] is False


def test_kl_penalty_uses_supplied_context_manager() -> None:
    """KL must invoke the context-manager factory exactly once per
    teacher forward (disable_adapter ON during forward, OFF after).
    """
    model = _ToyModel()
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(1, 2, 4)
    labels = torch.zeros(1, 2, dtype=torch.long)
    student_logits = model(x).logits

    counter = {"enter": 0, "exit": 0}

    @contextlib.contextmanager
    def factory():
        counter["enter"] += 1
        with model.disable_adapter():
            yield
        counter["exit"] += 1

    kl.compute(
        model=model,
        student_logits=student_logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=factory,
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    assert counter == {"enter": 1, "exit": 1}


def test_kl_penalty_temperature_t_squared_scaling_is_applied() -> None:
    """We apply Hinton-style T^2 scaling so the gradient magnitude is
    approximately temperature-invariant. That means at higher T the
    *raw* KL on softer distributions is smaller, but multiplied by T^2
    the reported penalty grows. Verify the direction so a future refactor
    that drops T^2 fails loudly here.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    with torch.no_grad():
        model.delta.add_(torch.randn_like(model.delta) * 2.0)  # non-uniform drift
    x = torch.randn(1, 4, 4)
    labels = torch.zeros(1, 4, dtype=torch.long)
    student_logits = model(x).logits

    common = dict(
        model=model,
        student_logits=student_logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    low_t = KLPenalty(temperature=1.0).compute(**common).item()
    high_t = KLPenalty(temperature=4.0).compute(**common).item()
    # With T^2 scaling, the *unscaled* KL at T=4 is ~16x smaller than at
    # T=1, but after the T^2 multiplier the high-T value is generally
    # *larger* than low-T (when the underlying distributions are not
    # totally degenerate). We assert this direction so the T^2 line in
    # the implementation can't be silently dropped.
    assert high_t > low_t


def test_kl_penalty_no_positions_returns_zero() -> None:
    """All-masked batch → zero KL (no contribution; must not divide by 0)."""
    model = _ToyModel()
    kl = KLPenalty(temperature=1.0)
    x = torch.randn(1, 3, 4)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    student_logits = model(x).logits
    loss = kl.compute(
        model=model,
        student_logits=student_logits,
        inputs={"inputs_embeds": x},
        labels=labels,
        teacher_ctx_factory=lambda: model.disable_adapter(),
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]).logits,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


# =========================================================================
# build_regularizers — factory wiring
# =========================================================================


def test_build_regularizers_disabled_returns_none_state() -> None:
    """When the config is fully disabled, build returns a state with no
    KL and no L2 — compute_loss can fast-path to identity.
    """
    from src.config import RegularizationConfig

    cfg = RegularizationConfig()  # disabled defaults
    state = build_regularizers(cfg, model=nn.Linear(2, 2))
    assert state.kl is None
    assert state.l2 is None
    assert state.enabled is False


def test_build_regularizers_enabled_constructs_both() -> None:
    from src.config import RegularizationConfig

    cfg = RegularizationConfig(
        kl_enabled=True,
        kl_weight=0.05,
        kl_temperature=1.0,
        l2_enabled=True,
        l2_weight=1.0e-4,
    )
    model = nn.Linear(4, 4)
    state = build_regularizers(cfg, model=model)
    assert isinstance(state.kl, KLPenalty)
    assert isinstance(state.l2, WeightL2Anchor)
    assert state.enabled is True


def test_build_regularizers_l2_only() -> None:
    """KL disabled, L2 enabled → state has L2 but no KL."""
    from src.config import RegularizationConfig

    cfg = RegularizationConfig(kl_enabled=False, l2_enabled=True, l2_weight=1.0e-4)
    state = build_regularizers(cfg, model=nn.Linear(4, 4))
    assert state.kl is None
    assert state.l2 is not None
    assert state.enabled is True


def test_build_regularizers_l2_anchors_only_trainable() -> None:
    """The L2 anchor must skip frozen params (the vast majority of a
    LoRA setup). Mix one trainable + one frozen tensor; only the
    trainable one shows up in the snapshot count.
    """
    from src.config import RegularizationConfig

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    # Freeze the second layer entirely.
    for p in model[1].parameters():
        p.requires_grad_(False)

    cfg = RegularizationConfig(l2_enabled=True, l2_weight=1.0e-4)
    state = build_regularizers(cfg, model=model)
    # 2 trainable tensors in layer 0 (weight + bias); 0 in layer 1.
    assert state.l2.num_anchored_params() == 2
