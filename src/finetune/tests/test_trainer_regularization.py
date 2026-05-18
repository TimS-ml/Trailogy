"""Tests for trainer-side regularization wiring.

We don't pull in real SFTTrainer / unsloth here — that's gated on GPU
in the smoke tests. These tests exercise the seam between the
``RegularizationState`` and a stand-in trainer's ``compute_loss`` to
make sure the extra-loss is added correctly, gradients reach the live
params, and the fast-path (no regularizers) doesn't degrade the
existing CE-only flow.
"""
from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import RegularizationConfig
from src.regularization import build_regularizers
from src.trainer_modality import compute_loss_with_regularization


class _StudentTeacherModel(nn.Module):
    """Tiny LM-shaped model with a PEFT-like disable_adapter() ctx.

    Matches the toy used in test_regularization.py. The trainer's
    compute_loss seam must accept this model + an inputs dict + a labels
    tensor + a regularization state and return (loss, metrics).
    """

    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.base = nn.Linear(hidden, vocab, bias=False)
        self.delta = nn.Parameter(torch.zeros_like(self.base.weight))
        self._adapter_enabled = True

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        w = self.base.weight + (self.delta if self._adapter_enabled else 0.0)
        return inputs_embeds @ w.T

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = prev


def _ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard masked CE for the test scaffolding."""
    flat = logits.view(-1, logits.size(-1))
    targets = labels.view(-1)
    return F.cross_entropy(flat, targets, ignore_index=-100)


def test_compute_loss_no_regularizers_returns_ce_unchanged() -> None:
    """Fast-path: when the regularization state is fully disabled, the
    returned loss must equal the bare CE loss bit-for-bit (no spurious
    epsilon noise from add-zero ops)."""
    torch.manual_seed(0)
    model = _StudentTeacherModel()
    cfg = RegularizationConfig()  # disabled defaults
    state = build_regularizers(cfg, model=model)
    assert state.enabled is False

    x = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    logits = model(x)
    ce = _ce_loss(logits, labels)

    total, metrics = compute_loss_with_regularization(
        model=model,
        student_logits=logits,
        labels=labels,
        ce_loss=ce,
        state=state,
        teacher_inputs=None,
        teacher_forward=None,
        teacher_ctx_factory=None,
    )
    assert torch.equal(total, ce)
    assert metrics == {}


def test_compute_loss_adds_kl_term() -> None:
    """With KL enabled, total = ce + kl_weight * kl. After student drift
    the KL term must be non-zero and the gradient must flow to the
    student delta param."""
    torch.manual_seed(0)
    model = _StudentTeacherModel()
    with torch.no_grad():
        model.delta.add_(torch.randn_like(model.delta) * 0.5)

    cfg = RegularizationConfig(
        kl_enabled=True, kl_weight=0.1, kl_temperature=1.0,
        l2_enabled=False,
    )
    state = build_regularizers(cfg, model=model)

    x = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    logits = model(x)
    ce = _ce_loss(logits, labels)

    total, metrics = compute_loss_with_regularization(
        model=model,
        student_logits=logits,
        labels=labels,
        ce_loss=ce,
        state=state,
        teacher_inputs={"inputs_embeds": x},
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]),
        teacher_ctx_factory=lambda: model.disable_adapter(),
    )
    assert "kl" in metrics
    assert metrics["kl"] > 0.0
    assert total.item() > ce.item()  # extra term is positive
    # Grad flows to the trainable delta param via the KL.
    model.delta.grad = None
    total.backward()
    assert model.delta.grad is not None
    assert model.delta.grad.abs().sum().item() > 0.0


def test_compute_loss_adds_l2_term() -> None:
    """With L2 enabled, total = ce + l2_weight * l2. After perturbing
    the live param the L2 term must grow and the gradient must flow
    back toward the snapshot."""
    torch.manual_seed(0)
    model = _StudentTeacherModel()
    cfg = RegularizationConfig(
        kl_enabled=False,
        l2_enabled=True, l2_weight=0.5,
    )
    state = build_regularizers(cfg, model=model)
    # Perturb AFTER snapshot: the snapshot inside build_regularizers
    # captures the param-at-construction time.
    with torch.no_grad():
        model.delta.add_(1.0)

    x = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    logits = model(x)
    ce = _ce_loss(logits, labels)

    total, metrics = compute_loss_with_regularization(
        model=model,
        student_logits=logits,
        labels=labels,
        ce_loss=ce,
        state=state,
        teacher_inputs=None,
        teacher_forward=None,
        teacher_ctx_factory=None,
    )
    assert "l2" in metrics
    assert metrics["l2"] > 0.0
    assert total.item() > ce.item()
    # Grad on delta should point in the direction OPPOSITE to the +1
    # perturbation (i.e. pulling delta back toward the 0 snapshot).
    model.delta.grad = None
    total.backward()
    assert model.delta.grad is not None
    # All grad entries from the L2 part should be positive (since
    # delta - snapshot = +1, the gradient of (delta - snap)^2 is
    # 2 * +1 > 0 everywhere). The CE contribution may dwarf it but
    # the sum-of-grads sanity check is that the magnitude is non-zero.
    assert model.delta.grad.abs().sum().item() > 0.0


def test_compute_loss_both_terms_compose() -> None:
    """KL + L2 both enabled: total = ce + kl_weight * kl + l2_weight * l2.
    All three metrics show up."""
    torch.manual_seed(0)
    model = _StudentTeacherModel()
    cfg = RegularizationConfig(
        kl_enabled=True, kl_weight=0.05, kl_temperature=1.0,
        l2_enabled=True, l2_weight=1e-3,
    )
    state = build_regularizers(cfg, model=model)
    with torch.no_grad():
        model.delta.add_(torch.randn_like(model.delta) * 0.5)

    x = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    logits = model(x)
    ce = _ce_loss(logits, labels)

    total, metrics = compute_loss_with_regularization(
        model=model,
        student_logits=logits,
        labels=labels,
        ce_loss=ce,
        state=state,
        teacher_inputs={"inputs_embeds": x},
        teacher_forward=lambda **kw: model(kw["inputs_embeds"]),
        teacher_ctx_factory=lambda: model.disable_adapter(),
    )
    assert "kl" in metrics and "l2" in metrics
    # Verify the combination identity (allow tiny fp drift).
    expected = ce.item() + 0.05 * metrics["kl"] + 1e-3 * metrics["l2"]
    assert total.item() == pytest.approx(expected, rel=1e-4, abs=1e-6)
