"""Regularizers for SFT: KL-penalty + L2-weight-anchor (v3).

Two regularizers, both optional, both opted in via the
``regularization:`` block in the training YAML.

1. ``KLPenalty`` (KL distillation against the base model)
   -----------------------------------------------------
   Computes ``KL(student ‖ teacher)`` on the same training batch the
   cross-entropy loss is computed on, where:
     * student = the trained model (with adapter ON)
     * teacher = the same model with adapter OFF (PEFT's
       ``model.disable_adapter()`` context manager) — i.e. the
       pretrained base model with no extra weights to load.

   This mirrors what RLHF / DPO use to keep the policy from drifting too
   far off the SFT reference, applied here at SFT time to keep us from
   drifting too far off the *pretrained* reference. The wording "general
   distribution" in early discussions just refers to the fact that the
   teacher (= base model) was trained on a general distribution; the KL
   itself is computed on whatever inputs the current batch contains.

   The label mask drives the position-level reduction: KL is summed only
   over positions where ``labels != -100``, then averaged by the count
   of those positions. An all-masked batch returns 0 (e.g. an
   accidentally over-truncated text-only batch).

   Memory: zero extra GPU bytes. The teacher forward runs with
   ``torch.no_grad()`` AND through PEFT's adapter-disabled context, so
   it reuses the live model's parameters and tensors without materializing
   a second copy.

2. ``WeightL2Anchor`` (elastic weight consolidation / L2 toward init)
   ------------------------------------------------------------------
   At trainer init, snapshot every trainable parameter. During each
   ``compute_loss`` call, add ``sum_p ||p - p_init||_F^2`` as a penalty
   so the optimizer is pulled back toward the snapshot. For LoRA delta
   params (LoRA-A/B, initialized small/zero), the snapshot value is
   approximately 0 → this is equivalent to L2 weight decay on the LoRA
   path. For ``modules_to_save`` full-rank params (projector,
   last-N vision-encoder layers), the snapshot is the pretrained value
   → meaningful anchoring against drift, which is exactly the EWC use
   case from continual learning.

   The snapshot lives in the same dtype as the live param (typically
   bf16 for our setup), so the memory cost is ≈ 1× trainable-param-count
   bytes. For Gemma 4 E2B + LoRA r=256 + projector + last-2 vision
   layers that's roughly 200-300 MB — fits inside the 4090 24 GB budget
   without disturbing batch size.

The two regularizers are composed by ``RegularizationState``; the
``build_regularizers`` factory constructs the right combination from the
``RegularizationConfig`` dataclass. The trainer code (see
``trainer_modality.py``) only needs to call ``state.compute_extra_loss``
once per ``compute_loss`` call.

Both are pure-torch — no trl/unsloth/peft imports at module-import time
(the disable_adapter context manager is supplied by the caller as a
``teacher_ctx_factory`` callable, so this file stays importable on CPU
boxes that don't have those libraries installed).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeightL2Anchor
# ---------------------------------------------------------------------------


class WeightL2Anchor:
    """L2 penalty pulling each trainable parameter toward its initial value.

    Snapshot ``p_init`` is taken at construction time as a **detached
    copy** of each registered param (NOT a view — that would track the
    live param and make the penalty trivially zero). The snapshot lives
    on the same device + dtype as the live param.

    Params with ``requires_grad=False`` at construction time are
    silently skipped — this keeps the anchor cheap on the frozen
    majority of the base model.

    The loss is the **sum** of squared Frobenius norms across all
    anchored params (no per-param normalization). The caller controls
    the global magnitude via the ``l2_weight`` coefficient in the
    config.
    """

    def __init__(self, named_params: Iterable[Tuple[str, nn.Parameter]]) -> None:
        self._snapshots: Dict[str, torch.Tensor] = {}
        self._live: Dict[str, nn.Parameter] = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            # ``detach().clone()`` decouples from autograd AND from the
            # live storage. We keep the snapshot in the live param's
            # dtype/device — for bf16 LoRA that's bf16, so the anchor
            # doesn't double the memory footprint vs an fp32 snapshot.
            self._snapshots[name] = p.detach().clone()
            self._live[name] = p

    def num_anchored_params(self) -> int:
        return len(self._snapshots)

    def compute(self) -> torch.Tensor:
        """Return the L2-toward-snapshot penalty, summed across params.

        Returns a 0-d tensor of dtype matching the first live param
        (typically bf16). When no params are registered, returns 0.0 on
        CPU; the caller is expected to ``.to(device)`` if needed (in
        practice this branch is never hit because the trainer only
        constructs the anchor when at least one param is trainable).
        """
        if not self._snapshots:
            return torch.zeros((), dtype=torch.float32)
        # Compute per-param ||p - p_init||_F^2 in float32 to avoid bf16
        # underflow when individual deltas are small; the *result* is
        # cast back to the live dtype by the caller's loss combination
        # (which is generally fp32 anyway because compute_loss returns
        # fp32-promoted).
        total = None
        for name, snap in self._snapshots.items():
            live = self._live[name]
            diff = (live - snap).to(torch.float32)
            term = diff.pow(2).sum()
            total = term if total is None else total + term
        return total  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# KLPenalty
# ---------------------------------------------------------------------------


class KLPenalty:
    """KL(student ‖ teacher) on the supervised positions of the current batch.

    The teacher forward is dispatched via two caller-supplied callables
    so this module doesn't depend on PEFT / transformers / trl:

      * ``teacher_ctx_factory()`` — a no-arg callable that returns a
        context manager. Typically ``lambda: model.disable_adapter()``
        in production. While inside the context the trained model
        behaves like its base. The context is entered AND exited around
        a single teacher forward call.
      * ``teacher_forward(**kwargs)`` — invoked inside the context (and
        inside ``torch.no_grad()``); must return the teacher's logits
        tensor of shape ``[B, T, V]``.

    The KL is computed in float32 (numerically robust) and reduced as:

        kl = sum_over_supervised_positions KL(softmax(s/T) ‖ softmax(t/T))
             / max(1, num_supervised_positions)

    Temperature ``T`` softens both distributions identically — at higher
    T the student-teacher distance shrinks, useful when the base model
    is very confident and the student needs room to specialize.

    Memory: the masked ``[N, V]`` fp32 buffers are still sizeable for
    Gemma 4 (V≈262 K → one fp32 row ≈ 1 MB), but ``N`` is the count of
    supervised tokens after the ``labels != -100`` mask — typically a
    few hundred per micro-batch in SFT, not ``B*T``. At a realistic
    N≈600 the five concurrent fp32 buffers (``s_fp32``, ``t_fp32``,
    ``s_log``, ``t_log``, ``elementwise``) sum to ~3 GB peak, which
    fits comfortably on a 24 GB GPU alongside the bf16 weights and
    optimizer state.

    By default (``chunk_size=None``) the fp32 math runs in one shot
    over all N positions — fastest, but peak memory scales linearly
    with N. When ``chunk_size`` is set to a positive int, the fp32
    log_softmax / kl_div is split into chunks of that many rows,
    bounding peak to ~5 × chunk_size × V × 4 bytes at the cost of
    extra kernel launches. The KL value is exact — chunking just
    defers the sum.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"KL temperature must be positive (got {temperature})")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"KL chunk_size must be positive (got {chunk_size})")
        self.temperature = float(temperature)
        self.chunk_size: Optional[int] = int(chunk_size) if chunk_size is not None else None

    def compute(
        self,
        *,
        model: nn.Module,
        student_logits: torch.Tensor,
        inputs: Dict[str, Any],
        labels: torch.Tensor,
        teacher_ctx_factory: Callable[[], ContextManager[Any]],
        teacher_forward: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Return a 0-d float32 KL tensor.

        ``student_logits`` is provided by the caller (already computed
        as part of the cross-entropy forward — no double-forward of the
        student needed). ``inputs`` is the kwargs dict that will be
        passed verbatim to ``teacher_forward`` inside the context.

        Memory note: the naive implementation materializes a full
        [B, T, V] fp32 tensor TWICE (student + teacher log_softmax).
        For Gemma 4 (vocab ~262 K), at B=4, T=1024 that's 8 GB total —
        enough to OOM a 24 GB 4090 mid-training. To avoid this we MASK
        FIRST (drop the unsupervised positions, ``[N, V]`` where
        N << B*T) before any fp32 conversion, and we ``del`` the
        teacher's full [B, T, V] tensor immediately after the slice so
        the big bf16 allocation is reclaimable before the fp32 math
        materializes its own buffers.
        """
        # Build the position mask once; positions with label == -100 are
        # excluded from the KL average (these are typically system /
        # user prompt tokens that aren't supervised by CE either).
        # ``mask`` shape matches ``labels``: [B, T] (full unshifted) or
        # [B, T-1] (HF-shifted). student_logits matches whichever shape
        # the caller passes — we just need them to align on (B, T*).
        mask = (labels != -100)
        n_positions = int(mask.sum().item())
        if n_positions == 0:
            # Nothing to penalize. Return 0 on the same device/dtype
            # as student_logits so the loss combination doesn't have to
            # branch on this.
            return student_logits.new_zeros((), dtype=torch.float32)

        # Slice the student to the supervised positions ONLY before any
        # fp32 conversion. Result: [N, V] (e.g. [N=300, V=262K] instead
        # of [B=4, T=1024, V=262K]) — ~13x smaller per fp32 buffer for
        # a typical SFT label pattern.
        masked_student = student_logits[mask]  # [N, V] bf16

        # Teacher forward: no grad + adapter-off via the supplied context
        # manager. The teacher distribution is detached even if the user
        # forgets to use no_grad, because we wrap explicitly.
        with torch.no_grad():
            with teacher_ctx_factory():
                teacher_logits = teacher_forward(**inputs)
            masked_teacher = teacher_logits[mask]  # [N, V] bf16
            # Drop the full [B, T, V] tensor IMMEDIATELY so the
            # subsequent fp32 conversions don't double-up with it. This
            # is the difference between fitting bs=4 with KL on a 4090
            # and OOMing during the SDPA attention call.
            del teacher_logits

        # Compute in float32 to dodge bf16 softmax underflow on
        # high-vocab models (Gemma 4 vocab is ~262 K).
        #
        # F.kl_div signature reminder: ``F.kl_div(input=log_q,
        # target=log_p, log_target=True, reduction='none')`` returns
        # the per-element ``p * (log_p - log_q)`` summed over the last
        # (vocab) dim, i.e. ``KL(p ‖ q)``. We want
        # ``KL(student ‖ teacher)``, so input=log_teacher,
        # target=log_student.
        t = self.temperature

        if self.chunk_size is None:
            # Unchunked path — fastest, but peak fp32 buffer is
            # ~5 × N × V × 4 bytes. At N≈600 / V≈262 K that's ~3 GB,
            # which fits on a 24 GB GPU.
            s_fp32 = masked_student.to(torch.float32) / t
            t_fp32 = masked_teacher.to(torch.float32) / t
            s_log = F.log_softmax(s_fp32, dim=-1)
            t_log = F.log_softmax(t_fp32, dim=-1)
            elementwise = F.kl_div(
                t_log,
                s_log,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)  # [N]
            kl_sum = elementwise.sum()
        else:
            # Chunked path — bounds peak fp32 to
            # ~5 × chunk_size × V × 4 bytes at the cost of extra
            # kernel launches. The KL value is exact (we just defer
            # the sum across chunks).
            n = masked_student.shape[0]
            kl_sum = torch.zeros((), dtype=torch.float32, device=masked_student.device)
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                s_chunk = masked_student[start:end].to(torch.float32) / t
                t_chunk = masked_teacher[start:end].to(torch.float32) / t
                s_log = F.log_softmax(s_chunk, dim=-1)
                t_log = F.log_softmax(t_chunk, dim=-1)
                elementwise = F.kl_div(
                    t_log,
                    s_log,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)  # [chunk]
                kl_sum = kl_sum + elementwise.sum()
                del s_chunk, t_chunk, s_log, t_log, elementwise

        # T^2 scaling is the standard distillation correction (Hinton et al.
        # 2015); it keeps the KL gradient magnitude approximately invariant
        # to temperature so the kl_weight coefficient is interpretable
        # across T values.
        kl = kl_sum * (t * t) / max(1, n_positions)
        return kl


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass
class RegularizationState:
    """Bundle of constructed regularizers + their coefficients.

    The trainer calls ``compute_extra_loss(...)`` once per step. When
    ``enabled`` is False, the trainer can fast-path: don't even materialize
    an extra-loss tensor.

    Diagnostic-only KL: when ``kl is not None`` but ``kl_weight == 0`` and
    ``kl_log_only`` is True, the KL teacher forward + math still runs
    (every ``kl_log_every_n_steps`` optimizer steps) and the value is
    surfaced via ``metrics["kl"]``, but the loss is unaffected. This is
    the path for "always record train/reg_kl on memory-tight boxes"
    without paying the per-step teacher-forward memory cost.
    """

    kl: Optional[KLPenalty]
    kl_weight: float
    l2: Optional[WeightL2Anchor]
    l2_weight: float
    # Diagnostic-only logging fields (default == no-op).
    kl_log_only: bool = False
    kl_log_every_n_steps: int = 1

    @property
    def enabled(self) -> bool:
        return self.kl is not None or self.l2 is not None

    def _should_run_kl_this_step(self, global_step: Optional[int]) -> bool:
        """Decide whether to actually run the teacher forward this step.

        When KL contributes to loss (kl_weight > 0): always run.
        When kl_log_only and kl_weight == 0: run only every Nth step.
        global_step=None means "caller didn't pass it" — be conservative
        and run (matches the pre-change behavior).
        """
        if self.kl is None:
            return False
        if self.kl_weight > 0:
            return True
        if not self.kl_log_only:
            return False
        if global_step is None:
            return True
        if self.kl_log_every_n_steps <= 1:
            return True
        # Trigger at step 0 (first observation) and every Nth thereafter.
        return (global_step % self.kl_log_every_n_steps) == 0

    def compute_extra_loss(
        self,
        *,
        model: nn.Module,
        student_logits: Optional[torch.Tensor],
        inputs: Optional[Dict[str, Any]],
        labels: Optional[torch.Tensor],
        teacher_ctx_factory: Optional[Callable[[], ContextManager[Any]]],
        teacher_forward: Optional[Callable[..., torch.Tensor]],
        global_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined ``kl_weight * KL + l2_weight * L2``.

        Returns ``(extra_loss, metrics_dict)`` where ``metrics_dict``
        contains the raw (un-weighted) values for logging.

        Any KL inputs may be None when KL is disabled. The L2 anchor needs
        no inputs (it reads its registered live params + snapshots).

        ``global_step`` is consulted only when kl_log_only=True and
        kl_weight==0 — see _should_run_kl_this_step. Caller may pass
        ``self.state.global_step`` from the HF Trainer subclass.
        """
        metrics: Dict[str, float] = {}
        total: Optional[torch.Tensor] = None

        run_kl = self._should_run_kl_this_step(global_step)
        if run_kl:
            assert self.kl is not None  # _should_run_kl_this_step guarantees
            assert student_logits is not None, "KL run but student_logits is None"
            assert inputs is not None, "KL run but inputs is None"
            assert labels is not None, "KL run but labels is None"
            assert teacher_ctx_factory is not None, "KL run but teacher_ctx_factory is None"
            assert teacher_forward is not None, "KL run but teacher_forward is None"
            kl_val = self.kl.compute(
                model=model,
                student_logits=student_logits,
                inputs=inputs,
                labels=labels,
                teacher_ctx_factory=teacher_ctx_factory,
                teacher_forward=teacher_forward,
            )
            metrics["kl"] = float(kl_val.detach().item())
            # Only contribute to loss when kl_weight > 0. The diagnostic-
            # only path (kl_weight == 0) skips the multiply-by-zero
            # branch entirely — avoids dragging a dead graph node into
            # the optimizer's backward pass.
            if self.kl_weight > 0:
                term = self.kl_weight * kl_val
                total = term if total is None else total + term

        if self.l2 is not None:
            l2_val = self.l2.compute()
            metrics["l2"] = float(l2_val.detach().item())
            term = self.l2_weight * l2_val
            # If KL also enabled, harmonize device/dtype to match.
            if total is not None:
                term = term.to(total.device).to(total.dtype)
            total = term if total is None else total + term

        if total is None:
            # No regularizers contributing to loss this step — degenerate
            # to a 0 tensor so the caller's add doesn't crash. This is
            # the common path on the diagnostic-only off-steps.
            total = torch.zeros((), dtype=torch.float32)
        return total, metrics


def build_regularizers(cfg: Any, model: nn.Module) -> RegularizationState:
    """Construct ``RegularizationState`` from a ``RegularizationConfig``.

    ``cfg`` is duck-typed to avoid a hard import of ``src.config`` from
    here — that lets the test file mock the config with a tiny stand-in
    when needed. In production callers pass the real
    ``FinetuneConfig.regularization`` instance.

    Diagnostic-only KL: when ``kl_enabled=False`` but ``kl_log_only=True``,
    a KLPenalty is still constructed and kl_weight is forced to 0.0 so
    the value is logged but doesn't contribute to the optimizer's loss.
    """
    kl_log_only = bool(getattr(cfg, "kl_log_only", False))
    kl_enabled = bool(getattr(cfg, "kl_enabled", False))

    kl: Optional[KLPenalty] = None
    if kl_enabled or kl_log_only:
        kl = KLPenalty(
            temperature=getattr(cfg, "kl_temperature", 1.0),
            chunk_size=getattr(cfg, "kl_chunk_size", None),
        )

    # When log-only path is active without the full KL path, force the
    # weight to 0 regardless of what the yaml says — keeps the loss
    # bit-identical to the kl-disabled baseline.
    effective_kl_weight = (
        float(getattr(cfg, "kl_weight", 0.0)) if kl_enabled else 0.0
    )

    l2: Optional[WeightL2Anchor] = None
    if getattr(cfg, "l2_enabled", False):
        l2 = WeightL2Anchor(named_params=list(model.named_parameters()))
        log.info(
            "WeightL2Anchor: snapshotted %d trainable param tensors as init reference.",
            l2.num_anchored_params(),
        )

    if kl_log_only and not kl_enabled:
        log.info(
            "KL diagnostic-only mode: teacher forward will run every "
            "%d optimizer step(s); KL value logged as train/reg_kl but "
            "kl_weight forced to 0 (no loss contribution).",
            int(getattr(cfg, "kl_log_every_n_steps", 1)),
        )

    return RegularizationState(
        kl=kl,
        kl_weight=effective_kl_weight,
        l2=l2,
        l2_weight=float(getattr(cfg, "l2_weight", 0.0)),
        kl_log_only=kl_log_only,
        kl_log_every_n_steps=int(getattr(cfg, "kl_log_every_n_steps", 1)),
    )
