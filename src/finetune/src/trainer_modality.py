"""ModalityAware trainer integration for v2/v3 SFT (mix-100k / mix-50k).

Glues together:
- ``ModalityAwareBatchSampler``  (src/batch_sampler.py) — homogeneous batches
- ``ModalityAwareCollator``       (src/data.py)         — dispatch by modality
- ``SFTTrainer``                  (trl, lazy import)    — actual training loop
- ``RegularizationState``         (src/regularization.py) — v3 KL + L2 penalties

The trainer subclass is loaded only at training time (lazy import of SFTTrainer
inside ``make_modality_aware_sft_trainer_class`` so CI / dry-run paths that
don't have trl/unsloth installed can still import this module).

Why the factory function rather than a top-level class: ``SFTTrainer`` lives
inside ``trl`` which itself imports ``transformers`` heavyweights and pulls in
GPU detection at import time. Keeping the subclass construction lazy means
the rest of the test suite can import ``trainer_modality`` (e.g. for the
``build_modality_aware_dataloader`` helper) without paying that cost or
needing GPU at all.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, ContextManager, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regularization seam (v3)
#
# Kept as a free function (not a method on the trainer subclass) so it can
# be unit-tested without standing up a real SFTTrainer / trl / unsloth.
# The trainer subclass below calls into this from its compute_loss override.
# ---------------------------------------------------------------------------


def compute_loss_with_regularization(
    *,
    model: nn.Module,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_loss: torch.Tensor,
    state: Any,  # RegularizationState — duck-typed to avoid circular import
    teacher_inputs: Optional[Dict[str, Any]],
    teacher_forward: Optional[Callable[..., torch.Tensor]],
    teacher_ctx_factory: Optional[Callable[[], ContextManager[Any]]],
    global_step: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine CE loss with the v3 regularization terms.

    Returns ``(total_loss, metrics)`` where ``metrics`` carries the raw
    KL / L2 values for trainer logging.

    Fast-path: when ``state.enabled`` is False, returns ``(ce_loss, {})``
    by identity — no graph nodes added, no spurious add-zero ops that
    could perturb numerics.

    ``global_step`` is forwarded to RegularizationState.compute_extra_loss
    so the diagnostic-only KL path (kl_log_only=True with kl_weight=0)
    can skip the teacher forward on non-aligned steps to bound the
    per-step memory ceiling.
    """
    if not state.enabled:
        return ce_loss, {}

    extra, metrics = state.compute_extra_loss(
        model=model,
        student_logits=student_logits,
        inputs=teacher_inputs,
        labels=labels,
        teacher_ctx_factory=teacher_ctx_factory,
        teacher_forward=teacher_forward,
        global_step=global_step,
    )
    # Cast extra to ce_loss's device + dtype for a numerically clean add.
    # ce_loss is usually fp32 (HF Trainer promotes), extra may be fp32
    # already; the .to() is a no-op in the common case.
    extra = extra.to(device=ce_loss.device, dtype=ce_loss.dtype)
    return ce_loss + extra, metrics


def build_modality_aware_dataloader(
    dataset: Sequence,
    batch_size: int,
    has_image_fn: Callable[[Any], bool],
    collator: Callable,
    length_fn: Optional[Callable[[Any], int]] = None,
    seed: int = 42,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Construct a torch DataLoader backed by ``ModalityAwareBatchSampler``.

    Standalone factory so the dataloader construction can be unit-tested
    without standing up a full SFTTrainer. The trainer subclass below
    simply forwards to this function from its ``get_train_dataloader``.
    """
    from torch.utils.data import DataLoader  # local import: torch heavyweight

    from src.batch_sampler import ModalityAwareBatchSampler

    sampler = ModalityAwareBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        has_image_fn=has_image_fn,
        length_fn=length_fn,
        seed=seed,
        drop_last=drop_last,
    )
    log.info(
        "ModalityAwareBatchSampler: %d image / %d text records => %d batches",
        sampler.n_image_records,
        sampler.n_text_records,
        len(sampler),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_modality_aware_sft_trainer_class(
    seed: int,
    regularization_state: Optional[Any] = None,
):
    """Factory that returns a ModalityAwareSFTTrainer subclass.

    Subclass overrides:
      - ``get_train_dataloader``: swaps HF's LengthGroupedSampler /
        RandomSampler for ModalityAwareBatchSampler.
      - ``compute_loss``: adds the v3 KL + L2 regularizers on top of the
        parent CE loss when ``regularization_state.enabled`` is True.

    Eval dataloader stays default — eval batches are small and
    eval_strategy='steps' handles per-modality reporting via the
    ``eval_dataset = {key: ds}`` dict feature (no custom sampler needed).

    ``regularization_state`` is supplied at trainer-class-construction
    time (not as a runtime kwarg) so the override doesn't have to fish
    it out of ``self.args``. When None, the compute_loss override is
    bit-identical to ``SFTTrainer.compute_loss``.
    """
    from trl import SFTTrainer  # type: ignore[import-not-found]

    class ModalityAwareSFTTrainer(SFTTrainer):  # type: ignore[misc, valid-type]
        """SFTTrainer that yields modality-homogeneous training batches.

        Pairs with ``ModalityAwareCollator`` (the ``data_collator`` arg
        passed at construction time): the sampler guarantees each batch
        is all-image or all-text, and the collator dispatches to the
        right underlying collator (UnslothVisionDataCollator for image,
        DataCollatorForLanguageModeling for text-only).
        """

        def get_train_dataloader(self):
            from src.data import record_has_image

            if self.train_dataset is None:
                raise ValueError(
                    "ModalityAwareSFTTrainer requires self.train_dataset"
                )

            def _length_fn(rec):
                # Pre-populated by the group_by_length code path in
                # finetune.py main; falls back to 0 if absent (no
                # length-sort in that case).
                return rec.get("length", 0)

            return build_modality_aware_dataloader(
                dataset=self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                has_image_fn=record_has_image,
                collator=self.data_collator,
                length_fn=_length_fn,
                seed=seed,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def get_eval_dataloader(self, eval_dataset=None):
            """Modality-aware eval dataloader (v3 — required for mixed eval buckets).

            The first run of the v3 mix surfaced this: ``val_nonplant.jsonl``
            holds BOTH llava (image-having) and smoltalk (text-only) records,
            so an eval batch from that bucket naturally contains both
            modalities. Our ``ModalityAwareCollator`` is intentionally strict
            about homogeneity (the assertion catches mis-wired train batches),
            so a mixed eval batch crashes mid-evaluate with
            ``ModalityAwareCollator got a mixed-modality batch (N image-having,
            M text-only)``.

            Fix: route eval through the SAME ``ModalityAwareBatchSampler``
            used for training. The sampler groups by modality, so every
            eval batch is homogeneous and the collator dispatches cleanly.
            Eval doesn't need to shuffle for correctness — but the sampler
            does shuffle deterministically by ``seed``, which is harmless
            for eval (the loss is reduced over the whole eval set).

            ``eval_dataset`` arg can be:
              * ``None`` → use ``self.eval_dataset`` (single Dataset or dict)
              * ``str``  → use ``self.eval_dataset[key]`` (multi-eval-dataset)
              * ``Dataset`` → use as-is (HF's contract)
            """
            from src.data import record_has_image

            # Resolve to the actual Dataset object, mirroring HF Trainer's
            # own logic in get_eval_dataloader.
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError(
                    "Trainer: evaluation requires an eval_dataset."
                )
            if isinstance(eval_dataset, str):
                resolved = self.eval_dataset[eval_dataset]
            elif eval_dataset is not None:
                resolved = eval_dataset
            else:
                resolved = self.eval_dataset

            def _length_fn(rec):
                return rec.get("length", 0)

            return build_modality_aware_dataloader(
                dataset=resolved,
                batch_size=self.args.per_device_eval_batch_size,
                has_image_fn=record_has_image,
                collator=self.data_collator,
                length_fn=_length_fn,
                seed=seed,
                # Eval intentionally doesn't drop_last so every sample
                # contributes to the eval loss reduction.
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            # Fast-path: no regularizers → exactly the parent behaviour.
            # This keeps v2 configs bit-identical to the pre-v3 trainer.
            if regularization_state is None or not regularization_state.enabled:
                return super().compute_loss(
                    model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            # Force unsloth to materialize raw logits THIS step. Unsloth
            # 2024.11+ saves memory by returning an ``EmptyLogits``
            # sentinel during training; the env var below opts back in.
            # ``unsloth.models.vision::for_training`` sets it to "0"
            # whenever the model enters training mode, so we have to
            # re-set per step (it's cheap — os.environ write).
            import os as _os
            _os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

            # SAVE labels BEFORE calling super().compute_loss — trl /
            # unsloth may pop or transform 'labels' from inputs inside
            # the parent compute_loss, so the local copy is the only
            # reliable handle for the KL position mask.
            #
            # Use duck-typing rather than ``isinstance(inputs, dict)``:
            # HF / unsloth pass ``BatchEncoding`` (which inherits from
            # ``collections.UserDict``, NOT ``dict``), so an isinstance
            # check against dict returns False and we'd silently miss
            # the labels — that bug already burnt one training launch.
            try:
                labels = inputs.get("labels", None)
            except AttributeError:
                labels = None

            # Slow path: run the parent's forward + CE, then add the
            # regularizers. We need the student logits + labels, which
            # SFTTrainer.compute_loss only exposes via return_outputs=True.
            ce_loss, outputs = super().compute_loss(
                model, inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )

            # Pull the student logits out of the outputs. HF's convention:
            # outputs.logits is the [B, T, V] tensor. With unsloth before
            # the env-var bump above kicked in this may still be the
            # ``EmptyLogits`` sentinel — guard with ``torch.is_tensor``.
            student_logits = getattr(outputs, "logits", None)
            student_logits_ok = (
                student_logits is not None
                and torch.is_tensor(student_logits)
            )

            if regularization_state.kl is not None and (
                not student_logits_ok or labels is None
            ):
                # KL needs both; drop to CE-only with a one-time warning
                # so the run doesn't crash but the issue is loud.
                if not getattr(self, "_kl_skip_warned", False):
                    log.warning(
                        "KL penalty enabled but compute_loss got "
                        "student_logits=%s (tensor=%s), labels=%s — "
                        "falling back to CE-only for this step. "
                        "Investigate if it persists.",
                        type(student_logits).__name__ if student_logits is not None else None,
                        student_logits_ok,
                        type(labels).__name__ if labels is not None else None,
                    )
                    self._kl_skip_warned = True
                if return_outputs:
                    return ce_loss, outputs
                return ce_loss

            # Build a forward closure for the teacher. We feed back the
            # same ``inputs`` dict (sans 'labels' to avoid the parent
            # computing CE inside the teacher forward). PEFT-wrapped
            # models expose ``.disable_adapter()`` directly; unsloth's
            # FastModel wraps PEFT so the call chains through. If the
            # model has no disable_adapter (e.g. full FT), we'd need a
            # separate base model — that path isn't supported yet and
            # is rejected by the config validator.
            # Same duck-typing concern as for ``labels`` above —
            # BatchEncoding is not isinstance(dict).
            try:
                teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            except AttributeError:
                teacher_inputs = None

            def _teacher_forward(**kw):
                out = model(**kw)
                return out.logits if hasattr(out, "logits") else out

            teacher_ctx_factory: Optional[Callable[[], ContextManager[Any]]]
            if hasattr(model, "disable_adapter"):
                teacher_ctx_factory = model.disable_adapter
            else:
                teacher_ctx_factory = None
                if regularization_state.kl is not None:
                    if not getattr(self, "_kl_no_adapter_warned", False):
                        log.warning(
                            "KL penalty enabled but model has no "
                            ".disable_adapter() context — KL requires a "
                            "PEFT-wrapped model. Falling back to L2 only."
                        )
                        self._kl_no_adapter_warned = True

            total_loss, metrics = compute_loss_with_regularization(
                model=model,
                student_logits=student_logits,
                labels=labels,
                ce_loss=ce_loss,
                state=regularization_state,
                teacher_inputs=teacher_inputs if teacher_ctx_factory is not None else None,
                teacher_forward=_teacher_forward if teacher_ctx_factory is not None else None,
                teacher_ctx_factory=teacher_ctx_factory,
                # Forwarded so the diagnostic-only KL path can decide
                # whether the current step is a logging step. Trainer's
                # self.state.global_step is the optimizer step counter
                # (HF Trainer convention).
                global_step=getattr(getattr(self, "state", None), "global_step", None),
            )

            # Accumulate raw KL / L2 values into a rolling-window dict
            # that the trainer log callback drains every logging_steps.
            # Logging every step is too noisy; aggregation matches HF's
            # own loss-aggregation cadence.
            buf = getattr(self, "_reg_metric_buffer", None)
            if buf is None:
                buf = {"sum": {}, "count": 0}
                self._reg_metric_buffer = buf
            for key, val in metrics.items():
                buf["sum"][key] = buf["sum"].get(key, 0.0) + float(val)
            buf["count"] += 1

            if return_outputs:
                return total_loss, outputs
            return total_loss

        def log(self, logs, *args, **kwargs):  # type: ignore[override]
            """Inject rolling-window-averaged KL/L2 into the trainer log.

            HF Trainer calls ``self.log({'loss': avg_loss, ...})`` every
            ``logging_steps`` (in fact whenever it wants to emit, which
            includes the eval cadence). We piggy-back on that call to
            drain the regularization metric buffer and add ``reg_kl`` /
            ``reg_l2`` keys with the window-averaged values.

            The buffer is cleared after each drain so the next window
            starts fresh.
            """
            buf = getattr(self, "_reg_metric_buffer", None)
            if buf is not None and buf["count"] > 0:
                n = buf["count"]
                for key, total in buf["sum"].items():
                    logs[f"reg_{key}"] = total / n
                # Reset window.
                self._reg_metric_buffer = {"sum": {}, "count": 0}
            return super().log(logs, *args, **kwargs)

    return ModalityAwareSFTTrainer
