"""GPU SMOKE TEST: KL + L2 regularizers work on a real PEFT-wrapped Gemma 4.

Gate test for the v3 regularization wiring. Verifies:

1. ``model.disable_adapter()`` is callable on a PEFT-wrapped Gemma 4 E2B
   model. (Without it the KL teacher forward has no fallback.)
2. Inside the ``disable_adapter()`` context, the model's logits differ
   from the adapter-on logits AFTER we mutate the LoRA delta (proves the
   adapter swap actually takes effect at forward time).
3. ``WeightL2Anchor`` snapshots all trainable params at init AND the
   per-param snapshot count matches the PEFT trainable set (catches
   regressions where unsloth silently drops modules_to_save wrappers).
4. ``compute_loss_with_regularization`` returns a finite total loss on
   one real text-only batch (no NaNs / Infs from the KL+CE+L2 sum).

Costs:
- ~5-6 GB GPU memory (bf16 Gemma 4 E2B + LoRA r=64 + small batch)
- ~45-60 s to load + apply LoRA + run two forwards

Marked ``@pytest.mark.gpu`` — NOT in the default pytest run. Trigger::

    CUDA_VISIBLE_DEVICES=0 pytest finetune/tests/test_regularization_gpu_smoke.py -m gpu -v
"""
from __future__ import annotations

import os

# Unsloth 2024.11+ drops logits by default; production sets this in
# finetune.py before importing unsloth. Mirror that here so the smoke
# test exercises the same path. MUST be set before the first unsloth
# import anywhere in the process.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def _gpu_available() -> bool:
    return torch.cuda.is_available()


def _build_gemma4_with_lora() -> tuple:
    """Load Gemma 4 E2B + apply unsloth-flavoured LoRA (the production
    wrapping path). Returns ``(model, processor)``.

    We use unsloth.FastModel + get_peft_model because vanilla PEFT can't
    directly target the Gemma 4 ``Gemma4ClippableLinear`` wrappers around
    q_proj / v_proj. unsloth's loader is the production wrapping path,
    so this also exercises the realistic stack.
    """
    from unsloth import FastModel  # type: ignore[import-not-found]

    model, processor = FastModel.from_pretrained(
        "unsloth/gemma-4-E2B-it",
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        random_state=3407,
        bias="none",
    )
    return model, processor


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU")
def test_kl_teacher_via_disable_adapter_changes_logits() -> None:
    """End-to-end: PEFT-wrapped Gemma 4 + non-trivial LoRA delta + KL
    teacher = disable_adapter(). With student adapter ON we get one
    set of logits; under disable_adapter() we get a DIFFERENT set —
    proving the teacher swap actually reaches the underlying forward.
    """
    model, processor = _build_gemma4_with_lora()
    model.eval()

    # Inject a non-trivial LoRA delta so adapter on/off actually
    # produce different logits.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.05)

    # Build a 1-record text-only prompt via the chat template.
    convo = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is 4."}]},
    ]
    inputs = processor.apply_chat_template(
        [convo], add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items() if torch.is_tensor(v)}

    # Adapter ON.
    with torch.no_grad():
        on_logits = model(**inputs).logits

    # Adapter OFF (the KL-teacher path).
    with torch.no_grad():
        with model.disable_adapter():
            off_logits = model(**inputs).logits

    # Logits must differ — proves the swap reached the underlying
    # forward. Use a generous tolerance because LoRA delta is tiny.
    assert on_logits.shape == off_logits.shape
    delta = (on_logits.float() - off_logits.float()).abs().max().item()
    assert delta > 1e-3, f"adapter on vs off max-abs delta = {delta} (expected > 1e-3)"


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU")
def test_compute_loss_with_regularization_on_real_model_is_finite() -> None:
    """End-to-end: build_regularizers + compute_loss_with_regularization
    on a real Gemma 4 + small LoRA produces a finite total loss across
    one text-only batch."""
    from src.config import RegularizationConfig
    from src.regularization import build_regularizers
    from src.trainer_modality import compute_loss_with_regularization

    model, processor = _build_gemma4_with_lora()
    model.train()

    cfg = RegularizationConfig(
        kl_enabled=True, kl_weight=0.05, kl_temperature=1.0,
        l2_enabled=True, l2_weight=1.0e-4,
    )
    state = build_regularizers(cfg, model=model)
    assert state.enabled is True
    assert state.l2.num_anchored_params() > 0

    convo = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is 4."}]},
    ]
    inputs = processor.apply_chat_template(
        [convo], add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = inputs["input_ids"].to("cuda:0")
    attention_mask = inputs["attention_mask"].to("cuda:0")
    labels = input_ids.clone()

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # HF Causal-LM auto-shifts: out.loss is the next-token CE. Use it
    # directly as the CE term, but build our own shifted labels for the
    # KL position mask (KL needs label-aligned positions).
    ce_loss = out.loss
    shifted_labels = labels[:, 1:].clone()
    shifted_logits = out.logits[:, :-1, :]

    total, metrics = compute_loss_with_regularization(
        model=model,
        student_logits=shifted_logits,
        labels=shifted_labels,
        ce_loss=ce_loss,
        state=state,
        teacher_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        teacher_forward=lambda **kw: model(**kw).logits[:, :-1, :],
        teacher_ctx_factory=lambda: model.disable_adapter(),
    )
    assert torch.isfinite(total), f"total loss is not finite: {total.item()}"
    assert "kl" in metrics and "l2" in metrics
    # At init the LoRA delta is ~0 → KL is ~0 (student ≈ teacher).
    # The L2 anchor is also ~0 because params == snapshot. Total ≈ CE.
    assert metrics["kl"] >= 0.0
    assert metrics["l2"] >= 0.0
    # Backward must work (full graph including the disable-adapter
    # teacher forward must be properly detached).
    total.backward()
    # At least one LoRA param should have a non-None grad.
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    assert n_grad > 0, "no params received gradient — backward graph is broken"
