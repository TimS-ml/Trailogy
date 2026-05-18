"""SMOKE TEST: Gemma 4 E2B accepts a text-only batch (no pixel_values).

This is the GATE for the v2 ModalityAware skip-vision approach. If
``Gemma4ForConditionalGeneration.forward`` rejects ``pixel_values=None``
or the processor's chat template produces a corrupt text-only sequence,
the entire B-approach is dead and we have to fall back to the dummy
image v1 method.

Costs:
- ~5 GB GPU memory for Gemma 4 E2B in bf16
- ~30 s to load
- ~2 s per forward pass

Marked ``@pytest.mark.gpu`` so it's NOT in the default pytest run.
Trigger explicitly::

    CUDA_VISIBLE_DEVICES=0 pytest finetune/tests/test_text_only_forward_smoke.py -m gpu -v
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def _gpu_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU")
def test_gemma4_text_only_batch_forward_returns_finite_loss():
    """Build a 2-record text-only batch, call model.forward, assert loss
    is finite (proves the vision-skip path actually works end-to-end).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_name = "unsloth/gemma-4-E2B-it"

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # Build two text-only conversations using Gemma 4's chat template.
    convos = [
        [
            {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is 4."}]},
        ],
        [
            {"role": "user", "content": [{"type": "text", "text": "Hello there."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
        ],
    ]
    text_inputs = [
        processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in convos
    ]

    # Critical: pass images=None (or omit images entirely) — no pixel_values
    # should land in the batch.
    enc = processor(
        text=text_inputs,
        images=None,
        return_tensors="pt",
        padding=True,
    )
    enc = {k: v.to("cuda:0") for k, v in enc.items() if hasattr(v, "to")}

    # Sanity: pixel_values must NOT be in the batch — that's the whole
    # point of vision-skip.
    assert "pixel_values" not in enc, (
        "text-only batch unexpectedly carries pixel_values — processor is "
        "auto-injecting an image somewhere"
    )

    labels = enc["input_ids"].clone()
    # Use -100 to mask padding from the loss (HF convention).
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    with torch.no_grad():
        out = model(**enc, labels=labels)

    assert torch.isfinite(out.loss), (
        f"text-only forward returned non-finite loss: {out.loss.item()}"
    )
    # Loss should be reasonable (not zero, not enormous) for a 2-token-target
    # text-only batch on a healthy model.
    loss_val = float(out.loss.item())
    assert 0.0 < loss_val < 50.0, f"suspect loss value: {loss_val}"


@pytest.mark.skipif(not _gpu_available(), reason="requires GPU")
def test_gemma4_mixed_image_and_text_batches_are_independently_processed():
    """Smoke: an image batch (with pixel_values) and a text-only batch
    (without) can be processed sequentially through the same model
    without state leakage. This confirms the trainer can alternate
    batches across modalities.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    model_name = "unsloth/gemma-4-E2B-it"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # Text-only batch.
    text_convo = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
    ]
    text_input = processor.apply_chat_template(
        text_convo, tokenize=False, add_generation_prompt=False
    )
    enc_text = processor(text=[text_input], images=None, return_tensors="pt", padding=True)
    enc_text = {k: v.to("cuda:0") for k, v in enc_text.items() if hasattr(v, "to")}
    labels_text = enc_text["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels_text[labels_text == processor.tokenizer.pad_token_id] = -100

    # Image batch.
    img = Image.new("RGB", (960, 672), color=(128, 128, 128))
    img_convo = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is this?"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Gray."}]},
    ]
    img_input = processor.apply_chat_template(
        img_convo, tokenize=False, add_generation_prompt=False
    )
    enc_img = processor(text=[img_input], images=[img], return_tensors="pt", padding=True)
    enc_img = {k: v.to("cuda:0") for k, v in enc_img.items() if hasattr(v, "to")}
    labels_img = enc_img["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels_img[labels_img == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        out_text = model(**enc_text, labels=labels_text)
        out_img = model(**enc_img, labels=labels_img)

    assert torch.isfinite(out_text.loss)
    assert torch.isfinite(out_img.loss)
