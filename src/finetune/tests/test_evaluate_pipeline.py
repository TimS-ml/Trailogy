"""Regression tests for eval/train parity and eval plumbing.

These tests intentionally avoid loading torch/transformers/unsloth by
monkeypatching evaluate.py's lazy globals. They exercise our wrapper logic,
not the upstream model implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import evaluate


class _FakeModel:
    device = "cpu"

    def eval(self):
        return self


class _FakeProcessor:
    pass


def _patch_eval_lazy_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate, "_import_deps", lambda: None)
    monkeypatch.setattr(evaluate, "_torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setattr(evaluate, "_BitsAndBytesConfig", object)
    monkeypatch.setattr(evaluate, "_PeftModel", None)


def test_load_model_applies_train_chat_template_to_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval must use the same gemma-4 chat template as finetune.py.

    The model path does not matter here; the invariant is that every loaded
    processor is normalized through evaluate._apply_train_chat_template().
    """

    _patch_eval_lazy_deps(monkeypatch)
    processor = _FakeProcessor()
    calls = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return processor

    class FakeImageTextModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeModel()

    monkeypatch.setattr(evaluate, "_AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(evaluate, "_AutoModelForImageTextToText", FakeImageTextModel, raising=False)
    monkeypatch.setattr(
        evaluate,
        "_apply_train_chat_template",
        lambda p: calls.append(p) or p,
        raising=False,
    )

    _model, returned_processor = evaluate.load_model(
        base_model="fake/gemma4", adapter_path=None, quantize=False, use_unsloth=False
    )

    assert returned_processor is processor
    assert calls == [processor]


def test_non_unsloth_eval_uses_multimodal_image_text_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-unsloth eval must not use AutoModelForCausalLM for Gemma 4 VLM.

    AutoModelForCausalLM can silently drop/ignore vision modules. The safe
    class is AutoModelForImageTextToText.
    """

    _patch_eval_lazy_deps(monkeypatch)

    class BadCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):  # pragma: no cover - should not run
            raise AssertionError("AutoModelForCausalLM must not be used for VLM eval")

    class GoodImageTextModel:
        called = False

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            cls.called = True
            return _FakeModel()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeProcessor()

    monkeypatch.setattr(evaluate, "_AutoModelForCausalLM", BadCausalLM)
    monkeypatch.setattr(evaluate, "_AutoModelForImageTextToText", GoodImageTextModel, raising=False)
    monkeypatch.setattr(evaluate, "_AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(evaluate, "_apply_train_chat_template", lambda p: p, raising=False)

    evaluate.load_model("fake/gemma4", adapter_path=None, quantize=False, use_unsloth=False)

    assert GoodImageTextModel.called is True


def test_evaluate_batch_passes_configured_max_new_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_generate_response(*_args, max_new_tokens: int, **_kwargs) -> str:
        seen.append(max_new_tokens)
        return "This is Eastern Hemlock."

    monkeypatch.setattr(evaluate, "generate_response", fake_generate_response)

    sample = {
        "image": "/tmp/plant.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "This is Eastern Hemlock."},
        ],
    }

    evaluate.evaluate_batch(
        model=object(),
        processor=object(),
        test_data=[sample],
        batch_size=1,
        max_new_tokens=17,
    )

    assert seen == [17]


def test_load_test_data_can_match_train_require_image_filter(tmp_path: Path) -> None:
    jsonl = tmp_path / "eval.jsonl"
    image_record = {
        "image": "/tmp/plant.jpg",
        "conversations": [
            {"role": "user", "content": "What plant?"},
            {"role": "assistant", "content": "This is Eastern Hemlock."},
        ],
    }
    text_only_record = {
        "conversations": [
            {"role": "user", "content": "What trail is this?"},
            {"role": "assistant", "content": "A hiking trail."},
        ],
    }
    jsonl.write_text(
        json.dumps(image_record) + "\n" + json.dumps(text_only_record) + "\n"
    )

    records = evaluate.load_test_data(str(jsonl), require_image=True)

    assert records == [image_record]


def test_build_eval_prompt_no_prefix_backward_compat() -> None:
    """Backward-compat contract: a model trained WITHOUT
    ``prompt_prefixes`` (pre-v3 baselines, plain v2 configs) must still
    eval cleanly under the v4 code path. The eval prompt for both an
    image sample and a text-only sample must come through with the
    user text verbatim when ``prompt_prefixes=None``.

    Pinned alongside ``test_data_prompt_prefix.test_no_prefix_when_prompt_prefixes_is_none``
    so the eval-side and the build-side guarantees move together.
    """
    image_sample = {
        "image": "/tmp/plant.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    text_sample = {
        # No 'image' field at all — pre-v3 text-only record.
        "conversations": [
            {"role": "user", "content": "Tell me about hiking."},
            {"role": "assistant", "content": "Hiking is..."},
        ],
    }

    img_messages, img_user_msg, img_ref = evaluate._build_eval_prompt(
        image_sample, prompt_prefixes=None
    )
    txt_messages, txt_user_msg, txt_ref = evaluate._build_eval_prompt(
        text_sample, prompt_prefixes=None
    )

    # Image path: image block + verbatim user text (no marker).
    img_text_block = next(
        b for b in img_messages[0]["content"] if b["type"] == "text"
    )
    assert img_text_block["text"] == "What plant is this?"
    assert img_user_msg == "What plant is this?"
    assert img_ref == "Acer rubrum."

    # Text-only path: text-only block, verbatim user text (no marker).
    txt_text_block = txt_messages[0]["content"][0]
    assert txt_text_block["text"] == "Tell me about hiking."
    assert txt_user_msg == "Tell me about hiking."
    assert txt_ref == "Hiking is..."


def test_evaluate_batch_uses_full_conversation_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages = []

    def fake_generate_response(*_args, messages=None, max_new_tokens: int, **_kwargs) -> str:
        captured_messages.append(messages)
        return "This is Sugar Maple."

    monkeypatch.setattr(evaluate, "generate_response", fake_generate_response)
    sample = {
        "image": "/tmp/plant.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "It might be a maple."},
            {"role": "user", "content": "Which species?"},
            {"role": "assistant", "content": "This is Sugar Maple."},
        ],
    }

    results = evaluate.evaluate_batch(
        model=object(),
        processor=object(),
        test_data=[sample],
        batch_size=1,
        max_new_tokens=32,
    )

    assert results[0]["question"] == "Which species?"
    assert results[0]["reference"] == "This is Sugar Maple."
    assert captured_messages == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/plant.jpg"},
                    {"type": "text", "text": "What plant is this?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "It might be a maple."}]},
            {"role": "user", "content": [{"type": "text", "text": "Which species?"}]},
        ]
    ]
