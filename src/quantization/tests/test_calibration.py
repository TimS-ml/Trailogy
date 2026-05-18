"""Tests for ``src.common.calibration``.

The module is route-agnostic — same loaders should be safe for GPTQ,
AWQ, dynamic_quant, and the future B.1 bridge. These tests cover the
public API directly; ``test_gptq_calibration.py`` exercises the same
helpers via the GPTQ-side wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.calibration import (
    CalibrationDataLeakError,
    build_calibration_dataset,
    load_plantnet_calibration,
    reject_calibration_leak,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_reject_calibration_leak_rejects_val(tmp_path):
    p = tmp_path / "val.jsonl"
    p.touch()
    with pytest.raises(CalibrationDataLeakError, match="val.jsonl"):
        reject_calibration_leak(p)


@pytest.mark.parametrize(
    "name,match",
    [
        ("eval.jsonl", "eval"),
        ("test.jsonl", "test"),
        ("plantnet-50k-val.jsonl", "val"),
        ("plantnet-50k-eval.jsonl", "eval"),
        ("plantnet-50k-test.jsonl", "test"),
        ("plantnet-canonical-species.jsonl", "canonical"),
    ],
)
def test_reject_calibration_leak_rejects_other_eval_sources(tmp_path, name, match):
    p = tmp_path / name
    p.touch()
    with pytest.raises(CalibrationDataLeakError, match=match):
        reject_calibration_leak(p)


def test_reject_calibration_leak_rejects_eval_named_parent(tmp_path):
    p = tmp_path / "val.split" / "train.jsonl"
    with pytest.raises(CalibrationDataLeakError, match="val"):
        reject_calibration_leak(p)


def test_reject_calibration_leak_rejects_overfit100(tmp_path):
    p = tmp_path / "plantnet-overfit100-train.jsonl"
    p.touch()
    with pytest.raises(CalibrationDataLeakError, match="overfit100"):
        reject_calibration_leak(p)


def test_reject_calibration_leak_accepts_train(tmp_path):
    p = tmp_path / "train.jsonl"
    p.touch()
    reject_calibration_leak(p)  # no raise


def test_load_plantnet_calibration_returns_text_records(tmp_path):
    p = tmp_path / "train.jsonl"
    _write_jsonl(
        p,
        [
            {
                "image": "/tmp/p.jpg",
                "conversations": [
                    {"role": "user", "content": "What plant is this?"},
                    {"role": "assistant", "content": "Quercus robur."},
                ],
            }
        ],
    )
    calib = load_plantnet_calibration(p, n_samples=10, seed=0)
    assert len(calib) == 1
    assert "Quercus robur" in calib[0]["text"]


def test_build_calibration_dataset_with_zero_text_skips_wikitext(tmp_path):
    """Smoke-test that n_calib_text=0 short-circuits before any
    ``datasets`` import — keeps the test offline.
    """
    p = tmp_path / "train.jsonl"
    _write_jsonl(
        p,
        [
            {
                "image": "/tmp/p.jpg",
                "conversations": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ],
    )
    out = build_calibration_dataset(
        n_calib_plantnet=1,
        n_calib_text=0,
        calib_seq_len=512,
        calib_seed=0,
        plantnet_jsonl=p,
    )
    assert len(out) == 1


def test_build_calibration_dataset_empty_when_no_source(tmp_path):
    """No plantnet jsonl AND no text → empty list, no exception."""
    out = build_calibration_dataset(
        n_calib_plantnet=10,
        n_calib_text=0,
        calib_seq_len=512,
        calib_seed=0,
        plantnet_jsonl=None,
    )
    assert out == []


def test_load_plantnet_calibration_uses_tokenizer_chat_template(tmp_path):
    """When a tokenizer is supplied, calibration text comes from
    ``apply_chat_template``, not the raw flatten fallback.
    """
    p = tmp_path / "train.jsonl"
    _write_jsonl(
        p,
        [
            {
                "conversations": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ],
            }
        ],
    )

    class FakeTokenizer:
        def __init__(self):
            self.calls: list[list[dict]] = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            self.calls.append(messages)
            assert tokenize is False
            # Distinctive sentinel string the test asserts on.
            return "<TEMPLATE>" + " | ".join(f"{m['role']}/{m['content']}" for m in messages)

    tok = FakeTokenizer()
    calib = load_plantnet_calibration(p, n_samples=10, seed=0, tokenizer=tok)
    assert len(calib) == 1
    assert calib[0]["text"].startswith("<TEMPLATE>")
    assert tok.calls and tok.calls[0][0]["role"] == "user"


def test_load_plantnet_calibration_strips_image_blocks_for_text_tokenizer(tmp_path):
    """Image-content blocks must be stripped before ``apply_chat_template``;
    text tokenizers reject ``{"type": "image"}`` blocks.
    """
    p = tmp_path / "train.jsonl"
    _write_jsonl(
        p,
        [
            {
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "What plant?"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "Acer."}]},
                ],
            }
        ],
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kw):
            # Assert every content is a plain string after stripping.
            for m in messages:
                assert isinstance(m["content"], str), f"Got non-string content: {m['content']!r}"
            return "OK:" + "|".join(m["content"] for m in messages)

    calib = load_plantnet_calibration(p, n_samples=10, seed=0, tokenizer=FakeTokenizer())
    assert len(calib) == 1
    assert "What plant?" in calib[0]["text"]
    assert "Acer." in calib[0]["text"]


def test_load_plantnet_calibration_falls_back_when_template_raises(tmp_path):
    """If ``apply_chat_template`` raises, the loader degrades gracefully
    to the plain flatten formatter rather than dropping the sample.
    """
    p = tmp_path / "train.jsonl"
    _write_jsonl(
        p,
        [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "World"},
                ],
            }
        ],
    )

    class BrokenTokenizer:
        def apply_chat_template(self, *a, **kw):
            raise RuntimeError("nope")

    calib = load_plantnet_calibration(p, n_samples=10, seed=0, tokenizer=BrokenTokenizer())
    assert len(calib) == 1
    # Flatten format: "user: Hello\nassistant: World"
    assert "user: Hello" in calib[0]["text"]
    assert "assistant: World" in calib[0]["text"]
    # Must not contain the old fake marker.
    assert "<|turn>" not in calib[0]["text"]
