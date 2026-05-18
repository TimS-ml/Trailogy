"""Tests for v2 multi-val loading + modality helpers in src/data.py.

These cover the data-side glue needed by the trainer's multi-eval-dataset
feature (eval_<key>_loss per modality) and by ModalityAwareBatchSampler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from src.data import (
    load_vision_dataset_dict,
    record_has_image,
    ModalityAwareCollator,
)


# ---------------------------------------------------------------------------
# record_has_image
# ---------------------------------------------------------------------------

def test_record_has_image_for_image_block_present():
    rec = {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "/x.jpg"},
            {"type": "text", "text": "Q"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]}
    assert record_has_image(rec) is True


def test_record_has_image_for_text_only():
    rec = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]}
    assert record_has_image(rec) is False


def test_record_has_image_empty_messages():
    assert record_has_image({"messages": []}) is False
    assert record_has_image({}) is False


# ---------------------------------------------------------------------------
# load_vision_dataset_dict
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def multi_val_files(tmp_path):
    plant_p = tmp_path / "val_plant.jsonl"
    nonplant_p = tmp_path / "val_nonplant.jsonl"
    negative_p = tmp_path / "val_negative.jsonl"

    _write_jsonl(plant_p, [
        {"image": "/img/plant_0.jpg", "conversations": [
            {"role": "user", "content": "What plant?"},
            {"role": "assistant", "content": "Rose."},
        ], "source": "plant"},
        {"image": "/img/plant_1.jpg", "conversations": [
            {"role": "user", "content": "What plant?"},
            {"role": "assistant", "content": "Oak."},
        ], "source": "plant"},
    ])
    _write_jsonl(nonplant_p, [
        {"image": None, "conversations": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ], "source": "smoltalk"},
        {"image": "/img/llava_0.jpg", "conversations": [
            {"role": "user", "content": "Caption?"},
            {"role": "assistant", "content": "A book."},
        ], "source": "llava"},
    ])
    _write_jsonl(negative_p, [
        {"image": "/img/neg_0.jpg", "conversations": [
            {"role": "user", "content": "What plant?"},
            {"role": "assistant", "content": "I don't see a plant."},
        ], "source": "negative"},
    ])
    return {"plant": plant_p, "nonplant": nonplant_p, "negative": negative_p}


def test_load_vision_dataset_dict_returns_named_partitions(multi_val_files):
    val_files = {k: str(v) for k, v in multi_val_files.items()}
    out = load_vision_dataset_dict(val_files)
    assert set(out.keys()) == {"plant", "nonplant", "negative"}
    assert len(out["plant"]) == 2
    assert len(out["nonplant"]) == 2
    assert len(out["negative"]) == 1
    # Each value is a list of unsloth-format messages records.
    for partition in out.values():
        for rec in partition:
            assert "messages" in rec


def test_load_vision_dataset_dict_handles_missing_file(multi_val_files, tmp_path):
    val_files = {
        "plant": str(multi_val_files["plant"]),
        "negative": str(tmp_path / "does_not_exist.jsonl"),
    }
    with pytest.raises(FileNotFoundError):
        load_vision_dataset_dict(val_files)


def test_load_vision_dataset_dict_does_not_require_image_for_text_only(
    multi_val_files,
):
    """nonplant val (smoltalk text-only) must survive — image=None is
    valid in v2 because the sampler will route into a vision-skip batch."""
    val_files = {k: str(v) for k, v in multi_val_files.items()}
    out = load_vision_dataset_dict(val_files)
    # If require_image were True, the text-only smoltalk record would
    # be dropped; assert nonplant kept BOTH records.
    assert len(out["nonplant"]) == 2


# ---------------------------------------------------------------------------
# ModalityAwareCollator
# ---------------------------------------------------------------------------

class _FakeVisionCollator:
    def __init__(self):
        self.calls = []
    def __call__(self, batch):
        self.calls.append(("vision", len(batch)))
        return {"kind": "vision", "n": len(batch)}


class _FakeTextCollator:
    def __init__(self):
        self.calls = []
    def __call__(self, batch):
        self.calls.append(("text", len(batch)))
        return {"kind": "text", "n": len(batch)}


def _img_rec(i):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": f"/img/{i}.jpg"},
            {"type": "text", "text": "Q"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]}


def _txt_rec(i):
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": f"Q{i}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
    ]}


def test_modality_collator_dispatches_image_batch_to_vision_collator():
    vc = _FakeVisionCollator()
    tc = _FakeTextCollator()
    collator = ModalityAwareCollator(vision_collator=vc, text_collator=tc)
    out = collator([_img_rec(0), _img_rec(1), _img_rec(2)])
    assert out["kind"] == "vision"
    assert vc.calls == [("vision", 3)]
    assert tc.calls == []


def test_modality_collator_dispatches_text_batch_to_text_collator():
    vc = _FakeVisionCollator()
    tc = _FakeTextCollator()
    collator = ModalityAwareCollator(vision_collator=vc, text_collator=tc)
    out = collator([_txt_rec(0), _txt_rec(1)])
    assert out["kind"] == "text"
    assert vc.calls == []
    assert tc.calls == [("text", 2)]


def test_modality_collator_rejects_mixed_batch():
    """Defensive assertion: if a mixed batch leaks through (e.g. sampler
    misconfigured), fail loudly rather than silently feed Gemma4Processor
    a mixed batch that would crash deep inside the model forward."""
    vc = _FakeVisionCollator()
    tc = _FakeTextCollator()
    collator = ModalityAwareCollator(vision_collator=vc, text_collator=tc)
    with pytest.raises(ValueError) as exc:
        collator([_img_rec(0), _txt_rec(1)])
    assert "mixed" in str(exc.value).lower() or "modality" in str(exc.value).lower()


def test_modality_collator_empty_batch_does_not_call_either():
    vc = _FakeVisionCollator()
    tc = _FakeTextCollator()
    collator = ModalityAwareCollator(vision_collator=vc, text_collator=tc)
    with pytest.raises(ValueError):
        collator([])
