from __future__ import annotations

from pathlib import Path

import pytest

from data_mix.src.smoltalk_sampler import sample_smoltalk_records
from data_mix.src.schema import validate_record


def _hf_row(user_text: str, asst_text: str, extra_turns: int = 0):
    """Mimics one HF smol-smoltalk record: list of {role, content} dicts."""
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": asst_text},
    ]
    for k in range(extra_turns):
        msgs.append({"role": "user", "content": f"follow-up {k}"})
        msgs.append({"role": "assistant", "content": f"reply {k}"})
    return {"messages": msgs}


def _fake_stream(n: int):
    for i in range(n):
        yield _hf_row(f"question {i}", f"answer {i}", extra_turns=i % 3)


def test_sample_keeps_only_first_turn_pair():
    out = sample_smoltalk_records(stream=_fake_stream(50), total=10, seed=0)
    assert len(out) == 10
    for rec in out:
        validate_record(rec)
        assert rec["source"] == "smoltalk"
        # v2: text-only records carry image=None (was: dummy image path).
        # ModalityAwareBatchSampler routes these into vision-skip batches.
        assert rec["image"] is None
        assert len(rec["conversations"]) == 2
        assert rec["conversations"][0]["role"] == "user"
        assert rec["conversations"][1]["role"] == "assistant"


def test_total_exceeds_stream_returns_all():
    out = sample_smoltalk_records(stream=_fake_stream(5), total=100, seed=0)
    assert len(out) == 5


def test_skips_records_with_no_assistant_turn():
    def bad_stream():
        yield {"messages": [{"role": "user", "content": "only user"}]}
        yield _hf_row("good u", "good a")
        yield {"messages": []}
        yield _hf_row("good u2", "good a2")

    out = sample_smoltalk_records(stream=bad_stream(), total=10, seed=0)
    assert len(out) == 2


def test_image_field_is_none_not_missing():
    # Defensive: schema.validate_record raises on missing 'image' key but
    # accepts explicit None. Confirm the sampler emits explicit None
    # rather than dropping the key.
    out = sample_smoltalk_records(stream=_fake_stream(1), total=1, seed=0)
    assert "image" in out[0]
    assert out[0]["image"] is None
