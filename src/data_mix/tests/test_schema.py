from __future__ import annotations

import pytest

from data_mix.src.schema import RecordError, validate_record, ALLOWED_SOURCES


def _good_record():
    return {
        "image": "/abs/path/to/img.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Eastern Hemlock."},
        ],
        "source": "plant",
    }


def test_validate_record_accepts_well_formed():
    validate_record(_good_record())  # no raise


def test_allowed_sources_exact():
    # v2 added 'llava' as a valid source (replaces cambrian for new mixes;
    # cambrian retained for v1 backward compat).
    # v3 added 'offline_qa' as a valid source (tiny persona-shaping
    # corpus sourced from assets/data_offline_qa/offline_qa.json).
    assert ALLOWED_SOURCES == {
        "plant", "cambrian", "llava", "smoltalk", "negative", "offline_qa",
    }


def test_validate_record_accepts_offline_qa_source():
    """v3: offline_qa is text-only (image=None), like smoltalk."""
    from data_mix.src.schema import validate_record
    rec = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "Are you ChatGPT?"},
            {"role": "assistant", "content": "No, I'm an offline on-device model."},
        ],
        "source": "offline_qa",
    }
    validate_record(rec)  # no raise


def test_validate_record_accepts_image_none_for_text_only():
    # v2: smoltalk text-only records carry image=None so the trainer can
    # route them to a vision-skip batch (see ModalityAwareBatchSampler).
    rec = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
        ],
        "source": "smoltalk",
    }
    validate_record(rec)  # no raise


def test_validate_record_accepts_llava_source():
    rec = _good_record()
    rec["source"] = "llava"
    validate_record(rec)  # no raise


def test_validate_record_accepts_multi_turn():
    # v2 LLaVA sampler preserves multi-turn conversations.
    rec = {
        "image": "/abs/path/img.jpg",
        "conversations": [
            {"role": "user",      "content": "What is this?"},
            {"role": "assistant", "content": "A book."},
            {"role": "user",      "content": "Who wrote it?"},
            {"role": "assistant", "content": "Donna Eden."},
        ],
        "source": "llava",
    }
    validate_record(rec)  # no raise


@pytest.mark.parametrize(
    "mutate, msg_fragment",
    [
        # v2: empty string still rejected (degenerate); only None is the
        # new accepted no-image marker.
        (lambda r: r.__setitem__("image", ""), "image"),
        (lambda r: r.__setitem__("image", 42), "image"),
        (lambda r: r.pop("conversations"), "conversations"),
        (lambda r: r.__setitem__("conversations", []), "conversations"),
        (lambda r: r.__setitem__("source", "weather"), "source"),
        (lambda r: r.pop("source"), "source"),
        (
            lambda r: r.__setitem__(
                "conversations",
                [{"role": "user", "content": "x"}],  # missing assistant
            ),
            "conversations",
        ),
        (
            lambda r: r.__setitem__(
                "conversations",
                [
                    {"role": "user", "content": "x"},
                    {"role": "user", "content": "y"},
                ],
            ),
            "alternation",
        ),
    ],
)
def test_validate_record_rejects(mutate, msg_fragment):
    r = _good_record()
    mutate(r)
    with pytest.raises(RecordError) as exc:
        validate_record(r)
    assert msg_fragment in str(exc.value).lower()


def test_validate_record_rejects_missing_image_key():
    # Pop test moved out of the parametrize because the message changed —
    # missing key vs. wrong type are now distinguishable error paths
    # (see schema.py).
    r = _good_record()
    r.pop("image")
    with pytest.raises(RecordError) as exc:
        validate_record(r)
    assert "image" in str(exc.value).lower()
