"""Tests for src.data — JSONL → unsloth `messages` conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import (
    build_vision_messages,
    iter_jsonl,
    load_vision_dataset,
    summarize_dataset,
)


def test_image_user_message_is_a_content_list() -> None:
    record = {
        "image": "/abs/path/img.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Eastern Hemlock."},
        ],
    }
    out = build_vision_messages(record)
    assert "messages" in out
    assert len(out["messages"]) == 2

    user_msg = out["messages"][0]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0] == {"type": "image", "image": "/abs/path/img.jpg"}
    assert user_msg["content"][1] == {"type": "text", "text": "What plant is this?"}

    asst_msg = out["messages"][1]
    assert asst_msg["role"] == "assistant"
    assert asst_msg["content"] == [{"type": "text", "text": "Eastern Hemlock."}]


def test_legacy_image_placeholder_is_stripped() -> None:
    record = {
        "image": "/img.jpg",
        "conversations": [
            {"role": "user", "content": "<image>\nWhat is this?"},
            {"role": "assistant", "content": "A flower."},
        ],
    }
    out = build_vision_messages(record)
    user_text = out["messages"][0]["content"][1]["text"]
    assert "<image>" not in user_text
    assert user_text == "What is this?"


def test_no_image_record_stays_text_only() -> None:
    record = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "Tell me about basalt."},
            {"role": "assistant", "content": "Basalt is a volcanic rock."},
        ],
    }
    out = build_vision_messages(record)
    user_msg = out["messages"][0]
    assert user_msg["content"] == [{"type": "text", "text": "Tell me about basalt."}]
    # No image block anywhere
    for msg in out["messages"]:
        for block in msg["content"]:
            assert block["type"] != "image"


def test_image_attached_only_to_first_user_turn() -> None:
    record = {
        "image": "/img.jpg",
        "conversations": [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Follow-up?"},
            {"role": "assistant", "content": "Sure."},
        ],
    }
    out = build_vision_messages(record)
    msgs = out["messages"]
    # First user message has the image block
    assert any(b["type"] == "image" for b in msgs[0]["content"])
    # Second user message does not
    assert all(b["type"] != "image" for b in msgs[2]["content"])


def test_empty_conversations_raises() -> None:
    with pytest.raises(ValueError, match="conversations"):
        build_vision_messages({"image": None, "conversations": []})


def test_unexpected_role_raises() -> None:
    with pytest.raises(ValueError, match="role"):
        build_vision_messages(
            {
                "image": None,
                "conversations": [
                    {"role": "robot", "content": "hi"},
                ],
            }
        )


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"a": 1}\n'
        "\n"
        '{"a": 2}\n'
        "\n"
    )
    rows = list(iter_jsonl(p))
    assert rows == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_bad_json_raises_with_lineno(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot-json\n')
    with pytest.raises(ValueError, match=":2:"):
        list(iter_jsonl(p))


def test_load_vision_dataset_respects_max_samples(tmp_path: Path) -> None:
    p = tmp_path / "train.jsonl"
    rows = []
    for i in range(10):
        rows.append(
            json.dumps(
                {
                    "image": f"/img/{i}.jpg",
                    "conversations": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ],
                }
            )
        )
    p.write_text("\n".join(rows) + "\n")
    out = load_vision_dataset(p, max_samples=3)
    assert len(out) == 3
    assert out[0]["messages"][0]["content"][1]["text"] == "q0"


def test_load_vision_dataset_require_image_drops_text_only(
    tmp_path: Path,
) -> None:
    """When require_image=True, text-only records (image=null) are dropped.

    Regression test for the production crash:
        ValueError: Received inconsistently sized batches of images (7) and
        text (8)
    Gemma4Processor + UnslothVisionDataCollator reject mixed batches, so
    hiking-Q&A text-only records merged into the PlantNet vision data must
    be filtered at load time.
    """
    p = tmp_path / "train.jsonl"
    rows = [
        json.dumps({
            "image": "/img/0.jpg",
            "conversations": [
                {"role": "user", "content": "q0"},
                {"role": "assistant", "content": "a0"},
            ],
        }),
        json.dumps({
            "image": None,  # text-only — must be dropped
            "conversations": [
                {"role": "user", "content": "hiking q"},
                {"role": "assistant", "content": "hiking a"},
            ],
        }),
        json.dumps({
            "image": "/img/2.jpg",
            "conversations": [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        }),
        json.dumps({
            # 'image' key missing entirely — must also be dropped
            "conversations": [
                {"role": "user", "content": "no-image-key"},
                {"role": "assistant", "content": "..."},
            ],
        }),
    ]
    p.write_text("\n".join(rows) + "\n")

    # Default behavior (backward compat): all 4 records kept.
    assert len(load_vision_dataset(p)) == 4

    # With require_image, only the 2 records with a non-empty image path.
    out = load_vision_dataset(p, require_image=True)
    assert len(out) == 2
    texts = [rec["messages"][0]["content"][1]["text"] for rec in out]
    assert texts == ["q0", "q2"]


def test_load_vision_dataset_require_image_max_samples_counts_kept(
    tmp_path: Path,
) -> None:
    """max_samples should count records that survive require_image filtering,
    not raw lines read. Otherwise a JSONL with many text-only records at the
    head could exhaust the budget before any vision records are loaded.
    """
    p = tmp_path / "train.jsonl"
    rows = []
    # 3 text-only records first
    for i in range(3):
        rows.append(json.dumps({
            "image": None,
            "conversations": [
                {"role": "user", "content": f"text{i}"},
                {"role": "assistant", "content": "..."},
            ],
        }))
    # 5 image records
    for i in range(5):
        rows.append(json.dumps({
            "image": f"/img/{i}.jpg",
            "conversations": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        }))
    p.write_text("\n".join(rows) + "\n")

    out = load_vision_dataset(p, max_samples=2, require_image=True)
    assert len(out) == 2
    # First two surviving records are the first two image records.
    texts = [rec["messages"][0]["content"][1]["text"] for rec in out]
    assert texts == ["q0", "q1"]


def test_summarize_dataset_counts() -> None:
    records = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/x.jpg"},
                        {"type": "text", "text": "q"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            ]
        },
    ]
    stats = summarize_dataset(records)
    assert stats == {
        "records": 2,
        "with_image": 1,
        "user_turns": 2,
        "assistant_turns": 2,
    }
