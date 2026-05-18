"""Tests for ``TextOnlyChatCollator`` — applies the chat template to
text-only records (raw ``messages`` dicts) and produces a tokenized,
padded batch with label masking, matching what the LM loss head expects.

The bug this collator fixes: the v2 ModalityAware path wired plain
``DataCollatorForLanguageModeling(tokenizer)`` as the text branch.
That collator assumes records already carry ``input_ids`` (e.g. from
a prior ``dataset.map(tokenize)`` step). Our text-only records carry
raw ``messages: [{role, content: [...]}]`` instead, so the collator
fails with::

    ValueError: You should supply an encoding ... that includes
    input_ids, but you provided ['messages', 'length']

``TextOnlyChatCollator`` closes the gap: chat_template → tokenize →
pad → label tensor in one pass.
"""
from __future__ import annotations

from typing import Any, List

import pytest

import torch

from src.data import TextOnlyChatCollator


# ---------------------------------------------------------------------------
# Fake processor / tokenizer — just enough surface to exercise the collator
# without pulling in a real HF tokenizer (those are slow + heavyweight).
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tokenize a string by whitespace; vocab is a counter, no special tokens."""

    pad_token_id = 0

    def __init__(self):
        self._vocab = {"<pad>": 0}

    def _enc_one(self, s: str) -> list[int]:
        toks = s.split()
        ids = []
        for t in toks:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab)
            ids.append(self._vocab[t])
        return ids

    def __call__(self, texts, padding=True, return_tensors="pt", **kwargs):
        encs = [self._enc_one(t) for t in texts]
        maxlen = max(len(e) for e in encs)
        input_ids = []
        attn = []
        for e in encs:
            pad_n = maxlen - len(e)
            input_ids.append(e + [self.pad_token_id] * pad_n)
            attn.append([1] * len(e) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _FakeProcessor:
    """Mimics HF's processor.apply_chat_template(tokenize=False)."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False, "collator must request tokenize=False (we tokenize separately)"
        # Flatten the messages into a simple string: "role: text \n role: text"
        parts = []
        for msg in messages:
            role = msg["role"]
            blocks = msg["content"]
            if isinstance(blocks, list):
                text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            else:
                text = blocks
            parts.append(f"{role}: {text}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _text_only_record(user_text: str, asst_text: str) -> dict:
    """Build a record in the same shape build_vision_messages emits."""
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": asst_text}]},
        ],
    }


def test_collator_produces_input_ids_and_labels() -> None:
    """Smallest correctness check: the collator output has the four keys
    the trainer expects (input_ids, attention_mask, labels), and labels
    align with input_ids (per the LM convention)."""
    proc = _FakeProcessor()
    collator = TextOnlyChatCollator(proc)
    batch = [
        _text_only_record("hi there", "hello back"),
        _text_only_record("what is 2 plus 2", "four"),
    ]
    out = collator(batch)
    assert "input_ids" in out
    assert "attention_mask" in out
    assert "labels" in out
    assert out["input_ids"].shape == out["labels"].shape
    assert out["input_ids"].shape == out["attention_mask"].shape


def test_collator_masks_pad_positions_in_labels() -> None:
    """Padded positions in input_ids must get label=-100 so the CE loss
    doesn't backprop through them.

    Build two records of different lengths; the shorter one ends up
    padded; assert the padded positions have label == -100 and the
    non-pad positions have label == input_id."""
    proc = _FakeProcessor()
    collator = TextOnlyChatCollator(proc)
    batch = [
        _text_only_record("short", "x"),
        _text_only_record("a b c d e f g h", "y z w"),
    ]
    out = collator(batch)
    attn = out["attention_mask"]
    labels = out["labels"]
    ids = out["input_ids"]

    # Padded positions (attn==0) → label == -100.
    pad_mask = attn == 0
    assert pad_mask.any(), "test setup should have produced at least one padded position"
    assert (labels[pad_mask] == -100).all()

    # Non-padded positions → label == input_id (LM-style next-token training).
    non_pad = attn == 1
    assert torch.equal(labels[non_pad], ids[non_pad])


def test_collator_handles_smoltalk_style_string_content() -> None:
    """smoltalk records (before v2 schema migration) used to have
    ``content`` as a flat string instead of a list of blocks. The
    collator must accept both — the fake processor's
    apply_chat_template handles both, so we just verify the collator
    doesn't crash on the string-content shape."""
    proc = _FakeProcessor()
    collator = TextOnlyChatCollator(proc)
    batch = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        },
    ]
    out = collator(batch)
    assert out["input_ids"].shape[0] == 1


def test_collator_single_record_batch() -> None:
    """A batch of size 1 must still produce 2D tensors (B=1, T)."""
    proc = _FakeProcessor()
    collator = TextOnlyChatCollator(proc)
    batch = [_text_only_record("solo q", "solo a")]
    out = collator(batch)
    assert out["input_ids"].dim() == 2
    assert out["input_ids"].shape[0] == 1


def test_collator_empty_batch_raises() -> None:
    proc = _FakeProcessor()
    collator = TextOnlyChatCollator(proc)
    with pytest.raises(ValueError):
        collator([])
