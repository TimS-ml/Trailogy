"""Tests for the ``prompt_prefixes`` mechanism — camera-state input gate.

Design (v4): conditional-FT input gate keyed on **image presence** rather
than ``record.source``. At training time, every record gets a literal
marker prepended to the first user turn:

    image present (vision record)  → ``[camera=on] <user text>``
    no image     (text-only record) → ``[camera=off] <user text>``

At inference time the deployment path prepends the same marker
(``[camera=on] <user prompt>`` when an image is captured,
``[camera=off] <user prompt>`` otherwise) so the model sees the same
two-state contract it was trained with.

The marker is independent of question topic. ``"[camera=on] 这是什么植物"``
(asking about a plant in the photo) and ``"[camera=on] 今天天气如何"``
(asking about the sky in the photo) both carry ``[camera=on]`` — what
the tag reflects is the modality state, not the topic.

Config surface (unchanged outer shape):

    data:
      prompt_prefixes:
        camera_on:  "[camera=on] "
        camera_off: "[camera=off] "

``prompt_prefixes=None`` keeps v2 behaviour (no prefix anywhere).
Missing key in the dict (e.g. only ``camera_on`` set) → no prefix for
the other branch.
"""
from __future__ import annotations

import pytest

from src.data import build_vision_messages


# ---------------------------------------------------------------------------
# Default (no prefix) — backward compat with v2 configs
# ---------------------------------------------------------------------------


def test_no_prefix_when_prompt_prefixes_is_none() -> None:
    """v2 behaviour: when no prefix dict is supplied the user text is
    passed through verbatim, regardless of image presence."""
    rec = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(rec)
    user_blocks = out["messages"][0]["content"]
    text_block = next(b for b in user_blocks if b["type"] == "text")
    assert text_block["text"] == "What plant is this?"


# ---------------------------------------------------------------------------
# Prefix injection — image-presence dispatch
# ---------------------------------------------------------------------------


def test_image_record_gets_camera_on_prefix() -> None:
    """Any record carrying an image path triggers the ``camera_on`` prefix
    regardless of what the user is asking about."""
    rec = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=on] What plant is this?"


def test_image_record_camera_on_prefix_independent_of_topic() -> None:
    """The gate is a modality-state flag, not a topic classifier. An image
    record where the user asks about weather still gets ``[camera=on]``."""
    rec = {
        "image": "/sky.jpg",
        "conversations": [
            {"role": "user", "content": "今天天气如何"},
            {"role": "assistant", "content": "看起来是多云。"},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=on] 今天天气如何"


def test_text_only_record_gets_camera_off_prefix() -> None:
    """Records with no image (``image`` field missing or falsy) get the
    ``camera_off`` prefix."""
    rec = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=off] Hello!"


def test_missing_image_field_treated_as_camera_off() -> None:
    """Legacy records that omit the ``image`` field entirely (rather than
    setting it to ``None``) still get ``camera_off`` — falsy/missing both
    count as "no image"."""
    rec = {
        # No 'image' key at all.
        "conversations": [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the chicken..."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=off] Tell me a joke."


# ---------------------------------------------------------------------------
# Partial / asymmetric dict — missing key = no prefix on that branch
# ---------------------------------------------------------------------------


def test_camera_off_key_missing_means_no_prefix_for_text_only() -> None:
    """If the config only sets ``camera_on``, text-only records get no
    prefix (rather than crashing or guessing). Asymmetric configs are
    valid — e.g. an ablation where only the vision branch is gated."""
    rec = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi."},
        ],
    }
    out = build_vision_messages(
        rec, prompt_prefixes={"camera_on": "[camera=on] "}
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "Hello!"


def test_camera_on_key_missing_means_no_prefix_for_image() -> None:
    """Symmetric: if the config only sets ``camera_off``, image records
    get no prefix."""
    rec = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(
        rec, prompt_prefixes={"camera_off": "[camera=off] "}
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "What plant is this?"


def test_empty_prefix_string_is_noop() -> None:
    """Explicit empty string for a key is identical to omitting the key
    — no leading space, no garbage."""
    rec = {
        "image": None,
        "conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": ""},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "Hello!"


# ---------------------------------------------------------------------------
# Source field is no longer the dispatch key
# ---------------------------------------------------------------------------


def test_source_field_does_not_affect_prefix_dispatch() -> None:
    """v4 contract: the legacy ``record.source`` field (kept for telemetry
    / multi-val routing) MUST NOT influence which prefix is selected.
    Two image records with different ``source`` values must both receive
    the same ``camera_on`` prefix."""
    rec_plant = {
        "source": "plant",
        "image": "/p.jpg",
        "conversations": [
            {"role": "user", "content": "What plant?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    rec_llava = {
        "source": "llava",
        "image": "/l.jpg",
        "conversations": [
            {"role": "user", "content": "What's in this image?"},
            {"role": "assistant", "content": "A cat."},
        ],
    }
    pp = {"camera_on": "[camera=on] ", "camera_off": "[camera=off] "}
    plant_text = next(
        b for b in build_vision_messages(rec_plant, prompt_prefixes=pp)["messages"][0]["content"]
        if b["type"] == "text"
    )
    llava_text = next(
        b for b in build_vision_messages(rec_llava, prompt_prefixes=pp)["messages"][0]["content"]
        if b["type"] == "text"
    )
    assert plant_text["text"] == "[camera=on] What plant?"
    assert llava_text["text"] == "[camera=on] What's in this image?"


# ---------------------------------------------------------------------------
# Multi-turn safety + placeholder-strip ordering
# ---------------------------------------------------------------------------


def test_prefix_only_added_to_first_user_turn_not_assistant() -> None:
    """Multi-turn safety: only the FIRST user turn gets the prefix, not
    the assistant turn and not any subsequent user turn. The gate fires
    once per conversation."""
    rec = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "A flower."},
            {"role": "user", "content": "What colour?"},
            {"role": "assistant", "content": "Red."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    msgs = out["messages"]
    # First user gets the prefix.
    user0_text = next(b for b in msgs[0]["content"] if b["type"] == "text")
    assert user0_text["text"] == "[camera=on] What is in this image?"
    # Assistant: untouched.
    asst0_text = msgs[1]["content"][0]
    assert asst0_text["text"] == "A flower."
    # Second user turn: NO prefix (the gate already fired at turn 0).
    user1_text = msgs[2]["content"][0]
    assert user1_text["text"] == "What colour?"


# ---------------------------------------------------------------------------
# Per-record prefix_key override — future-proofing for multi-axis tags
# ---------------------------------------------------------------------------


def test_prefix_key_override_takes_precedence_over_image_presence() -> None:
    """A record carrying an explicit ``prefix_key`` field bypasses the
    default image-presence dispatch. Use case: a future data-prep stage
    pre-computes multi-axis tags (e.g. ``camera_on_plant_true``) and
    writes them per-record without changing the dispatcher."""
    rec = {
        "image": "/x.jpg",
        "prefix_key": "camera_on_plant_true",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={
            "camera_on":             "[camera=on] ",
            "camera_off":            "[camera=off] ",
            "camera_on_plant_true":  "[camera=on, plant=true] ",
        },
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=on, plant=true] What plant is this?"


def test_prefix_key_override_unknown_key_falls_through_to_empty() -> None:
    """If the override names a key that's missing from the dict, the
    dispatch falls through to "no prefix" — same fallback semantics as
    the default-key path. Guards against typos in pre-baked records."""
    rec = {
        "image": "/x.jpg",
        "prefix_key": "camera_on_planet_true",  # typo: "planet" not "plant"
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={
            "camera_on":             "[camera=on] ",
            "camera_on_plant_true":  "[camera=on, plant=true] ",
        },
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    # No prefix injected (key not found) — image_path is non-empty but
    # the override took precedence over the camera_on default.
    assert text_block["text"] == "What plant is this?"


def test_prefix_key_override_empty_or_missing_uses_image_presence_default() -> None:
    """An empty-string or absent ``prefix_key`` field is treated as
    "no override" — the image-presence default applies. Backward-compat
    guarantee for legacy records that don't know about the field."""
    rec_absent = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    rec_empty = {
        "image": "/x.jpg",
        "prefix_key": "",
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    pp = {"camera_on": "[camera=on] ", "camera_off": "[camera=off] "}
    out_absent = build_vision_messages(rec_absent, prompt_prefixes=pp)
    out_empty = build_vision_messages(rec_empty, prompt_prefixes=pp)
    text_absent = next(b for b in out_absent["messages"][0]["content"] if b["type"] == "text")
    text_empty = next(b for b in out_empty["messages"][0]["content"] if b["type"] == "text")
    assert text_absent["text"] == "[camera=on] What plant is this?"
    assert text_empty["text"] == "[camera=on] What plant is this?"


def test_prefix_preserves_image_placeholder_strip() -> None:
    """Regression guard: the legacy ``<image>\\n`` placeholder strip must
    still fire BEFORE the prefix is prepended, so the output is
    ``[camera=on] What plant ...``, not
    ``[camera=on] <image>\\nWhat plant ...``."""
    rec = {
        "image": "/x.jpg",
        "conversations": [
            {"role": "user", "content": "<image>\nWhat plant is this?"},
            {"role": "assistant", "content": "Acer rubrum."},
        ],
    }
    out = build_vision_messages(
        rec,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )
    text_block = next(b for b in out["messages"][0]["content"] if b["type"] == "text")
    assert text_block["text"] == "[camera=on] What plant is this?"
