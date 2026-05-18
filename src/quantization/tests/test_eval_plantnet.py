"""PlantNet eval — tests that don't require model loading.

We provide a synthetic JSONL + a fake ModelHandle that returns canned
predictions. Verifies aggregation logic and per-sample bookkeeping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.eval.model_loaders import ModelHandle
from src.eval.plantnet import (
    PlantNetConfig,
    _last_assistant_text,
    _strip_trailing_assistant,
    run,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _records_with_string_conv() -> list[dict]:
    # Mirror the JSONL format used by prepare_plantnet.py: conversations
    # with text content as strings, no image content blocks (eval-time
    # image is at top-level "image" path).
    return [
        {
            "image": "/tmp/plant1.jpg",
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant", "content": "This is Quercus robur. Nice find!"},
            ],
        },
        {
            "image": "/tmp/plant2.jpg",
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant", "content": "This is Acer rubrum."},
            ],
        },
    ]


def _fake_handle(predictions: list[str]) -> ModelHandle:
    """Returns one canned prediction per call, in order."""
    state = {"i": 0}

    def infer(messages, image_path=None, max_new_tokens=128):
        i = state["i"]
        state["i"] += 1
        return predictions[i] if i < len(predictions) else ""

    return ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    )


def test_last_assistant_text_string_content():
    conv = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
    ]
    assert _last_assistant_text(conv) == "Hello there"


def test_last_assistant_text_block_content():
    conv = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]
    assert _last_assistant_text(conv) == "Hello!"


def test_strip_trailing_assistant():
    conv = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},  # multiple trailing assistants
    ]
    out = _strip_trailing_assistant(conv)
    assert [m["role"] for m in out] == ["user"]


def test_run_aggregates_species_match(tmp_path, monkeypatch):
    """End-to-end with a fake handle. Skip if the dependency import path
    (``finetune.src.evaluate``) can't be resolved on this box."""
    # Make sure the finetune package is importable as ``finetune.src.evaluate``.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from finetune.src import evaluate as _evaluate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"finetune.src.evaluate not importable: {e}")

    jsonl = tmp_path / "val.jsonl"
    _write_jsonl(jsonl, _records_with_string_conv())

    # Prediction 1 matches species; prediction 2 does not.
    handle = _fake_handle([
        "This is Quercus robur.",         # match
        "This is some other plant.",       # no match
    ])
    cfg = PlantNetConfig(val_jsonl=jsonl)

    # The plantnet runner also tries to load PIL via generate_response
    # → mock that out. The fake handle never calls into the real
    # generate, so this is enough.
    result = run(handle, cfg)

    assert result.n == 2
    assert result.species_matches == 1
    assert 0.4 < result.species_match < 0.6
    assert len(result.per_sample) == 2
    assert result.per_sample[0]["species_match"] is True
    assert result.per_sample[1]["species_match"] is False


def test_run_injects_image_block_into_user_messages(tmp_path):
    """Regression: when the JSONL record carries an ``image`` field but
    the user-turn ``content`` is a plain string (the format produced by
    ``prepare_plantnet.py``), the messages handed to ``handle.infer_text``
    MUST include a ``{"type": "image"}`` content block on the user turn.

    Without this, ``apply_chat_template`` produces a prompt without the
    image-soft-token reservation, and the HF processor errors with::

        Image features and image tokens do not match, tokens: 0, features: 280
    """
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from finetune.src import evaluate as _evaluate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"finetune.src.evaluate not importable: {e}")

    captured: list[dict] = []

    def infer(messages, image_path=None, max_new_tokens=128):
        captured.append({"messages": messages, "image_path": image_path})
        return "This is Quercus robur."

    handle = ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    )

    jsonl = tmp_path / "val.jsonl"
    _write_jsonl(jsonl, _records_with_string_conv()[:1])
    cfg = PlantNetConfig(val_jsonl=jsonl)

    run(handle, cfg)

    assert len(captured) == 1
    sent_messages = captured[0]["messages"]
    # Find the user turn the runner gave to the handle.
    user_turns = [m for m in sent_messages if m.get("role") == "user"]
    assert user_turns, f"no user turn passed to infer_text: {sent_messages!r}"
    user_content = user_turns[-1]["content"]
    # User content must be a list of content blocks (multimodal format),
    # not a plain string. And one of those blocks must be the image
    # placeholder so the chat template reserves soft tokens for it.
    assert isinstance(user_content, list), (
        "user content must be a list of blocks for VLM inference, "
        f"got {type(user_content).__name__}: {user_content!r}"
    )
    has_image_block = any(
        isinstance(b, dict) and b.get("type") == "image" for b in user_content
    )
    assert has_image_block, (
        "user turn missing {'type': 'image'} block — chat template will "
        "produce a prompt without image soft-token reservation, causing "
        "'Image features and image tokens do not match' at inference."
    )
    # And the image path on the runner must still be forwarded.
    assert captured[0]["image_path"] == "/tmp/plant1.jpg"


def test_run_applies_prompt_prefix_camera_on_for_image_records(tmp_path):
    """v4 conditional-FT contract: models trained with
    ``prompt_prefixes={camera_on: "[camera=on] ", ...}`` only respond
    correctly when eval-time prompts carry the same marker. The
    PlantNet runner must accept a ``prompt_prefixes`` config and route
    it through to ``build_vision_messages`` so the first user turn's
    text starts with ``[camera=on] `` for image-bearing records.

    Without this, eval on a v4-trained checkpoint scores ~0 because
    the model never sees the camera-state gate it was trained on.
    Pinned because we hit exactly this on baseline-v2-step4000.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from finetune.src import evaluate as _evaluate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"finetune.src.evaluate not importable: {e}")

    captured: list[dict] = []

    def infer(messages, image_path=None, max_new_tokens=128):
        captured.append({"messages": messages, "image_path": image_path})
        return "This is Quercus robur."

    handle = ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    )

    jsonl = tmp_path / "val.jsonl"
    _write_jsonl(jsonl, _records_with_string_conv()[:1])
    cfg = PlantNetConfig(
        val_jsonl=jsonl,
        prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
    )

    run(handle, cfg)

    assert len(captured) == 1
    sent_messages = captured[0]["messages"]
    user_turns = [m for m in sent_messages if m.get("role") == "user"]
    assert user_turns, f"no user turn passed to infer_text: {sent_messages!r}"
    user_content = user_turns[-1]["content"]
    # First user turn must carry the [camera=on] prefix in its text block.
    text_blocks = [
        b for b in user_content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert text_blocks, f"no text block on user turn: {user_content!r}"
    assert text_blocks[0]["text"].startswith("[camera=on] "), (
        "First user turn must be prefixed with '[camera=on] ' when "
        "the record carries an image and prompt_prefixes is configured; "
        f"got: {text_blocks[0]['text']!r}"
    )


def test_run_applies_prompt_prefix_camera_off_for_text_only_records(tmp_path):
    """Same contract on the text-only branch: a record with
    ``image=None`` should get ``[camera=off] `` (when configured), not
    ``[camera=on] `` and not nothing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from finetune.src import evaluate as _evaluate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"finetune.src.evaluate not importable: {e}")

    captured: list[dict] = []

    def infer(messages, image_path=None, max_new_tokens=128):
        captured.append({"messages": messages, "image_path": image_path})
        return "Hi!"

    handle = ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    )

    # plantnet's load_test_data has require_image=True by default; we
    # exercise the same path with a text-only record by patching
    # require_image off via a custom record list. The plantnet runner
    # itself calls load_test_data(require_image=True), so this branch
    # is reachable only via mix datasets. We still pin the dispatcher
    # contract here so a future enable-text-only flag is correct.
    from src.eval import plantnet as _pn

    text_only_records = [{
        "image": None,
        "conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there."},
        ],
    }]

    # Monkey-patch load_test_data inside the module to return our
    # text-only record set.
    import finetune.src.evaluate as _ev
    orig_load = _ev.load_test_data
    _ev.load_test_data = lambda path, require_image=False: text_only_records
    try:
        jsonl = tmp_path / "val.jsonl"
        jsonl.write_text("")  # path must exist
        cfg = PlantNetConfig(
            val_jsonl=jsonl,
            prompt_prefixes={"camera_on": "[camera=on] ", "camera_off": "[camera=off] "},
        )
        _pn.run(handle, cfg)
    finally:
        _ev.load_test_data = orig_load

    assert len(captured) == 1
    sent_messages = captured[0]["messages"]
    user_turns = [m for m in sent_messages if m.get("role") == "user"]
    assert user_turns
    text_blocks = [
        b for b in user_turns[-1]["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert text_blocks
    assert text_blocks[0]["text"].startswith("[camera=off] "), (
        f"text-only record must get '[camera=off] ' prefix; got: "
        f"{text_blocks[0]['text']!r}"
    )


def test_run_no_prefix_when_prompt_prefixes_none(tmp_path):
    """Bit-identical-by-default: when ``prompt_prefixes`` is unset
    (``None``), the runner must NOT inject any marker. This pins the
    backward-compat path for pre-v4 checkpoints."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from finetune.src import evaluate as _evaluate  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"finetune.src.evaluate not importable: {e}")

    captured: list[dict] = []

    def infer(messages, image_path=None, max_new_tokens=128):
        captured.append({"messages": messages, "image_path": image_path})
        return "This is Quercus robur."

    handle = ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    )

    jsonl = tmp_path / "val.jsonl"
    _write_jsonl(jsonl, _records_with_string_conv()[:1])
    cfg = PlantNetConfig(val_jsonl=jsonl)  # prompt_prefixes defaults to None

    run(handle, cfg)

    user_turns = [m for m in captured[0]["messages"] if m.get("role") == "user"]
    text_blocks = [
        b for b in user_turns[-1]["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert not text_blocks[0]["text"].startswith("[camera="), (
        "default (prompt_prefixes=None) must not inject any camera marker; "
        f"got: {text_blocks[0]['text']!r}"
    )
