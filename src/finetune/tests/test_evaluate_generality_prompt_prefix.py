"""Tests for ``eval/evaluate_generality_plantnet.py``'s v4 camera-state
prompt-prefix dispatch.

Context: models trained with the v4 ``data.prompt_prefixes`` contract
see ``[camera=on] `` on every image-bearing user prompt and
``[camera=off] `` on every text-only user prompt during training. At
eval time those markers must be re-injected with the same dispatch
rule, or the model behaves like base.

Image-bearing eval domains (plant / llava / refusal) historically got
``[camera=on] `` via the single ``--prompt_prefix`` flag. Text-only
eval domains (mmlu / aime / text_chat) got nothing — which corrupts
the catastrophic-forgetting story because the canary scores reflect
the model on out-of-distribution input gates, not on its real
behaviour.

These tests pin the v4 dispatch behaviour for ``generate_response``
and the per-domain wiring in ``main``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make ``finetune`` importable as a top-level package.
_FT_ROOT = Path(__file__).resolve().parents[1]
if str(_FT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FT_ROOT))

from eval import evaluate_generality_plantnet as eg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(user_text: str, image_path: str | None) -> dict:
    rec: dict = {
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "ok"},
        ]
    }
    if image_path is not None:
        rec["image"] = image_path
    return rec


def _capture_user_content_via_handle():
    """Returns (handle, captured_list). The handle.infer_text records
    every messages list it's called with so the test can inspect what
    the runner sent."""
    captured: list[dict] = []

    def infer(messages=None, image_path=None, max_new_tokens=128, **_kw):
        captured.append({"messages": messages, "image_path": image_path})
        return "stub response"

    handle = MagicMock()
    handle.infer_text = infer
    return handle, captured


# ---------------------------------------------------------------------------
# generate_response — camera_on dispatch (image present)
# ---------------------------------------------------------------------------


def test_generate_response_injects_camera_on_for_image_record():
    """An image-bearing record must get ``[camera=on] `` prepended to
    the user text when ``prompt_prefix_camera_on`` is set."""
    handle, captured = _capture_user_content_via_handle()
    rec = _record("What plant is this?", image_path="/tmp/plant.jpg")

    eg.generate_response(
        handle,
        backend="hf_quant",  # any path that exposes infer_text
        record=rec,
        max_new_tokens=64,
        prompt_prefix_camera_on="[camera=on] ",
        prompt_prefix_camera_off="[camera=off] ",
    )

    assert len(captured) == 1
    msgs = captured[0]["messages"]
    user_text = _extract_user_text(msgs)
    assert user_text.startswith("[camera=on] "), (
        f"image-bearing record must be prefixed with '[camera=on] '; "
        f"got: {user_text!r}"
    )


# ---------------------------------------------------------------------------
# generate_response — camera_off dispatch (no image)
# ---------------------------------------------------------------------------


def test_generate_response_injects_camera_off_for_text_only_record():
    """A text-only record (no ``image`` key, or image_path=None) MUST
    get ``[camera=off] `` prepended when configured. This is the bug
    that drove the v4 fix: mmlu / aime / text_chat domains were
    getting raw text, so v4-trained models scored on
    out-of-distribution input gates rather than their real behaviour.
    """
    handle, captured = _capture_user_content_via_handle()
    rec = _record("What is the capital of France?", image_path=None)

    eg.generate_response(
        handle,
        backend="hf_quant",
        record=rec,
        max_new_tokens=64,
        prompt_prefix_camera_on="[camera=on] ",
        prompt_prefix_camera_off="[camera=off] ",
    )

    assert len(captured) == 1
    user_text = _extract_user_text(captured[0]["messages"])
    assert user_text.startswith("[camera=off] "), (
        f"text-only record must be prefixed with '[camera=off] '; "
        f"got: {user_text!r}"
    )


def test_generate_response_camera_off_only_branch_left_alone_for_image():
    """When only ``prompt_prefix_camera_off`` is supplied (asymmetric
    ablation), an image-bearing record must NOT pick up the off-marker
    by accident — it gets no prefix."""
    handle, captured = _capture_user_content_via_handle()
    rec = _record("What plant?", image_path="/tmp/plant.jpg")

    eg.generate_response(
        handle,
        backend="hf_quant",
        record=rec,
        max_new_tokens=64,
        prompt_prefix_camera_on=None,
        prompt_prefix_camera_off="[camera=off] ",
    )

    user_text = _extract_user_text(captured[0]["messages"])
    assert not user_text.startswith("[camera=") and user_text.startswith("What plant?"), (
        f"image record must not get camera_off prefix; got: {user_text!r}"
    )


def test_generate_response_camera_on_only_branch_left_alone_for_text():
    """Inverse of the above: only ``camera_on`` set → text-only record
    has no prefix, not ``[camera=on] `` (modality mismatch would be
    worse than nothing)."""
    handle, captured = _capture_user_content_via_handle()
    rec = _record("Hi", image_path=None)

    eg.generate_response(
        handle,
        backend="hf_quant",
        record=rec,
        max_new_tokens=64,
        prompt_prefix_camera_on="[camera=on] ",
        prompt_prefix_camera_off=None,
    )

    user_text = _extract_user_text(captured[0]["messages"])
    assert not user_text.startswith("[camera="), (
        f"text-only record must not be tagged with camera_on; got: {user_text!r}"
    )


# ---------------------------------------------------------------------------
# Backward-compat: legacy ``prompt_prefix`` arg still treated as
# camera_on (image-only). Pre-v4 callers passing the old single string
# must continue to work bit-identically.
# ---------------------------------------------------------------------------


def test_generate_response_legacy_prompt_prefix_arg_treated_as_camera_on():
    """Old call sites (sweep scripts before this commit) pass a single
    ``prompt_prefix="[camera=on] "`` kwarg. The wrapper must keep
    treating that as the image-only camera_on prefix for
    backward-compat — no changes to existing run scripts required."""
    handle, captured = _capture_user_content_via_handle()
    rec_img = _record("What plant?", image_path="/tmp/plant.jpg")
    rec_txt = _record("Hi", image_path=None)

    eg.generate_response(
        handle, backend="hf_quant", record=rec_img,
        max_new_tokens=64, prompt_prefix="[camera=on] ",
    )
    eg.generate_response(
        handle, backend="hf_quant", record=rec_txt,
        max_new_tokens=64, prompt_prefix="[camera=on] ",
    )

    assert _extract_user_text(captured[0]["messages"]).startswith("[camera=on] ")
    # Text-only branch with legacy arg → no prefix (preserves pre-v4 behaviour).
    text_only = _extract_user_text(captured[1]["messages"])
    assert not text_only.startswith("[camera="), (
        f"legacy prompt_prefix must NOT leak onto text-only records; "
        f"got: {text_only!r}"
    )


def test_generate_response_no_prefix_when_both_none():
    """Bit-identical-by-default: both flags ``None`` → no injection,
    matching pre-v4 behaviour for old checkpoints."""
    handle, captured = _capture_user_content_via_handle()
    rec = _record("What plant?", image_path="/tmp/plant.jpg")

    eg.generate_response(
        handle, backend="hf_quant", record=rec, max_new_tokens=64,
    )

    user_text = _extract_user_text(captured[0]["messages"])
    assert user_text == "What plant?"


# ---------------------------------------------------------------------------
# CLI plumbing — ``--prompt_prefix_camera_off`` flag exists
# ---------------------------------------------------------------------------


def test_cli_help_lists_prompt_prefix_camera_off_flag(capsys):
    """``--help`` must advertise the new flag so operators discover it.

    Using ``--help`` exits with SystemExit(0) after writing usage to
    stdout; that's the cheapest way to verify argparse wiring without
    coupling the test to the rest of ``main()``'s body.
    """
    with pytest.raises(SystemExit) as excinfo:
        # Re-using the real parser via main() is the contract that
        # matters — operators run this CLI, not a re-built parser.
        old_argv = sys.argv
        try:
            sys.argv = ["evaluate_generality_plantnet.py", "--help"]
            eg.main()
        finally:
            sys.argv = old_argv
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    assert "--prompt_prefix_camera_off" in help_text, (
        "argparse --help must list --prompt_prefix_camera_off so "
        "sweep scripts (and humans) can discover the v4 text-only "
        f"flag. Got: {help_text!r}"
    )
    # And the legacy --prompt_prefix flag stays for backward-compat.
    assert "--prompt_prefix " in help_text or "--prompt_prefix\n" in help_text, (
        "legacy --prompt_prefix flag must still appear in --help"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_text(messages: list[dict]) -> str:
    """Pull the first user turn's text out of a messages list,
    tolerating both string-content and content-block list shapes."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        return ""
    return ""
