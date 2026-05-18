"""Sample HuggingFaceH4/llava-instruct-mix-vsft into general + negative pools.

Replaces ``cambrian_sampler`` for v2 (100K / 50K mixes). Key differences:

- LLaVA-mix-vsft schema: ``messages[i].content`` is a list of
  ``{type: "text"|"image", text?: str, index?: int}`` blocks (not a
  flat string). We flatten text-segments per turn into a single string
  and drop image segments (the separate ``images`` field carries the PIL).
- Multi-turn conversations are PRESERVED. iOS deploys 10-turn dialogue
  (`GemmaService` carries `maxHistoryMessages=20`); SFT exposure to
  multi-turn patterns aligns training with deploy.
- **No plant-like filter**. Per user spec, non-plant images don't need
  filtering. LLaVA-mix contains very few plant images anyway, and the
  Negative bucket's refusal template handles incidental leakage.
- Rows with ``len(images) != 1`` are dropped to maintain the single-image
  invariant shared by plant/negative buckets and to keep the data
  collator simple.

Output schema (matches data_mix/src/schema.py):
    {
        "image": "<image_root>/llava/<rid>.jpg",
        "conversations": [
            {"role": "user", "content": "<flattened text>"},
            {"role": "assistant", "content": "..."},
            ... # multi-turn preserved
        ],
        "source": "llava",
    }

Output negative pool: ``List[Path]`` of resized image paths (no records);
``mix.py`` hands them to ``negative_builder.build_negative_records`` which
constructs the refusal template.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from data_mix.src.image_resize import persist_pil_to_disk
from data_mix.src.schema import validate_record


# ---------------------------------------------------------------------------
# Content flattening
# ---------------------------------------------------------------------------

def _flatten_content(content) -> str:
    """LLaVA-mix-vsft content block -> single string.

    Joins all ``{"type": "text", "text": ...}`` segments into one string
    (preserves internal whitespace) and drops ``{"type": "image", ...}``
    segments. Strips leading/trailing whitespace from the joined result.

    Defensive: if ``content`` is already a flat string (cached/older
    splits, future schema drift), return it stripped.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _first_turn_after_flatten(messages: list) -> tuple[str, str] | None:
    """Verify the first two turns are a usable user/assistant pair after
    flattening. Returns ``(user_text, asst_text)`` or ``None`` to skip."""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    u, a = messages[0], messages[1]
    if not (isinstance(u, dict) and isinstance(a, dict)):
        return None
    if u.get("role") != "user" or a.get("role") != "assistant":
        return None
    u_text = _flatten_content(u.get("content"))
    a_text = _flatten_content(a.get("content"))
    if not u_text or not a_text:
        return None
    return u_text, a_text


def _flatten_all_turns(messages: list) -> List[dict] | None:
    """Flatten every turn's content to a string, preserving turn order.

    Returns the unified-schema ``conversations`` list, or ``None`` if any
    turn is malformed / produces an empty string. Empty strings are
    treated as a row-level failure because the schema validator rejects
    them and there's no recovery.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    expected = ("user", "assistant")
    out: List[dict] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        if role != expected[i % 2]:
            # Bail on broken alternation rather than try to repair.
            return None
        text = _flatten_content(msg.get("content"))
        if not text:
            return None
        out.append({"role": role, "content": text})
    # Must end on assistant (even count of turns).
    if len(out) % 2 != 0:
        return None
    return out


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _row_id(row: dict, fallback_text: str) -> str:
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        return rid
    return hashlib.sha1(fallback_text.encode("utf-8")).hexdigest()[:16]


def _row_single_image_pil(row: dict) -> Image.Image | None:
    """Return the single PIL image for this row, or None to skip.

    Enforces the ``len(images) == 1`` invariant. The dataset card and
    sample inspection both showed length-1 in practice, but this guards
    against rare multi-image or no-image rows that would break the
    single-image schema downstream.
    """
    imgs = row.get("images")
    if isinstance(imgs, list) and len(imgs) == 1:
        candidate = imgs[0]
        if isinstance(candidate, Image.Image):
            return candidate
        # HF can hand back dicts with raw bytes in some cached forms.
        if isinstance(candidate, dict) and "bytes" in candidate:
            from io import BytesIO
            return Image.open(BytesIO(candidate["bytes"]))
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_llava_records(
    stream: Iterable[dict],
    n_general: int,
    n_negative: int,
    image_root: Path,
    seed: int,  # unused; caller seeds the HF stream
) -> Dict[str, list]:
    """Pull records from a LLaVA-mix-vsft stream into two pools.

    Returns ``{"general": List[dict], "negative": List[Path]}``. The
    asymmetric return shape matches ``cambrian_sampler.sample_cambrian_records``
    so ``mix.py`` can swap samplers transparently.
    """
    image_root = Path(image_root)
    general: List[dict] = []
    negative_paths: List[Path] = []

    for row in stream:
        if len(general) >= n_general and len(negative_paths) >= n_negative:
            break

        pair = _first_turn_after_flatten(row.get("messages", []))
        if pair is None:
            continue
        user_text, _ = pair

        pil = _row_single_image_pil(row)
        if pil is None:
            continue

        rid = _row_id(row, fallback_text=user_text)

        if len(general) < n_general:
            convs = _flatten_all_turns(row.get("messages", []))
            if convs is None:
                continue
            dst = image_root / "llava" / f"{rid}.jpg"
            persist_pil_to_disk(pil, dst)
            rec = {
                "image": str(dst),
                "conversations": convs,
                "source": "llava",
            }
            validate_record(rec)
            general.append(rec)
        elif len(negative_paths) < n_negative:
            dst = image_root / "negative" / f"{rid}.jpg"
            persist_pil_to_disk(pil, dst)
            negative_paths.append(dst)

    return {"general": general, "negative": negative_paths}


def open_llava_stream(seed: int):
    """Real HF streaming entry point. Lazy import so unit tests don't
    require network or the ``datasets`` install path."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "HuggingFaceH4/llava-instruct-mix-vsft",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=5_000)
