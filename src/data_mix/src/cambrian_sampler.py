"""Sample nyu-visionx/Cambrian-10M into two pools:

- 'general': diverse non-plant VQA records, become the 30% bucket.
- 'negative': non-plant images that will be re-prompted with the
  plant-id question by negative_builder.

Plant detection is a best-effort substring match (see
PLANT_NEGATIVE_FILTER_TOKENS) on the concatenated conversation text.
Cambrian has no clean botany label, so this is noise reduction, not a
guarantee.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from data_mix.src.image_resize import persist_pil_to_disk, resize_to_trained_shape
from data_mix.src.schema import validate_record

# Tokens we want to *exclude* from the Cambrian general / negative pools.
# v1 used plain substring matching — false-positive-heavy ("plant" matched
# "plantation"/"plantain"/"transplant", "tree" matched "decision tree",
# "leaf" matched "leaflet"). We now use an explicit allow-list of stem +
# permissible suffixes, anchored at word boundaries.
#
# Kept as a tuple for the unit test that asserts lowercase tokens.
PLANT_NEGATIVE_FILTER_TOKENS = ("plant", "flower", "botan", "tree", "leaf")

# Word-boundary regex with explicit allowed suffixes per stem:
#   plant/plants, flower/flowers, botany/botanical/botanist,
#   tree/trees, leaf/leafy, leaves.
# This intentionally REJECTS "plantation", "plantain", "transplant",
# "treetop", "streetlight", "leaflet" — they share a prefix with a stem
# but aren't plant references. "decision tree" still matches "tree" by
# design (we treat it as ambiguous and err on the side of exclusion).
_PLANT_FILTER_REGEX = re.compile(
    r"\b(?:"
    r"plants?"
    r"|flowers?"
    r"|botan(?:y|ical|ist)"
    r"|trees?"
    r"|leaf(?:y|s)?|leaves"
    r")\b",
    flags=re.IGNORECASE,
)


def is_plant_like(text: str) -> bool:
    """True iff ``text`` contains a plant-related token at word boundaries.

    Matches:  plant(s), flower(s), botany/botanical/botanist, tree(s),
              leaf/leafy, leaves.
    Rejects:  plantation, plantain, transplant, treetop, streetlight,
              leaflet, powerplant.
    """
    return _PLANT_FILTER_REGEX.search(text) is not None


def _row_id(row: dict) -> str:
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        return rid
    # Fallback: hash of concatenated conversation text.
    text = "|".join(
        t.get("content", "") for t in row.get("conversations", [])
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _row_image_pil(row: dict) -> Image.Image | None:
    img = row.get("image")
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and "bytes" in img:
        from io import BytesIO
        return Image.open(BytesIO(img["bytes"]))
    return None


def _first_turn_pair(row: dict) -> tuple[str, str] | None:
    convs = row.get("conversations") or row.get("messages") or []
    if len(convs) < 2:
        return None
    u, a = convs[0], convs[1]
    if (
        u.get("role") == "user"
        and a.get("role") == "assistant"
        and isinstance(u.get("content"), str)
        and isinstance(a.get("content"), str)
        and u["content"]
        and a["content"]
    ):
        return u["content"], a["content"]
    return None


def _persist_image(pil: Image.Image, dest: Path) -> Path:
    """Backward-compat alias of ``persist_pil_to_disk`` for v1 callers."""
    return persist_pil_to_disk(pil, dest)


def sample_cambrian_records(
    stream: Iterable[dict],
    n_general: int,
    n_negative: int,
    image_root: Path,
    seed: int,  # unused; caller seeds the HF stream
) -> Dict[str, list]:
    """Split a Cambrian-like stream into two pools.

    Returns ``{"general": List[dict], "negative": List[Path]}``.

    The two pools have intentionally different element types:

    - ``general`` records are full unified-schema dicts with the
      original Cambrian user/assistant pair and ``source: "cambrian"``;
      they ship straight into the Cambrian bucket.
    - ``negative`` is only a list of resized image paths. The
      orchestrator hands these to ``negative_builder.build_negative_records``
      which constructs fresh records with the fixed refusal prompt +
      response and ``source: "negative"``. The Cambrian conversation
      text for negative seeds is intentionally discarded, so returning
      anything more than the path would be dead metadata.
    """
    image_root = Path(image_root)
    general: List[dict] = []
    negative_paths: List[Path] = []

    for row in stream:
        if len(general) >= n_general and len(negative_paths) >= n_negative:
            break

        pair = _first_turn_pair(row)
        if pair is None:
            continue
        user_text, asst_text = pair
        plantish = is_plant_like(user_text + " " + asst_text)
        if plantish:
            continue  # exclude from both pools

        pil = _row_image_pil(row)
        if pil is None:
            continue

        rid = _row_id(row)

        if len(general) < n_general:
            dst = image_root / "cambrian" / f"{rid}.jpg"
            _persist_image(pil, dst)
            rec = {
                "image": str(dst),
                "conversations": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text},
                ],
                "source": "cambrian",
            }
            validate_record(rec)
            general.append(rec)
        elif len(negative_paths) < n_negative:
            dst = image_root / "negative" / f"{rid}.jpg"
            _persist_image(pil, dst)
            negative_paths.append(dst)

    return {"general": general, "negative": negative_paths}


def open_cambrian_stream(seed: int):
    """Real HF datasets streaming entry point. Lazy-import."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "nyu-visionx/Cambrian-10M",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=5_000)
