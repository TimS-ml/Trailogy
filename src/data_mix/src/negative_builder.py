"""Build 'negative' bucket records: non-plant image + plant-id prompt
+ fixed refusal response."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from data_mix.src.schema import validate_record

NEGATIVE_PROMPT: str = "What plant species is shown in this image?"
NEGATIVE_RESPONSE: str = (
    "I don't see an identifiable plant in this image. Please provide a "
    "clear image of a plant for identification."
)


def build_negative_records(image_paths: Sequence[Path]) -> List[dict]:
    out: List[dict] = []
    for p in image_paths:
        rec = {
            "image": str(p),
            "conversations": [
                {"role": "user", "content": NEGATIVE_PROMPT},
                {"role": "assistant", "content": NEGATIVE_RESPONSE},
            ],
            "source": "negative",
        }
        validate_record(rec)
        out.append(rec)
    return out
