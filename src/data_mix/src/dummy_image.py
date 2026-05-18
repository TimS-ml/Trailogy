"""Generate the shared mid-gray placeholder image used to bind smoltalk
text-only records into a multimodal-shaped batch.

Why: Gemma4Processor + UnslothVisionDataCollator assert
``len(images) == len(text)`` per batch (see
``finetune/src/data.py``). A constant grey image is the cheapest way to
keep that assertion satisfied while contributing ~zero gradient through
the visual pathway.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

DUMMY_IMAGE_SIZE_HW: Tuple[int, int] = (960, 672)  # (height, width)
DUMMY_IMAGE_BASENAME: str = "dummy_gray_960x672.jpg"


def ensure_dummy_image(image_root: Path) -> Path:
    """Create the dummy image under ``image_root`` if missing.

    Returns the absolute path to the JPEG.
    """
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    out = image_root / DUMMY_IMAGE_BASENAME
    if out.exists() and out.stat().st_size > 0:
        return out
    height, width = DUMMY_IMAGE_SIZE_HW
    im = Image.new("RGB", (width, height), color=(128, 128, 128))
    im.save(out, format="JPEG", quality=92)
    return out
