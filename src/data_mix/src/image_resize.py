"""Stretch-resize images to the trained vision shape (960x672, H x W).

Intentionally duplicates the logic of
``finetune/src/prepare_plantnet.py::resize_image_to_disk`` to keep the
data_mix module loosely coupled to the finetune subtree. If you change
the trained vision shape, you must update three places (see
hikeCompanion/AGENTS.md "Finetune -> MLX export contract").
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

TRAINED_VISION_HW: Tuple[int, int] = (960, 672)  # (height, width)


def resize_to_trained_shape(src: Path, dst: Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    target_h, target_w = TRAINED_VISION_HW
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        resized = rgb.resize((target_w, target_h), Image.BICUBIC)
        suffix = dst.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            resized.save(dst, format="JPEG", quality=92)
        elif suffix == ".png":
            resized.save(dst, format="PNG", optimize=True)
        elif suffix == ".webp":
            resized.save(dst, format="WEBP", quality=92)
        else:
            resized.save(dst)
    return dst


def persist_pil_to_disk(pil: Image.Image, dest: Path) -> Path:
    """Save a PIL image to ``dest`` after stretch-resizing to trained shape.

    Idempotent: if ``dest`` already exists and is non-empty, returns
    immediately. Cleans up the intermediate ``.raw.jpg`` even on failure.

    Shared between ``cambrian_sampler._persist_image`` (v1) and
    ``llava_sampler`` (v2) — both samplers receive PIL images from
    streaming HF datasets and must materialize them at the trained shape.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".raw.jpg")
    try:
        pil.convert("RGB").save(tmp, format="JPEG", quality=92)
        resize_to_trained_shape(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
