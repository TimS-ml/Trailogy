from __future__ import annotations

from pathlib import Path

from PIL import Image

from data_mix.src.dummy_image import (
    DUMMY_IMAGE_BASENAME,
    DUMMY_IMAGE_SIZE_HW,
    ensure_dummy_image,
)


def test_constants_match_trained_shape():
    assert DUMMY_IMAGE_BASENAME == "dummy_gray_960x672.jpg"
    assert DUMMY_IMAGE_SIZE_HW == (960, 672)


def test_ensure_dummy_image_creates_file(tmp_path: Path):
    out = ensure_dummy_image(tmp_path)
    assert out == tmp_path / DUMMY_IMAGE_BASENAME
    assert out.exists()
    with Image.open(out) as im:
        # PIL .size is (width, height); we stored 960x672 (h,w)
        assert im.size == (672, 960)
        assert im.mode == "RGB"
        # All pixels mid-gray
        px = list(im.getdata())[:10]
        for r, g, b in px:
            assert r == g == b == 128


def test_ensure_dummy_image_is_idempotent(tmp_path: Path):
    first = ensure_dummy_image(tmp_path)
    mtime1 = first.stat().st_mtime_ns
    second = ensure_dummy_image(tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == mtime1  # not rewritten
