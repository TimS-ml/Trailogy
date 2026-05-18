from __future__ import annotations

from pathlib import Path

from PIL import Image

from data_mix.src.image_resize import TRAINED_VISION_HW, resize_to_trained_shape


def test_constant_matches_spec():
    assert TRAINED_VISION_HW == (960, 672)


def test_resize_writes_target_shape(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (1234, 567), color=(20, 30, 40)).save(src, "JPEG", quality=90)
    dst = tmp_path / "out.jpg"
    resize_to_trained_shape(src, dst)
    with Image.open(dst) as im:
        assert im.size == (672, 960)  # (w, h)


def test_resize_skips_if_exists_and_nonempty(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(src, "JPEG", quality=90)
    dst = tmp_path / "out.jpg"
    Image.new("RGB", (50, 50), color=(255, 255, 255)).save(dst, "JPEG", quality=90)
    mtime_before = dst.stat().st_mtime_ns
    resize_to_trained_shape(src, dst)
    assert dst.stat().st_mtime_ns == mtime_before  # untouched


def test_resize_creates_parent_dir(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (10, 10), color=(1, 2, 3)).save(src, "JPEG", quality=90)
    dst = tmp_path / "nested" / "deeper" / "out.jpg"
    resize_to_trained_shape(src, dst)
    assert dst.exists()
