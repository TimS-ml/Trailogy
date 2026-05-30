"""Tests for vit_baseline.dataset pure scanning / class-map helpers (no torch)."""

from __future__ import annotations

import pytest

from vit_baseline.dataset import build_class_to_idx, list_class_dirs, scan_split


def _make_imagefolder(root, layout):
    """layout: {split: {class: n_images}} -> writes empty .jpg files."""
    for split, classes in layout.items():
        for cls, n in classes.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(b"")


def test_build_class_to_idx_is_sorted_and_stable(tmp_path) -> None:
    _make_imagefolder(tmp_path, {"train": {"zebra_plant": 2, "agarita": 1, "maple": 3}})
    c2i = build_class_to_idx(tmp_path / "train")
    assert list(c2i.keys()) == ["agarita", "maple", "zebra_plant"]
    assert c2i == {"agarita": 0, "maple": 1, "zebra_plant": 2}


def test_scan_split_counts_and_labels(tmp_path) -> None:
    _make_imagefolder(tmp_path, {"train": {"a": 2, "b": 3}})
    c2i = build_class_to_idx(tmp_path / "train")
    samples = scan_split(tmp_path / "train", c2i)
    assert len(samples) == 5
    labels = sorted(lbl for _, lbl in samples)
    assert labels == [0, 0, 1, 1, 1]


def test_scan_split_drops_unknown_val_classes(tmp_path) -> None:
    _make_imagefolder(tmp_path, {"train": {"a": 1, "b": 1}, "val": {"a": 1, "b": 1, "ghost": 4}})
    c2i = build_class_to_idx(tmp_path / "train")
    val = scan_split(tmp_path / "val", c2i)
    # ghost is not in train; its 4 images must be skipped.
    assert len(val) == 2
    assert all(lbl in (0, 1) for _, lbl in val)


def test_scan_split_caps(tmp_path) -> None:
    _make_imagefolder(tmp_path, {"train": {"a": 10, "b": 10}})
    c2i = build_class_to_idx(tmp_path / "train")
    assert len(scan_split(tmp_path / "train", c2i, max_images_per_class=3)) == 6
    assert len(scan_split(tmp_path / "train", c2i, max_samples=5)) == 5


def test_list_class_dirs_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list_class_dirs(tmp_path / "nope")
