"""Tests for src.prepare_plantnet — image discovery, sampling, JSONL emission."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

# prepare_plantnet is run as a script; import its functions directly.
from src.prepare_plantnet import (
    DEFAULT_TRAINED_VISION_HW,
    build_conversation,
    class_stratified_split,
    discover_images,
    load_species_map,
    parse_resize_arg,
    resize_image_to_disk,
    stratified_sample,
)


def _make_plantnet_tree(root: Path, species: dict[str, int]) -> None:
    """Create a fake PlantNet train/ tree with `species → count` images."""
    train = root / "train"
    train.mkdir()
    for sid, n in species.items():
        d = train / sid
        d.mkdir()
        for i in range(n):
            (d / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic


def test_discover_images_finds_all(tmp_path: Path) -> None:
    _make_plantnet_tree(tmp_path, {"sp_1": 3, "sp_2": 5})
    classes = discover_images(tmp_path / "train")
    assert set(classes.keys()) == {"sp_1", "sp_2"}
    assert len(classes["sp_1"]) == 3
    assert len(classes["sp_2"]) == 5


def test_discover_images_ignores_non_images(tmp_path: Path) -> None:
    _make_plantnet_tree(tmp_path, {"sp_1": 2})
    (tmp_path / "train" / "sp_1" / "notes.txt").write_text("ignore me")
    classes = discover_images(tmp_path / "train")
    assert len(classes["sp_1"]) == 2


def test_stratified_sample_round_robins() -> None:
    rng = random.Random(0)
    classes = {
        "a": [Path(f"a/{i}.jpg") for i in range(5)],
        "b": [Path(f"b/{i}.jpg") for i in range(5)],
        "c": [Path(f"c/{i}.jpg") for i in range(5)],
    }
    samples = stratified_sample(classes, max_samples=6, rng=rng)
    assert len(samples) == 6
    # Round-robin: should hit each class at least once before repeats.
    first_three = {sid for sid, _ in samples[:3]}
    assert first_three == {"a", "b", "c"}


def test_stratified_sample_caps_at_total() -> None:
    rng = random.Random(0)
    classes = {"a": [Path("a/0.jpg"), Path("a/1.jpg")]}
    samples = stratified_sample(classes, max_samples=100, rng=rng)
    assert len(samples) == 2  # capped at available


# ---------------------------------------------------------------------------
# class_stratified_split — per-class val carve-out
#
# Replaces the v1 random `samples[:val_count]` slice. v1 left up to 23%
# of species absent from val (random tail of long-tail dataset misses
# sparse classes). v2 carves out per-species so val species set is
# guaranteed a subset of train species set.
# ---------------------------------------------------------------------------

def _samples_from_classes(classes: dict[str, list[Path]]) -> list[tuple[str, Path]]:
    """Flatten a {sid: [paths]} dict into the (sid, path) tuple list shape
    that class_stratified_split consumes (matches stratified_sample output)."""
    return [(sid, p) for sid, paths in classes.items() for p in paths]


def test_class_stratified_split_val_species_equals_train_species() -> None:
    """The whole point of the new split: every species in train is also
    in val (no orphan species). This guarantees the SFT sees and is
    evaluated on the same vocabulary of classes."""
    rng = random.Random(0)
    classes = {
        f"sp_{i:03d}": [Path(f"sp_{i:03d}/{j}.jpg") for j in range(10)]
        for i in range(50)
    }
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.1, rng=rng)
    train_species = {sid for sid, _ in train}
    val_species = {sid for sid, _ in val}
    assert val_species == train_species
    assert val_species == set(classes.keys())


def test_class_stratified_split_per_class_density_proportional() -> None:
    """Each species contributes floor(N * val_ratio) (min 1) to val.

    For uniform N=10 / val_ratio=0.1: every species contributes 1 to val,
    9 to train.
    """
    rng = random.Random(0)
    classes = {
        f"sp_{i:03d}": [Path(f"sp_{i:03d}/{j}.jpg") for j in range(10)]
        for i in range(50)
    }
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.1, rng=rng)

    from collections import Counter
    train_by_sid = Counter(sid for sid, _ in train)
    val_by_sid = Counter(sid for sid, _ in val)
    for sid in classes:
        assert val_by_sid[sid] == 1, (
            f"species {sid}: val={val_by_sid[sid]} (expected 1)"
        )
        assert train_by_sid[sid] == 9, (
            f"species {sid}: train={train_by_sid[sid]} (expected 9)"
        )


def test_class_stratified_split_single_image_species_goes_to_train_only() -> None:
    """Species with only 1 image cannot contribute to both train and val
    without leakage. v2 policy: train only (so the model still sees the
    rare class). val species set therefore EXCLUDES single-image species.

    Trade-off accepted: this means val species set is slightly smaller
    than the full species set when some species are truly single-image
    in the pool. But the asymmetry is honest (test-on-unseen would be
    impossible) and dwarfed by v1's much larger 23% coverage gap.
    """
    rng = random.Random(0)
    classes = {
        "common":     [Path(f"common/{i}.jpg") for i in range(10)],
        "rare":       [Path("rare/0.jpg")],     # single image
    }
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.1, rng=rng)

    train_species = {sid for sid, _ in train}
    val_species = {sid for sid, _ in val}
    assert "rare" in train_species
    assert "rare" not in val_species
    assert "common" in train_species
    assert "common" in val_species


def test_class_stratified_split_disjoint_images() -> None:
    """An image cannot appear in both train and val (no leakage)."""
    rng = random.Random(0)
    classes = {
        f"sp_{i:03d}": [Path(f"sp_{i:03d}/{j}.jpg") for j in range(5)]
        for i in range(20)
    }
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.2, rng=rng)
    train_paths = {p for _, p in train}
    val_paths = {p for _, p in val}
    assert train_paths.isdisjoint(val_paths)
    # And together they reconstitute the input (no records dropped).
    assert train_paths | val_paths == {p for _, p in samples}


def test_class_stratified_split_is_deterministic() -> None:
    """Same input + same seed -> bit-identical split.

    Required for the 'byte-identical JSONL across machines' invariant
    in prepare_plantnet_50k.sh.
    """
    classes = {
        f"sp_{i:03d}": [Path(f"sp_{i:03d}/{j}.jpg") for j in range(5)]
        for i in range(20)
    }
    samples = _samples_from_classes(classes)
    rng_a = random.Random(42)
    train_a, val_a = class_stratified_split(samples, val_ratio=0.2, rng=rng_a)
    rng_b = random.Random(42)
    train_b, val_b = class_stratified_split(samples, val_ratio=0.2, rng=rng_b)
    assert train_a == train_b
    assert val_a == val_b


def test_class_stratified_split_long_tail_distribution() -> None:
    """Approximates the real PlantNet long-tail (1 / 27 / 146 imgs/class).
    Verifies the algorithm survives the actual shape we'll feed it."""
    rng = random.Random(0)
    # 100 head species at 50 imgs + 200 mid species at 20 imgs + 100 tail
    # species at 2 imgs + 100 singletons.
    classes: dict[str, list[Path]] = {}
    for i in range(100):
        classes[f"head_{i:03d}"] = [Path(f"head_{i:03d}/{j}.jpg") for j in range(50)]
    for i in range(200):
        classes[f"mid_{i:03d}"]  = [Path(f"mid_{i:03d}/{j}.jpg") for j in range(20)]
    for i in range(100):
        classes[f"tail_{i:03d}"] = [Path(f"tail_{i:03d}/{j}.jpg") for j in range(2)]
    for i in range(100):
        classes[f"sing_{i:03d}"] = [Path(f"sing_{i:03d}/0.jpg")]

    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.1, rng=rng)

    train_species = {sid for sid, _ in train}
    val_species = {sid for sid, _ in val}

    # Head + mid + tail species (each with >=2 imgs) must all be in val.
    # Singletons go to train only.
    expected_val_species = {sid for sid in classes if len(classes[sid]) >= 2}
    assert val_species == expected_val_species
    assert len(val_species) == 400  # 100 head + 200 mid + 100 tail
    # Singletons (100) are in train only.
    assert all(f"sing_{i:03d}" in train_species for i in range(100))


def test_class_stratified_split_val_ratio_zero_returns_all_train() -> None:
    """val_ratio=0 = no val carved out. Useful when the caller wants to
    later concatenate with an externally-provided val."""
    rng = random.Random(0)
    classes = {"a": [Path(f"a/{i}.jpg") for i in range(10)]}
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=0.0, rng=rng)
    assert len(train) == 10
    assert len(val) == 0


def test_class_stratified_split_val_ratio_one_returns_all_val() -> None:
    """val_ratio=1.0 = everything to val, train empty.

    This is the symmetric edge case to val_ratio=0.0 and is used by
    ``prepare_plantnet_50k.sh`` Step 2 to promote ALL of PlantNet
    ``test/`` to ``test.jsonl`` via a staging dir. Without this branch,
    the per-species "leave ≥1 in train" rule + "single-image → train
    only" rule silently divert some test images to the staging-dir
    train.jsonl (which the shell script discards), and we lose those
    test records — especially every single-image species.
    """
    rng = random.Random(0)
    classes = {
        "multi": [Path(f"multi/{i}.jpg") for i in range(10)],
        "pair": [Path("pair/0.jpg"), Path("pair/1.jpg")],
        "single_a": [Path("single_a/0.jpg")],
        "single_b": [Path("single_b/0.jpg")],
    }
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=1.0, rng=rng)
    # All 14 records end up in val; train is empty.
    assert len(train) == 0
    assert len(val) == 14
    # All species are represented in val (singletons included).
    val_species = {sid for sid, _ in val}
    assert val_species == {"multi", "pair", "single_a", "single_b"}


def test_class_stratified_split_val_ratio_one_preserves_singletons() -> None:
    """Specific regression test: at val_ratio=1.0, single-image species
    must NOT be diverted to train. The first version of v2 applied the
    'single-image → train only' rule unconditionally; that broke Step 2
    of prepare_plantnet_50k.sh (test.jsonl lost ~6% of species).
    """
    rng = random.Random(0)
    classes = {f"sing_{i:03d}": [Path(f"sing_{i:03d}/0.jpg")] for i in range(100)}
    samples = _samples_from_classes(classes)
    train, val = class_stratified_split(samples, val_ratio=1.0, rng=rng)
    assert len(train) == 0
    assert len(val) == 100


def test_build_conversation_no_image_placeholder() -> None:
    rng = random.Random(0)
    rec = build_conversation(Path("/tmp/x.jpg"), "Acer rubrum", rng)
    assert "image" in rec
    assert rec["conversations"][0]["role"] == "user"
    # No legacy <image> placeholder — we now express the image as a content
    # block in src.data.build_vision_messages.
    assert "<image>" not in rec["conversations"][0]["content"]
    # Species name is in the assistant turn.
    assert "Acer rubrum" in rec["conversations"][1]["content"]


def test_load_species_map_handles_missing(tmp_path: Path, caplog) -> None:
    out = load_species_map(str(tmp_path / "nope.json"))
    assert out == {}


def test_load_species_map_returns_dict(tmp_path: Path) -> None:
    p = tmp_path / "sp.json"
    p.write_text(json.dumps({"123": "Acer rubrum", "456": "Quercus alba"}))
    out = load_species_map(str(p))
    assert out["123"] == "Acer rubrum"


def test_built_record_round_trips_through_data_module(tmp_path: Path) -> None:
    """End-to-end: prepare_plantnet output → build_vision_messages."""
    from src.data import build_vision_messages

    rng = random.Random(42)
    rec = build_conversation(Path("/x.jpg"), "Test species", rng)
    msgs = build_vision_messages(rec)["messages"]
    # First user msg has image + text blocks.
    user = msgs[0]
    assert any(b["type"] == "image" for b in user["content"])
    assert any(b["type"] == "text" for b in user["content"])
    # Assistant msg has only text.
    asst = msgs[1]
    assert asst["content"] == [{"type": "text", "text": rec["conversations"][1]["content"]}]


# ---------------------------------------------------------------------------
# parse_resize_arg
# ---------------------------------------------------------------------------


def test_default_trained_vision_hw_matches_ios_runtime() -> None:
    """The default must match scripts/fetch-gemma.sh and export_mlx.py."""
    # 960 height × 672 width — matches the app model-fetch script patch.
    assert DEFAULT_TRAINED_VISION_HW == (960, 672)


def test_parse_resize_arg_default_format() -> None:
    assert parse_resize_arg("960x672") == (960, 672)
    assert parse_resize_arg("960X672") == (960, 672)
    assert parse_resize_arg("  960x672  ") == (960, 672)


def test_parse_resize_arg_alt_separators() -> None:
    assert parse_resize_arg("960*672") == (960, 672)
    assert parse_resize_arg("960,672") == (960, 672)


@pytest.mark.parametrize("disabled", ["none", "None", "off", "false", "no", "", None])
def test_parse_resize_arg_disable_values(disabled) -> None:
    assert parse_resize_arg(disabled) is None


@pytest.mark.parametrize("bad", ["nope", "960", "960x", "x672", "abc x def", "0x100", "100x-50"])
def test_parse_resize_arg_invalid_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_resize_arg(bad)


# ---------------------------------------------------------------------------
# resize_image_to_disk — needs PIL but exercises a tiny synthetic image.
# ---------------------------------------------------------------------------


def _write_solid_jpeg(path: Path, size_wh: tuple[int, int], color=(120, 200, 80)) -> None:
    """Write a tiny solid-color JPEG at `size_wh` (PIL ordering: width, height)."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", size_wh, color).save(path, format="JPEG", quality=80)


def test_resize_image_stretches_to_target(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _write_solid_jpeg(src, size_wh=(320, 200))  # 320 wide × 200 tall

    dest = tmp_path / "out" / "resized.jpg"
    out_path = resize_image_to_disk(src, dest, target_hw=(960, 672))

    assert out_path == dest
    assert dest.exists()

    from PIL import Image
    with Image.open(dest) as im:
        # PIL .size is (width, height); we expect 672 × 960.
        assert im.size == (672, 960)


def test_resize_image_skips_when_dest_present(tmp_path: Path) -> None:
    """Re-runs of the prep script should be incremental — resized files are
    treated as a cache."""
    src = tmp_path / "src.jpg"
    _write_solid_jpeg(src, size_wh=(100, 100))
    dest = tmp_path / "cached.jpg"
    _write_solid_jpeg(dest, size_wh=(672, 960))  # pre-existing "cached" file

    cached_size = dest.stat().st_size
    cached_mtime = dest.stat().st_mtime

    out = resize_image_to_disk(src, dest, target_hw=(960, 672))
    assert out == dest
    # File was not rewritten — same size + mtime.
    assert dest.stat().st_size == cached_size
    assert dest.stat().st_mtime == cached_mtime


def test_resize_image_creates_parent_dirs(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _write_solid_jpeg(src, size_wh=(100, 100))

    dest = tmp_path / "deeply" / "nested" / "dir" / "img.jpg"
    resize_image_to_disk(src, dest, target_hw=(960, 672))
    assert dest.exists()


def test_resize_image_converts_alpha_to_rgb(tmp_path: Path) -> None:
    """RGBA / palette inputs must come out as RGB (matches HF do_convert_rgb=True)."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "rgba.png"
    Image.new("RGBA", (50, 80), (10, 20, 30, 128)).save(src, format="PNG")

    dest = tmp_path / "out.jpg"
    resize_image_to_disk(src, dest, target_hw=(960, 672))

    with Image.open(dest) as im:
        assert im.mode == "RGB"
        assert im.size == (672, 960)
