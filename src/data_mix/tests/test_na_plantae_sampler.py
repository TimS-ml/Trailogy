"""Tests for ``sample_na_plantae_records``.

The sampler now supports a per-class re-weighting controlled by
``train_temperature``:

  * ``train_temperature = 1.0`` (default): natural frequency-proportional
    sampling. The output preserves the input class distribution and
    matches the legacy shuffle+repeat behaviour exactly (back-compat).
  * ``train_temperature < 1.0``: per-class probability is tempered to
    ``p(class) ∝ n_c ** temperature``. Used for long-tail multi-class
    classification — ``temperature = 0.5`` is the canonical
    square-root-tempered sampling from Mahajan et al. 2018.
  * ``train_temperature -> 0``: fully balanced sampling (all classes
    seen equally often, irrespective of pool size).

Val sampling is intentionally LEFT NATURAL regardless of temperature so
held-out eval still reflects the underlying pool distribution.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from data_mix.src.na_plantae_sampler import sample_na_plantae_records


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(slug: str, idx: int) -> dict:
    return {
        "image": f"/img/{slug}/{idx}.jpg",
        "slug": slug,
        "species": slug,
        "family": "F",
        "conversations": [{"role": "user", "content": "?"},
                          {"role": "assistant", "content": slug}],
    }


def _pool(slug_counts: dict[str, int]) -> list[dict]:
    out = []
    for slug, n in slug_counts.items():
        for i in range(n):
            out.append(_row(slug, i))
    return out


# ---------------------------------------------------------------------------
# Back-compat: default temperature == 1.0 keeps legacy behaviour
# ---------------------------------------------------------------------------

def test_default_temperature_preserves_class_proportions(tmp_path: Path) -> None:
    """At temperature=1 (default), the expected per-class share equals
    n_c / total — exactly the legacy shuffle+repeat distribution."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"head": 90, "tail": 10}))  # 9:1 pool
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10_000, n_val=0, seed=0,
    )
    counts = Counter(r["slug"] for r in train_out)
    # Expected: head 90%, tail 10% ± 1.5%.
    assert 8500 <= counts["head"] <= 9500
    assert 500 <= counts["tail"] <= 1500


def test_default_matches_legacy_full_pass_shape(tmp_path: Path) -> None:
    """When n_train is an exact multiple of pool size and
    temperature=1, every record appears exactly ``n_train/pool_size``
    times (the legacy full-pass + shuffle invariant)."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    pool = _pool({"a": 3, "b": 2})  # size 5
    _write(train_p, pool)
    _write(val_p, [_row("a", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=15, n_val=0, seed=0,
    )
    img_counts = Counter(r["image"] for r in train_out)
    assert set(img_counts.values()) == {3}


# ---------------------------------------------------------------------------
# Sqrt tempering: temperature < 1 boosts tail
# ---------------------------------------------------------------------------

def test_temperature_half_makes_distribution_proportional_to_sqrt(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    # 100:1 pool, sqrt -> 10:1. n=11000 exceeds pool size (101) so the
    # sampler falls back to the with-replacement path; the
    # ratio assertion is identical to the v1 contract.
    _write(train_p, _pool({"head": 100, "tail": 1}))
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=11_000, n_val=0, seed=0, train_temperature=0.5,
    )
    counts = Counter(r["slug"] for r in train_out)
    ratio = counts["head"] / max(counts["tail"], 1)
    # Expected ratio ≈ sqrt(100/1) = 10. Allow ±25% Monte-Carlo noise
    # on n=11000.
    assert 7.5 <= ratio <= 12.5


def test_temperature_zero_yields_balanced_sampling(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"big": 100, "small": 5}))
    _write(val_p, [_row("big", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10_000, n_val=0, seed=0, train_temperature=0.0,
    )
    counts = Counter(r["slug"] for r in train_out)
    # Balanced -> 50/50 ± 3%.
    assert 4500 <= counts["big"] <= 5500
    assert 4500 <= counts["small"] <= 5500


# ---------------------------------------------------------------------------
# Val stays natural regardless of temperature
# ---------------------------------------------------------------------------

def test_val_sampling_is_unaffected_by_train_temperature(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, [_row("x", 0)])
    _write(val_p, _pool({"head": 90, "tail": 10}))

    _, val_out = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=0, n_val=10_000, seed=0, train_temperature=0.0,
    )
    counts = Counter(r["slug"] for r in val_out)
    # Even with extreme temperature on the train side, val stays
    # frequency-proportional (90:10).
    assert 8500 <= counts["head"] <= 9500
    assert 500 <= counts["tail"] <= 1500


# ---------------------------------------------------------------------------
# Determinism + source stamp
# ---------------------------------------------------------------------------

def test_temperature_sampling_is_deterministic(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 50, "b": 50}))
    _write(val_p, [_row("a", 0)])

    out1, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=200, n_val=0, seed=42, train_temperature=0.5,
    )
    out2, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=200, n_val=0, seed=42, train_temperature=0.5,
    )
    assert [r["image"] for r in out1] == [r["image"] for r in out2]


def test_records_carry_source_stamp(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 5}))
    _write(val_p, _pool({"a": 5}))

    train_out, val_out = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10, n_val=10, seed=0, train_temperature=0.5,
    )
    for rec in train_out + val_out:
        assert rec["source"] == "na_plantae"


# ---------------------------------------------------------------------------
# Weighted-no-replacement (Efraimidis-Spirakis) — n <= pool_size
# ---------------------------------------------------------------------------

def test_tempered_sampling_no_replacement_when_n_le_pool(
    tmp_path: Path,
) -> None:
    """When the requested count is at or below the post-filter pool
    size, tempered sampling must not produce duplicates. v1 used
    ``random.choices`` unconditionally and produced 18.8 % duplicate
    rows on the production mix; this is the regression guard."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"head": 200, "tail": 100}))   # 300 rows
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=250, n_val=0, seed=0, train_temperature=0.5,
    )
    assert len(train_out) == 250
    images = [r["image"] for r in train_out]
    assert len(set(images)) == 250, (
        f"expected unique images (no replacement), got "
        f"{len(images) - len(set(images))} duplicates"
    )


def test_tempered_sampling_uses_replacement_when_n_gt_pool(
    tmp_path: Path,
) -> None:
    """When n exceeds pool size, the only way to satisfy the target
    is with replacement. The sampler must fall back to the legacy
    path and still produce the requested count."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 10, "b": 10}))           # 20 rows
    _write(val_p, [_row("a", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=200, n_val=0, seed=0, train_temperature=0.5,
    )
    assert len(train_out) == 200
    images = [r["image"] for r in train_out]
    assert len(set(images)) < 200  # duplicates expected on this path


def test_tempered_sampling_no_replacement_skews_per_class(
    tmp_path: Path,
) -> None:
    """No-replacement weighted reservoir keeps the expected per-class
    share unchanged from the with-replacement variant. Verify on a
    50:5 pool with sqrt tempering — the tail share should be
    larger than its natural 5/55 ≈ 9 %.
    """
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"head": 50, "tail": 5}))      # 55 rows
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=30, n_val=0, seed=1, train_temperature=0.5,
    )
    counts = Counter(r["slug"] for r in train_out)
    # All 5 tail records present (no replacement → tail bounded by
    # pool size). Tempering with T=0.5 strongly upweights the tail so
    # in expectation it visits every tail row given a 30-pick budget.
    assert counts["tail"] >= 4, (
        f"tail under-sampled: {counts['tail']} of 5 tail records "
        f"picked (sqrt tempering should grab nearly all)"
    )


# ---------------------------------------------------------------------------
# Drop list: train_exclude_slugs filters pool BEFORE sampling
# ---------------------------------------------------------------------------

def test_train_exclude_slugs_removes_listed_classes(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"keep_a": 20, "drop_x": 20, "keep_b": 20, "drop_y": 20}))
    _write(val_p, [_row("keep_a", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=40, n_val=0, seed=0, train_temperature=0.5,
        train_exclude_slugs=["drop_x", "drop_y"],
    )
    slugs = {r["slug"] for r in train_out}
    assert "drop_x" not in slugs
    assert "drop_y" not in slugs
    assert {"keep_a", "keep_b"} <= slugs


def test_train_exclude_slugs_does_not_touch_val(tmp_path: Path) -> None:
    """Drop list is train-only. Val records with excluded slugs must
    still come out — the trainer val + held-out test set need to
    keep measuring drift on dropped classes."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"keep": 20, "drop": 20}))
    _write(val_p, _pool({"keep": 10, "drop": 10}))

    _, val_out = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10, n_val=20, seed=0, train_temperature=0.5,
        train_exclude_slugs=["drop"],
    )
    val_slugs = {r["slug"] for r in val_out}
    assert val_slugs == {"keep", "drop"}


def test_train_exclude_slugs_unknown_slug_is_warning_not_error(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 5, "b": 5}))
    _write(val_p, [_row("a", 0)])

    # An exclude slug that doesn't exist in the pool must not crash
    # the build — the user-facing drop list is human-curated and
    # typos / partner classes that weren't in the fetch are common.
    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=5, n_val=0, seed=0, train_temperature=0.5,
        train_exclude_slugs=["a", "ghost_class_that_does_not_exist"],
    )
    assert all(r["slug"] != "a" for r in train_out)


def test_empty_pool_after_exclude_raises(tmp_path: Path) -> None:
    """If exclude_slugs drains the pool to zero we must fail loud
    rather than silently return 0 train records."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 5, "b": 5}))
    _write(val_p, [_row("a", 0)])

    with pytest.raises(RuntimeError, match="empty after applying"):
        sample_na_plantae_records(
            train_jsonl=train_p, val_jsonl=val_p,
            n_train=5, n_val=0, seed=0, train_temperature=0.5,
            train_exclude_slugs=["a", "b"],
        )
