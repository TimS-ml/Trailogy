"""Tests for the per-species image-count caps applied by
``prepare_na_plantae._apply_species_caps``.

The cap layer runs AFTER ``_process_split`` / ``_process_flat`` so it
operates uniformly on the post-rollup binomial slug, regardless of how
many variety-level sub-folders contributed images to a parent species.

Semantics:
  * ``min_imgs_per_species``: drop every row of any species whose total
    (train + val + test) count is strictly less than this threshold.
    Tail classes too small to learn from cleanly are removed entirely
    rather than left in the corpus to be memorized.
  * ``max_imgs_per_species``: trim train rows (preserving val + test)
    so the species' total count is at most this value. Head dominance
    is reduced without disturbing the held-out splits.
"""
from __future__ import annotations

import random

from data_mix.src import prepare_na_plantae as prep


def _rec(slug: str, idx: int) -> dict:
    return {
        "image": f"/fake/{slug}/{idx}.jpg",
        "slug": slug,
        "species": slug.replace("_", " "),
        "family": "Testaceae",
        "conversations": [{"role": "user", "content": "?"},
                          {"role": "assistant", "content": "ans"}],
    }


def _build_splits(counts: dict[str, dict[str, int]]) -> dict[str, list[dict]]:
    """counts[slug][split] = N rows."""
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for slug, per in counts.items():
        for split, n in per.items():
            for i in range(n):
                splits[split].append(_rec(slug, i))
    return splits


def _build_summary(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        slug: {**per, "total": sum(per.values())}
        for slug, per in counts.items()
    }


# ---------------------------------------------------------------------------
# Drop-below-min
# ---------------------------------------------------------------------------

def test_drops_species_below_min_imgs() -> None:
    counts = {
        "small_class":   {"train": 1, "val": 0, "test": 1},   # total 2
        "ok_class":      {"train": 28, "val": 5, "test": 5},  # total 38
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=30, max_imgs=10**9, rng=random.Random(0),
    )
    surviving = {r["slug"] for s in out_splits.values() for r in s}
    assert surviving == {"ok_class"}
    assert "small_class" not in out_summary
    assert out_summary["ok_class"]["total"] == 38


def test_no_min_threshold_keeps_everything() -> None:
    counts = {
        "a": {"train": 1, "val": 0, "test": 0},
        "b": {"train": 2, "val": 0, "test": 0},
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=0, max_imgs=10**9, rng=random.Random(0),
    )
    assert sum(len(v) for v in out_splits.values()) == 3
    assert set(out_summary) == {"a", "b"}


# ---------------------------------------------------------------------------
# Cap-above-max
# ---------------------------------------------------------------------------

def test_caps_species_above_max_imgs_trims_train_first() -> None:
    counts = {
        "huge": {"train": 380, "val": 10, "test": 10},   # total 400
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=0, max_imgs=120, rng=random.Random(7),
    )
    # Val + test untouched.
    assert sum(1 for r in out_splits["val"] if r["slug"] == "huge") == 10
    assert sum(1 for r in out_splits["test"] if r["slug"] == "huge") == 10
    # Train trimmed so total == max.
    assert sum(1 for r in out_splits["train"] if r["slug"] == "huge") == 100
    assert out_summary["huge"]["total"] == 120
    assert out_summary["huge"]["train"] == 100


def test_cap_no_op_when_total_below_max() -> None:
    counts = {
        "mid": {"train": 50, "val": 5, "test": 5},  # total 60
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=0, max_imgs=120, rng=random.Random(0),
    )
    assert sum(1 for r in out_splits["train"] if r["slug"] == "mid") == 50
    assert out_summary["mid"]["total"] == 60


def test_cap_when_val_test_alone_exceed_max() -> None:
    """Edge case: val+test already exceed max. Train trimmed to 0 but
    val/test are NEVER touched (held-out integrity is non-negotiable)."""
    counts = {
        "weird": {"train": 50, "val": 80, "test": 80},  # total 210, val+test=160
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=0, max_imgs=120, rng=random.Random(0),
    )
    assert sum(1 for r in out_splits["train"] if r["slug"] == "weird") == 0
    assert sum(1 for r in out_splits["val"] if r["slug"] == "weird") == 80
    assert sum(1 for r in out_splits["test"] if r["slug"] == "weird") == 80


# ---------------------------------------------------------------------------
# Multi-species + determinism
# ---------------------------------------------------------------------------

def test_combined_min_and_max_filter_multi_species() -> None:
    counts = {
        "tiny":   {"train": 5,   "val": 1,  "test": 0},   # total 6  -> drop (<30)
        "ok":     {"train": 60,  "val": 5,  "test": 5},   # total 70 -> keep as-is
        "huge":   {"train": 300, "val": 10, "test": 10},  # total 320 -> cap to 120
    }
    splits = _build_splits(counts)
    summary = _build_summary(counts)
    out_splits, out_summary = prep._apply_species_caps(
        splits, summary, min_imgs=30, max_imgs=120, rng=random.Random(1),
    )
    assert set(out_summary) == {"ok", "huge"}
    assert out_summary["ok"]["total"] == 70
    assert out_summary["huge"]["total"] == 120


def test_apply_caps_is_deterministic_under_same_seed() -> None:
    counts = {
        "h": {"train": 200, "val": 5, "test": 5},
    }
    splits1 = _build_splits(counts)
    splits2 = _build_splits(counts)
    summary = _build_summary(counts)
    o1, _ = prep._apply_species_caps(
        splits1, summary, min_imgs=0, max_imgs=100, rng=random.Random(42),
    )
    o2, _ = prep._apply_species_caps(
        splits2, summary, min_imgs=0, max_imgs=100, rng=random.Random(42),
    )
    train1 = [r["image"] for r in o1["train"]]
    train2 = [r["image"] for r in o2["train"]]
    assert train1 == train2


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_defaults_30_and_120() -> None:
    """Defaults documented in help text — guards against accidental
    config drift between training runs."""
    import argparse
    import io
    import contextlib

    ap = argparse.ArgumentParser()
    prep._add_species_cap_args(ap)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            ap.parse_args(["--help"])
        except SystemExit:
            pass
    help_text = buf.getvalue()
    args = ap.parse_args([])
    assert args.min_imgs_per_species == 30
    assert args.max_imgs_per_species == 120
