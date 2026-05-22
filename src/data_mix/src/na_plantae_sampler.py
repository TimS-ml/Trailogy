"""Sample NA Plantae records from the prepared na_plantae JSONL.

Since the NA Plantae corpus is long-tailed (per-class image counts span
roughly 1.5 orders of magnitude even after the prepare-step caps), the
sampler supports per-class re-weighting via ``train_temperature``:

  * ``train_temperature = 1.0`` (default) — natural shuffle-and-repeat.
    Each record is sampled with uniform weight, so per-class share in
    the output equals per-class share in the pool. Bit-compatible with
    the legacy implementation.
  * ``train_temperature < 1.0`` — temper the distribution toward
    balanced. The per-record weight becomes ``n_class ** (T - 1)``,
    making the expected per-class share proportional to ``n_class ** T``.
    The canonical square-root tempering of Mahajan et al. 2018 is
    ``T = 0.5``.
  * ``train_temperature -> 0`` — fully balanced, every class shows up
    equally often.

Tempered sampling uses Efraimidis-Spirakis weighted-reservoir (no
replacement) when ``n_train <= len(filtered_pool)``, falling back to
``random.choices`` (with replacement) only when the target count
exceeds the available pool size. The old code path used
``random.choices`` unconditionally; on the production mix-50k v1 build
that produced ~19 % duplicate-image rows (some images visited 6 times)
even though the pool was more than 2x larger than the requested count.
The reservoir variant gives the same expected per-class share without
the duplicates.

The sampler also accepts a ``train_exclude_slugs`` set, applied to the
train pool BEFORE weighting. Excluded slugs simply don't enter the
pool — there is no special-case during sampling. Val sampling is
intentionally untouched by both ``train_exclude_slugs`` and
``train_temperature`` so eval loss keeps measuring the underlying
class distribution.

Records always get a ``source: "na_plantae"`` stamp.
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

log = logging.getLogger("data_mix.na_plantae_sampler")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _natural_oversample(
    pool: List[dict], n: int, rng: random.Random,
) -> List[dict]:
    """Legacy shuffle-and-repeat: full passes through a re-shuffled
    pool, plus a partial tail. Preserves natural class proportions."""
    if n <= 0 or not pool:
        return []
    out: List[dict] = []
    full_passes, remainder = divmod(n, len(pool))
    for _ in range(full_passes):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        out.extend(shuffled)
    if remainder:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        out.extend(shuffled[:remainder])
    return out


def _weighted_reservoir(
    pool: List[dict], weights: List[float], n: int, rng: random.Random,
) -> List[dict]:
    """Efraimidis-Spirakis weighted reservoir (no-replacement).

    For each item compute ``key_i = u_i ** (1 / w_i)`` with
    ``u_i ~ Uniform(0,1)``, then take the n items with the largest
    keys. Equivalent (in distribution over selected sets) to drawing n
    items without replacement with per-item probability proportional
    to ``w_i``. Stdlib-only; same RNG as the rest of the sampler.

    Requires ``n <= len(pool)``. Caller must guarantee that.
    """
    assert n <= len(pool), (
        f"_weighted_reservoir: n={n} exceeds pool size {len(pool)}; "
        f"caller must dispatch to the with-replacement path instead."
    )
    # Slightly inflate any zero weights so log/division is well-defined.
    # In practice the caller produces strictly-positive weights because
    # each pool row's class has at least one member (itself), so this
    # branch is defensive.
    eps = 1e-12
    keys = [
        rng.random() ** (1.0 / max(w, eps))
        for w in weights
    ]
    # Negative key for descending sort (largest key wins). Using a key
    # function rather than reverse=True avoids a second list build.
    order = sorted(range(len(pool)), key=lambda i: keys[i], reverse=True)
    return [pool[i] for i in order[:n]]


def _tempered_sample(
    pool: List[dict],
    n: int,
    temperature: float,
    rng: random.Random,
) -> List[dict]:
    """Per-class tempered sampling.

    For each record, ``weight = n_class[record's slug] ** (T - 1)``.
    Aggregated across the class, the expected per-class share is
    proportional to ``n_class ** T``.

    Dispatch:

      * ``n <= len(pool)`` — Efraimidis-Spirakis weighted reservoir
        (no replacement). The expected per-class share matches
        ``random.choices`` with the same weights, but no image is
        repeated within the output.
      * ``n > len(pool)``  — fall back to ``random.choices`` (with
        replacement). The only way to satisfy a target larger than
        the pool. Logged as a warning since this is rarely the
        intent on a multi-tens-of-thousands-row pool.
    """
    if n <= 0 or not pool:
        return []
    counts = Counter(rec.get("slug", "") for rec in pool)
    if not counts:
        return []
    exponent = temperature - 1.0
    weights = [counts[rec.get("slug", "")] ** exponent for rec in pool]

    if n <= len(pool):
        return _weighted_reservoir(pool, weights, n, rng)

    log.warning(
        "_tempered_sample: n=%d exceeds pool size %d; "
        "falling back to with-replacement sampling. Some images "
        "will appear multiple times in the output.",
        n, len(pool),
    )
    return rng.choices(pool, weights=weights, k=n)


def _apply_exclude_slugs(
    pool: List[dict],
    exclude: Iterable[str] | None,
) -> List[dict]:
    """Drop train pool records whose ``slug`` is in ``exclude``.

    Returns the filtered list and logs the per-slug drop counts so
    operators can confirm the drop list landed on the expected classes.
    Slugs in ``exclude`` that don't match any record are warned about
    individually (catches typos in the config without failing the
    build).
    """
    if not exclude:
        return pool
    exclude_set = {s for s in exclude if s}
    if not exclude_set:
        return pool

    pool_slugs = Counter(rec.get("slug", "") for rec in pool)
    missing = sorted(s for s in exclude_set if s not in pool_slugs)
    if missing:
        log.warning(
            "na_plantae train_exclude_slugs: %d slug(s) not present in "
            "pool, ignored: %s",
            len(missing), missing,
        )

    filtered = [rec for rec in pool if rec.get("slug", "") not in exclude_set]
    dropped = len(pool) - len(filtered)
    matched = sorted(s for s in exclude_set if s in pool_slugs)
    log.info(
        "na_plantae train_exclude_slugs: dropped %d records across %d "
        "matched slug(s): %s",
        dropped, len(matched), matched,
    )
    return filtered


def sample_na_plantae_records(
    train_jsonl: Path,
    val_jsonl: Path,
    n_train: int,
    n_val: int,
    seed: int,
    train_temperature: float = 1.0,
    train_exclude_slugs: Iterable[str] | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Read na_plantae train/val JSONLs, stamp source, and sample to
    the requested counts.

    ``train_temperature`` controls per-class re-weighting on the train
    pool (see module docstring). ``train_exclude_slugs`` removes
    matching slugs from the train pool BEFORE sampling. Val records
    are never excluded and val sampling is always natural — the
    eval-side measurement must keep seeing the full class distribution
    so per-class drift attributable to the drop list stays observable.
    """
    train_raw = _read_jsonl(train_jsonl)
    val_raw = _read_jsonl(val_jsonl)

    if n_train > 0 and not train_raw:
        raise RuntimeError(f"na_plantae train JSONL is empty: {train_jsonl}")
    if n_val > 0 and not val_raw:
        raise RuntimeError(f"na_plantae val JSONL is empty: {val_jsonl}")

    # Stamp source on every record (in-place is fine; the lists are
    # local to this call).
    for rec in train_raw:
        rec["source"] = "na_plantae"
    for rec in val_raw:
        rec["source"] = "na_plantae"

    train_raw = _apply_exclude_slugs(train_raw, train_exclude_slugs)
    if n_train > 0 and not train_raw:
        raise RuntimeError(
            "na_plantae train pool is empty after applying "
            "train_exclude_slugs; refusing to silently return 0 train "
            "records."
        )

    rng_t = random.Random(seed)
    rng_v = random.Random(seed + 1)

    if train_temperature == 1.0:
        # Legacy path. Preserve bit-identical shuffle order to the
        # pre-temperature implementation for back-compat — same RNG,
        # same upstream shuffle, same divmod tail.
        rng_t.shuffle(train_raw)
        train_out = _natural_oversample(train_raw, n_train, rng_t)
    else:
        train_out = _tempered_sample(
            train_raw, n_train, train_temperature, rng_t,
        )

    # Val: always natural. Preserves the legacy contract that
    # eval_<key>_loss reflects the pool's class distribution.
    rng_v.shuffle(val_raw)
    val_out = _natural_oversample(val_raw, n_val, rng_v)

    return train_out, val_out
