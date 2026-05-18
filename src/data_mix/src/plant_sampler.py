"""Subsample the existing PlantNet enriched JSONL into the 'plant' bucket.

- Per-class cap (default 30) limits images per species id.
- Three prompt variants assigned uniformly at random.
- Image paths are reused verbatim from the source JSONL (no copy).

Two entry points:

- ``sample_plant_records(jsonl_path, total, per_class_cap, seed)``
  v1 API: single JSONL -> single pool. Used by v1 mix-20k flow where
  ``mix.py`` then random-slices the pool into mix-train + mix-val.

- ``sample_plant_records_split(train_jsonl, val_jsonl, n_train, n_val,
  per_class_cap, seed)``
  v2 API: two JSONLs -> two pools, no slicing between them. Used by v2
  mix-50k / mix-100k flow where the upstream prepare script already
  carved a per-species val out of the 50K stratified pool, so we read
  each side as-is and apply the cap independently.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from data_mix.src.schema import validate_record

PROMPT_VARIANTS: Tuple[str, str, str] = (
    "What plant species is shown in this image?",
    "Identify this plant.",
    "Can you tell me what plant this is and describe its key features?",
)


def extract_species_id(image_path: str) -> str:
    """Species id is the parent directory name in PlantNet's layout."""
    return Path(image_path).parent.name


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cap_pool_by_class(
    raw: List[dict], per_class_cap: int, rng: random.Random
) -> List[dict]:
    """Apply per-class cap to a flat record list.

    Shuffles the input first so each class's surviving slice is a random
    sample (not biased toward the file's first N rows per class).
    Returns a deterministic, shuffled pool.
    """
    shuffled = list(raw)
    rng.shuffle(shuffled)
    by_sid: dict[str, List[dict]] = defaultdict(list)
    for r in shuffled:
        sid = extract_species_id(r["image"])
        if len(by_sid[sid]) < per_class_cap:
            by_sid[sid].append(r)
    pool: List[dict] = []
    for sid in sorted(by_sid):
        pool.extend(by_sid[sid])
    rng.shuffle(pool)
    return pool


def _apply_prompt_variants(pool: List[dict], rng: random.Random) -> List[dict]:
    """Wrap raw PlantNet records into the unified-schema 'plant' bucket
    with a uniformly-random one of three prompt variants per record."""
    out: List[dict] = []
    for r in pool:
        variant_idx = rng.randrange(3)
        user_text = PROMPT_VARIANTS[variant_idx]
        original_assistant = r["conversations"][-1]["content"]
        if variant_idx == 2:
            # "key features" -> keep the full description verbatim.
            assistant_text = original_assistant
        else:
            # Concise: drop trailing description sentence(s) if any —
            # the first sentence is the species identification.
            #
            # ASSUMPTION: PlantNet-enriched assistant text begins with
            # "This is <CommonName>." then either a Latin name or a longer
            # description. The first ". " boundary cleanly separates the
            # identification from the description.
            #
            # We verified empirically (audit on 2026-05-15) that 0/45000
            # rows in `english-desc-v2/train.jsonl` contain abbreviation
            # patterns like " sp.", " L.", " var." that would cause early
            # truncation. If the source data ever changes to include such
            # patterns, this heuristic must be revisited.
            assistant_text = original_assistant.split(". ")[0].rstrip(".") + "."
        rec = {
            "image": r["image"],
            "conversations": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "source": "plant",
        }
        validate_record(rec)
        out.append(rec)
    return out


def sample_plant_records(
    jsonl_path: Path,
    total: int,
    per_class_cap: int,
    seed: int,
) -> List[dict]:
    """v1 API: single-source sampler used by mix-20k (cambrian flow)."""
    raw = _read_jsonl(jsonl_path)
    rng = random.Random(seed)
    pool = _cap_pool_by_class(raw, per_class_cap, rng)
    pool = pool[:total]
    return _apply_prompt_variants(pool, rng)


def sample_plant_records_split(
    train_jsonl: Path,
    val_jsonl: Path,
    n_train: int,
    n_val: int,
    per_class_cap: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """v2 API: dual-source sampler — train pool from ``train_jsonl``, val
    pool from ``val_jsonl``. Each pool gets its OWN per_class_cap (no
    cross-pool slicing).

    Assumes the upstream prepare script (``prepare_plantnet*.py`` with
    ``class_stratified_split``) already produced a clean per-species
    train/val split with disjoint images. We just need to apply the
    bucket's per_class_cap independently and emit prompt-variant
    wrappers.

    Raises ``ValueError`` if either capped pool is too small to satisfy
    the requested n_train / n_val. The caller likely typo'd a per_class_cap
    or pointed at the wrong file.
    """
    # Two separate RNG streams keyed on the same seed so the val side's
    # variant assignment doesn't depend on the train side's pool size.
    train_rng = random.Random(seed)
    # Use a different stream salt for val so the two sides aren't lock-
    # stepped on the same draw sequence (otherwise val variant idx 0/1/2
    # would correlate with train variant idx 0/1/2).
    val_rng = random.Random(seed ^ 0xDEADBEEF)

    train_raw = _read_jsonl(train_jsonl)
    val_raw = _read_jsonl(val_jsonl)

    train_pool = _cap_pool_by_class(train_raw, per_class_cap, train_rng)
    val_pool = _cap_pool_by_class(val_raw, per_class_cap, val_rng)

    if len(train_pool) < n_train:
        raise ValueError(
            f"train pool has only {len(train_pool)} records after cap "
            f"({per_class_cap}), need {n_train}. Source: {train_jsonl}"
        )
    if len(val_pool) < n_val:
        raise ValueError(
            f"val pool has only {len(val_pool)} records after cap "
            f"({per_class_cap}), need {n_val}. Source: {val_jsonl}"
        )

    train_out = _apply_prompt_variants(train_pool[:n_train], train_rng)
    val_out = _apply_prompt_variants(val_pool[:n_val], val_rng)
    return train_out, val_out
