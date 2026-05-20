"""Sample NA Plantae records from the prepared na_plantae JSONL.

Since the NA Plantae dataset is small (~440 train, ~55 val), this sampler
oversamples (repeats) records to hit the requested count. Each record
gets ``source: "na_plantae"`` stamped before returning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_na_plantae_records(
    train_jsonl: Path,
    val_jsonl: Path,
    n_train: int,
    n_val: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Read na_plantae train/val JSONLs, stamp source, oversample to target counts."""
    rng = random.Random(seed)

    train_raw = _read_jsonl(train_jsonl)
    val_raw = _read_jsonl(val_jsonl)

    if not train_raw:
        raise RuntimeError(f"na_plantae train JSONL is empty: {train_jsonl}")
    if not val_raw:
        raise RuntimeError(f"na_plantae val JSONL is empty: {val_jsonl}")

    # Stamp source on every record
    for rec in train_raw:
        rec["source"] = "na_plantae"
    for rec in val_raw:
        rec["source"] = "na_plantae"

    # Oversample train
    train_out: List[dict] = []
    if n_train > 0:
        rng_t = random.Random(seed)
        rng_t.shuffle(train_raw)
        # Repeat full passes + partial tail
        full_passes, remainder = divmod(n_train, len(train_raw))
        for _ in range(full_passes):
            shuffled = list(train_raw)
            rng_t.shuffle(shuffled)
            train_out.extend(shuffled)
        if remainder:
            shuffled = list(train_raw)
            rng_t.shuffle(shuffled)
            train_out.extend(shuffled[:remainder])

    # Oversample val
    val_out: List[dict] = []
    if n_val > 0:
        rng_v = random.Random(seed + 1)
        rng_v.shuffle(val_raw)
        full_passes, remainder = divmod(n_val, len(val_raw))
        for _ in range(full_passes):
            shuffled = list(val_raw)
            rng_v.shuffle(shuffled)
            val_out.extend(shuffled)
        if remainder:
            shuffled = list(val_raw)
            rng_v.shuffle(shuffled)
            val_out.extend(shuffled[:remainder])

    return train_out, val_out
