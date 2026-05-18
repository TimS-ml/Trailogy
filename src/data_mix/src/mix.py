# data_mix/src/mix.py
"""Orchestrate the 4-bucket mix build: read config, run each sampler,
shuffle, split train/val, write JSONL + build report.

v2 changes:
- Config ``source: llava`` (default ``cambrian`` for v1 backward compat)
  dispatches to ``llava_sampler`` instead of ``cambrian_sampler``.
- In addition to the combined ``val.jsonl``, writes three split-by-source
  val files (``val_plant.jsonl`` / ``val_nonplant.jsonl`` /
  ``val_negative.jsonl``) for the trainer's multi-eval-dataset feature.
- ``build_report.json`` records the per-source val paths so downstream
  finetune configs can reference them programmatically.

Run as a module:
    cd src && python -m data_mix.src.mix --config data_mix/configs/mix-100k.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from data_mix.src.cambrian_sampler import (
    open_cambrian_stream,
    sample_cambrian_records,
)
from data_mix.src.env_paths import HIKECOMPANION_ROOT, resolve_paths
from data_mix.src.llava_sampler import (
    open_llava_stream,
    sample_llava_records,
)
from data_mix.src.negative_builder import build_negative_records
from data_mix.src.offline_qa_sampler import sample_offline_qa_records
from data_mix.src.plant_sampler import (
    sample_plant_records,
    sample_plant_records_split,
)
from data_mix.src.schema import validate_record
from data_mix.src.smoltalk_sampler import (
    open_smoltalk_stream,
    sample_smoltalk_records,
)

log = logging.getLogger("data_mix.mix")

ALLOWED_GENERAL_SOURCES = frozenset({"cambrian", "llava"})

# Sources that contribute to the "nonplant" val partition (general VQA +
# text-only chat — i.e. everything that exercises non-plant capabilities).
# offline_qa is intentionally NOT in here — it gets its own val_offline_qa
# bucket because the persona signal is conceptually different from
# "general nonplant" (negative bucket is general refusal; offline_qa is
# specifically the "I'm an on-device AI" persona).
NONPLANT_SOURCES = frozenset({"cambrian", "llava", "smoltalk"})


@dataclass(frozen=True)
class MixConfig:
    source: str                      # "cambrian" (v1) | "llava" (v2)
    plant_train: int
    plant_val: int
    plant_per_class_cap: int
    general_train: int               # was: cambrian_train
    general_val: int                 # was: cambrian_val
    smoltalk_train: int
    smoltalk_val: int
    negative_train: int
    negative_val: int
    seed: int
    # v3: offline_qa is a tiny (~42 entries) persona corpus that sits
    # OUTSIDE the main 45/30/15/10 ratio. We include the whole corpus
    # (no random subsample, no oversample) and let the bucket appear
    # at small absolute count, controlled only by val_ratio.
    # ``offline_qa_path`` is optional — when None the bucket is skipped
    # entirely so v1/v2 configs that don't reference it remain
    # bit-identical to their previous output.
    offline_qa_path: str | None = None
    offline_qa_val_ratio: float = 0.1


def _load_config(path: Path) -> MixConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    source = raw.get("source", "cambrian")
    if source not in ALLOWED_GENERAL_SOURCES:
        raise ValueError(
            f"unknown 'source' in {path}: {source!r}; "
            f"expected one of {sorted(ALLOWED_GENERAL_SOURCES)}"
        )
    # The YAML key for the general bucket matches the source name to keep
    # the config self-documenting (v1 mix-20k uses 'cambrian:', v2 uses
    # 'llava:').
    if source not in raw:
        raise ValueError(
            f"config {path} declares source={source!r} but has no "
            f"matching '{source}:' block"
        )
    general_section = raw[source]
    offline_qa_section = raw.get("offline_qa") or {}
    offline_qa_path = offline_qa_section.get("path")
    if offline_qa_path:
        offline_qa_path = Path(offline_qa_path)
        if not offline_qa_path.is_absolute():
            offline_qa_path = HIKECOMPANION_ROOT / offline_qa_path
        offline_qa_path = str(offline_qa_path.resolve())
    return MixConfig(
        source=source,
        plant_train=raw["plant"]["train"],
        plant_val=raw["plant"]["val"],
        plant_per_class_cap=raw["plant"]["per_class_cap"],
        general_train=general_section["train"],
        general_val=general_section["val"],
        smoltalk_train=raw["smoltalk"]["train"],
        smoltalk_val=raw["smoltalk"]["val"],
        negative_train=raw["negative"]["train"],
        negative_val=raw["negative"]["val"],
        seed=raw["seed"],
        offline_qa_path=offline_qa_path,
        offline_qa_val_ratio=float(offline_qa_section.get("val_ratio", 0.1)),
    )


def _write_jsonl(records: List[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for rec in records:
            validate_record(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class InsufficientPoolError(RuntimeError):
    """A bucket pool ran short of (n_train + n_val) records."""


def _split_train_val(records: list, n_train: int, n_val: int) -> tuple[list, list]:
    """Slice a deterministic pool into disjoint train + val partitions.

    Raises ``InsufficientPoolError`` (NOT ``AssertionError``) when the pool
    is short — assertions are stripped under ``python -O`` and would
    silently return wrong-size partitions in that mode.
    """
    if len(records) < n_train + n_val:
        raise InsufficientPoolError(
            f"pool has only {len(records)} records but need {n_train + n_val} "
            f"(n_train={n_train}, n_val={n_val})"
        )
    return records[:n_train], records[n_train : n_train + n_val]


def _partition_val_by_source(val_records: List[dict]) -> Dict[str, List[dict]]:
    """Split a shuffled val list by source into the eval buckets:

      - plant       -> plant-species ID (degradation watch)
      - nonplant    -> general VQA + text-only chat (forget watch)
      - negative    -> refusal template (over/under-refusal watch)
      - offline_qa  -> "I'm an offline AI" persona (persona watch; v3)

    Used to write val_<key>.jsonl files for the trainer's
    ``eval_dataset = {key: ds}`` multi-eval feature. The trainer logs
    ``eval_<key>_loss`` per bucket so we can spot, e.g., degradation of
    the persona signal independently of plant-ID accuracy.

    Empty buckets are still emitted to the output dict (downstream writer
    handles empty files; finetune's config rejects an empty val_files
    entry only if explicitly listed).
    """
    out: Dict[str, List[dict]] = {
        "plant": [], "nonplant": [], "negative": [], "offline_qa": [],
    }
    for rec in val_records:
        src = rec["source"]
        if src == "plant":
            out["plant"].append(rec)
        elif src == "negative":
            out["negative"].append(rec)
        elif src == "offline_qa":
            out["offline_qa"].append(rec)
        elif src in NONPLANT_SOURCES:
            out["nonplant"].append(rec)
        else:
            raise ValueError(f"val record has unexpected source: {src!r}")
    return out


def build_mix(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    paths = resolve_paths()
    rng = random.Random(cfg.seed)

    log.info("paths: %s", paths)
    log.info("source: %s", cfg.source)

    # --- Plant ---
    # v2: dual-source sampler if PLANTNET_VAL_JSONL resolves to an existing
    # file (the per-species val output of prepare_plantnet_50k.sh). Falls
    # back to the v1 single-source + random slice if no val.jsonl exists
    # next to plantnet_jsonl — keeps mix-20k (the v1 cambrian config) and
    # older smoke tests working without a re-prep.
    if paths.plantnet_val_jsonl.exists():
        plant_train, plant_val = sample_plant_records_split(
            train_jsonl=paths.plantnet_jsonl,
            val_jsonl=paths.plantnet_val_jsonl,
            n_train=cfg.plant_train,
            n_val=cfg.plant_val,
            per_class_cap=cfg.plant_per_class_cap,
            seed=cfg.seed,
        )
        log.info(
            "Plant bucket: per-species dual-source split "
            "(train=%s, val=%s)",
            paths.plantnet_jsonl,
            paths.plantnet_val_jsonl,
        )
    else:
        log.warning(
            "Plant bucket: PLANTNET_VAL_JSONL %s missing — falling back to "
            "v1 single-source + random slice. Re-run prepare_plantnet_50k.sh "
            "to produce a per-species val.jsonl and remove this warning.",
            paths.plantnet_val_jsonl,
        )
        plant_pool = sample_plant_records(
            jsonl_path=paths.plantnet_jsonl,
            total=cfg.plant_train + cfg.plant_val,
            per_class_cap=cfg.plant_per_class_cap,
            seed=cfg.seed,
        )
        plant_train, plant_val = _split_train_val(
            plant_pool, cfg.plant_train, cfg.plant_val
        )

    # --- smoltalk (text-only; image=None) ---
    # v2: no longer needs a dummy image; ModalityAwareBatchSampler in
    # finetune routes image=None records into vision-skip batches.
    smol_stream = open_smoltalk_stream(seed=cfg.seed)
    smol_pool = sample_smoltalk_records(
        stream=smol_stream,
        total=cfg.smoltalk_train + cfg.smoltalk_val,
        seed=cfg.seed,
    )
    smol_train, smol_val = _split_train_val(
        smol_pool, cfg.smoltalk_train, cfg.smoltalk_val
    )

    # --- General bucket (Cambrian for v1, LLaVA for v2) + negative seeds ---
    if cfg.source == "llava":
        gen_stream = open_llava_stream(seed=cfg.seed)
        pools = sample_llava_records(
            stream=gen_stream,
            n_general=cfg.general_train + cfg.general_val,
            n_negative=cfg.negative_train + cfg.negative_val,
            image_root=paths.image_root,
            seed=cfg.seed,
        )
    else:  # source == "cambrian"
        gen_stream = open_cambrian_stream(seed=cfg.seed)
        pools = sample_cambrian_records(
            stream=gen_stream,
            n_general=cfg.general_train + cfg.general_val,
            n_negative=cfg.negative_train + cfg.negative_val,
            image_root=paths.image_root,
            seed=cfg.seed,
        )
    gen_train, gen_val = _split_train_val(
        pools["general"], cfg.general_train, cfg.general_val
    )

    # Negative: build fresh refusal records from the seed image paths.
    # ``pools["negative"]`` is already a List[Path] for both samplers.
    neg_records = build_negative_records(pools["negative"])
    neg_train, neg_val = _split_train_val(
        neg_records, cfg.negative_train, cfg.negative_val
    )

    # --- v3: offline_qa persona bucket (tiny, sits OUTSIDE the main
    # 45/30/15/10 ratio). Whole corpus included; only the train/val
    # split is configurable. Skipped entirely when cfg.offline_qa_path
    # is None so v1/v2 configs are bit-identical.
    offline_qa_train: List[dict] = []
    offline_qa_val: List[dict] = []
    if cfg.offline_qa_path:
        log.info("offline_qa: loading from %s", cfg.offline_qa_path)
        offline_qa_train, offline_qa_val = sample_offline_qa_records(
            json_path=cfg.offline_qa_path,
            val_ratio=cfg.offline_qa_val_ratio,
            seed=cfg.seed,
        )
        log.info(
            "offline_qa: %d train / %d val (full corpus, no oversample)",
            len(offline_qa_train), len(offline_qa_val),
        )

    # --- Combine and shuffle ---
    train_all = plant_train + gen_train + smol_train + neg_train + offline_qa_train
    val_all = plant_val + gen_val + smol_val + neg_val + offline_qa_val
    rng.shuffle(train_all)
    rng.shuffle(val_all)

    train_path = paths.output_root / "train.jsonl"
    val_path = paths.output_root / "val.jsonl"
    report_path = paths.output_root / "build_report.json"

    _write_jsonl(train_all, train_path)
    _write_jsonl(val_all, val_path)

    # v2: also write per-eval-bucket val files so finetune can run
    # multi-eval-dataset and report eval_<key>_loss per modality.
    # v3: the ``offline_qa`` partition is only emitted when the bucket
    # is configured AND non-empty — keeps v1/v2 outputs bit-identical.
    val_partitions = _partition_val_by_source(val_all)
    val_files: Dict[str, str] = {}
    for key, recs in val_partitions.items():
        if key == "offline_qa" and not recs:
            # Backward compat: only materialize val_offline_qa.jsonl when
            # the bucket has at least one record (i.e. offline_qa is
            # actually configured + has val_ratio > 0 + corpus N >= 2).
            continue
        # Always write the file for the legacy buckets, even if empty —
        # downstream config knows to skip empties rather than crash on
        # missing path.
        dest = paths.output_root / f"val_{key}.jsonl"
        _write_jsonl(recs, dest)
        val_files[key] = str(dest)

    report = {
        "seed": cfg.seed,
        "source": cfg.source,
        "train_total": len(train_all),
        "val_total": len(val_all),
        "train_by_source": _count_by_source(train_all),
        "val_by_source": _count_by_source(val_all),
        "paths": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "val_files": val_files,
            "image_root": str(paths.image_root),
            "plantnet_source": str(paths.plantnet_jsonl),
        },
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    log.info("wrote %d train, %d val to %s", len(train_all), len(val_all), paths.output_root)
    log.info("multi-val splits: %s", {k: len(v) for k, v in val_partitions.items()})
    return report_path


def _count_by_source(records: List[dict]) -> dict:
    out: dict = {}
    for r in records:
        out[r["source"]] = out.get(r["source"], 0) + 1
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    report = build_mix(args.config)
    print(report)


if __name__ == "__main__":
    main()
