#!/usr/bin/env python3
"""Convert PlantNet-300K (ImageFolder layout) into instruction-tuning JSONL for Gemma 4 VLM finetuning.

Expected PlantNet-300K layout:
    <root>/{train,val,test}/<species_id>/<image>.jpg

Outputs train.jsonl and val.jsonl with Gemma chat-format conversations.

Train/inference vision parity
-----------------------------
By default we **pre-resize** every image to 960x672 (height x width) and
write the resized copies to ``<output_dir>/images_resized/<split>/<sid>/``,
then point the JSONL at those copies. This matches mlx-swift-lm's iOS
runtime behaviour exactly:

  * mlx-swift-lm's Gemma4Processor.preprocess does a fixed-size
    ``resampleBicubic`` to whatever ``processor_config.json`` says — currently
    960x672 (matching the app model-fetch script's trained-size patch).
  * HF's Gemma4ImageProcessor (used during unsloth training) does
    aspect-ratio-preserving resize to a variable patch grid.

If we trained on aspect-preserved images and deployed on aspect-stretched
ones, the LoRA would learn against a different visual feature distribution
than it sees at inference. Pre-resizing at data-prep time is the cheapest
way to guarantee train/deploy parity.

Pass ``--resize_to none`` to disable (only useful if you've fixed the
mlx-swift-lm processor to mirror HF's aspect-ratio-preserving behaviour).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question / answer templates
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES = [
    "What plant is this?",
    "Can you identify this species?",
    "What am I looking at?",
    "Describe this plant.",
    "Do you know what kind of plant this is?",
    "I found this on the trail — what is it?",
    "What species is this plant?",
    "Can you tell me about this plant I just spotted?",
    "I saw this growing near the trail. Any idea what it is?",
    "What's the name of this plant?",
    "Help me identify this — is it a common species?",
    "I'm curious about this plant. What can you tell me?",
]

ANSWER_TEMPLATES = [
    "This is {species}. You'll find it in many temperate habitats.",
    "You're looking at {species}. Nice find!",
    "That's {species}. It's fairly common along trails in this kind of environment.",
    "Good eye — this is {species}.",
    "This appears to be {species}. It's a well-known species among botanists.",
    "You've spotted {species}. Keep an eye out for more along the trail.",
    "That looks like {species}. It tends to grow in areas with moderate sunlight.",
    "This is {species}. Hikers often notice it because of its distinctive look.",
    "Looks like {species} to me. A great example of local flora.",
    "That's {species}. It's one of the species frequently catalogued in plant surveys.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_species_map(path: str | None) -> dict[str, str]:
    """Load species_id -> scientific name mapping. Returns empty dict on failure."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        log.warning("Species map not found at %s — using folder names.", path)
        return {}
    with open(p) as f:
        raw = json.load(f)
    # The standard PlantNet JSON maps string species_id -> name.
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    log.warning("Unexpected species map format; ignoring.")
    return {}


def discover_images(split_dir: Path) -> dict[str, list[Path]]:
    """Return {species_id: [image_paths]} for one split directory.

    Both the species-directory iteration AND the per-species image
    iteration are explicitly sorted so the returned dict is identical
    across filesystems (macOS APFS vs Linux ext4 etc.). Without the
    inner sort, ``species_dir.iterdir()`` returns images in
    filesystem-dependent order, which breaks the script's
    "byte-identical JSONL across machines" promise documented in
    ``hikeCompanion/finetune/scripts/run/prepare_plantnet_50k.sh``
    (see § Determinism in its header).
    """
    classes: dict[str, list[Path]] = defaultdict(list)
    if not split_dir.is_dir():
        return classes
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        sid = species_dir.name
        for img in sorted(species_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                classes[sid].append(img)
    return classes


def stratified_sample(
    classes: dict[str, list[Path]], max_samples: int, rng: random.Random
) -> list[tuple[str, Path]]:
    """Sample up to *max_samples* images, stratified by class.

    Strategy: round-robin one sample per class, repeat until budget exhausted
    or all images consumed.
    """
    # Shuffle within each class
    pool: dict[str, list[Path]] = {}
    for sid, imgs in classes.items():
        shuffled = list(imgs)
        rng.shuffle(shuffled)
        pool[sid] = shuffled

    result: list[tuple[str, Path]] = []
    remaining = max_samples

    # Round-robin passes
    while remaining > 0:
        made_progress = False
        for sid in sorted(pool.keys()):
            if not pool[sid]:
                continue
            result.append((sid, pool[sid].pop()))
            remaining -= 1
            made_progress = True
            if remaining <= 0:
                break
        if not made_progress:
            break

    return result


def class_stratified_split(
    samples: list[tuple[str, Path]],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Per-species stratified train/val split (v2).

    Replaces the v1 ``samples[:val_count]`` random slice that left ~23 % of
    species absent from val (see the train/eval data-mismatch note's
    "Fourth finding" for the full diagnosis). For each species, hold out
    ``max(1, floor(N * val_ratio))`` images for val (capped so train
    keeps ≥1) and put the rest in train.

    Properties guaranteed:
      - ``val_species_set`` ⊆ ``train_species_set`` — every val species is
        also in train (no orphans the model never saw).
      - Single-image species → train only. Better to train a rare class
        than to evaluate on a class the model has never been shown.
      - Per-species val density is proportional to ``val_ratio`` of the
        species' image count in the pool (modulo the floor + min(1) rule).
      - Disjoint by image — no record appears in both train and val.
      - Deterministic given the same ``samples`` order and ``rng`` seed.

    Image selection within a species is by sorted filename to keep the
    split byte-identical across machines (mirrors the inner sort in
    ``discover_images`` that fixes filesystem ordering drift, see the
    train-eval data-mismatch note §"Second bug").
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(
            f"val_ratio must be in [0, 1], got {val_ratio}"
        )

    # Group by species, preserving deterministic image order.
    by_sid: dict[str, list[Path]] = defaultdict(list)
    for sid, p in samples:
        by_sid[sid].append(p)

    train: list[tuple[str, Path]] = []
    val: list[tuple[str, Path]] = []

    for sid in sorted(by_sid):
        imgs = sorted(by_sid[sid], key=lambda p: p.name)
        n = len(imgs)
        if val_ratio == 1.0:
            # Symmetric edge case to val_ratio=0.0: caller wants
            # everything in the "val" half (used by the test-stage
            # promotion in prepare_plantnet_50k.sh — Step 2 stages with
            # val_ratio=1.0 then renames the staged val.jsonl to
            # test.jsonl). Without this branch the per-species
            # "leave ≥1 in train" + "single-image → train only" rules
            # silently divert ~6% of species to a staging train.jsonl
            # that the shell wrapper discards.
            val.extend((sid, p) for p in imgs)
            continue
        if val_ratio == 0.0 or n < 2:
            # No val carve-out: either the caller opted out, or this is a
            # single-image species (cannot contribute to both sides
            # without leakage).
            train.extend((sid, p) for p in imgs)
            continue
        n_val = max(1, int(n * val_ratio))
        n_val = min(n_val, n - 1)  # always leave ≥1 for train
        for p in imgs[:n_val]:
            val.append((sid, p))
        for p in imgs[n_val:]:
            train.append((sid, p))

    # Shuffle the two halves so the within-species deterministic order
    # doesn't bias the downstream JSONL emit order.
    rng.shuffle(train)
    rng.shuffle(val)

    # Tripwire: with 0 < val_ratio < 1, val species set must be a non-empty
    # subset of train species set. Catches accidental drop of val_ratio
    # validation or a future refactor that breaks the invariant. At
    # val_ratio=1.0 train is intentionally empty (test-stage promotion),
    # so val cannot be a subset of train — skip the check.
    if 0.0 < val_ratio < 1.0:
        train_species = {sid for sid, _ in train}
        val_species = {sid for sid, _ in val}
        if not val_species.issubset(train_species):
            orphans = val_species - train_species
            raise AssertionError(
                f"class_stratified_split produced val species not present "
                f"in train: {sorted(orphans)[:5]} (and {len(orphans) - 5} "
                f"more)" if len(orphans) > 5 else
                f"class_stratified_split produced val species not present "
                f"in train: {sorted(orphans)}"
            )

    return train, val


# ---------------------------------------------------------------------------
# Pre-resize for train/inference parity (Bug 5 in the export pipeline review)
# ---------------------------------------------------------------------------

# Trained vision shape — must mirror the app model-fetch script's trained-size patch
# and ``finetune/src/export_mlx.py:TRAINED_VISION_SIZE``. (height, width).
DEFAULT_TRAINED_VISION_HW: Tuple[int, int] = (960, 672)


def parse_resize_arg(s: str | None) -> Optional[Tuple[int, int]]:
    """Parse ``--resize_to`` value into a (height, width) tuple or None.

    Accepts the formats:
      * ``"960x672"`` / ``"960X672"`` — explicit HxW
      * ``"none"`` / ``"off"`` / ``"false"`` / ``""`` — disable resize
      * ``None`` — disable resize

    Raises ``ValueError`` for anything else, including non-positive sizes.
    """
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("", "none", "off", "false", "no"):
        return None
    sep = None
    for candidate in ("x", "X", "*", ","):
        if candidate.lower() in s:
            sep = candidate.lower()
            break
    if sep is None:
        raise ValueError(
            f"--resize_to must be HxW (e.g. '960x672') or 'none'; got {s!r}"
        )
    parts = s.split(sep)
    if len(parts) != 2:
        raise ValueError(
            f"--resize_to expects exactly two integers separated by 'x'; got {s!r}"
        )
    try:
        h, w = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--resize_to integers unparseable: {s!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"--resize_to dimensions must be positive; got {h}x{w}")
    return (h, w)


def resize_image_to_disk(
    src_path: Path,
    dest_path: Path,
    target_hw: Tuple[int, int],
) -> Path:
    """Stretch-resize ``src_path`` to ``target_hw`` (height, width) at ``dest_path``.

    Stretches without preserving aspect ratio — this is intentional, to
    match mlx-swift-lm's fixed-size ``resampleBicubic`` exactly. If you fix
    mlx-swift-lm to preserve aspect ratio, this function should change too.

    Skips if ``dest_path`` already exists with non-zero size.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    from PIL import Image  # lazy import so module load works without PIL

    target_h, target_w = target_hw
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as im:
        # Convert to RGB to drop alpha / palette modes — mirrors HF
        # processor's do_convert_rgb=True and mlx-swift-lm's sRGB conversion.
        rgb = im.convert("RGB")
        # PIL .resize takes (width, height). BICUBIC matches the
        # processor_config.json ``resample: 3`` field used everywhere in
        # the Gemma 4 stack.
        resized = rgb.resize((target_w, target_h), Image.BICUBIC)
        # Preserve original suffix where reasonable, but always re-encode.
        suffix = dest_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            resized.save(dest_path, format="JPEG", quality=92)
        elif suffix == ".png":
            resized.save(dest_path, format="PNG", optimize=True)
        elif suffix == ".webp":
            resized.save(dest_path, format="WEBP", quality=92)
        else:
            resized.save(dest_path)
    return dest_path


def build_conversation(
    image_path: Path,
    species_name: str,
    rng: random.Random,
) -> dict:
    """Build a single JSONL record in Gemma chat format."""
    question = rng.choice(QUESTION_TEMPLATES)
    answer = rng.choice(ANSWER_TEMPLATES).format(species=species_name)
    # NOTE: no `<image>\n` placeholder — the unsloth vision data collator
    # injects image soft-tokens via the structural content block produced by
    # `src/data.py:build_vision_messages`. A leading placeholder would
    # double-reserve image tokens in Gemma 4's chat template.
    return {
        "image": str(image_path.resolve()),
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PlantNet-300K into instruction-tuning JSONL."
    )
    parser.add_argument(
        "--plantnet_root",
        type=str,
        required=True,
        help="Path to PlantNet-300K root (contains train/, val/, test/).",
    )
    parser.add_argument(
        "--species_map",
        type=str,
        default=None,
        help="Path to plantnet300K_species_id_2_name.json (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Max training samples to emit (default: 5000 — first-run cap).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of samples held out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--resize_to",
        type=str,
        default="960x672",
        help=(
            "Pre-resize images to HxW before writing the JSONL. The default "
            "960x672 matches the iOS runtime's mlx-swift-lm fixed-stretch "
            "preprocessing, ensuring the LoRA trains on the same visual "
            "feature distribution it will see at deploy time. "
            "Pass 'none' to skip resizing (only safe if mlx-swift-lm has "
            "been patched to do aspect-ratio-preserving resize)."
        ),
    )
    parser.add_argument(
        "--resized_image_root",
        type=str,
        default=None,
        help=(
            "Where to write resized image copies. Defaults to "
            "<output_dir>/images_resized/. Reused across runs so resize "
            "is incremental."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.plantnet_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target_hw = parse_resize_arg(args.resize_to)
    if target_hw is None:
        log.info(
            "Image resize: DISABLED. Train/inference vision distribution may "
            "differ unless mlx-swift-lm has been patched to preserve aspect "
            "ratio. See AGENTS.md and the export pipeline review for context."
        )
        resized_root: Optional[Path] = None
    else:
        resized_root = (
            Path(args.resized_image_root)
            if args.resized_image_root
            else out / "images_resized"
        )
        resized_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "Image resize: ENABLED. Pre-stretching every image to %dx%d (HxW) "
            "and writing to %s. This matches the iOS runtime's fixed-resize.",
            target_hw[0], target_hw[1], resized_root,
        )

    # Load species name mapping
    species_map = load_species_map(args.species_map)
    log.info("Species map: %d entries loaded.", len(species_map))

    # Discover images from training split
    train_dir = root / "train"
    if not train_dir.is_dir():
        log.error("Training split not found at %s", train_dir)
        raise SystemExit(1)

    classes = discover_images(train_dir)
    total_images = sum(len(v) for v in classes.values())
    log.info(
        "Discovered %d images across %d classes in %s.",
        total_images,
        len(classes),
        train_dir,
    )

    # Stratified sample
    samples = stratified_sample(classes, args.max_samples, rng)
    log.info("Sampled %d images (budget: %d).", len(samples), args.max_samples)

    # Per-species stratified train/val split (v2; replaces v1 random slice).
    # v1 used `samples[:val_count]` after `rng.shuffle(samples)` which left
    # ~23 % of species absent from val (random tail of a long-tail dataset
    # misses sparse classes). v2 carves out per species so val species set
    # ⊆ train species set. See class_stratified_split docstring.
    train_samples, val_samples = class_stratified_split(
        samples, val_ratio=args.val_ratio, rng=rng,
    )
    train_species = {sid for sid, _ in train_samples}
    val_species = {sid for sid, _ in val_samples}
    if args.val_ratio == 1.0:
        # Test-stage promotion path: train is intentionally empty.
        log.info(
            "All-to-val mode (val_ratio=1.0): 0 train / %d val (%d species).",
            len(val_samples),
            len(val_species),
        )
    else:
        log.info(
            "Per-species split: %d train / %d val "
            "(species coverage: train=%d, val=%d; %d single-image species → train only)",
            len(train_samples),
            len(val_samples),
            len(train_species),
            len(val_species),
            len(train_species) - len(val_species),
        )

    # Build conversations and write JSONL
    stats: dict[str, int] = {"train": 0, "val": 0, "classes_seen": set()}  # type: ignore[dict-item]

    resize_failures = 0
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        out_path = out / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for sid, img_path in split_samples:
                species_name = species_map.get(sid, sid)

                # If resize is enabled, materialize a stretched copy at the
                # iOS-runtime shape and emit *its* path into the JSONL so the
                # unsloth vision data collator loads the matching pixels.
                final_image_path = img_path
                if target_hw is not None and resized_root is not None:
                    dest = resized_root / split_name / sid / img_path.name
                    try:
                        final_image_path = resize_image_to_disk(
                            img_path, dest, target_hw
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        # PlantNet has occasional truncated JPEGs. Log and
                        # fall back to the original; better than aborting
                        # the whole prep run.
                        resize_failures += 1
                        log.warning(
                            "Resize failed for %s (%s); falling back to original.",
                            img_path, exc,
                        )

                record = build_conversation(final_image_path, species_name, rng)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats[split_name] += 1
                stats["classes_seen"].add(sid)  # type: ignore[union-attr]
        log.info("Wrote %d records to %s", stats[split_name], out_path)

    if resize_failures:
        log.warning(
            "%d image(s) could not be resized — JSONL points at the originals "
            "for those entries. Inspect logs above for details.",
            resize_failures,
        )

    log.info("--- Summary ---")
    log.info("  Train samples : %d", stats["train"])
    log.info("  Val samples   : %d", stats["val"])
    log.info("  Classes seen  : %d / %d", len(stats["classes_seen"]), len(classes))  # type: ignore[arg-type]
    log.info("  Output dir    : %s", out)


if __name__ == "__main__":
    main()
