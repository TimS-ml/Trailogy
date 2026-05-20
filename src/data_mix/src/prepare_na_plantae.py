#!/usr/bin/env python3
"""Prepare the NA-Plantae supplement / primary dataset.

Produces ``{train,val,test}.jsonl`` files in the same schema as
``finetune/src/prepare_plantnet.py``::

    {
      "image": "<abs path>",
      "conversations": [
        {"role": "user", "content": "<question template>"},
        {"role": "assistant", "content": "<species description>"}
      ]
    }

Reads species descriptions from a YAML file (operator-supplied; one
record per slug with at minimum ``slug``, ``common_name``, ``species``,
``family``, ``answer`` fields — see ``assets/na_plantae/descriptions.yaml``
for the canonical schema). Each species folder under ``--source_root``
is matched by slug. Species that don't have a description entry are
skipped with a WARN (typical when ``na_plantae_fetch.py`` pulls all
Plantae in a region — the fetcher does not curate by species).

Source layouts (auto-detected):
  * **split**  (current na_plantae_fetch.py output):
        <source_root>/{train,val,test}/<slug>/<file>.jpg
    The split boundary is respected (the fetcher already split
    per-observation_id to avoid train/test leakage).
  * **flat**   (legacy / manual curation):
        <source_root>/<slug>/<file>.jpg
    prepare runs its own deterministic per-species random split using
    ``--train_per_species`` / ``--val_per_species`` / ``--test_per_species``.

Storage convention: defaults sit OUTSIDE the repo at
``<repo>/../data/inaturalist_na_plantae{,_prepared}/``. Override via the
CLI flags or via the ``TRAILOGY_DATA_ROOT`` env var (consumed by
``data_mix.src.env_paths``).

Usage::

    python -m data_mix.src.prepare_na_plantae \\
        --source_root  $TRAILOGY_DATA_ROOT/inaturalist_na_plantae \\
        --descriptions assets/na_plantae/descriptions.yaml \\
        --output_dir   $TRAILOGY_DATA_ROOT/inaturalist_na_plantae_prepared \\
        --resize_to 960x672

``--resize_to`` mirrors ``prepare_plantnet.py`` — pre-resize every image
to the iOS-runtime shape so training and deploy see the same visual
distribution. Set to ``none`` to disable.

Output layout::

    <output_dir>/
        train.jsonl
        val.jsonl
        test.jsonl
        filter_report.json
        images_resized/
            train/<slug>/*.jpg
            val/<slug>/*.jpg
            test/<slug>/*.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml  # type: ignore

# Default external paths derived via env_paths so they stay in sync
# with the rest of the data_mix toolchain. Imported lazily inside
# main() (so module import does not depend on env vars being set).
log = logging.getLogger("prepare_na_plantae")

QUESTION_TEMPLATES = [
    "What plant is this?",
    "Can you identify this species?",
    "What am I looking at?",
    "Describe this tree.",
    "Do you know what kind of tree this is?",
    "I found this on the trail — what is it?",
    "What species is this tree?",
    "Can you tell me about this tree I just spotted?",
    "I saw this growing near the trail. Any idea what it is?",
    "What's the name of this tree?",
    "Help me identify this — is it a common species?",
    "I'm curious about this tree. What can you tell me?",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
DEFAULT_RESIZE_HW = (960, 672)  # height, width — iOS runtime contract
SPLIT_NAMES = ("train", "val", "test")


def load_descriptions(path: Path) -> dict[str, dict[str, Any]]:
    """Return {slug: record} mapping."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    trees = raw.get("trees", [])
    out = {}
    for t in trees:
        slug = t["slug"]
        if slug in out:
            raise ValueError(f"duplicate slug in descriptions: {slug}")
        out[slug] = t
    return out


def detect_source_layout(source_root: Path) -> str:
    """``split`` if any of train/val/test exist as subdirs of source_root;
    otherwise ``flat``."""
    for s in SPLIT_NAMES:
        if (source_root / s).is_dir():
            return "split"
    return "flat"


def discover_species_images_flat(
    source_root: Path, slugs: list[str]
) -> dict[str, list[Path]]:
    """{slug: sorted [Path, ...]} reading <source_root>/<slug>/*.jpg."""
    out: dict[str, list[Path]] = {}
    for slug in slugs:
        d = source_root / slug
        if not d.is_dir():
            log.warning("species folder missing (flat): %s", d)
            out[slug] = []
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        out[slug] = files
        log.info("  %s: %d images", slug, len(files))
    return out


def discover_species_images_split(
    source_root: Path, slugs: list[str]
) -> dict[str, dict[str, list[Path]]]:
    """{slug: {split: sorted [Path, ...]}} reading
    <source_root>/<split>/<slug>/*.jpg.

    Slugs that don't appear in ANY split are logged once. Slugs that
    appear under directories not listed in ``slugs`` (i.e. species
    the fetcher pulled but no description exists for) are surfaced as
    a single WARN with the dropped count.
    """
    out: dict[str, dict[str, list[Path]]] = {s: {} for s in slugs}
    fetched_slugs_with_imgs: set[str] = set()
    for split_name in SPLIT_NAMES:
        split_dir = source_root / split_name
        if not split_dir.is_dir():
            continue
        for slug_dir in sorted(split_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            files = sorted(
                p for p in slug_dir.iterdir() if p.suffix.lower() in IMG_EXTS
            )
            if not files:
                continue
            fetched_slugs_with_imgs.add(slug_dir.name)
            if slug_dir.name not in out:
                # Fetcher pulled this slug but we have no description
                # for it; track for the post-walk WARN.
                continue
            out[slug_dir.name][split_name] = files

    dropped = fetched_slugs_with_imgs - set(slugs)
    if dropped:
        log.warning(
            "Skipping %d fetched species without descriptions (extend "
            "assets/na_plantae/descriptions.yaml to include them): %s",
            len(dropped),
            ", ".join(sorted(dropped)[:10])
            + (" ..." if len(dropped) > 10 else ""),
        )
    for slug in slugs:
        per_split = out.get(slug) or {}
        total = sum(len(v) for v in per_split.values())
        if total == 0:
            log.warning(
                "  %s: no images in any split under %s",
                slug, source_root,
            )
        else:
            log.info(
                "  %s: %d images (%s)",
                slug, total,
                " ".join(f"{s}={len(per_split.get(s, []))}" for s in SPLIT_NAMES),
            )
    return out


def maybe_resize_and_save(
    src: Path, dst: Path, resize_hw: tuple[int, int] | None
) -> None:
    """Copy or pre-resize an image to dst. Skips if dst already exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize_hw is None:
        # Hard-link if possible (zero-byte cost on most filesystems),
        # else copy.
        try:
            dst.hardlink_to(src)
        except (OSError, AttributeError):
            import shutil
            shutil.copy2(src, dst)
        return
    # Resize. Pillow handles JPEG / PNG / WEBP fine.
    from PIL import Image  # type: ignore

    h, w = resize_hw
    with Image.open(src) as im:
        im = im.convert("RGB")
        im_resized = im.resize((w, h), Image.LANCZOS)
        im_resized.save(dst, "JPEG", quality=92)


def parse_resize(s: str) -> tuple[int, int] | None:
    if s.lower() in ("none", "off", "no", ""):
        return None
    parts = s.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--resize_to expects WxH or 'none'; got {s!r}"
        )
    w, h = int(parts[0]), int(parts[1])
    return (h, w)  # store as (H, W)


def _build_record(
    *,
    image_dst: Path,
    slug: str,
    desc: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    return {
        "image": str(image_dst.resolve()),
        "slug": slug,
        "species": desc["species"],
        "family": desc["family"],
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": desc["answer"].strip()},
        ],
    }


def _process_flat(
    args: argparse.Namespace,
    descs: dict[str, dict[str, Any]],
    slugs: list[str],
) -> tuple[
    dict[str, list[dict]], dict[str, dict[str, int]], random.Random
]:
    """Per-species random split (legacy flat-source mode)."""
    species_imgs = discover_species_images_flat(args.source_root, slugs)
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES}
    summary: dict[str, dict[str, int]] = {}

    for slug in slugs:
        imgs = species_imgs[slug]
        if not imgs:
            log.warning("  %s: no images, skipping species.", slug)
            continue
        # Stable per-slug seed: Python's builtin ``hash()`` is salted
        # per-interpreter, so reusing ``--seed`` across machines /
        # processes would produce different splits. Hash via sha256.
        slug_offset = int(
            hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8], 16
        ) % 100000
        rng_local = random.Random(args.seed + slug_offset)
        idxs = list(range(len(imgs)))
        rng_local.shuffle(idxs)

        n_tr = min(args.train_per_species, len(idxs))
        n_va = min(args.val_per_species, max(0, len(idxs) - n_tr))
        n_te = min(args.test_per_species, max(0, len(idxs) - n_tr - n_va))

        plan = (
            ("train", idxs[:n_tr]),
            ("val", idxs[n_tr:n_tr + n_va]),
            ("test", idxs[n_tr + n_va:n_tr + n_va + n_te]),
        )

        rec = descs[slug]
        for split_name, sel in plan:
            for i in sel:
                src = imgs[i]
                dst = (
                    args.output_dir
                    / "images_resized" / split_name / slug
                    / f"{src.stem}.jpg"
                )
                maybe_resize_and_save(src, dst, args.resize_to)
                question = rng.choice(QUESTION_TEMPLATES)
                splits[split_name].append(_build_record(
                    image_dst=dst, slug=slug, desc=rec, question=question,
                ))
        summary[slug] = {
            "train": n_tr, "val": n_va, "test": n_te,
            "total": n_tr + n_va + n_te,
        }
        log.info("  %s: train=%d val=%d test=%d", slug, n_tr, n_va, n_te)
    return splits, summary, rng


def _process_split(
    args: argparse.Namespace,
    descs: dict[str, dict[str, Any]],
    slugs: list[str],
) -> tuple[
    dict[str, list[dict]], dict[str, dict[str, int]], random.Random
]:
    """Respect fetcher-pre-computed splits (split-source mode)."""
    per_slug_per_split = discover_species_images_split(args.source_root, slugs)
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {s: [] for s in SPLIT_NAMES}
    summary: dict[str, dict[str, int]] = {}

    for slug in slugs:
        per_split = per_slug_per_split.get(slug) or {}
        if not any(per_split.values()):
            continue
        rec = descs[slug]
        per_summary = {"train": 0, "val": 0, "test": 0, "total": 0}
        for split_name in SPLIT_NAMES:
            imgs = per_split.get(split_name, [])
            for src in imgs:
                dst = (
                    args.output_dir
                    / "images_resized" / split_name / slug
                    / f"{src.stem}.jpg"
                )
                maybe_resize_and_save(src, dst, args.resize_to)
                question = rng.choice(QUESTION_TEMPLATES)
                splits[split_name].append(_build_record(
                    image_dst=dst, slug=slug, desc=rec, question=question,
                ))
            per_summary[split_name] = len(imgs)
        per_summary["total"] = sum(
            per_summary[s] for s in SPLIT_NAMES
        )
        if per_summary["total"] > 0:
            summary[slug] = per_summary
    return splits, summary, rng


def main(argv: list[str] | None = None) -> int:
    # Lazy import so module-level import doesn't pay env_paths cost.
    from data_mix.src.env_paths import (
        HIKECOMPANION_ROOT, external_data_root,
    )

    _ext = external_data_root()
    default_source = _ext / "inaturalist_na_plantae"
    default_output = _ext / "inaturalist_na_plantae_prepared"
    default_descriptions = (
        HIKECOMPANION_ROOT / "assets" / "na_plantae" / "descriptions.yaml"
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--source_root", type=Path, default=default_source,
        help="Directory with the raw images. Auto-detects split "
             "(<root>/{train,val,test}/<slug>/) vs flat (<root>/<slug>/) "
             f"layout. Default: {default_source}",
    )
    ap.add_argument(
        "--descriptions", type=Path, default=default_descriptions,
        help="YAML with the species answer + metadata. "
             f"Default: {default_descriptions}",
    )
    ap.add_argument(
        "--output_dir", type=Path, default=default_output,
        help=f"Where to write {{train,val,test}}.jsonl + "
             f"images_resized/. Default: {default_output}",
    )
    ap.add_argument(
        "--source_layout", choices=("auto", "flat", "split"),
        default="auto",
        help="Override source layout detection. Default: auto.",
    )
    ap.add_argument("--train_per_species", type=int, default=40,
                    help="(flat layout only) per-species train cap.")
    ap.add_argument("--val_per_species", type=int, default=5,
                    help="(flat layout only) per-species val cap.")
    ap.add_argument("--test_per_species", type=int, default=5,
                    help="(flat layout only) per-species test cap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resize_to",
        type=parse_resize,
        default=(DEFAULT_RESIZE_HW[0], DEFAULT_RESIZE_HW[1]),
        help="WxH to pre-resize images. 'none' disables. "
             "Default: 960x672 (iOS runtime).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    descs = load_descriptions(args.descriptions)
    log.info("Loaded %d species descriptions from %s",
             len(descs), args.descriptions)
    slugs = sorted(descs.keys())

    layout = args.source_layout
    if layout == "auto":
        layout = detect_source_layout(args.source_root)
    log.info("Source layout: %s (root=%s)", layout, args.source_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if layout == "split":
        splits, summary, rng = _process_split(args, descs, slugs)
    else:
        splits, summary, rng = _process_flat(args, descs, slugs)

    # Shuffle train globally so species aren't clumped together.
    rng.shuffle(splits["train"])

    for split_name in SPLIT_NAMES:
        rows = splits[split_name]
        path = args.output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info("Wrote %s (%d rows).", path, len(rows))

    log.info("")
    log.info("=== Summary ===")
    tot_tr = sum(s["train"] for s in summary.values())
    tot_va = sum(s["val"] for s in summary.values())
    tot_te = sum(s["test"] for s in summary.values())
    log.info(
        "Total: train=%d val=%d test=%d (across %d species)",
        tot_tr, tot_va, tot_te, len(summary),
    )

    with open(args.output_dir / "filter_report.json", "w") as f:
        json.dump({
            "source_root":  str(args.source_root),
            "descriptions": str(args.descriptions),
            "source_layout": layout,
            "split_per_species": {
                "train": args.train_per_species,
                "val":   args.val_per_species,
                "test":  args.test_per_species,
            },
            "resize_to": (
                [args.resize_to[1], args.resize_to[0]]
                if args.resize_to else None
            ),
            "totals": {"train": tot_tr, "val": tot_va, "test": tot_te},
            "per_species": summary,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
