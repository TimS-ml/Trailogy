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


def synthesize_descriptions_from_observations(
    observations_jsonl: Path,
) -> dict[str, dict[str, Any]]:
    """Build {slug: record} for every unique slug in observations.jsonl.

    Used as a fallback so the 800-species bulk fetch from ``na_plantae_
    fetch.py`` doesn't require 800 hand-curated description entries.
    The synthesized records carry the minimum fields ``_build_record``
    consumes (``slug``, ``species``, ``family``, ``answer``):

      - ``species``  = the iNat ``scientific_name`` (Latin binomial)
      - ``family``   = ``"(unknown)"`` — iNat /v1/observations doesn't
        return family inline; populating it would cost ~800 extra
        /v1/taxa calls. The field is required by the output JSONL
        schema but not by the answer template here.
      - ``answer``   = a single-shot canonical assistant turn matching
        the shape of the curated answers in
        ``assets/na_plantae/descriptions.yaml`` (lead-in + Latin name
        recall), without the family clause.

    Curated entries from ``descriptions.yaml`` override the synthesized
    ones at merge time — see ``main()``.
    """
    out: dict[str, dict[str, Any]] = {}
    with open(observations_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            slug = rec.get("slug")
            if not slug or slug in out:
                continue
            common = (
                rec.get("common_name")
                or slug.replace("_", " ")
            )
            species = rec.get("scientific_name") or "(unknown)"
            answer = (
                f"Looks like {common} to me. {species}, commonly called "
                f"{common}, is a plant species found in North America."
            )
            out[slug] = {
                "slug":        slug,
                "common_name": common,
                "species":     species,
                "family":      "(unknown)",
                "answer":      answer,
            }
    return out


def load_rolled_descriptions(
    rolled_jsonl: Path,
) -> dict[str, dict[str, Any]]:
    """Build the ``{slug: desc}`` mapping from ``species_enriched_rolled.jsonl``.

    The answer template matches ``synthesize_descriptions_from_observations``
    so eval / training records look identical to the non-rollup path
    once the binomial substitution lands. We do NOT carry the binomial's
    Wikipedia / GBIF prose into the answer here — that lives in the RAG
    layer; the SFT target stays the short "Looks like X to me. Y, …"
    canonical form.
    """
    out: dict[str, dict[str, Any]] = {}
    with open(rolled_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            slug = rec.get("slug")
            if not slug or slug in out:
                continue
            common = (
                rec.get("common_name")
                or slug.replace("_", " ")
            )
            species = rec.get("scientific_name") or "(unknown)"
            answer = (
                f"Looks like {common} to me. {species}, commonly called "
                f"{common}, is a plant species found in North America."
            )
            out[slug] = {
                "slug":        slug,
                "common_name": common,
                "species":     species,
                "family":      "(unknown)",
                "answer":      answer,
            }
    return out


def build_slug_rewrite_map(
    rolled_rows: list[dict[str, Any]],
    observations_jsonl: Path,
) -> dict[str, str]:
    """Return ``{child_slug: parent_slug}`` for the rollup remap.

    The rolled JSONL records the parent's slug + its children's
    *scientific* names (``child_taxa``). We need the *child*'s slug to
    know which on-disk image folder to redirect into the parent. We
    pull that lookup from ``observations.jsonl`` (the fetcher records
    sci_name + slug per observation).

    Edge cases handled:
      * a ``child_taxa`` entry whose scientific name isn't in
        observations.jsonl (partial data) is silently skipped;
      * a child whose slug equals the parent's (iNat shares a single
        common-name slug across the species + its indicating subspecies
        — happened in our dataset for ``Eriophyllum confertiflorum``)
        emits no entry, since the rewrite would be a no-op.
    """
    sci_to_slug: dict[str, str] = {}
    with open(observations_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sci = rec.get("scientific_name")
            slug = rec.get("slug")
            if sci and slug and sci not in sci_to_slug:
                sci_to_slug[sci] = slug
    rewrite: dict[str, str] = {}
    for row in rolled_rows:
        parent_slug = row.get("slug")
        if not parent_slug:
            continue
        for child_sci in row.get("child_taxa") or []:
            child_slug = sci_to_slug.get(child_sci)
            if not child_slug or child_slug == parent_slug:
                continue
            rewrite[child_slug] = parent_slug
    return rewrite


def detect_source_layout(source_root: Path) -> str:
    """``split`` if any of train/val/test exist as subdirs of source_root;
    otherwise ``flat``."""
    for s in SPLIT_NAMES:
        if (source_root / s).is_dir():
            return "split"
    return "flat"


def discover_species_images_flat(
    source_root: Path,
    slugs: list[str],
    slug_rewrite: dict[str, str] | None = None,
) -> dict[str, list[Path]]:
    """``{slug: sorted [Path, ...]}`` reading ``<source_root>/<slug>/*.jpg``.

    When ``slug_rewrite`` is given, walk ALL subdirs of ``source_root``
    and merge any folder whose name appears as a key in the map into
    the mapped (parent) slug. Folders not in the map keep their name.
    Folders whose post-rewrite name isn't in ``slugs`` are ignored
    (they belong to species we don't have a description for).
    """
    if not slug_rewrite:
        out: dict[str, list[Path]] = {}
        for slug in slugs:
            d = source_root / slug
            if not d.is_dir():
                log.warning("species folder missing (flat): %s", d)
                out[slug] = []
                continue
            files = sorted(
                p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS
            )
            out[slug] = files
            log.info("  %s: %d images", slug, len(files))
        return out

    accepted = set(slugs)
    out = {s: [] for s in slugs}
    for slug_dir in sorted(source_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        target_slug = slug_rewrite.get(slug_dir.name, slug_dir.name)
        if target_slug not in accepted:
            continue
        out[target_slug].extend(
            sorted(p for p in slug_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        )
    for slug in slugs:
        if out[slug]:
            log.info("  %s: %d images", slug, len(out[slug]))
        else:
            log.warning("  %s: no images under %s", slug, source_root)
    return out


def discover_species_images_split(
    source_root: Path,
    slugs: list[str],
    slug_rewrite: dict[str, str] | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """``{slug: {split: sorted [Path, ...]}}`` reading
    ``<source_root>/<split>/<slug>/*.jpg``.

    When ``slug_rewrite`` is given, any sub-folder whose name is a key
    in the map is treated as belonging to the mapped (binomial)
    parent. Multiple child folders can therefore contribute to the
    same parent slug — that's the whole point of the rollup mode.

    Slugs that don't appear in ANY split (after rewrite) are logged
    once. Folders whose post-rewrite slug isn't in ``slugs`` (= no
    description for that species) are surfaced as a single WARN.
    """
    slug_rewrite = slug_rewrite or {}
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
            target_slug = slug_rewrite.get(slug_dir.name, slug_dir.name)
            if target_slug not in out:
                # Fetcher pulled this slug but we have no description
                # for it (and no rewrite into a known slug); track for
                # the post-walk WARN.
                continue
            out[target_slug].setdefault(split_name, []).extend(files)

    # In rollup mode, child slugs aren't in the `slugs` list (only
    # binomial parents are) but they're explicitly mapped — so they're
    # not "dropped", they were merged. Subtract those before warning.
    dropped = fetched_slugs_with_imgs - set(slugs) - set(slug_rewrite.keys())
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
    slug_rewrite: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[dict]], dict[str, dict[str, int]], random.Random
]:
    """Per-species random split (legacy flat-source mode)."""
    species_imgs = discover_species_images_flat(
        args.source_root, slugs, slug_rewrite=slug_rewrite,
    )
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


def _apply_species_caps(
    splits: dict[str, list[dict]],
    summary: dict[str, dict[str, int]],
    min_imgs: int,
    max_imgs: int,
    rng: random.Random,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """Filter ``splits`` + ``summary`` so each species' total image count
    is in ``[min_imgs, max_imgs]``.

    * Species with total < ``min_imgs`` are dropped from every split.
      Rationale: a tail class with only a handful of images can't
      generalize across photo angle / lighting / phenology — the model
      will either memorize the specific shots or learn nothing. Either
      way it steals capacity from learnable classes.
    * Species with total > ``max_imgs`` get their train rows trimmed
      (held-out val + test are NEVER touched — that would corrupt eval
      integrity). The trim is uniform-random across the train rows of
      that species using ``rng``.

    A ``min_imgs <= 0`` or ``max_imgs <= 0`` argument disables the
    respective bound (treats it as +inf / -inf).
    """
    # Group train indices by slug so we can sample down without
    # rebuilding the whole list per species.
    train_idx_by_slug: dict[str, list[int]] = {}
    for i, rec in enumerate(splits["train"]):
        train_idx_by_slug.setdefault(rec["slug"], []).append(i)

    drop_slugs: set[str] = set()
    train_drop_idxs: set[int] = set()

    effective_min = min_imgs if min_imgs > 0 else 0
    effective_max = max_imgs if max_imgs > 0 else 10**18

    for slug, per in summary.items():
        total = per.get("total", per["train"] + per["val"] + per["test"])
        if total < effective_min:
            drop_slugs.add(slug)
            continue
        if total > effective_max:
            # Trim train rows by ``overflow`` so total comes down to
            # exactly ``effective_max``. Never trim val/test.
            overflow = total - effective_max
            train_idxs = list(train_idx_by_slug.get(slug, []))
            n_drop = min(overflow, len(train_idxs))
            if n_drop > 0:
                drop_local = rng.sample(train_idxs, n_drop)
                train_drop_idxs.update(drop_local)

    new_splits: dict[str, list[dict]] = {
        "train": [
            r for i, r in enumerate(splits["train"])
            if r["slug"] not in drop_slugs and i not in train_drop_idxs
        ],
        "val": [r for r in splits["val"] if r["slug"] not in drop_slugs],
        "test": [r for r in splits["test"] if r["slug"] not in drop_slugs],
    }

    # Rebuild summary from the filtered splits to stay in sync.
    new_summary: dict[str, dict[str, int]] = {}
    for split_name, rows in new_splits.items():
        for r in rows:
            entry = new_summary.setdefault(
                r["slug"], {"train": 0, "val": 0, "test": 0, "total": 0},
            )
            entry[split_name] += 1
            entry["total"] += 1
    return new_splits, new_summary


def _add_species_cap_args(ap: argparse.ArgumentParser) -> None:
    """Register --min-imgs-per-species / --max-imgs-per-species on
    an existing argparse parser. Pulled out so the defaults can be
    introspected by both ``main()`` and the unit tests."""
    ap.add_argument(
        "--min-imgs-per-species", "--min_imgs_per_species",
        dest="min_imgs_per_species", type=int, default=30,
        help="Drop any species whose total (train+val+test) image "
             "count is below this threshold. Default: 30. "
             "Set 0 to disable.",
    )
    ap.add_argument(
        "--max-imgs-per-species", "--max_imgs_per_species",
        dest="max_imgs_per_species", type=int, default=120,
        help="Cap any species' total image count at this value by "
             "trimming train rows (val + test preserved). "
             "Default: 120. Set 0 to disable.",
    )


def _process_split(
    args: argparse.Namespace,
    descs: dict[str, dict[str, Any]],
    slugs: list[str],
    slug_rewrite: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[dict]], dict[str, dict[str, int]], random.Random
]:
    """Respect fetcher-pre-computed splits (split-source mode)."""
    per_slug_per_split = discover_species_images_split(
        args.source_root, slugs, slug_rewrite=slug_rewrite,
    )
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
    _add_species_cap_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resize_to",
        type=parse_resize,
        default=(DEFAULT_RESIZE_HW[0], DEFAULT_RESIZE_HW[1]),
        help="WxH to pre-resize images. 'none' disables. "
             "Default: 960x672 (iOS runtime).",
    )
    ap.add_argument(
        "--synthesize_missing", action=argparse.BooleanOptionalAction,
        default=True,
        help="When set (default), auto-synthesize description records "
             "from <source_root>/observations.jsonl for slugs not "
             "present in --descriptions. Curated entries always win. "
             "Set --no-synthesize_missing to recover the strict pre-v4 "
             "behaviour (drop fetched species without curated entries).",
    )
    ap.add_argument(
        "--observations_jsonl", type=Path, default=None,
        help="observations.jsonl to read for synthesized descriptions. "
             "Default: <source_root>/observations.jsonl.",
    )
    ap.add_argument(
        "--rollup-to-species", dest="rollup_to_species",
        type=Path, default=None,
        help="Path to species_enriched_rolled.jsonl from rollup_to_species.py. "
             "When set, every fetched <split>/<child_slug>/ image folder is "
             "merged into its binomial parent; output JSONL class_ids use the "
             "binomial slug. Replaces curated YAML + observations synthesis "
             "as the description source.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    slug_rewrite: dict[str, str] | None = None

    if args.rollup_to_species:
        # Rollup mode: rolled JSONL is the sole description source, and
        # the rewrite map redirects child trinomial folders into their
        # binomial parents. Curated YAML / observations-synthesize are
        # bypassed — the rolled file already absorbed both.
        if not args.rollup_to_species.exists():
            log.error("--rollup-to-species not found: %s",
                      args.rollup_to_species)
            return 2
        descs = load_rolled_descriptions(args.rollup_to_species)
        log.info(
            "Loaded %d binomial descriptions from %s",
            len(descs), args.rollup_to_species,
        )
        obs_path = (
            args.observations_jsonl
            or (args.source_root / "observations.jsonl")
        )
        if not obs_path.exists():
            log.error(
                "Rollup mode needs observations.jsonl (got: %s missing). "
                "Pass --observations_jsonl explicitly.",
                obs_path,
            )
            return 2
        with open(args.rollup_to_species) as f:
            rolled_rows = [json.loads(line) for line in f if line.strip()]
        slug_rewrite = build_slug_rewrite_map(rolled_rows, obs_path)
        log.info(
            "Built slug rewrite map: %d child slugs → binomial parents",
            len(slug_rewrite),
        )
    else:
        descs = load_descriptions(args.descriptions)
        log.info("Loaded %d curated species descriptions from %s",
                 len(descs), args.descriptions)

        if args.synthesize_missing:
            obs_path = (
                args.observations_jsonl
                or (args.source_root / "observations.jsonl")
            )
            if not obs_path.exists():
                log.warning(
                    "--synthesize_missing set but %s missing; falling back "
                    "to curated descriptions only.",
                    obs_path,
                )
            else:
                synth = synthesize_descriptions_from_observations(obs_path)
                n_new = 0
                for slug, rec in synth.items():
                    if slug not in descs:  # curated always wins
                        descs[slug] = rec
                        n_new += 1
                log.info(
                    "Synthesized %d additional descriptions from %s "
                    "(curated entries preserved as overrides). "
                    "Total now: %d.",
                    n_new, obs_path, len(descs),
                )

    slugs = sorted(descs.keys())

    layout = args.source_layout
    if layout == "auto":
        layout = detect_source_layout(args.source_root)
    log.info("Source layout: %s (root=%s)", layout, args.source_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if layout == "split":
        splits, summary, rng = _process_split(
            args, descs, slugs, slug_rewrite=slug_rewrite,
        )
    else:
        splits, summary, rng = _process_flat(
            args, descs, slugs, slug_rewrite=slug_rewrite,
        )

    # Per-species min/max image-count caps. Tail (< min) classes are
    # dropped (unlearnable); head (> max) classes have train rows
    # uniformly trimmed (val/test preserved).
    pre_classes = len(summary)
    pre_total = sum(s["total"] for s in summary.values())
    splits, summary = _apply_species_caps(
        splits,
        summary,
        min_imgs=args.min_imgs_per_species,
        max_imgs=args.max_imgs_per_species,
        rng=rng,
    )
    post_total = sum(s["total"] for s in summary.values())
    log.info(
        "Species caps applied (min=%d, max=%d): "
        "%d -> %d classes, %d -> %d images.",
        args.min_imgs_per_species, args.max_imgs_per_species,
        pre_classes, len(summary), pre_total, post_total,
    )

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
            "species_caps": {
                "min_imgs_per_species": args.min_imgs_per_species,
                "max_imgs_per_species": args.max_imgs_per_species,
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
