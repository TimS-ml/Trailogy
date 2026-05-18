#!/usr/bin/env python3
"""Convert PlantNet-300K into ENRICHED instruction-tuning JSONL.

Differences vs ``prepare_plantnet.py``:

  * **English common names instead of Latin scientific names.** Each species
    is rendered as a single English vernacular name (e.g. ``"Eastern Hemlock"``)
    sourced either from the V2 enriched CSV ``common_names`` column directly,
    or (legacy V1 path) from GBIF's ``vernacularNames`` API responses cached at
    the species cache directory (see ``--species_cache``).

  * **One-sentence description appended.** Prefer the Wikipedia summary;
    fall back to the GBIF description when Wikipedia is missing. Capped at
    400 characters (cut at the first sentence boundary, else hard-truncated
    with an ellipsis) so each sample fits comfortably under ``max_seq_length``.

  * **Species without an English vernacular are DROPPED.** This filters out
    species that are obscure flora with only non-English names.
    A separate ``filter_report.json`` records why each species was dropped.

  * **Species with no metadata at all are DROPPED.** Species present in the
    image tree but absent from ``species_metadata_enriched.csv`` are dropped.

The output JSONL format is identical to ``prepare_plantnet.py``: one record
per training image, with ``image`` + ``conversations`` keys, ready to be
loaded by ``src.data.load_vision_dataset``.

This script DOES NOT touch the original ``prepare_plantnet.py`` or its
output. Run both side by side; switch between them via the ``data.train_file``
field in your training YAML.

Data version support
--------------------

* **V2 (default):** Uses ``PlantNet-300K-data-v2/species_metadata_enriched.csv``
  which has ``common_names`` directly (semicolon-separated; first name is used).
  No GBIF cache needed. Species IDs are zero-padded class indices (``0000``--``0999``).

* **V1 (legacy):** Uses the GBIF cache at the species cache directory
  (see ``--species_cache``) to look up English vernacular names via
  ``gbif_usage_key``. Species IDs are raw PlantNet IDs.

Example (V2)
-------------

    python -m src.prepare_plantnet_enriched \\
        --plantnet_root  ../../PlantNet-300K-data-v2 \\
        --output_dir     data/english-desc/ \\
        --max_samples    50000 \\
        --val_ratio      0.1 \\
        --seed           42

Example (V1, legacy)
---------------------

    python -m src.prepare_plantnet_enriched \\
        --plantnet_root  ../../PlantNet-300K-data/plantnet_300K/images \\
        --enriched_csv   <path-to>/species_metadata_enriched.csv \\
        --species_cache  <path-to>/.plantnet_species_cache \\
        --data_version   v1 \\
        --output_dir     data/english-desc/ \\
        --max_samples    50000 \\
        --val_ratio      0.1 \\
        --seed           42

Then point a training YAML at the output:

    data:
      train_file: data/english-desc/train.jsonl
      val_file:   data/english-desc/val.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse helpers from prepare_plantnet.py to keep image discovery + sampling
# + resize behavior bit-identical with the legacy pipeline. This is the
# whole point of subclassing the data-prep layer rather than forking it.
from src.prepare_plantnet import (
    DEFAULT_TRAINED_VISION_HW,
    class_stratified_split,
    discover_images,
    parse_resize_arg,
    resize_image_to_disk,
    stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# CSV cell size can exceed default csv.field_size_limit (some GBIF
# best_description fields run > 31 KB). Bump before any DictReader use.
csv.field_size_limit(10_000_000)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_FINETUNE_ROOT = _HERE.parent
_APP_ROOT = _FINETUNE_ROOT.parent.parent  # .../App/

# V2 default: enriched CSV ships alongside the dataset itself.
DEFAULT_ENRICHED_CSV_V2 = (
    _APP_ROOT / "PlantNet-300K-data-v2" / "species_metadata_enriched.csv"
)

# V1 legacy defaults — kept for backwards compatibility.
# Point these at whatever directory holds the GBIF enrichment outputs
# (species_metadata_enriched.csv and .plantnet_species_cache/).
DEFAULT_ENRICHED_CSV_V1 = (
    _APP_ROOT
    / "plantnet-enrich"
    / "species_metadata_enriched.csv"
)
DEFAULT_SPECIES_CACHE = (
    _APP_ROOT
    / "plantnet-enrich"
    / ".plantnet_species_cache"
)

# Alias — callers that don't care about version can use this; it points at V2.
DEFAULT_ENRICHED_CSV = DEFAULT_ENRICHED_CSV_V2

DEFAULT_DESCRIPTION_MAX_CHARS = 400

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

# Answer = "<intro> <CommonName>. <Description>".
# Per design: English common name only, no Latin (so the iOS TTS path
# doesn't have to read out "Tsuga canadensis"). The description is one
# sentence pulled from Wikipedia / GBIF, capped at DEFAULT_DESCRIPTION_MAX_CHARS.
ANSWER_TEMPLATES_RICH = [
    "This is {common}. {description}",
    "You're looking at {common}. {description}",
    "That's {common}. {description}",
    "Good eye — this is {common}. {description}",
    "That looks like {common}. {description}",
    "Looks like {common} to me. {description}",
    "You've spotted {common}. {description}",
    "This appears to be {common}. {description}",
]

# ---------------------------------------------------------------------------
# English vernacular extraction (from GBIF cache)
# ---------------------------------------------------------------------------

# Languages GBIF tags English under. The bare "(EN)" suffix is what the
# Catalogue of Life source emits when language is blank — treat as English.
_ENGLISH_LANG_TAGS = {"en", "eng", "english"}
_EN_SUFFIX_RE = re.compile(r"\s*\(EN\)\s*$", re.IGNORECASE)


def _clean_vernacular(name: str) -> str:
    """Strip GBIF's '(EN)' suffix and surrounding whitespace."""
    return _EN_SUFFIX_RE.sub("", name or "").strip()


def _looks_english(item: Dict[str, object]) -> bool:
    """True if a GBIF vernacularNames record looks English-tagged.

    GBIF data is inconsistent: Catalogue of Life leaves ``language`` blank
    and appends ``" (EN)"`` to the name; other sources set ``language``
    properly. Accept both signals.
    """
    raw_lang = item.get("language") if isinstance(item, dict) else None
    lang = (raw_lang or "").strip().lower() if isinstance(raw_lang, str) else ""
    if lang in _ENGLISH_LANG_TAGS:
        return True
    name = item.get("vernacularName") if isinstance(item, dict) else None
    if isinstance(name, str) and name.strip().lower().endswith("(en)"):
        return True
    return False


def pick_english_vernacular(items: List[Dict[str, object]]) -> Optional[str]:
    """Pick the best English vernacular from a list of GBIF records.

    Strategy: first record whose ``language in {en, eng, english}`` or whose
    ``vernacularName`` ends with ``"(EN)"``. Returns the cleaned name in
    Title Case (because the GBIF data is wildly inconsistent — sometimes
    ``"wild lettuce"``, sometimes ``"Wild Lettuce"``, sometimes
    ``"WILD LETTUCE"``). Returns None if no English entry exists.
    """
    for item in items:
        if not isinstance(item, dict):
            continue
        if not _looks_english(item):
            continue
        name = item.get("vernacularName")
        if not isinstance(name, str):
            continue
        cleaned = _clean_vernacular(name)
        if not cleaned:
            continue
        # Normalize to Title Case. Skips ALL-CAPS only fix: "Wild lettuce" → "Wild Lettuce".
        return _title_case(cleaned)
    return None


def _title_case(s: str) -> str:
    """Title-case a vernacular name, but preserve already-mixed-case words.

    Standard ``str.title()`` would mangle ``"O'Brien"`` → ``"O'Brien"`` (OK)
    but also ``"McKinley"`` → ``"Mckinley"`` (NOT OK). We only flip words
    that are entirely lowercase or entirely uppercase; anything with mixed
    case is assumed to be deliberate.
    """
    words = []
    for w in s.split():
        if w.islower() or w.isupper():
            words.append(w.capitalize())
        else:
            words.append(w)
    return " ".join(words)


def build_taxonkey_to_english_index(cache_dir: Path) -> Dict[str, str]:
    """Walk the GBIF vernacularNames cache → ``{taxonKey(str): english_name}``.

    Only files matching ``gbif_vernacularNames_*.json`` are read. The first
    English-tagged record per taxonKey wins. Returns an empty dict (with a
    warning) if the cache directory is missing — caller should treat the
    empty index as "filter EVERYTHING out" so we fail loud, not silent.
    """
    index: Dict[str, str] = {}
    if not cache_dir.is_dir():
        log.warning(
            "Species cache directory %s does not exist. No English vernacular "
            "names will be available; every PlantNet species will be dropped. "
            "Populate the cache by running the PlantNet enrichment script "
            "(enrich_plantnet300k_species.py) first.",
            cache_dir,
        )
        return index

    pattern = os.path.join(str(cache_dir), "gbif_vernacularNames_*.json")
    files = glob.glob(pattern)
    n_with_eng = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Skipping unreadable cache file %s (%s)", fp, exc)
            continue
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            continue
        # taxonKey is shared across results in a given file (it's a per-taxon
        # vernacularNames query). Take the first one available.
        taxon_key = None
        for r in results:
            if isinstance(r, dict):
                tk = r.get("taxonKey")
                if tk is not None:
                    taxon_key = str(tk)
                    break
        if taxon_key is None:
            continue
        english = pick_english_vernacular(results)
        if english:
            index[taxon_key] = english
            n_with_eng += 1
    log.info(
        "GBIF vernacular cache: %d files scanned, %d taxa with English names.",
        len(files),
        n_with_eng,
    )
    return index


# ---------------------------------------------------------------------------
# V2 direct common-name extraction (from enriched CSV column)
# ---------------------------------------------------------------------------


def pick_first_common_name(common_names_field: str) -> Optional[str]:
    """Pick the first common name from a semicolon-separated ``common_names`` field.

    V2's ``species_metadata_enriched.csv`` has a ``common_names`` column with
    entries like ``"Acrid lettuce; Bitter lettuce; Great Lettuce; wild lettuce"``.
    We simply take the first non-empty entry and Title-Case it.

    Returns None if the field is empty or contains only whitespace / semicolons.
    """
    if not common_names_field:
        return None
    for name in common_names_field.split(";"):
        name = name.strip()
        if name:
            return _title_case(name)
    return None


# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------

# Heuristic sentence splitter: split on ". " / "! " / "? " followed by a
# capital letter or end of string. We don't need NLTK-quality segmentation
# here; the goal is just "first complete sentence under 400 chars".
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def extract_first_sentence(text: str, max_chars: int) -> str:
    """Return the first sentence of ``text`` (or hard-truncate to ``max_chars``).

    * Collapses internal whitespace (handles GBIF's ``\\r\\r`` artifacts).
    * Splits on sentence-final ``.``, ``!``, ``?`` followed by whitespace and
      an uppercase / digit start. The first chunk is the candidate.
    * If the candidate is still longer than ``max_chars``, hard-truncate to
      ``max_chars - 1`` characters and append ``"…"``.
    * If ``text`` is empty / whitespace, returns ``""``.
    """
    if not text:
        return ""
    # Normalize whitespace: GBIF descriptions contain literal "\r\r"
    # paragraph breaks and tabs that bloat the token count.
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return ""

    # Try to split on first sentence boundary.
    parts = _SENTENCE_END_RE.split(flat, maxsplit=1)
    first = parts[0].strip()

    if len(first) <= max_chars:
        return first

    # First sentence still too long — hard truncate. Try to end at a
    # word boundary near the cap.
    cutoff = max_chars - 1  # reserve 1 char for the ellipsis
    truncated = flat[:cutoff]
    last_space = truncated.rfind(" ")
    if last_space > cutoff * 0.7:  # only respect word boundary if reasonably close
        truncated = truncated[:last_space]
    return truncated.rstrip(" .,;:-") + "…"


# ---------------------------------------------------------------------------
# Enriched-metadata loading + filtering
# ---------------------------------------------------------------------------


@dataclass
class SpeciesInfo:
    """The minimum a single kept species needs for JSONL emission."""

    species_id: str
    common_name: str
    description: str
    latin: str  # kept for the filter report only (not used in answers)


@dataclass
class FilterReport:
    kept: List[str] = field(default_factory=list)
    dropped_no_metadata: List[str] = field(default_factory=list)
    dropped_no_english: List[Tuple[str, str]] = field(default_factory=list)  # (sid, latin)
    dropped_no_description: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kept_count": len(self.kept),
            "dropped_no_metadata_count": len(self.dropped_no_metadata),
            "dropped_no_english_count": len(self.dropped_no_english),
            "dropped_no_description_count": len(self.dropped_no_description),
            "kept_species_ids": sorted(self.kept, key=int) if all(s.isdigit() for s in self.kept) else sorted(self.kept),
            "dropped_no_metadata": sorted(self.dropped_no_metadata, key=lambda s: int(s) if s.isdigit() else s),
            "dropped_no_english": sorted(self.dropped_no_english),
            "dropped_no_description": sorted(self.dropped_no_description),
        }


def build_species_index(
    enriched_csv: Path,
    taxonkey_to_english: Dict[str, str],
    description_max_chars: int,
    known_species_ids: Optional[set[str]] = None,
    *,
    use_csv_common_names: bool = False,
) -> Tuple[Dict[str, SpeciesInfo], FilterReport]:
    """Build ``{species_id: SpeciesInfo}`` from the enriched CSV + cache index.

    Filtering rules (applied in this order, recorded in ``FilterReport``):

      1. Drop rows with no English common name. The name source depends on
         ``use_csv_common_names``:

         * **False (V1 legacy):** English name is looked up from
           ``taxonkey_to_english`` via the row's ``gbif_usage_key``.
         * **True (V2):** English name is the first entry in the CSV's
           ``common_names`` column (semicolon-separated).

      2. Drop rows whose ``wikipedia_summary`` AND ``best_description`` are
         both empty.

    Additionally, any species in ``known_species_ids`` (i.e. seen in the
    PlantNet image tree) but absent from the CSV is recorded as
    ``dropped_no_metadata`` — this is informational only; the caller already
    filters by the kept dict.
    """
    report = FilterReport()
    kept: Dict[str, SpeciesInfo] = {}
    seen_csv_ids: set[str] = set()

    if not enriched_csv.exists():
        raise FileNotFoundError(
            f"Enriched metadata CSV not found at {enriched_csv}. "
            "Ensure the species_metadata_enriched.csv exists at this path."
        )

    with open(enriched_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # V2 CSVs have common_names; V1 CSVs require gbif_usage_key.
        if use_csv_common_names:
            required_cols = {
                "species_id",
                "species",
                "common_names",
                "wikipedia_summary",
                "best_description",
            }
        else:
            required_cols = {
                "species_id",
                "species",
                "gbif_usage_key",
                "wikipedia_summary",
                "best_description",
            }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Enriched CSV {enriched_csv} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            sid = (row.get("species_id") or "").strip()
            latin = (row.get("species") or "").strip()
            wiki = (row.get("wikipedia_summary") or "").strip()
            gbif_desc = (row.get("best_description") or "").strip()
            seen_csv_ids.add(sid)

            # --- English common name ---
            if use_csv_common_names:
                common_names_raw = (row.get("common_names") or "").strip()
                english = pick_first_common_name(common_names_raw)
            else:
                usage_key = (row.get("gbif_usage_key") or "").strip()
                english = taxonkey_to_english.get(usage_key) if usage_key else None

            if not english:
                report.dropped_no_english.append((sid, latin))
                continue

            # Prefer Wikipedia summary, fall back to GBIF best_description.
            description = extract_first_sentence(wiki, description_max_chars)
            if not description:
                description = extract_first_sentence(gbif_desc, description_max_chars)
            if not description:
                report.dropped_no_description.append((sid, latin))
                continue

            kept[sid] = SpeciesInfo(
                species_id=sid,
                common_name=english,
                description=description,
                latin=latin,
            )
            report.kept.append(sid)

    # Species we saw in PlantNet image tree but couldn't find in CSV.
    if known_species_ids:
        for sid in sorted(known_species_ids - seen_csv_ids):
            report.dropped_no_metadata.append(sid)

    return kept, report


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------


def build_enriched_conversation(
    image_path: Path,
    species: SpeciesInfo,
    rng: random.Random,
) -> Dict[str, object]:
    """Build one JSONL record using ANSWER_TEMPLATES_RICH."""
    question = rng.choice(QUESTION_TEMPLATES)
    answer = rng.choice(ANSWER_TEMPLATES_RICH).format(
        common=species.common_name,
        description=species.description,
    )
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
        description=(
            "Convert PlantNet-300K into ENRICHED instruction-tuning JSONL "
            "using English common names + short descriptions, dropping "
            "species without English vernaculars."
        )
    )
    parser.add_argument(
        "--plantnet_root",
        type=str,
        required=True,
        help="Path to PlantNet-300K root (contains train/, val/, test/).",
    )
    parser.add_argument(
        "--enriched_csv",
        type=str,
        default=None,
        help=(
            "Path to species_metadata_enriched.csv. Default depends on "
            "--data_version: V2 → PlantNet-300K-data-v2/species_metadata_enriched.csv, "
            "V1 → plantnet-enrich/species_metadata_enriched.csv."
        ),
    )
    parser.add_argument(
        "--species_cache",
        type=str,
        default=str(DEFAULT_SPECIES_CACHE),
        help=(
            "Path to GBIF cache dir (.plantnet_species_cache). Only used "
            "with --data_version v1. The vernacularNames JSON files are read "
            "to filter species without English common names."
        ),
    )
    parser.add_argument(
        "--data_version",
        type=str,
        default="v2",
        choices=("v1", "v2"),
        help=(
            "Data layout version (default: v2). V2 reads common_names "
            "directly from the enriched CSV (no GBIF cache needed). "
            "V1 uses the GBIF cache for English name lookup."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help=(
            "Which image split to discover images from (default: train). "
            "Useful when only test/ is available (e.g. during V2 data sync)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write train.jsonl, val.jsonl, filter_report.json.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Max training samples to emit (default: 50000).",
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
            "Pre-resize images to HxW before writing the JSONL "
            "(default 960x672 matches iOS runtime). Pass 'none' to disable."
        ),
    )
    parser.add_argument(
        "--resized_image_root",
        type=str,
        default=None,
        help=(
            "Where to write resized image copies. Defaults to "
            "<output_dir>/images_resized/. Reused across runs."
        ),
    )
    parser.add_argument(
        "--description_max_chars",
        type=int,
        default=DEFAULT_DESCRIPTION_MAX_CHARS,
        help=(
            "Hard cap on description length (default: "
            f"{DEFAULT_DESCRIPTION_MAX_CHARS}). First-sentence cut + word "
            "boundary truncation if needed."
        ),
    )
    args = parser.parse_args()

    # Resolve enriched CSV default based on data version.
    if args.enriched_csv is None:
        if args.data_version == "v2":
            args.enriched_csv = str(DEFAULT_ENRICHED_CSV_V2)
        else:
            args.enriched_csv = str(DEFAULT_ENRICHED_CSV_V1)

    use_v2 = args.data_version == "v2"
    rng = random.Random(args.seed)
    root = Path(args.plantnet_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    target_hw = parse_resize_arg(args.resize_to)
    if target_hw is None:
        log.info("Image resize: DISABLED.")
        resized_root: Optional[Path] = None
    else:
        resized_root = (
            Path(args.resized_image_root)
            if args.resized_image_root
            else out / "images_resized"
        )
        resized_root.mkdir(parents=True, exist_ok=True)
        log.info(
            "Image resize: ENABLED. Stretching to %dx%d → %s",
            target_hw[0], target_hw[1], resized_root,
        )

    # 1) English vernacular names.
    if use_v2:
        log.info(
            "Data version: V2 — reading common_names from enriched CSV directly "
            "(GBIF cache not needed)."
        )
        taxonkey_to_english: Dict[str, str] = {}  # unused in V2 mode
    else:
        log.info("Data version: V1 — loading GBIF vernacular cache from %s ...", args.species_cache)
        taxonkey_to_english = build_taxonkey_to_english_index(Path(args.species_cache))

    # 2) Discover available species in the image tree.
    split_dir = root / args.split
    if not split_dir.is_dir():
        log.error("Image split '%s' not found at %s", args.split, split_dir)
        raise SystemExit(1)
    raw_classes = discover_images(split_dir)

    # V2 folders are zero-padded ("0000"–"0999") but the CSV uses unpadded
    # species_id ("0"–"999"). Normalize folder-derived IDs to match by
    # stripping leading zeros. V1 IDs are large integers (e.g. "1355868")
    # that don't have leading zeros, so this is a no-op for V1.
    classes: Dict[str, List[Path]] = {}
    for raw_sid, imgs in raw_classes.items():
        normalized = raw_sid.lstrip("0") or "0"  # "0000" → "0", "0042" → "42"
        classes[normalized] = imgs

    total_images = sum(len(v) for v in classes.values())
    log.info(
        "Discovered %d images across %d species in %s",
        total_images, len(classes), split_dir,
    )

    # 3) Build the keep-set from enriched CSV + cache.
    species_index, report = build_species_index(
        enriched_csv=Path(args.enriched_csv),
        taxonkey_to_english=taxonkey_to_english,
        description_max_chars=args.description_max_chars,
        known_species_ids=set(classes.keys()),
        use_csv_common_names=use_v2,
    )
    log.info(
        "Species filter: kept %d / dropped %d (no English: %d, no description: %d, no metadata: %d)",
        len(species_index),
        len(report.dropped_no_english) + len(report.dropped_no_description) + len(report.dropped_no_metadata),
        len(report.dropped_no_english),
        len(report.dropped_no_description),
        len(report.dropped_no_metadata),
    )

    # 4) Restrict image classes dict to kept species.
    classes_kept = {sid: imgs for sid, imgs in classes.items() if sid in species_index}
    if not classes_kept:
        raise SystemExit(
            "No PlantNet species survived filtering. Check that the GBIF "
            "cache and enriched CSV are populated; rerun the enrich script "
            "if needed."
        )
    images_after_filter = sum(len(v) for v in classes_kept.values())
    log.info(
        "Images after species filter: %d (was %d, %.1f%% retained).",
        images_after_filter,
        total_images,
        100.0 * images_after_filter / max(total_images, 1),
    )

    # 5) Stratified sample, per-species split, emit.
    samples = stratified_sample(classes_kept, args.max_samples, rng)
    log.info("Sampled %d images (budget: %d)", len(samples), args.max_samples)

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
        # Test-stage promotion path (val_ratio=1.0 → everything to val).
        # See class_stratified_split docstring + prepare_plantnet_50k.sh Step 2.
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

    stats: Dict[str, int] = {"train": 0, "val": 0}
    classes_seen: set[str] = set()
    resize_failures = 0

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        out_path = out / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for sid, img_path in split_samples:
                info = species_index[sid]
                final_image_path = img_path
                if target_hw is not None and resized_root is not None:
                    dest = resized_root / split_name / sid / img_path.name
                    try:
                        final_image_path = resize_image_to_disk(
                            img_path, dest, target_hw
                        )
                    except Exception as exc:
                        resize_failures += 1
                        log.warning(
                            "Resize failed for %s (%s); falling back to original.",
                            img_path, exc,
                        )
                record = build_enriched_conversation(final_image_path, info, rng)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats[split_name] += 1
                classes_seen.add(sid)
        log.info("Wrote %d records to %s", stats[split_name], out_path)

    if resize_failures:
        log.warning(
            "%d image(s) could not be resized — JSONL points at originals.",
            resize_failures,
        )

    # 6) Filter report (for transparency + reproducibility).
    report_path = out / "filter_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    log.info("Filter report → %s", report_path)

    log.info("--- Summary ---")
    log.info("  Train samples : %d", stats["train"])
    log.info("  Val samples   : %d", stats["val"])
    log.info("  Species used  : %d / %d kept after filter", len(classes_seen), len(species_index))
    log.info("  Output dir    : %s", out)


if __name__ == "__main__":
    main()
