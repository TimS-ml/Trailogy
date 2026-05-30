#!/usr/bin/env python3
"""Build the v2-enrich variant of the NA-Plantae prepared dataset.

Reads:
  * ``<input_root>/{train,val,test}.jsonl`` — the canonical prepared
    NA-Plantae JSONLs (per-image rows, conversations[1] is the short
    canonical caption "Looks like X to me. Y, commonly called X, is a
    plant species found in North America.").
  * ``<enriched_jsonl>`` — one row per species with GBIF / Wikipedia
    content (built by ``enrich_na_plantae.py``).

Writes:
  * ``<output_root>/{train,val,test}.jsonl`` — same rows as the input,
    but ``conversations[1]["content"]`` is rebuilt as a longer prose
    target that concatenates the content-bearing fields of the
    enrichment record.
  * ``<output_root>/images_resized`` — **symlink** to the input root's
    ``images_resized`` directory. Skipping the copy avoids duplicating
    ~16 GB of plant images. Train rows reference absolute image paths
    so the symlink isn't strictly required for them to load; it's
    here so downstream tools that scan ``<output_root>`` see the same
    layout the input has.
  * ``<output_root>/build_report.json`` — audit trail (input counts,
    enrichment coverage, slugs missing enrichment, average token
    counts before/after).

Eval-side contract:
  * Sentence 1 of the rebuilt caption is ALWAYS exactly
    ``"Looks like {common_name} to me."``. ``evaluate_generality.py``
    extracts ``pred_species`` from this position; if the leading
    template changes the entire plant scorer breaks silently.
  * The v2-enrich captions are intended for the **train** stream
    only. To keep ``eval_*_loss`` cross-mix comparable, route the
    eval/val files through the existing ``freeze_val_from`` flow at
    the mix-build layer (see ``mix-50k-v2.yaml``). This script
    rewrites val.jsonl + test.jsonl too for completeness, but the
    consumer (mix builder) is free to hardlink the v1/v2 val files
    over the top.

Field policy:
  Included content fields (per the rationale in
  ``02-datamix-sft/docs/`` v3 design notes):
    - ``scientific_name`` (+ ``accepted_scientific_name`` when different)
    - ``common_names`` (GBIF varietals; deduped against the primary)
    - ``wikipedia_summary`` (preferred) or ``best_description``
    - ``gbif_distribution`` (capped to first ``MAX_REGIONS`` regions)

  Explicitly EXCLUDED:
    - ``gbif_description`` — high false-positive rate for plant ID
      signal (Latin-typification history, Colombian-reserve site
      lists, herbarium specimen codes; all real examples).
    - ``gbif_profile`` — almost always
      ``"habitat: Terrestrial"`` repeated 6×, sometimes with a
      multilingual JSON blob mixed in (real example:
      ``lifeForm: {"lifeForm":["Árvore"],"habitat":["Terrícola"]}``
      in Portuguese for the ``balsam_fir`` record).
    - URLs / IDs / counts / source-of-source meta fields
      (``gbif_url``, ``wikipedia_url``, ``powo_search_url``,
      ``gbif_usage_key``, ``gbif_match_type``, ``gbif_confidence``,
      ``gbif_status``, ``n_observations``, ``n_photos``,
      ``fetch_status``, ``gbif_description_source``,
      ``gbif_description_language``, ``best_description_source``,
      ``wikipedia_title``, ``rank``, ``slug``).
    - ``rag_text`` (it's a pre-aggregation of the included fields in
      ``key: value`` format; the rebuild here uses prose form).

Usage::

    # Original v2enrich (long captions, ~800 chars / row)
    python src/data_mix/src/build_enriched_captions.py \\
        --input-root  data/inaturalist_na_plantae_prepared \\
        --enriched    data/inaturalist_na_plantae/species_enriched.jsonl \\
        --output-root data/inaturalist_na_plantae_prepared_v2enrich

    # v2enrich-lite (~300 chars / row, common name >=3x, scientific >=2x)
    python src/data_mix/src/build_enriched_captions.py \\
        --mode lite_nameboost \\
        --input-root  data/inaturalist_na_plantae_prepared \\
        --enriched    data/inaturalist_na_plantae/species_enriched.jsonl \\
        --output-root data/inaturalist_na_plantae_prepared_v2enrich_lite
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("data_mix.build_enriched_captions")

# Caps to keep the long tail under control. The 99th-percentile of
# distribution-region counts on the current 956-species enrichment is
# ~60 regions (Taraxacum officinale tops out at 70+); 10 covers the
# main native range for every species in the eval set without bloating
# the caption.
MAX_DISTRIBUTION_REGIONS = 10
# Hard cap on the assistant content so a single pathological enrichment
# record can't drag a whole training batch's padding cost up. Default
# allows ~1200 chars ≈ 200-250 tokens, which is ~10x the legacy 25-token
# target. Triggered cases get logged so they're not silent.
MAX_CONTENT_CHARS = 1200

# ---- lite_nameboost mode constants ------------------------------------
# Lite mode targets ~250-400 chars / row to pull the NA-Plantae
# token-mass share back toward the v2 baseline (~24 %) while keeping
# enrichment content for the model to attend to. The "name-boost"
# part: the common name appears 3x and the scientific name 2x per row,
# pushing the species-name token gradient share back from v2enrich's
# ~1.5 % toward v2's ~7-8 %.
MAX_CONTENT_CHARS_LITE = 500
MAX_DISTRIBUTION_REGIONS_LITE = 3
MAX_WIKI_SENTENCE_CHARS_LITE = 180

CAPTION_MODES = ("full", "lite_nameboost")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _dedupe_common_names(
    gbif_names_str: str | None, primary: str | None
) -> list[str]:
    if not gbif_names_str:
        return []
    names = [n.strip() for n in gbif_names_str.split(";") if n.strip()]
    out: list[str] = []
    # Case-only dedupe. Drops exact case-variants of the primary
    # common name (e.g. "Common Pitcher Plant" when primary is
    # "common pitcher plant") and within-list case duplicates. Does
    # NOT collapse hyphen / spacing variants — "red osier dogwood",
    # "Red-osier Dogwood", and "Redosier" all survive. That's
    # intentional for now: the model sees the spelling diversity as
    # additional signal. Tighter normalisation (collapse hyphens to
    # spaces, drop punctuation) is a follow-up if the variants prove
    # noisy in eval.
    seen: set[str] = {primary.lower()} if primary else set()
    for n in names:
        kl = n.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(n)
    return out


def _truncate_distribution(s: str | None, max_regions: int) -> str | None:
    if not s:
        return None
    regions = [r.strip() for r in s.split(";") if r.strip()]
    if not regions:
        return None
    if len(regions) > max_regions:
        n_extra = len(regions) - max_regions
        regions = regions[:max_regions]
        regions.append(f"and {n_extra} more regions")
    return "; ".join(regions)


def _first_sentence(text: str | None, max_chars: int) -> str | None:
    """Extract the first sentence of ``text``, hard-capped at ``max_chars``.

    Splits on the first ``". "`` boundary and returns ``"<first>."``. If
    the first sentence exceeds ``max_chars``, hard-truncate at
    ``max_chars - 1`` chars and append ``"…"``. Used by lite_nameboost
    mode to keep the Wikipedia blurb on a tight budget while preserving
    the leading clause that almost always contains the scientific
    name.
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    # Split on ". " (most Wikipedia summaries are well-punctuated). Fall
    # back to the whole string if there is no sentence boundary inside
    # max_chars.
    idx = s.find(". ")
    if 0 < idx < max_chars:
        first = s[: idx + 1]
    else:
        first = s
    if len(first) > max_chars:
        first = first[: max_chars - 1].rstrip() + "…"
    return first


def build_enriched_answer(
    common_name: str,
    scientific_name: str,
    enriched: dict[str, Any] | None,
) -> str:
    """Produce the rich assistant target for a single image row.

    When ``enriched`` is ``None`` (slug not present in
    species_enriched.jsonl), fall back to a compact version that
    keeps the eval anchor and the canonical scientific-name clause
    but drops the legacy "is a plant species found in North America"
    tail (per user instruction).
    """
    # Sentence 1 — DO NOT CHANGE. The eval scorer's species extractor
    # (Trailogy/src/finetune/eval/evaluate_generality.py:_SPECIES_PHRASE_RE)
    # depends on this leading "Looks like {X} to me." phrase to pull
    # pred_species out. Touch it and every plant eval breaks silently.
    parts: list[str] = [f"Looks like {common_name} to me."]

    if enriched is None:
        # Compact fallback: scientific name only.
        if scientific_name and scientific_name != "(unknown)":
            parts.append(
                f"{scientific_name}, commonly called {common_name}."
            )
        return " ".join(parts)

    # Scientific name + accepted name (when different).
    accepted = enriched.get("accepted_scientific_name") or scientific_name
    if accepted and accepted != scientific_name:
        parts.append(
            f"Scientific name: {scientific_name} "
            f"(accepted: {accepted})."
        )
    elif scientific_name and scientific_name != "(unknown)":
        parts.append(f"Scientific name: {scientific_name}.")

    # Other common names from GBIF.
    other = _dedupe_common_names(enriched.get("common_names"), common_name)
    if other:
        parts.append(f"Other common names: {'; '.join(other)}.")

    # Wikipedia summary / best_description (~86 % coverage).
    blurb = enriched.get("wikipedia_summary") or enriched.get(
        "best_description"
    )
    if blurb:
        parts.append(blurb.strip())

    # Distribution (truncated).
    dist = _truncate_distribution(
        enriched.get("gbif_distribution"), MAX_DISTRIBUTION_REGIONS
    )
    if dist:
        parts.append(f"Distribution: {dist}.")

    return " ".join(parts)


def build_lite_nameboost_answer(
    common_name: str,
    scientific_name: str,
    enriched: dict[str, Any] | None,
) -> str:
    """Lite caption variant: short prose with the species name repeated.

    Target shape (4 sentences, ~250-400 chars):
      [S1] Looks like {common} to me.                          # eval anchor
      [S2] {common} is also called {scientific}.               # name repeat
      [S3] {wikipedia_first_sentence}                          # ~scientific name
      [S4] {common} native range: {dist_top3}.                 # name repeat

    Guarantees per row (when fields are present):
      common name appears >= 3x, scientific name appears >= 2x.

    Skips any sentence whose source field is missing. Falls back to
    the compact form when ``enriched`` is ``None``.
    """
    # Sentence 1 — DO NOT CHANGE. Same eval-anchor invariant as
    # build_enriched_answer. The plant scorer regex anchors on
    # "Looks like {X} to me." at the start of generation.
    parts: list[str] = [f"Looks like {common_name} to me."]

    if enriched is None:
        # Compact fallback identical to the full-mode fallback so the
        # eval anchor + scientific-name clause survive.
        if scientific_name and scientific_name != "(unknown)":
            parts.append(
                f"{scientific_name}, commonly called {common_name}."
            )
        return " ".join(parts)

    sci = scientific_name if scientific_name and scientific_name != "(unknown)" else None

    # Sentence 2 — common-name to scientific-name binding. Only emitted
    # when both names are present and distinct (case-insensitive) to
    # avoid the degenerate "Foo is also called Foo." case.
    if sci and sci.lower() != common_name.lower():
        parts.append(f"{common_name} is also called {sci}.")

    # Sentence 3 — Wikipedia first sentence (capped). This is where the
    # enrichment value lives; almost always contains the scientific
    # name in the leading clause (Wikipedia convention).
    blurb = enriched.get("wikipedia_summary") or enriched.get(
        "best_description"
    )
    first = _first_sentence(blurb, MAX_WIKI_SENTENCE_CHARS_LITE)
    if first:
        parts.append(first)

    # Sentence 4 — native range with the common name reprised. Capped
    # at MAX_DISTRIBUTION_REGIONS_LITE regions so the tail stays small.
    dist = _truncate_distribution(
        enriched.get("gbif_distribution"), MAX_DISTRIBUTION_REGIONS_LITE
    )
    if dist:
        parts.append(f"{common_name} native range: {dist}.")

    return " ".join(parts)


# Dispatch table for the assistant-content builders. Keyed by the
# value of the ``--mode`` CLI flag (also exposed in CAPTION_MODES so
# argparse can validate the choices).
_BUILDERS = {
    "full": build_enriched_answer,
    "lite_nameboost": build_lite_nameboost_answer,
}


def _rebuild_row(
    row: dict[str, Any],
    enriched_by_slug: dict[str, dict[str, Any]],
    truncated_counter: Counter,
    mode: str = "full",
) -> tuple[dict[str, Any], bool]:
    """Return (new_row, had_enrichment).

    The image / slug / species / family fields are preserved verbatim.
    Only conversations[1].content is rebuilt; conversations[0]
    (the user question template) is unchanged so the question-side
    variety is preserved.

    ``mode`` selects the assistant-content builder:
      * ``"full"`` — original v2enrich rebuild (long, ~800 chars / row).
      * ``"lite_nameboost"`` — short rebuild with the species name
        repeated (~250-400 chars / row). See
        ``build_lite_nameboost_answer`` for the contract.
    """
    slug = row.get("slug", "")
    species = row.get("species") or "(unknown)"
    # Prefer the enrichment record's common_name (which is the
    # iNaturalist primary) over a slug-derived guess so capitalisation
    # / hyphenation matches GBIF.
    enriched = enriched_by_slug.get(slug)
    common = (
        enriched.get("common_name")
        if enriched and enriched.get("common_name")
        else slug.replace("_", " ")
    )
    try:
        builder = _BUILDERS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown caption mode {mode!r}; expected one of {CAPTION_MODES}"
        ) from exc
    answer = builder(common, species, enriched)
    cap = MAX_CONTENT_CHARS_LITE if mode == "lite_nameboost" else MAX_CONTENT_CHARS
    if len(answer) > cap:
        truncated_counter[slug] += 1
        answer = answer[: cap - 1].rstrip() + "…"

    new_convs = list(row.get("conversations", []))
    if len(new_convs) < 2:
        raise ValueError(
            f"row missing conversations[1]: {row.get('image')!r}"
        )
    # Replace the assistant turn only; preserve user-side question.
    new_convs[1] = {**new_convs[1], "content": answer}

    new_row = dict(row)
    new_row["conversations"] = new_convs
    return new_row, enriched is not None


def _symlink_images(input_root: Path, output_root: Path) -> str:
    src = (input_root / "images_resized").resolve()
    dst = output_root / "images_resized"
    if not src.is_dir():
        return f"images_resized not found at {src}; skipped symlink"
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return f"images_resized symlink already correct: {dst} -> {src}"
        return f"images_resized already exists at {dst}; not overwriting"
    output_root.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    return f"symlinked images_resized: {dst} -> {src}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild NA-Plantae captions from species_enriched.jsonl."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Existing prepared dir with {train,val,test}.jsonl + "
             "images_resized/.",
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        required=True,
        help="Path to species_enriched.jsonl built by enrich_na_plantae.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New prepared dir to write. Will be created.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which JSONL splits to rebuild. Default: all three.",
    )
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=0,
        help="If > 0, process at most this many rows per split. "
             "Use for smoke tests; production runs leave it 0.",
    )
    parser.add_argument(
        "--mode",
        choices=list(CAPTION_MODES),
        default="full",
        help="Caption-rebuild mode. 'full' = original v2enrich behaviour "
             "(~800 chars / row, long GBIF+Wikipedia prose). "
             "'lite_nameboost' = short prose (~300 chars / row) with the "
             "common name repeated >=3x and the scientific name >=2x; "
             "pulls the NA-Plantae token-mass share back toward the v2 "
             "baseline while preserving enrichment content.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    enriched_rows = _read_jsonl(args.enriched)
    enriched_by_slug = {
        r["slug"]: r for r in enriched_rows if r.get("slug")
    }
    log.info(
        "loaded enrichment for %d unique slugs from %s",
        len(enriched_by_slug),
        args.enriched,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    sym_msg = _symlink_images(args.input_root, args.output_root)
    log.info(sym_msg)

    if args.mode == "lite_nameboost":
        mode_caps = {
            "mode": args.mode,
            "max_distribution_regions": MAX_DISTRIBUTION_REGIONS_LITE,
            "max_content_chars": MAX_CONTENT_CHARS_LITE,
            "max_wiki_sentence_chars": MAX_WIKI_SENTENCE_CHARS_LITE,
        }
    else:
        mode_caps = {
            "mode": args.mode,
            "max_distribution_regions": MAX_DISTRIBUTION_REGIONS,
            "max_content_chars": MAX_CONTENT_CHARS,
        }

    report: dict[str, Any] = {
        "input_root": str(args.input_root.resolve()),
        "enriched": str(args.enriched.resolve()),
        "output_root": str(args.output_root.resolve()),
        "enriched_unique_slugs": len(enriched_by_slug),
        **mode_caps,
        "splits": {},
    }

    truncated_counter: Counter = Counter()

    for split in args.splits:
        src_path = args.input_root / f"{split}.jsonl"
        if not src_path.exists():
            log.warning("split %s not present at %s; skipping", split, src_path)
            continue
        rows_in = _read_jsonl(src_path)
        if args.max_rows_per_split > 0:
            rows_in = rows_in[: args.max_rows_per_split]
        log.info("rebuilding %s: %d input rows", split, len(rows_in))

        rebuilt: list[dict] = []
        missing_slugs: Counter = Counter()
        old_lens: list[int] = []
        new_lens: list[int] = []

        for row in rows_in:
            new_row, had_enrich = _rebuild_row(
                row, enriched_by_slug, truncated_counter, mode=args.mode,
            )
            rebuilt.append(new_row)
            old_lens.append(
                len(row["conversations"][1]["content"])
            )
            new_lens.append(
                len(new_row["conversations"][1]["content"])
            )
            if not had_enrich:
                missing_slugs[row.get("slug", "")] += 1

        out_path = args.output_root / f"{split}.jsonl"
        n_written = _write_jsonl(out_path, rebuilt)
        log.info("wrote %s: %d rows", out_path, n_written)

        report["splits"][split] = {
            "n_rows": n_written,
            "n_rows_missing_enrichment": int(sum(missing_slugs.values())),
            "n_unique_slugs_missing_enrichment": len(missing_slugs),
            "missing_slugs_sample": sorted(missing_slugs)[:20],
            "caption_chars_old": {
                "mean": round(statistics.fmean(old_lens), 1) if old_lens else 0,
                "median": int(statistics.median(old_lens)) if old_lens else 0,
                "max": max(old_lens) if old_lens else 0,
            },
            "caption_chars_new": {
                "mean": round(statistics.fmean(new_lens), 1) if new_lens else 0,
                "median": int(statistics.median(new_lens)) if new_lens else 0,
                "max": max(new_lens) if new_lens else 0,
            },
        }

    if truncated_counter:
        log.warning(
            "truncated %d rows across %d unique slugs at "
            "max_content_chars=%d (mode=%s)",
            sum(truncated_counter.values()),
            len(truncated_counter),
            mode_caps["max_content_chars"],
            args.mode,
        )
    report["n_rows_truncated"] = int(sum(truncated_counter.values()))
    report["n_unique_slugs_truncated"] = len(truncated_counter)
    report["truncated_slugs_sample"] = sorted(truncated_counter)[:20]

    report_path = args.output_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log.info("wrote build report: %s", report_path)


if __name__ == "__main__":
    main()
