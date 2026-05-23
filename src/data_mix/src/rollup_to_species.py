#!/usr/bin/env python3
"""Roll up an enriched na_plantae JSONL from mixed-rank taxa to binomial species.

iNaturalist's species_counts endpoint mixes ranks: of the 956 unique
taxa in our Plantae sweep, 156 are subspecies / variety / form (e.g.
``Acer negundo violaceum``, ``Pseudotsuga menziesii menziesii``). For
an on-device 800-class classifier those finer ranks are noise:

  * single-photo identification at subspecies / variety level is
    unreliable even for botanists;
  * GBIF / Wikipedia rarely carry separate text per subspecies, so
    descriptions are near-duplicate;
  * the parent binomial covers all of them and gives more training
    photos per class.

This script collapses the mixed-rank enriched JSONL down to one
record per binomial. For each binomial:

  * if the binomial itself was enriched, that row is the primary;
  * any trinomial children get listed under ``child_taxa`` and their
    observation / photo counts get added to the parent;
  * if the binomial had no description but a child did, the child's
    text fills the parent (with ``filled_from_child`` set);
  * orphan trinomials (no binomial parent in the input) are promoted
    in place so we never silently lose data.

Outputs ``species_enriched_rolled.jsonl`` + ``species_rag_docs_rolled.jsonl``
next to (or wherever ``--output-*`` points to) the inputs.

This script does NOT touch image directories — that's a separate step
inside ``prepare_na_plantae.py`` (gated by ``--rollup-to-species``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


# --- I/O helpers ----------------------------------------------------

def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic write — same pattern as enrich_na_plantae._write_jsonl."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


# --- Core rollup ----------------------------------------------------

def binomial_of(scientific_name: str) -> str:
    """``'Acer negundo violaceum' -> 'Acer negundo'``. Two-word names
    pass through unchanged. Empty / single-word inputs return as-is."""
    parts = (scientific_name or "").split()
    if len(parts) <= 2:
        return scientific_name or ""
    return " ".join(parts[:2])


def group_by_binomial(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group enriched rows by their binomial parent.

    Insertion order is preserved on the outer dict so the rolled output
    keeps roughly the same observation-count ordering as the input.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sci = row.get("scientific_name") or ""
        bn = binomial_of(sci)
        if not bn:
            continue
        groups.setdefault(bn, []).append(row)
    return groups


# Fields that are "description-shaped" — we want to fill them from a
# child only when the parent's value is empty.
_DESC_FIELDS = (
    "wikipedia_summary",
    "wikipedia_title",
    "wikipedia_url",
    "gbif_description",
    "gbif_description_source",
    "gbif_description_language",
    "gbif_distribution",
    "gbif_profile",
    "common_names",
    "best_description",
    "best_description_source",
)


def _first_nonempty(rows: list[dict[str, Any]], key: str) -> str:
    for r in rows:
        v = r.get(key)
        if v:
            return v
    return ""


def rollup_group(
    binomial: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge a list of rows that share a binomial into one record.

    The primary row is the one whose ``scientific_name`` equals
    ``binomial`` (i.e. the species-rank entry). If absent, the highest
    observation-count child is promoted into the slot and we tag
    ``promoted_from_child=True`` so the caller can audit.
    """
    if not rows:
        raise ValueError(f"rollup_group called with no rows for {binomial!r}")

    # Find the species-rank row (binomial == scientific_name) and split
    # the rest into children. Sort children for deterministic output.
    parent = None
    children: list[dict[str, Any]] = []
    for r in rows:
        if r.get("scientific_name") == binomial:
            parent = dict(r)  # copy — we'll mutate
        else:
            children.append(r)
    children.sort(key=lambda r: r.get("scientific_name", ""))

    promoted = False
    if parent is None:
        # Orphan binomial — promote the child with the most observations
        # so we never silently drop a row. Tag it for auditability.
        pick = max(children, key=lambda r: r.get("n_observations") or 0)
        parent = dict(pick)
        parent["scientific_name"] = binomial
        # The promoted child still needs to appear in child_taxa.
        promoted = True

    # Aggregate observation / photo counts across the whole group.
    parent["n_observations"] = sum(
        (r.get("n_observations") or 0) for r in rows
    )
    parent["n_photos"] = sum((r.get("n_photos") or 0) for r in rows)
    parent["child_taxa"] = [r["scientific_name"] for r in children]

    # Slug + common name: prefer parent, but if parent's is empty /
    # "unknown" (iNat fallback when no preferred_common_name), borrow a
    # child's. Common-name fallback uses the same row's slug to stay
    # consistent.
    if not parent.get("slug") or parent.get("slug") == "unknown":
        for c in children:
            if c.get("slug") and c["slug"] != "unknown":
                parent["slug"] = c["slug"]
                if not parent.get("common_name") or parent.get("common_name") == "?":
                    parent["common_name"] = c.get("common_name") or parent.get("common_name") or ""
                break
    if not parent.get("common_name") or parent.get("common_name") == "?":
        for c in children:
            if c.get("common_name") and c["common_name"] != "?":
                parent["common_name"] = c["common_name"]
                break

    # Description-shaped fields: keep the parent's where present, fill
    # from the first child that has the field. Tag the row when ANY
    # field was filled so audit reports can spot it.
    filled_from = ""
    parent_had_text = bool(
        parent.get("best_description") or parent.get("wikipedia_summary")
        or parent.get("gbif_description")
    )
    if not parent_had_text:
        for c in children:
            if c.get("best_description") or c.get("wikipedia_summary") \
                    or c.get("gbif_description"):
                for k in _DESC_FIELDS:
                    if not parent.get(k) and c.get(k):
                        parent[k] = c[k]
                filled_from = c.get("scientific_name", "")
                # Promote child status from no_description_found to ok
                # since we now have text for this binomial.
                if parent.get("best_description"):
                    parent["fetch_status"] = "ok"
                break

    parent["filled_from_child"] = filled_from
    if promoted:
        parent["promoted_from_child"] = True

    # ``rag_text`` is regenerated by the caller (build_rag_text_for_rolled)
    # if needed; leave the parent's existing text alone here. The parent's
    # text references the binomial which is still correct after rollup.
    return parent


# --- RAG doc rebuild ------------------------------------------------

def build_rolled_doc(
    rolled: dict[str, Any], original_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one RAG doc per rolled binomial.

    Reuses the original doc's ``text`` (= the GBIF/Wikipedia rag_text)
    when available, else falls back to ``rag_text`` on the rolled
    record. The doc ID is binomial-stable so downstream RAG indexes
    don't churn when subspecies enter/leave the dataset.
    """
    sci = rolled["scientific_name"]
    slug = rolled.get("slug") or "species"
    doc = {
        "id": f"na_plantae:{slug}:{_short_hash(sci)}",
        "scientific_name": sci,
        "common_name": rolled.get("common_name"),
        "slug": rolled.get("slug"),
        "child_taxa": rolled.get("child_taxa") or [],
        "n_observations": rolled.get("n_observations"),
        "n_photos": rolled.get("n_photos"),
        "text": (original_doc or {}).get("text") or rolled.get("rag_text", ""),
        "metadata": {
            "gbif_usage_key": rolled.get("gbif_usage_key"),
            "gbif_url": rolled.get("gbif_url"),
            "wikipedia_url": rolled.get("wikipedia_url"),
            "powo_search_url": rolled.get("powo_search_url"),
            "best_description_source": rolled.get("best_description_source"),
            "fetch_status": rolled.get("fetch_status"),
            "filled_from_child": rolled.get("filled_from_child") or "",
            "promoted_from_child": bool(rolled.get("promoted_from_child")),
        },
    }
    return doc


def rollup_enriched(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = group_by_binomial(rows)
    return [rollup_group(bn, grp) for bn, grp in groups.items()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    # I/O defaults follow the same convention as enrich_na_plantae —
    # sit next to observations.jsonl in the external data root. Direct
    # ``python path/to/rollup_to_species.py`` invocations don't have
    # ``data_mix`` on sys.path, so we fall back to the sibling-of-repo
    # convention with ``TRAILOGY_DATA_ROOT`` honored either way.
    import os
    try:
        from data_mix.src.env_paths import external_data_root
        _data_root = external_data_root()
    except ModuleNotFoundError:
        env_root = os.environ.get("TRAILOGY_DATA_ROOT")
        _script_repo = Path(__file__).resolve().parents[3]
        _data_root = (
            Path(env_root).expanduser().resolve()
            if env_root
            else (_script_repo.parent / "data").resolve()
        )
    _default_dir = _data_root / "inaturalist_na_plantae"

    ap.add_argument(
        "--input-enriched", type=Path,
        default=_default_dir / "species_enriched.jsonl",
        help="Path to enrich_na_plantae's output JSONL.",
    )
    ap.add_argument(
        "--input-docs", type=Path,
        default=_default_dir / "species_rag_docs.jsonl",
        help="Path to enrich_na_plantae's RAG docs JSONL.",
    )
    ap.add_argument(
        "--output-enriched", type=Path,
        default=_default_dir / "species_enriched_rolled.jsonl",
        help="Where to write the rolled enriched JSONL.",
    )
    ap.add_argument(
        "--output-docs", type=Path,
        default=_default_dir / "species_rag_docs_rolled.jsonl",
        help="Where to write the rolled RAG docs JSONL.",
    )
    ap.add_argument(
        "--report", type=Path, default=None,
        help="Optional path for a rollup audit JSON.",
    )
    args = ap.parse_args(argv)

    if not args.input_enriched.exists():
        print(
            f"ERROR: --input-enriched not found: {args.input_enriched}",
            file=sys.stderr,
        )
        return 2

    enriched = _read_jsonl(args.input_enriched)
    print(
        f"Loaded {len(enriched)} enriched rows from {args.input_enriched}",
        file=sys.stderr,
    )

    groups = group_by_binomial(enriched)
    rolled = [rollup_group(bn, grp) for bn, grp in groups.items()]
    print(
        f"Rolled to {len(rolled)} binomial species "
        f"(collapsed {len(enriched) - len(rolled)} trinomial taxa).",
        file=sys.stderr,
    )

    # Build the rolled docs JSONL. Reuse text from the original docs
    # JSONL when the binomial parent's doc is present, else regenerate
    # from the rolled enriched row.
    docs_by_sci: dict[str, dict[str, Any]] = {}
    if args.input_docs.exists():
        for d in _read_jsonl(args.input_docs):
            sci = d.get("scientific_name")
            if sci and sci not in docs_by_sci:
                docs_by_sci[sci] = d
    rolled_docs = [
        build_rolled_doc(row, docs_by_sci.get(row["scientific_name"]))
        for row in rolled
    ]

    _write_jsonl(args.output_enriched, rolled)
    _write_jsonl(args.output_docs, rolled_docs)
    print(f"Wrote {args.output_enriched}", file=sys.stderr)
    print(f"Wrote {args.output_docs}", file=sys.stderr)

    # Report
    n_filled = sum(1 for r in rolled if r.get("filled_from_child"))
    n_promoted = sum(1 for r in rolled if r.get("promoted_from_child"))
    n_no_desc = sum(
        1 for r in rolled if r.get("fetch_status") == "no_description_found"
    )
    print(
        f"Summary: {len(rolled)} binomial species  "
        f"filled_from_child={n_filled}  promoted={n_promoted}  "
        f"no_description_found={n_no_desc}",
        file=sys.stderr,
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "input_rows": len(enriched),
            "rolled_rows": len(rolled),
            "collapsed_trinomials": len(enriched) - len(rolled),
            "filled_from_child": n_filled,
            "promoted_from_child": n_promoted,
            "no_description_found": n_no_desc,
            "filled_examples": [
                {
                    "binomial": r["scientific_name"],
                    "filled_from_child": r["filled_from_child"],
                }
                for r in rolled if r.get("filled_from_child")
            ][:25],
            "promoted_examples": [
                r["scientific_name"]
                for r in rolled if r.get("promoted_from_child")
            ],
        }, indent=2))
        print(f"Wrote {args.report}", file=sys.stderr)

    print(str(args.output_enriched))
    print(str(args.output_docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
