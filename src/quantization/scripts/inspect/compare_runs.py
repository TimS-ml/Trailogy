#!/usr/bin/env python3
"""Aggregate per-variant ``eval.json`` files into one matrix.

After the two GPU scripts finish (and the Mac MLX scripts finish),
all variants' results live under ``RESULTS_ROOT`` as
``<variant>/eval.json``. This script:

1. Walks ``RESULTS_ROOT`` and loads every ``eval.json``.
2. Builds a single TSV/Markdown table with one row per variant.
3. Computes deltas vs the bf16 reference variant (default name:
   ``bf16_reference``).
4. Highlights any variant breaching the <4 GB ceiling or showing >10
   absolute pct points drop on PlantNet vs bf16 (tripwire).

Usage:
    python -m scripts.inspect.compare_runs
    python -m scripts.inspect.compare_runs --results_root quantization/results
    python -m scripts.inspect.compare_runs --output_md /tmp/matrix.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("compare_runs")


def _load_evals(results_root: Path) -> list[dict]:
    """Find every ``<variant>/eval.json`` and return the parsed list."""
    out: list[dict] = []
    for eval_file in sorted(results_root.glob("*/eval.json")):
        try:
            payload = json.loads(eval_file.read_text())
            payload["_variant_dir"] = str(eval_file.parent)
            out.append(payload)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load %s: %s", eval_file, e)
    return out


def _benchmark_value(payload: dict, bench: str, key: str) -> float | None:
    b = payload.get("benchmarks", {}).get(bench, {})
    val = b.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def build_matrix(payloads: list[dict], reference_variant: str = "bf16_reference") -> list[dict]:
    """Return a list of row dicts ready for table rendering."""
    # Find reference numbers
    ref = next((p for p in payloads if p.get("variant") == reference_variant), None)
    if ref is None:
        log.warning(
            "No reference variant %r found. Deltas will be blank.",
            reference_variant,
        )

    def get(p, bench, key):
        return _benchmark_value(p, bench, key)

    rows: list[dict] = []
    for p in payloads:
        row = {
            "variant": p.get("variant", "<unnamed>"),
            "backend": p.get("backend", "?"),
            "size_gb": p.get("model_size_gb"),
            "plantnet_match": get(p, "plantnet_val", "species_match"),
            "vqav2_acc": get(p, "vqav2_devtest", "accuracy"),
            "wikitext_ppl": get(p, "wikitext_ppl", "perplexity"),
        }
        if ref is not None and p is not ref:
            ref_pn = get(ref, "plantnet_val", "species_match")
            ref_vqa = get(ref, "vqav2_devtest", "accuracy")
            ref_ppl = get(ref, "wikitext_ppl", "perplexity")
            row["plantnet_delta"] = (
                (row["plantnet_match"] - ref_pn)
                if row["plantnet_match"] is not None and ref_pn is not None
                else None
            )
            row["vqav2_delta"] = (
                (row["vqav2_acc"] - ref_vqa)
                if row["vqav2_acc"] is not None and ref_vqa is not None
                else None
            )
            row["ppl_ratio"] = (
                (row["wikitext_ppl"] / ref_ppl)
                if row["wikitext_ppl"] is not None and ref_ppl is not None and ref_ppl > 0
                else None
            )
        # Tripwires
        warnings: list[str] = []
        if row["size_gb"] is not None and row["size_gb"] > 4.0:
            warnings.append(f"SIZE>{4.0}GB")
        if row.get("plantnet_delta") is not None and row["plantnet_delta"] < -0.10:
            warnings.append("PlantNet drop >10pts")
        if row.get("ppl_ratio") is not None and row["ppl_ratio"] > 2.0:
            warnings.append("PPL >2x ref")
        row["warnings"] = warnings
        rows.append(row)
    return rows


def render_markdown(rows: list[dict]) -> str:
    headers = [
        "variant", "backend", "size_gb",
        "plantnet_match", "plantnet_delta",
        "vqav2_acc", "vqav2_delta",
        "wikitext_ppl", "ppl_ratio",
        "warnings",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            elif isinstance(v, list):
                cells.append("; ".join(v) if v else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_tsv(rows: list[dict]) -> str:
    headers = [
        "variant", "backend", "size_gb",
        "plantnet_match", "plantnet_delta",
        "vqav2_acc", "vqav2_delta",
        "wikitext_ppl", "ppl_ratio",
        "warnings",
    ]
    lines = ["\t".join(headers)]
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h)
            if v is None:
                cells.append("")
            elif isinstance(v, list):
                cells.append("; ".join(v))
            else:
                cells.append(str(v))
        lines.append("\t".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("quantization/results"),
    )
    parser.add_argument(
        "--reference_variant",
        default="bf16_reference",
        help="Variant whose numbers are treated as the deltas baseline.",
    )
    parser.add_argument("--output_md", type=Path, help="If set, also write markdown to this path.")
    parser.add_argument("--output_tsv", type=Path, help="If set, also write TSV to this path.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.results_root.is_dir():
        log.error("Results root not found: %s", args.results_root)
        return 2

    payloads = _load_evals(args.results_root)
    if not payloads:
        log.error("No eval.json files found under %s", args.results_root)
        return 1

    rows = build_matrix(payloads, reference_variant=args.reference_variant)
    md = render_markdown(rows)
    print()
    print(md)
    print()

    if args.output_md:
        args.output_md.write_text(md + "\n")
        log.info("Wrote markdown table → %s", args.output_md)
    if args.output_tsv:
        args.output_tsv.write_text(render_tsv(rows) + "\n")
        log.info("Wrote TSV → %s", args.output_tsv)

    warned = [r for r in rows if r.get("warnings")]
    if warned:
        print()
        print("WARNINGS:")
        for r in warned:
            print(f"  {r['variant']}: {'; '.join(r['warnings'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
