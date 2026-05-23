#!/usr/bin/env python3
"""Re-download the iNaturalist images for an existing fetch tree at a
higher resolution than the API's default 75x75 ``square.jpg``.

The fetch script's ``--image-size`` flag (added 2026-05-20) controls
the URL substitution at sweep time. This script is the recovery path
for trees that were captured before that fix, or any time the
operator wants to upgrade an existing dataset in place without
re-walking species_counts.

It reads ``observations.jsonl`` from a ``na_plantae_fetch.py`` output
directory, substitutes ``square.jpg`` -> ``<size>.jpg`` in the URL,
and re-downloads each image atomically (write to ``.tmp`` then rename)
to the exact same on-disk path used by the original fetch. Files
larger than ``--min-existing-bytes`` are skipped — a 75x75 square is
~10 KB while ``large`` is ~700 KB, so a 50 KB floor distinguishes them
cleanly without re-touching files that are already at the target
quality.

Usage::

    python src/data_mix/scripts/refetch_inat_images.py \\
        --source-root  $TRAILOGY_DATA_ROOT/inaturalist_na_plantae \\
        --image-size   large \\
        --max-workers  32

The output tree layout is preserved (``<split>/<slug>/<obs>_<idx>.jpg``);
``observations.jsonl`` is rewritten in place with the new (larger) URL
so prepare-time tools that look at it see the consistent value.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests  # type: ignore
except ImportError:
    print(
        "ERROR: requests is required (pip install requests).",
        file=sys.stderr,
    )
    sys.exit(2)

# Reuse the URL helpers from the main fetch script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from na_plantae_fetch import (  # noqa: E402
    INAT_IMAGE_SIZES,
    DEFAULT_IMAGE_SIZE,
    _upgrade_inat_url,
)


_SCRIPT_REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = _SCRIPT_REPO.parent / "data" / "inaturalist_na_plantae"

HTTP_TIMEOUT_SEC = 30.0
# A 240 px ``small`` JPEG is ~30 KB; a 75x75 ``square`` is ~10 KB; a
# 1024 px ``large`` is ~600-900 KB. 50 KB cleanly distinguishes the
# thumbnail tier from anything ``medium`` or above.
DEFAULT_MIN_EXISTING_BYTES = 50_000


def _download_one(
    url: str,
    path: Path,
    min_existing_bytes: int,
    timeout: float,
) -> tuple[str, int]:
    """Return ``(status, bytes_written)``. status ∈ {ok, skipped,
    failed}. Writes atomically: download to ``.tmp`` then rename."""
    try:
        if path.exists() and path.stat().st_size >= min_existing_bytes:
            return ("skipped", 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return ("failed", 0)
        tmp.write_bytes(r.content)
        tmp.replace(path)
        return ("ok", len(r.content))
    except Exception:  # noqa: BLE001
        return ("failed", 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE,
        help=f"Output dir of na_plantae_fetch.py. Default: {DEFAULT_SOURCE}.",
    )
    ap.add_argument(
        "--image-size", choices=INAT_IMAGE_SIZES,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Target iNaturalist size. Default: {DEFAULT_IMAGE_SIZE!r}.",
    )
    ap.add_argument(
        "--max-workers", type=int, default=32,
        help="Thread pool size. Default: 32.",
    )
    ap.add_argument(
        "--min-existing-bytes", type=int,
        default=DEFAULT_MIN_EXISTING_BYTES,
        help=f"Skip files already larger than this. "
             f"Default: {DEFAULT_MIN_EXISTING_BYTES} (~50 KB; just "
             "above any 75x75 square thumbnail).",
    )
    ap.add_argument(
        "--observations-jsonl", type=Path, default=None,
        help="Path to observations.jsonl. Default: <source-root>/"
             "observations.jsonl.",
    )
    ap.add_argument(
        "--no-rewrite-observations", action="store_true",
        help="Do not rewrite observations.jsonl with the upgraded URLs. "
             "Default is to rewrite so downstream tools see consistent values.",
    )
    ap.add_argument(
        "--max-records", type=int, default=0,
        help="If > 0, process only the first N records (for smoke tests). "
             "Default: 0 (all).",
    )
    ap.add_argument(
        "--http-timeout", type=float, default=HTTP_TIMEOUT_SEC,
        help=f"HTTP timeout per request (s). Default: {HTTP_TIMEOUT_SEC}.",
    )
    args = ap.parse_args(argv)

    obs_path = (
        args.observations_jsonl
        or (args.source_root / "observations.jsonl")
    )
    if not obs_path.exists():
        print(f"ERROR: observations.jsonl not found: {obs_path}",
              file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    with obs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]

    print(
        f"== refetch ==\n"
        f"  source        : {args.source_root}\n"
        f"  observations  : {obs_path}\n"
        f"  records       : {len(records)}\n"
        f"  image-size    : {args.image_size}\n"
        f"  workers       : {args.max_workers}\n"
        f"  skip-if-bytes : >= {args.min_existing_bytes}",
        flush=True,
    )

    # Plan: for each record, upgrade URL + compute destination path.
    jobs: list[tuple[str, Path, int]] = []
    upgraded_urls: dict[int, str] = {}
    for i, rec in enumerate(records):
        url = rec.get("photo_url") or ""
        if not url:
            continue
        new_url = _upgrade_inat_url(url, args.image_size)
        upgraded_urls[i] = new_url
        split = rec.get("split") or "train"
        slug = rec.get("slug") or "unknown"
        obs_id = rec.get("observation_id")
        photo_idx = rec.get("photo_idx", 0)
        if obs_id is None:
            continue
        path = (
            args.source_root
            / split
            / slug
            / f"{obs_id}_{photo_idx}.jpg"
        )
        jobs.append((new_url, path, i))

    print(f"Queued {len(jobs)} download jobs.", flush=True)

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    total_bytes = 0
    failed_jobs: list[tuple[str, Path]] = []
    t0 = time.monotonic()

    def _wrap(args_tuple):
        url, path, _i = args_tuple
        return (
            _download_one(url, path, args.min_existing_bytes, args.http_timeout),
            url,
            path,
        )

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for i, ((status, n_bytes), url, path) in enumerate(ex.map(_wrap, jobs), 1):
            counts[status] += 1
            total_bytes += n_bytes
            if status == "failed":
                failed_jobs.append((url, path))
            if i % 1000 == 0 or i == len(jobs):
                dt = time.monotonic() - t0
                rate = i / dt if dt > 0 else 0.0
                eta = (len(jobs) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  [{i}/{len(jobs)}] ok={counts['ok']} "
                    f"skipped={counts['skipped']} failed={counts['failed']} "
                    f"({rate:.1f} req/s, ETA {eta/60:.1f} min)",
                    flush=True,
                )

    dt = time.monotonic() - t0
    print(
        f"\n== summary ==\n"
        f"  ok       : {counts['ok']}\n"
        f"  skipped  : {counts['skipped']}\n"
        f"  failed   : {counts['failed']}\n"
        f"  downloaded: {total_bytes / 1024 / 1024:.1f} MB\n"
        f"  wall time: {dt/60:.1f} min",
        flush=True,
    )

    # Persist the upgraded URLs back into observations.jsonl so downstream
    # tools and re-runs see consistent values.
    if not args.no_rewrite_observations and upgraded_urls:
        tmp = obs_path.with_suffix(obs_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for i, rec in enumerate(records):
                if i in upgraded_urls:
                    rec["photo_url"] = upgraded_urls[i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(obs_path)
        print(f"Rewrote {obs_path} with upgraded URLs.", flush=True)

    if failed_jobs:
        log = args.source_root / "refetch_failed.txt"
        log.write_text("\n".join(f"{u}\t{p}" for u, p in failed_jobs))
        print(f"Wrote {log} ({len(failed_jobs)} failures).", flush=True)

    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
