#!/usr/bin/env python3
"""Cleanup pass after ``refetch_inat_images.py``.

The refetch script's ``--skip-if-bytes 50000`` floor distinguishes 75x75
``square`` thumbnails (~10 KB) from any size at or above ``medium``.
Any file still under that floor when this script runs is one of:

  * A refetch that returned HTTP 200 + body but the body was somehow a
    thumbnail-shaped response (rare iNat / CDN edge case).
  * A refetch that returned non-200, in which case the atomic write
    left the original 75x75 file in place.
  * A photo iNaturalist no longer serves at ``large``.

For each such file we:

  1. Retry the download at ``original.jpg`` (highest fidelity, most
     likely to exist if anything does — iNat occasionally has the
     original but not the size-rendered variant).
  2. Fall back to ``medium.jpg`` (500x500).
  3. If both retries fail, delete the file so the prepare step drops
     it from the training corpus instead of feeding the upscaled
     thumbnail into the trainer.

After cleanup we print per-class counts so an operator can see if any
class dropped below a usable threshold.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
from pathlib import Path
from typing import Any

try:
    import requests  # type: ignore
except ImportError:
    print("ERROR: requests is required.", file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from na_plantae_fetch import _upgrade_inat_url  # noqa: E402


_SCRIPT_REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = _SCRIPT_REPO.parent / "data" / "inaturalist_na_plantae"
DEFAULT_MIN_BYTES = 50_000

# Order matters: try ``original`` first (most likely to exist), then
# ``medium`` (renders that exist on more recent photos). Skip ``large``
# because we already tried that in the main refetch pass.
RETRY_SIZES = ("original", "medium")


def _try_download(url: str, path: Path, timeout: float = 30.0) -> int:
    """Atomic download. Returns bytes written, or 0 on any failure."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or not r.content:
            return 0
        if len(r.content) < DEFAULT_MIN_BYTES:
            # Server returned a thumbnail-shaped response despite the
            # ``original`` / ``medium`` URL — don't write it.
            return 0
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(r.content)
        tmp.replace(path)
        return len(r.content)
    except Exception:  # noqa: BLE001
        return 0


def _retry_or_delete(orig_url: str, path: Path) -> tuple[str, int]:
    """Return ``(status, bytes)``. status ∈ {recovered, deleted, no_url}."""
    if not orig_url:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return ("deleted", 0)
    for size in RETRY_SIZES:
        retry_url = _upgrade_inat_url(orig_url, size)
        if retry_url == orig_url and size != "original":
            continue
        n = _try_download(retry_url, path)
        if n > 0:
            return ("recovered", n)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return ("deleted", 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--observations-jsonl", type=Path, default=None)
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument(
        "--warn-class-min", type=int, default=10,
        help="Print a WARNING for any class whose final image count "
             "falls at or below this threshold. Default: 10.",
    )
    args = ap.parse_args(argv)

    obs_path = (
        args.observations_jsonl
        or (args.source_root / "observations.jsonl")
    )
    if not obs_path.exists():
        print(f"ERROR: {obs_path} not found", file=sys.stderr)
        return 2

    # Build path -> URL map from observations.jsonl.
    path_to_url: dict[Path, str] = {}
    records: list[dict[str, Any]] = []
    with obs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)
            obs_id = r.get("observation_id")
            slug = r.get("slug") or "unknown"
            split = r.get("split") or "train"
            photo_idx = r.get("photo_idx", 0)
            if obs_id is None:
                continue
            path = (
                args.source_root / split / slug / f"{obs_id}_{photo_idx}.jpg"
            )
            path_to_url[path] = r.get("photo_url") or ""

    # Walk raw fetch tree, find files under threshold.
    too_small: list[Path] = []
    for split in ("train", "val", "test"):
        d = args.source_root / split
        if not d.is_dir():
            continue
        for p in d.rglob("*.jpg"):
            try:
                if p.stat().st_size < args.min_bytes:
                    too_small.append(p)
            except FileNotFoundError:
                pass

    print(
        f"Found {len(too_small)} files under {args.min_bytes} bytes "
        f"(refetch failures / leftover thumbnails).",
        flush=True,
    )

    # Retry pass.
    recovered = deleted = no_url = 0
    deleted_paths: list[Path] = []

    def _wrap(path: Path):
        url = path_to_url.get(path, "")
        status, n = _retry_or_delete(url, path)
        return path, status, n

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for i, (path, status, n) in enumerate(ex.map(_wrap, too_small), 1):
            if status == "recovered":
                recovered += 1
            elif status == "no_url":
                no_url += 1
                deleted_paths.append(path)
            else:
                deleted += 1
                deleted_paths.append(path)
            if i % 5 == 0 or i == len(too_small):
                print(
                    f"  [{i}/{len(too_small)}] recovered={recovered} "
                    f"deleted={deleted} no_url={no_url}",
                    flush=True,
                )

    # Filter observations.jsonl: drop records whose path was deleted.
    deleted_set = {str(p) for p in deleted_paths}
    if deleted_set:
        out_records = []
        for r in records:
            obs_id = r.get("observation_id")
            slug = r.get("slug") or "unknown"
            split = r.get("split") or "train"
            photo_idx = r.get("photo_idx", 0)
            path = str(
                args.source_root / split / slug / f"{obs_id}_{photo_idx}.jpg"
            )
            if path in deleted_set:
                continue
            out_records.append(r)
        tmp = obs_path.with_suffix(obs_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(obs_path)
        print(
            f"Rewrote observations.jsonl: "
            f"{len(records)} -> {len(out_records)} rows",
            flush=True,
        )

    # Per-class min counts so the operator can spot starved classes.
    counts: collections.Counter = collections.Counter()
    for split in ("train", "val", "test"):
        d = args.source_root / split
        if not d.is_dir():
            continue
        for slug_dir in d.iterdir():
            if slug_dir.is_dir():
                counts[slug_dir.name] += sum(
                    1 for _ in slug_dir.glob("*.jpg")
                )

    starved = [(s, c) for s, c in counts.items() if c <= args.warn_class_min]
    starved.sort(key=lambda x: x[1])

    print("\n=== summary ===", flush=True)
    print(f"  too-small files found: {len(too_small)}")
    print(f"  recovered           : {recovered}")
    print(f"  deleted             : {deleted}")
    print(f"  classes:               {len(counts)}")
    if counts:
        vals = sorted(counts.values())
        print(f"  min/median/max      : "
              f"{vals[0]} / {vals[len(vals)//2]} / {vals[-1]}")
    if starved:
        print(f"\n  WARNING: {len(starved)} classes <= "
              f"{args.warn_class_min} images:")
        for s, c in starved[:30]:
            print(f"    {s:35s} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
