#!/usr/bin/env python3
"""Fetch plant observations from iNaturalist.

**Default scope: US-wide research-grade Plantae, stratified by species.**
We diverged from the upstream-reference ``na_plantae_fetch.py`` (Pittsburgh
regional + flat `created_at desc` pagination) for two reasons:

  1. A tight regional sweep returns ~4-8 species (recent test: 4
     species across 176 obs).
  2. Flat pagination by ``created_at desc`` / ``votes desc`` is heavily
     long-tail dominated (35 species across 10 000 obs in our US-wide
     probe), AND iNaturalist hard-caps deep pagination at ~10 000
     results (403 after ``page=50`` at ``per_page=200``).

The new flow is **stratified by species**:

  1. Call ``/v1/observations/species_counts`` with the filter to get
     the top-N species by observation count in the region.
  2. For each species, pull up to ``--obs-per-species`` research-grade
     observations via ``/v1/observations?taxon_id=...``.
  3. Aggregate, split per-observation_id into train/val/test, download.

The geographic centre / radius knobs are still here as optional CLI
flags (``--lat``, ``--lng``, ``--radius``) — see the **Pittsburgh
example** below for the regional-fetch invocation. They are commented
out of ``DEFAULT_PARAMS`` so a default run uses ``place_id`` instead.

Default request params::

    {
        "taxon_name":    "Plantae",
        "quality_grade": "research",
        "photos":        "true",
        "place_id":      1,              # iNaturalist place id for the US
        # "lat":         40.4406,        # Pittsburgh example — set via --lat
        # "lng":         -79.9959,       #   --lng
        # "radius":      80,             #   --radius (km)
        "per_page":      200,
        "page":          1,
        "order_by":      "created_at",
        "order":         "desc",
    }

Output layout (default ``--output-dir`` is
``<repo>/../data/inaturalist_na_plantae/``, the Trailogy convention —
relocate the whole external data root via ``TRAILOGY_DATA_ROOT``)::

    <output_dir>/
        observations.jsonl       # one JSON line per photo
        fetch_report.json
        {train,val,test}.jsonl   # same JSONL bucketed by split
        train/<slug>/<obs_id>_<photo_idx>.jpg
        val/<slug>/<obs_id>_<photo_idx>.jpg
        test/<slug>/<obs_id>_<photo_idx>.jpg

The slug = sanitised ``preferred_common_name`` (falls back to the
Latin name). Split assignment is on a per-OBSERVATION key (so all
photos of the same observation always land in the same split — no
train/test leakage from near-duplicate frames of one plant).

Usage::

    # Default sweep — US-wide research-grade Plantae, stop at
    # ~150 species. Downloads images into <output_dir>/{train,val,test}/.
    python src/data_mix/scripts/na_plantae_fetch.py

    # Same sweep, metadata only (no image bytes on disk).
    python src/data_mix/scripts/na_plantae_fetch.py --no-download

    # Pittsburgh-only example (matches the upstream reference's scope).
    python src/data_mix/scripts/na_plantae_fetch.py \\
        --place-id 0 --lat 40.4406 --lng -79.9959 --radius 80

Network: 1 species_counts call + N per-taxon obs fetches, 0.5-1 s
inter-call sleep. Default (150 species × 50 obs each ≈ 7500 photos)
runs in ~3-5 min and lands ~75 MB on disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
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


OBS_URL = "https://api.inaturalist.org/v1/observations"
SPECIES_COUNTS_URL = "https://api.inaturalist.org/v1/observations/species_counts"

# iNaturalist place IDs (https://www.inaturalist.org/places). Useful
# anchor points; override via ``--place-id``.
PLACE_ID_US = 1               # United States (canonical, includes territories)
PLACE_ID_CONTIGUOUS_US = 46   # Contiguous USA (sometimes called "Lower 48")

# iNaturalist canonical taxon IDs. Plantae kingdom = 47126; we use
# ``taxon_id`` (which descends into the whole subtree) instead of
# ``taxon_name`` because the latter does fuzzy/substring matching and
# leaks species whose common name happens to contain "plant" — e.g.
# the "north american tarnished plant bug" (an *insect*, taxon_id
# under Animalia).
TAXON_ID_PLANTAE = 47126

# --- Default request params. Reference ``na_plantae_fetch.py`` was
#     Pittsburgh-only with flat pagination; we relax to US-wide
#     species-stratified sampling by default to lift the species count
#     from ~4-8 (single-region) or ~35 (flat-paginated US) to ~150+.
#     The Pittsburgh lat/lng/radius lines are intentionally kept
#     (commented) as the regional-fetch example. ---
DEFAULT_PARAMS: dict[str, Any] = {
    "taxon_id":      TAXON_ID_PLANTAE,
    "quality_grade": "research",
    "photos":        "true",
    "place_id":      PLACE_ID_US,     # United States (drop or override
                                      #   to widen / narrow scope)
    # Reference (Pittsburgh) — set via --lat / --lng / --radius:
    # "lat":           40.4406,
    # "lng":           -79.9959,
    # "radius":        80,             # km
    "per_page":      200,
    "order_by":      "created_at",
    "order":         "desc",
}

DEFAULT_TARGET_SPECIES = 150
DEFAULT_OBS_PER_SPECIES = 50

# iNaturalist's ``photos[*].url`` field returns the 75x75 ``square.jpg``
# thumbnail by default — useless for vision training (a 12x upscale to
# the 960x672 prep target leaves only blur). Substitute the trailing
# ``square.jpg`` with one of the larger sizes below before download.
# ``large`` is the smallest size that exceeds the 960-pixel-on-long-side
# iOS runtime target (yields 771x1024 typical), so no upscale needed.
INAT_IMAGE_SIZES = ("square", "small", "medium", "large", "original")
DEFAULT_IMAGE_SIZE = "large"


def _upgrade_inat_url(url: str, size: str) -> str:
    """Rewrite an iNat photo URL to request ``<size>.jpg`` instead of
    ``square.jpg``. Non-iNat URLs and already-upgraded URLs pass through
    unchanged.
    """
    if not url or size == "square":
        return url
    # Both inaturalist-open-data.s3.amazonaws.com and static.inaturalist.org
    # use the trailing-filename convention.
    for s in INAT_IMAGE_SIZES:
        suffix = f"/{s}.jpg"
        if url.endswith(suffix):
            return url[: -len(suffix)] + f"/{size}.jpg"
    return url

# Backward-compat alias: some callers still import ``BASE``.
BASE = OBS_URL

# Default cache lives OUTSIDE the repo at ``<repo>/../data/inaturalist_na_plantae/``
# (sibling-of-repo convention shared with env_paths.py). Override via
# the ``--output-dir`` CLI flag or by symlinking ``<repo>/../data/`` to
# wherever real storage lives.
#
# script path: <repo>/src/data_mix/scripts/na_plantae_fetch.py
#   parents[3] = <repo>
#   parents[3].parent = <repo>'s parent (= external data root parent)
_SCRIPT_REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = _SCRIPT_REPO.parent / "data" / "inaturalist_na_plantae"

# Default 80/10/10 split, mirrors the per-species counts in the legacy
# prepare_na_plantae recipe (40 train / 5 val / 5 test out of 50).
DEFAULT_SPLIT_RATIOS = (0.80, 0.10, 0.10)
SPLIT_NAMES = ("train", "val", "test")

HTTP_TIMEOUT_SEC = 30.0
INTER_PAGE_SLEEP_SEC = 1.0
# iNaturalist publishes a soft cap of ~60 requests/minute for /v1/*.
# Empirically ~40 req/min still trips 429 normal_throttling under sustained
# load, so we target a conservative ~30 req/min (= 2.0 s between calls)
# for the metadata walk. Image downloads go to a different host
# (static.inaturalist.org) and are not counted here.
INTER_TAXON_SLEEP_SEC = 2.0
# 429 retry policy. iNat's response sometimes includes a Retry-After
# header in seconds; we honour it when present, otherwise back off
# exponentially starting at this base.
HTTP_RETRY_MAX_ATTEMPTS = 5
HTTP_RETRY_BASE_SEC     = 10.0


def _get_with_retry(
    url: str, params: dict[str, Any] | None = None,
) -> requests.Response:
    """``requests.get`` with HTTP 429 / 5xx retry-with-backoff.

    On 429 we honour the ``Retry-After`` response header (iNat sets
    this on sustained-throttle events); on other transient errors we
    use exponential backoff. After HTTP_RETRY_MAX_ATTEMPTS we re-raise
    so the caller sees the original error.
    """
    last_exc: Exception | None = None
    delay = HTTP_RETRY_BASE_SEC
    for attempt in range(1, HTTP_RETRY_MAX_ATTEMPTS + 1):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code == 429 or r.status_code >= 500:
                # Surface as exception so the retry branch below runs.
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(
                    f"  [retry {attempt}/{HTTP_RETRY_MAX_ATTEMPTS}] "
                    f"HTTP {r.status_code} on {url} — sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                delay = min(delay * 2, 120.0)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= HTTP_RETRY_MAX_ATTEMPTS:
                break
            print(
                f"  [retry {attempt}/{HTTP_RETRY_MAX_ATTEMPTS}] "
                f"{type(exc).__name__}: {exc} — sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise last_exc if last_exc else RuntimeError(
        f"_get_with_retry: exhausted retries for {url}"
    )


def slugify(name: str) -> str:
    """Lowercase, collapse non-alnum to ``_``, trim. Empty -> 'unknown'."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "unknown"


def derive_slug(taxon: dict[str, Any]) -> str:
    """Prefer ``preferred_common_name`` (Red Maple -> ``red_maple``),
    fall back to the Latin ``name`` (Acer rubrum -> ``acer_rubrum``)."""
    common = (taxon or {}).get("preferred_common_name") or ""
    sci = (taxon or {}).get("name") or ""
    return slugify(common) or slugify(sci)


def stable_bucket(key: str, seed: int) -> float:
    """Map a string key + seed to a stable float in [0, 1).

    Uses sha256 so the result is reproducible across processes (Python's
    builtin ``hash()`` is salted per-interpreter and would split the
    same observation into different folders on re-runs).
    """
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_split(
    key: str,
    seed: int,
    ratios: tuple[float, float, float],
) -> str:
    """Deterministic train/val/test assignment from a key + seed."""
    r_train, r_val, _ = ratios
    b = stable_bucket(key, seed)
    if b < r_train:
        return "train"
    if b < r_train + r_val:
        return "val"
    return "test"


def fetch_top_species(
    base_params: dict[str, Any], n_species: int
) -> list[dict[str, Any]]:
    """Call ``/v1/observations/species_counts`` to rank species by
    observation count under the given filter.

    iNaturalist's ``species_counts`` endpoint caps ``per_page`` at
    500. For larger N we paginate (page=1,2,...).

    Returns a list of taxon dicts (keys: ``id``, ``name``,
    ``preferred_common_name``, ``rank``) for the top ``n_species``.

    Pagination dedup: iNat's ``species_counts`` does NOT guarantee a
    stable per-row ordering across pages — when many species tie on
    ``observations_count`` (large 17K-obs plateau under our Plantae
    filter), a given species can land on page 1 AND page 2. Empirically
    ~20% of raw rows duplicate between pages 1 and 2 at per_page=500.
    Without dedup the caller silently gets fewer unique species than
    requested. We track ``taxon.id`` in a ``seen`` set and only count
    fresh IDs toward ``n_species``. We also keep ``per_page`` pinned to
    the API max (500) so each round-trip pulls maximum new material;
    shrinking per_page near the end stalls when remaining slots happen
    to be filled with duplicates.
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    page = 1
    while len(out) < n_species:
        params = dict(base_params)
        # Always request the API max — dedup means some rows are dropped
        # and we want each request to maximize fresh species per RTT.
        params["per_page"] = 500
        params["page"] = page
        # ``species_counts`` has its own ordering convention
        # (``observations_count desc`` by default). Forwarding the
        # obs-endpoint ``order_by=created_at`` would degrade the
        # discovery to "species that uploaded recently" instead of
        # "most-observed species in the region".
        params.pop("order_by", None)
        params.pop("order", None)
        # ``photos=true`` filters out photo-less obs from the count too,
        # so the species we discover are exactly the ones we can then
        # download images for.
        r = _get_with_retry(SPECIES_COUNTS_URL, params=params)
        results = r.json().get("results", []) or []
        if not results:
            break
        for rec in results:
            taxon = rec.get("taxon") or {}
            tid = taxon.get("id")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            out.append(taxon)
            if len(out) >= n_species:
                break
        page += 1
        if len(out) >= n_species:
            break
        time.sleep(INTER_PAGE_SLEEP_SEC)
    return out[:n_species]


def fetch_observations_for_taxon(
    base_params: dict[str, Any],
    taxon_id: int,
    max_obs: int,
    *,
    max_pages_per_taxon: int = 10,
) -> list[dict[str, Any]]:
    """Pull up to ``max_obs`` research-grade observations for one taxon.

    Drops ``taxon_name`` from ``base_params`` (``taxon_id`` is more
    specific) and overrides ``page`` / ``per_page`` to walk per-200.
    Pages are capped at ``max_pages_per_taxon`` as a safety net even
    if the taxon has more obs than we need.
    """
    params = dict(base_params)
    params.pop("taxon_name", None)
    params["taxon_id"] = taxon_id
    out: list[dict[str, Any]] = []
    page = 1
    while len(out) < max_obs and page <= max_pages_per_taxon:
        params["page"] = page
        params["per_page"] = min(200, max_obs - len(out))
        r = _get_with_retry(OBS_URL, params=params)
        results = r.json().get("results", []) or []
        if not results:
            break
        out.extend(results)
        page += 1
        if len(out) >= max_obs:
            break
        time.sleep(INTER_PAGE_SLEEP_SEC * 0.5)  # gentler intra-taxon
    return out[:max_obs]


def flatten_photos(
    observations: list[dict[str, Any]],
    *,
    seed: int,
    split_ratios: tuple[float, float, float],
    image_size: str = DEFAULT_IMAGE_SIZE,
) -> list[dict[str, Any]]:
    """Explode obs[*].photos[*] into per-photo records.

    Matches the upstream reference's ``all_records`` schema
    (observation_id, scientific_name, common_name, rank, photo_url,
    license_code, observed_on, place_guess, lat, lng) and appends three
    fields used by the downloader: ``slug``, ``photo_idx``, ``split``.

    The split key is the ``observation_id`` so all photos of the same
    observation land in the same split (avoids train/test leakage from
    near-duplicate frames).
    """
    out: list[dict[str, Any]] = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        slug = derive_slug(taxon)
        obs_id = obs.get("id")
        split = assign_split(str(obs_id), seed, split_ratios)
        for photo_idx, photo in enumerate(obs.get("photos", [])):
            out.append({
                "observation_id":  obs_id,
                "scientific_name": taxon.get("name"),
                "common_name":     taxon.get("preferred_common_name"),
                "rank":            taxon.get("rank"),
                "photo_url":       _upgrade_inat_url(
                    photo.get("url") or "", image_size,
                ),
                "license_code":    photo.get("license_code"),
                "observed_on":     obs.get("observed_on"),
                "place_guess":     obs.get("place_guess"),
                "lat": obs.get("geojson", {}).get("coordinates", [None, None])[1],
                "lng": obs.get("geojson", {}).get("coordinates", [None, None])[0],
                # Trailogy additions for the downloader / downstream
                # consumers; these do not affect the upstream record
                # schema (first 10 keys above).
                "slug":      slug,
                "photo_idx": photo_idx,
                "split":     split,
            })
    return out


def _download_one(job: tuple[str, Path]) -> bool:
    url, path = job
    if path.exists():
        return True  # idempotent
    try:
        r = requests.get(url, timeout=15.0)
        if r.status_code == 200 and r.content:
            path.write_bytes(r.content)
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def parse_split_ratios(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--split-ratios expects 'train,val,test'; got {s!r}"
        )
    try:
        ratios = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--split-ratios entries must be floats: {e}"
        ) from None
    total = sum(ratios)
    if total <= 0:
        raise argparse.ArgumentTypeError(
            f"--split-ratios sum must be > 0; got {total}"
        )
    if not abs(total - 1.0) < 1e-6:
        # Auto-normalise so e.g. '40,5,5' works as a shorthand.
        ratios = tuple(r / total for r in ratios)
    return ratios  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Top-level output dir (kept OUTSIDE the repo). "
             f"Contains observations.jsonl, fetch_report.json, and "
             f"train/val/test/<slug>/<obs>_<idx>.jpg subtrees. "
             f"Default: <repo>/../data/inaturalist_na_plantae/ "
             f"(resolved at script-load time to "
             f"{str(DEFAULT_OUTPUT_DIR)!r}).",
    )
    # --- Geographic filter. Default: place_id=1 (United States).
    #     Lat/lng/radius are optional regional overrides — when any of
    #     them is given, the per-page request uses them and place_id
    #     is dropped unless --place-id is also explicit. ---
    ap.add_argument(
        "--place-id", type=int, default=DEFAULT_PARAMS["place_id"],
        help=f"iNaturalist place_id (https://www.inaturalist.org/places). "
             f"Default: {DEFAULT_PARAMS['place_id']} (United States). "
             "Pass 0 to disable the place filter (paired with --lat / "
             "--lng / --radius for a regional fetch).",
    )
    ap.add_argument(
        "--lat", type=float, default=None,
        help="Latitude of the search centre (regional override; e.g. "
             "40.4406 for Pittsburgh). Default: unset (use --place-id).",
    )
    ap.add_argument(
        "--lng", type=float, default=None,
        help="Longitude of the search centre (regional override; e.g. "
             "-79.9959 for Pittsburgh). Default: unset (use --place-id).",
    )
    ap.add_argument(
        "--radius", type=float, default=None,
        help="Search radius in km (regional override; e.g. 80 km for a "
             "city-sized sweep). Default: unset (use --place-id).",
    )
    # --- Taxon filter. Default: --taxon-id Plantae kingdom. ---
    ap.add_argument(
        "--taxon-id", type=int, default=DEFAULT_PARAMS["taxon_id"],
        help=f"iNaturalist taxon_id (the filter descends into the "
             f"whole subtree). Default: {DEFAULT_PARAMS['taxon_id']} "
             "(Plantae kingdom).",
    )
    ap.add_argument(
        "--taxon-name", default=None,
        help="Override the --taxon-id filter with a taxon_name string "
             "(WARNING: does fuzzy / substring matching server-side; "
             "'Plantae' leaks non-plant species whose common name "
             "contains 'plant'. Prefer --taxon-id for precise filters). "
             "Default: unset.",
    )
    ap.add_argument(
        "--quality-grade", default=DEFAULT_PARAMS["quality_grade"],
        help=f"iNaturalist quality_grade filter. Default: "
             f"{DEFAULT_PARAMS['quality_grade']!r}.",
    )
    ap.add_argument(
        "--order-by", default=DEFAULT_PARAMS["order_by"],
        help=f"iNaturalist order_by. Default: "
             f"{DEFAULT_PARAMS['order_by']!r}.",
    )
    ap.add_argument(
        "--order", default=DEFAULT_PARAMS["order"],
        help=f"iNaturalist order direction. Default: "
             f"{DEFAULT_PARAMS['order']!r}.",
    )

    # --- Split + download knobs (Trailogy extensions). ---
    ap.add_argument(
        "--split-ratios", type=parse_split_ratios,
        default=DEFAULT_SPLIT_RATIOS,
        help="Train/val/test ratios as 'train,val,test'. Auto-normalised "
             f"if sum != 1. Default: "
             f"{','.join(str(r) for r in DEFAULT_SPLIT_RATIOS)}.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the per-observation split hash. Same seed + same "
             "observation_id -> same split. Default: 42.",
    )
    ap.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Skip image downloads; write only observations.jsonl and "
             "fetch_report.json. Default: download is on.",
    )
    ap.set_defaults(download=True)
    ap.add_argument(
        "--target-species", type=int, default=DEFAULT_TARGET_SPECIES,
        help=f"Number of distinct species to pull (discovered via "
             f"species_counts, ranked by observation count under the "
             f"filter). Default: {DEFAULT_TARGET_SPECIES}.",
    )
    ap.add_argument(
        "--obs-per-species", type=int, default=DEFAULT_OBS_PER_SPECIES,
        help=f"Max observations to pull per species. Default: "
             f"{DEFAULT_OBS_PER_SPECIES}. Each obs typically has 1-2 "
             "photos, so ~50 obs -> ~75 photos per species.",
    )
    ap.add_argument(
        "--max-pages-per-taxon", type=int, default=10,
        help="Safety cap on pagination depth per taxon. Default: 10. "
             "Only kicks in for very popular taxa with thousands of obs.",
    )
    ap.add_argument(
        "--max-workers", type=int, default=32,
        help="Thread pool size for parallel image downloads. Default: 32.",
    )
    ap.add_argument(
        "--image-size", choices=INAT_IMAGE_SIZES, default=DEFAULT_IMAGE_SIZE,
        help=f"iNaturalist image size to request. Default: "
             f"{DEFAULT_IMAGE_SIZE!r} (~1024 px long side; matches the "
             "iOS runtime resize without upscale). The API returns "
             "75x75 'square' thumbnails by default, which is too low-res "
             "for vision training.",
    )
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_params: dict[str, Any] = {
        "quality_grade": args.quality_grade,
        "photos":        "true",
        "order_by":      args.order_by,
        "order":         args.order,
    }
    # Taxon filter: --taxon-name overrides --taxon-id when given.
    if args.taxon_name:
        base_params["taxon_name"] = args.taxon_name
    else:
        base_params["taxon_id"] = args.taxon_id
    # Geographic filter: regional override (lat+lng+radius) wins over
    # place_id when all three are given; otherwise place_id is used
    # unless explicitly disabled with --place-id 0.
    regional = (
        args.lat is not None and args.lng is not None and args.radius is not None
    )
    if regional:
        base_params["lat"] = args.lat
        base_params["lng"] = args.lng
        base_params["radius"] = args.radius
    if args.place_id and not regional:
        base_params["place_id"] = args.place_id

    geo_str = (
        f"lat={args.lat} lng={args.lng} radius={args.radius} km"
        if regional
        else f"place_id={base_params.get('place_id') or '<none>'}"
    )
    taxon_str = (
        f"taxon_name={args.taxon_name!r}"
        if args.taxon_name
        else f"taxon_id={args.taxon_id}"
    )
    print(
        f"== iNaturalist stratified fetch ==\n"
        f"  scope  : {geo_str}\n"
        f"  taxon  : {taxon_str}  quality={args.quality_grade!r}\n"
        f"  target : species={args.target_species}  "
        f"obs/species={args.obs_per_species}\n"
        f"  splits : train/val/test = "
        f"{tuple(round(r, 4) for r in args.split_ratios)}  seed={args.seed}\n"
        f"  output : {args.output_dir}  download={args.download}",
        flush=True,
    )

    print(
        f"Step 1: discover top {args.target_species} species via "
        f"species_counts ...",
        flush=True,
    )
    top_taxa = fetch_top_species(base_params, args.target_species)
    print(f"  -> got {len(top_taxa)} species", flush=True)
    if not top_taxa:
        print("ERROR: species_counts returned 0 results.", file=sys.stderr)
        return 3

    print(
        f"Step 2: fetch up to {args.obs_per_species} obs per species "
        f"({len(top_taxa)} taxa × ~1 call) ...",
        flush=True,
    )
    observations: list[dict[str, Any]] = []
    for i, taxon in enumerate(top_taxa, 1):
        taxon_id = taxon["id"]
        slug_preview = (
            taxon.get("preferred_common_name") or taxon.get("name") or "?"
        )
        obs = fetch_observations_for_taxon(
            base_params, taxon_id, args.obs_per_species,
            max_pages_per_taxon=args.max_pages_per_taxon,
        )
        observations.extend(obs)
        if i % 10 == 0 or i == len(top_taxa):
            print(
                f"  [{i:3d}/{len(top_taxa)}] {slug_preview!r}: "
                f"+{len(obs)} obs (running total {len(observations)})",
                flush=True,
            )
        # Inter-taxon throttle: keeps us well under iNat's 60 req/min cap.
        if i < len(top_taxa):
            time.sleep(INTER_TAXON_SLEEP_SEC)
    photo_records = flatten_photos(
        observations,
        seed=args.seed,
        split_ratios=args.split_ratios,
        image_size=args.image_size,
    )
    print(
        f"Found {len(observations)} observations "
        f"-> {len(photo_records)} photo records.",
        flush=True,
    )

    # Materialise metadata BEFORE downloading so a Ctrl-C mid-download
    # still leaves a useful trace of what was planned.
    obs_jsonl = args.output_dir / "observations.jsonl"
    with obs_jsonl.open("w") as f:
        for rec in photo_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {obs_jsonl}", flush=True)

    # Pre-bucket by split so we can also write per-split JSONLs (handy
    # for downstream consumers that just want one split).
    per_split_records: dict[str, list[dict[str, Any]]] = {
        s: [] for s in SPLIT_NAMES
    }
    for rec in photo_records:
        per_split_records[rec["split"]].append(rec)
    for split_name, recs in per_split_records.items():
        split_jsonl = args.output_dir / f"{split_name}.jsonl"
        with split_jsonl.open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(
            f"Wrote {split_jsonl} ({len(recs)} records)",
            flush=True,
        )

    requested_by_split: dict[str, dict[str, int]] = {
        s: {} for s in SPLIT_NAMES
    }
    jobs: list[tuple[str, Path]] = []
    for rec in photo_records:
        url = rec["photo_url"]
        if not url:
            continue
        split = rec["split"]
        slug = rec["slug"]
        (args.output_dir / split / slug).mkdir(parents=True, exist_ok=True)
        path = (
            args.output_dir
            / split
            / slug
            / f"{rec['observation_id']}_{rec['photo_idx']}.jpg"
        )
        jobs.append((url, path))
        requested_by_split[split][slug] = (
            requested_by_split[split].get(slug, 0) + 1
        )

    if not args.download:
        print(
            "--no-download set: skipping image downloads. "
            "(observations.jsonl + per-split JSONLs already on disk.)",
            flush=True,
        )
        _write_report(
            args, observations, photo_records,
            requested=requested_by_split,
            delivered={s: {} for s in SPLIT_NAMES},
            total_bytes_by_split={s: 0 for s in SPLIT_NAMES},
            n_succ=0, n_jobs=len(jobs),
        )
        return 0

    print(
        f"Downloading {len(jobs)} images with {args.max_workers} workers...",
        flush=True,
    )
    n_succ = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for ok in ex.map(_download_one, jobs):
            if ok:
                n_succ += 1

    delivered_by_split: dict[str, dict[str, int]] = {s: {} for s in SPLIT_NAMES}
    total_bytes_by_split: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        split_dir = args.output_dir / split_name
        if not split_dir.is_dir():
            continue
        for slug_dir in sorted(split_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            files = list(slug_dir.glob("*.jpg"))
            delivered_by_split[split_name][slug_dir.name] = len(files)
            total_bytes_by_split[split_name] += sum(
                f.stat().st_size for f in files
            )

    print()
    print("=== Summary ===")
    for split_name in SPLIT_NAMES:
        req_total = sum(requested_by_split[split_name].values())
        del_total = sum(delivered_by_split[split_name].values())
        bytes_mb = total_bytes_by_split[split_name] / 1024 / 1024
        print(
            f"  {split_name:5s}  species={len(requested_by_split[split_name]):4d}  "
            f"requested={req_total:5d}  delivered={del_total:5d}  "
            f"size={bytes_mb:7.1f} MB"
        )
    print(
        f"Overall: {n_succ}/{len(jobs)} downloads succeeded "
        f"(idempotent: re-running skips files already on disk)."
    )

    _write_report(
        args, observations, photo_records,
        requested=requested_by_split,
        delivered=delivered_by_split,
        total_bytes_by_split=total_bytes_by_split,
        n_succ=n_succ, n_jobs=len(jobs),
    )
    return 0


def _write_report(
    args: argparse.Namespace,
    observations: list[dict[str, Any]],
    photo_records: list[dict[str, Any]],
    *,
    requested: dict[str, dict[str, int]],
    delivered: dict[str, dict[str, int]],
    total_bytes_by_split: dict[str, int],
    n_succ: int,
    n_jobs: int,
) -> None:
    report = args.output_dir / "fetch_report.json"
    report.write_text(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "request_params": {
            "taxon_id":      None if args.taxon_name else args.taxon_id,
            "taxon_name":    args.taxon_name,
            "quality_grade": args.quality_grade,
            "photos":        "true",
            "place_id":      args.place_id or None,
            "lat":           args.lat,
            "lng":           args.lng,
            "radius":        args.radius,
            "order_by":      args.order_by,
            "order":         args.order,
        },
        "strategy":               "species_counts -> per-taxon obs fetch",
        "target_species":         args.target_species,
        "obs_per_species":        args.obs_per_species,
        "max_pages_per_taxon":    args.max_pages_per_taxon,
        "download":               args.download,
        "split_ratios":           list(args.split_ratios),
        "split_seed":             args.seed,
        "n_observations":         len(observations),
        "n_photo_records":        len(photo_records),
        "n_photos_requested":     n_jobs,
        "n_photos_delivered":     n_succ,
        "requested_by_split":     requested,
        "delivered_by_split":     delivered,
        "total_bytes_by_split":   total_bytes_by_split,
    }, indent=2))
    print(f"Wrote: {report}")


if __name__ == "__main__":
    sys.exit(main())
