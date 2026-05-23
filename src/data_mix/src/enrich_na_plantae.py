#!/usr/bin/env python3
"""Enrich na_plantae observations with per-species text descriptions.

Consumes the JSONL output of ``scripts/na_plantae_fetch.py`` (one record
per photo, grouped by ``scientific_name`` / ``slug``), then for each
unique species:

  1. Matches the scientific name against the GBIF Backbone (kingdom
     filter defaults to ``Plantae`` since this corpus is iNaturalist's
     Plantae sweep).
  2. Pulls English descriptions, vernacular names, distributions, and
     species profiles from GBIF when available.
  3. Falls back to the English Wikipedia summary for a readable blurb.
  4. Emits one enriched JSONL record per species, plus a RAG-ready
     ``text`` field that concatenates the structured fields.

Outputs (in ``--output-dir``, default
``<output_dir>/`` where ``<output_dir>`` is the same dir that holds
``observations.jsonl``)::

    species_enriched.jsonl   # one JSON line per unique species
    species_rag_docs.jsonl   # RAG-ready docs (id, text, metadata)
    .species_cache/          # http response cache (resume-friendly)

Both output files are flushed after every species so an interrupted
run loses at most the in-flight species. Re-running picks up where the
previous run left off (resume key = scientific_name).

Network budget: ~5 GBIF + 2 Wikipedia calls per species, throttled to
keep us under public-API soft caps. For ~950 species the full sweep
takes ~30-40 min cold; subsequent runs hit the JSON cache and finish
in under a minute.

Usage::

    # Default: read <repo>/../data/inaturalist_na_plantae/observations.jsonl,
    # write species_enriched.jsonl + species_rag_docs.jsonl next to it.
    python src/data_mix/src/enrich_na_plantae.py

    # Custom location + a quick smoke test on the first 20 species:
    python src/data_mix/src/enrich_na_plantae.py \\
        --input /path/to/observations.jsonl \\
        --max-species 20

    # Force a refetch (ignore the on-disk JSON cache):
    python src/data_mix/src/enrich_na_plantae.py --refresh-cache
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
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


GBIF_API = "https://api.gbif.org/v1"
POWO_SEARCH = "https://powo.science.kew.org/results?q={q}"

DEFAULT_USER_AGENT = (
    "trailogy-na-plantae-enrich/0.1 "
    "(set --user-agent with your email or project URL)"
)

# Default I/O lives next to ``observations.jsonl``. Script path is
# ``<repo>/src/data_mix/src/enrich_na_plantae.py``; parents[3] = <repo>.
_SCRIPT_REPO = Path(__file__).resolve().parents[3]

# GBIF / Wikipedia tagging conventions.
ENGLISH_LANG_TAGS = {"en", "eng", "english"}

# Catalogue-of-Life packs multiple translations into one row with a
# trailing "(XX)" language marker per chunk (e.g.
# "Weiden-Lattich (DE); Least Lettuce (EN)"). The regex parses one chunk.
_LANG_SUFFIX_RE = re.compile(r"^(.*?)\s*\(([A-Z]{2,3})\)\s*$")


def _clean_text(value: Any) -> str:
    """Strip HTML, collapse whitespace, decode entities."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class HttpClient:
    """Tiny GET-only HTTP client with on-disk JSON caching + retry/backoff.

    The cache key is ``cache_prefix + sha1(url?sorted_params)`` so the
    same logical request hits the same cache file across runs. Errors
    and 404s are cached too (with ``_error`` / ``_status_code`` keys)
    to keep "missing on this source" responses cheap on re-runs.
    """

    cache_dir: Path
    sleep: float
    timeout: float
    user_agent: str
    refresh: bool = False
    max_retries: int = 5

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent or DEFAULT_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        })

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        cache_prefix: str = "http",
    ) -> Any | None:
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        key = (
            cache_prefix
            + "_"
            + _short_hash(
                url + "?" + urllib.parse.urlencode(sorted(params.items()))
            )
            + ".json"
        )
        cache_path = self.cache_dir / key

        if cache_path.exists() and not self.refresh:
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                pass

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 404:
                    data = {"_status_code": 404, "_url": resp.url}
                    cache_path.write_text(
                        json.dumps(data, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return data
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt == self.max_retries - 1:
                        print(
                            f"  [http] giving up after HTTP "
                            f"{resp.status_code} on {url}",
                            file=sys.stderr, flush=True,
                        )
                        return {
                            "_error": f"HTTP {resp.status_code}",
                            "_status_code": resp.status_code,
                            "_url": resp.url,
                            "_params": params,
                        }
                    # Honor Retry-After header on 429 (Wikipedia + GBIF
                    # both send it); else exponential backoff capped at
                    # 60 s so a brief throttle doesn't stall the run.
                    retry_after = resp.headers.get("Retry-After") or ""
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 0.0
                    if wait <= 0:
                        wait = min(2 ** attempt + self.sleep, 60.0)
                    print(
                        f"  [http] retry {attempt + 1}/{self.max_retries} "
                        f"HTTP {resp.status_code} on {url} "
                        f"sleeping {wait:.1f}s",
                        file=sys.stderr, flush=True,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    data.setdefault("_url", resp.url)
                cache_path.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding="utf-8",
                )
                if self.sleep:
                    time.sleep(self.sleep)
                return data
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    print(
                        f"  [http] giving up after {type(exc).__name__} "
                        f"on {url}: {exc}",
                        file=sys.stderr, flush=True,
                    )
                    return {
                        "_error": repr(exc),
                        "_url": url,
                        "_params": params,
                    }
                print(
                    f"  [http] retry {attempt + 1}/{self.max_retries} "
                    f"{type(exc).__name__} on {url}",
                    file=sys.stderr, flush=True,
                )
                time.sleep((2 ** attempt) + self.sleep)
        return None


def default_input_path() -> Path:
    """Resolve the default observations path using the shared data root.

    The package import path is available under pytest / ``python -m``.
    The fallback keeps direct script execution working while preserving
    the same ``TRAILOGY_DATA_ROOT`` override semantics.
    """
    try:
        from data_mix.src.env_paths import external_data_root

        root = external_data_root()
    except ModuleNotFoundError:
        env_root = os.environ.get("TRAILOGY_DATA_ROOT")
        root = (
            Path(env_root).expanduser().resolve()
            if env_root
            else (_SCRIPT_REPO.parent / "data").resolve()
        )
    return root / "inaturalist_na_plantae" / "observations.jsonl"


# --- GBIF helpers ----------------------------------------------------

def gbif_match(
    client: HttpClient,
    scientific_name: str,
    kingdom: str | None = "Plantae",
) -> dict[str, Any]:
    """Match a scientific name against the GBIF Backbone.

    ``kingdom='Plantae'`` is the right default here — the upstream fetch
    is filtered to iNaturalist's Plantae kingdom — and reduces
    cross-kingdom homonym collisions (e.g. an animal genus that shares
    a Latin name).
    """
    params: dict[str, Any] = {"name": scientific_name, "verbose": "true"}
    if kingdom:
        params["kingdom"] = kingdom
    return client.get_json(
        f"{GBIF_API}/species/match",
        params=params,
        cache_prefix="gbif_match",
    ) or {}


def gbif_extension(
    client: HttpClient,
    usage_key: Any,
    endpoint: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch one of the per-species GBIF extension endpoints
    (descriptions / vernacularNames / distributions / speciesProfiles).

    Returns ``[]`` on 404 / network error so the caller can keep
    advancing through other extensions for the same species.
    """
    if not usage_key:
        return []
    data = client.get_json(
        f"{GBIF_API}/species/{usage_key}/{endpoint}",
        params={"limit": limit},
        cache_prefix=f"gbif_{endpoint}",
    ) or {}
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if data.get("_status_code") == 404 or data.get("_error"):
        return []
    results = data.get("results")
    if isinstance(results, list):
        return results
    if isinstance(data, list):
        return data
    return []


def _extract_english_names(name: str, lang: str) -> list[str]:
    """Return clean English vernacular names from one GBIF vernacular row.

    Handles three GBIF/CoL idioms:

      1. ``language`` ∈ {en, eng, english}: the whole string is English.
      2. Untagged row with a single trailing ``(EN)``: strip the marker.
      3. Untagged row with packed multilingual chunks separated by
         ``;``, each tagged ``(XX)``: keep only ``(EN)`` chunks.

    Rows tagged with a non-English language, or untagged rows with no
    ``(EN)`` marker anywhere, return ``[]`` and are dropped by callers.
    """
    if not name:
        return []
    if lang.lower() in ENGLISH_LANG_TAGS:
        return [name]
    chunks = [c.strip() for c in name.split(";") if c.strip()]
    tagged = [(_LANG_SUFFIX_RE.match(c), c) for c in chunks]
    if any(m for m, _ in tagged):
        out: list[str] = []
        for m, _c in tagged:
            if m and m.group(2) == "EN":
                stripped = m.group(1).strip()
                if stripped:
                    out.append(stripped)
        return out
    return []


def pick_gbif_description(
    items: list[dict[str, Any]],
    english_only: bool = True,
) -> tuple[str, str, str]:
    """Pick the best English (or longest, if ``english_only=False``)
    description from a list of GBIF description records.

    Returns ``(description, source, language)``. When ``english_only``
    is True we drop entries explicitly tagged as a non-English
    language. Entries with an empty ``language`` field are kept: many
    GBIF English-language datasets leave the tag blank.
    """
    candidates: list[tuple[int, str, str, str]] = []
    for item in items:
        desc = _clean_text(
            item.get("description")
            or item.get("descriptionText")
            or item.get("text")
        )
        if not desc:
            continue
        lang = _clean_text(item.get("language"))
        lang_lc = lang.lower()
        is_eng = lang_lc in ENGLISH_LANG_TAGS
        if english_only and lang and not is_eng:
            continue
        source = _clean_text(
            item.get("source")
            or item.get("sourceTaxonKey")
            or item.get("datasetTitle")
        )
        score = len(desc)
        if is_eng:
            score += 10000
        candidates.append((score, desc, source, lang))
    if not candidates:
        return "", "", ""
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, desc, source, lang = candidates[0]
    return desc, source, lang


def collect_common_names(
    items: list[dict[str, Any]],
    max_names: int = 12,
    english_only: bool = True,
) -> str:
    """Join unique English vernacular names with semicolons.

    Stable order: rows explicitly tagged English first (cleanest single
    names), then rows where the ``(EN)`` marker has to be parsed out
    of a packed multilingual chunk; alphabetical within each bucket.
    """
    names: list[str] = []
    seen: set[str] = set()

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        lang = _clean_text(item.get("language")).lower()
        name = _clean_text(item.get("vernacularName"))
        return (0 if lang in ENGLISH_LANG_TAGS else 1, name.lower())

    for item in sorted(items, key=sort_key):
        raw_name = _clean_text(item.get("vernacularName"))
        lang = _clean_text(item.get("language"))
        if not raw_name:
            continue
        if english_only:
            candidates = _extract_english_names(raw_name, lang)
        else:
            candidates = [raw_name]
        for nm in candidates:
            key = nm.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(nm)
            if len(names) >= max_names:
                break
        if len(names) >= max_names:
            break
    return "; ".join(names)


def collect_distributions(
    items: list[dict[str, Any]], max_items: int = 20
) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for item in items:
        loc = _clean_text(
            item.get("locationID")
            or item.get("locality")
            or item.get("country")
            or item.get("area")
        )
        status = _clean_text(
            item.get("status") or item.get("establishmentMeans")
        )
        if not loc:
            continue
        text = f"{loc} ({status})" if status else loc
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        parts.append(text)
        if len(parts) >= max_items:
            break
    return "; ".join(parts)


def collect_profiles(items: list[dict[str, Any]], max_items: int = 8) -> str:
    parts: list[str] = []
    for item in items[:max_items]:
        kv: list[str] = []
        for k in (
            "habitat",
            "lifeForm",
            "isMarine",
            "isFreshwater",
            "isTerrestrial",
        ):
            if k in item and item[k] not in (None, ""):
                kv.append(f"{k}: {item[k]}")
        if kv:
            parts.append(", ".join(kv))
    return "; ".join(parts)


# --- Wikipedia helpers ----------------------------------------------

def wikipedia_summary(
    client: HttpClient,
    scientific_name: str,
    lang: str = "en",
    query_suffix: str = "plant",
) -> tuple[str, str, str]:
    """Fetch a Wikipedia summary for ``scientific_name``.

    Returns ``(extract_text, page_title, page_url)`` — empty strings if
    no acceptable page is found.

    ``query_suffix`` biases full-text search toward one sense of the
    name. We default to ``"plant"`` because the upstream fetch is
    Plantae-only. Empty string for mixed-taxa lists.
    """
    api = f"https://{lang}.wikipedia.org/w/api.php"
    rest = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary"
    query = f'"{scientific_name}" {query_suffix}'.strip()
    data = client.get_json(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        },
        cache_prefix=f"wiki_{lang}_search",
    ) or {}
    search = (
        ((data.get("query") or {}).get("search")) or []
        if isinstance(data, dict)
        else []
    )
    if not search:
        return "", "", ""

    genus_species = scientific_name.lower().split()[:2]
    chosen_title = ""
    for hit in search:
        title = _clean_text(hit.get("title"))
        hay = (title + " " + _clean_text(hit.get("snippet"))).lower()
        if (
            title.lower() == scientific_name.lower()
            or all(x in hay for x in genus_species)
        ):
            chosen_title = title
            break
    if not chosen_title:
        return "", "", ""

    summary = client.get_json(
        f"{rest}/{urllib.parse.quote(chosen_title.replace(' ', '_'), safe='')}",
        params={},
        cache_prefix=f"wiki_{lang}_summary",
    ) or {}
    if summary.get("type") == "disambiguation":
        url = (
            summary.get("content_urls", {})
            .get("desktop", {})
            .get("page", "")
        )
        return "", chosen_title, url
    text = _clean_text(summary.get("extract"))
    url = ((summary.get("content_urls") or {}).get("desktop") or {}).get(
        "page", ""
    )
    return text, chosen_title, url


def make_powo_search_url(scientific_name: str) -> str:
    return POWO_SEARCH.format(q=urllib.parse.quote_plus(scientific_name))


# --- Per-species enrichment -----------------------------------------

def build_rag_text(species: dict[str, str], enriched: dict[str, str]) -> str:
    """Concatenate structured fields into a single RAG-ready text block."""
    chunks = [
        f"Scientific name: {species.get('scientific_name', '')}",
        f"Common name (iNaturalist): {species.get('common_name', '')}",
        f"Slug: {species.get('slug', '')}",
    ]
    for label, key in [
        ("Accepted scientific name", "accepted_scientific_name"),
        ("Common names (GBIF)", "common_names"),
        ("GBIF description", "gbif_description"),
        ("Wikipedia summary", "wikipedia_summary"),
        ("Distribution", "gbif_distribution"),
        ("Profile", "gbif_profile"),
    ]:
        if enriched.get(key):
            chunks.append(f"{label}: {enriched[key]}")
    sources: list[str] = []
    for key in ("gbif_url", "wikipedia_url", "powo_search_url"):
        if enriched.get(key):
            sources.append(enriched[key])
    if sources:
        chunks.append("Sources: " + " | ".join(sources))
    return "\n".join(chunks)


def enrich_species(
    client: HttpClient,
    species: dict[str, str],
    wiki_lang: str = "en",
    kingdom: str | None = "Plantae",
    wiki_query_suffix: str = "plant",
    english_only: bool = True,
) -> dict[str, str]:
    """Enrich one species record. Always returns a populated dict; empty
    string fields mean the source had no usable text for that species.
    """
    sci = _clean_text(species.get("scientific_name") or "")
    result: dict[str, str] = {
        "gbif_usage_key": "",
        "gbif_match_type": "",
        "gbif_confidence": "",
        "gbif_status": "",
        "accepted_scientific_name": "",
        "gbif_url": "",
        "common_names": "",
        "gbif_description": "",
        "gbif_description_source": "",
        "gbif_description_language": "",
        "gbif_distribution": "",
        "gbif_profile": "",
        "wikipedia_title": "",
        "wikipedia_summary": "",
        "wikipedia_url": "",
        "powo_search_url": make_powo_search_url(sci),
        "best_description": "",
        "best_description_source": "",
        "rag_text": "",
        "fetch_status": "ok",
    }
    if not sci:
        result["fetch_status"] = "missing scientific name"
        return result

    match = gbif_match(client, sci, kingdom=kingdom)
    if match.get("_error"):
        result["fetch_status"] = "gbif_match_error"
    usage_key = (
        match.get("acceptedUsageKey")
        or match.get("usageKey")
        or match.get("speciesKey")
    )
    result["gbif_usage_key"] = str(usage_key or "")
    result["gbif_match_type"] = _clean_text(match.get("matchType"))
    result["gbif_confidence"] = str(match.get("confidence", ""))
    result["gbif_status"] = _clean_text(match.get("status"))
    result["accepted_scientific_name"] = _clean_text(
        match.get("species")
        or match.get("canonicalName")
        or match.get("scientificName")
    )
    if usage_key:
        result["gbif_url"] = f"https://www.gbif.org/species/{usage_key}"

        descriptions = gbif_extension(client, usage_key, "descriptions")
        desc, desc_source, desc_lang = pick_gbif_description(
            descriptions, english_only=english_only,
        )
        result["gbif_description"] = desc
        result["gbif_description_source"] = desc_source
        result["gbif_description_language"] = desc_lang

        vernacular = gbif_extension(client, usage_key, "vernacularNames")
        result["common_names"] = collect_common_names(
            vernacular, english_only=english_only,
        )

        distributions = gbif_extension(client, usage_key, "distributions")
        result["gbif_distribution"] = collect_distributions(distributions)

        profiles = gbif_extension(client, usage_key, "speciesProfiles")
        result["gbif_profile"] = collect_profiles(profiles)

    wiki_text, wiki_title, wiki_url = wikipedia_summary(
        client, sci, lang=wiki_lang, query_suffix=wiki_query_suffix,
    )
    result["wikipedia_title"] = wiki_title
    result["wikipedia_summary"] = wiki_text
    result["wikipedia_url"] = wiki_url

    # Priority: Wikipedia lead paragraph first (consistent "readable
    # one-paragraph summary" shape), GBIF description as fallback. Both
    # raw fields stay populated under ``wikipedia_summary`` /
    # ``gbif_description`` so downstream consumers can keep both.
    if result["wikipedia_summary"]:
        result["best_description"] = result["wikipedia_summary"]
        result["best_description_source"] = f"Wikipedia-{wiki_lang}"
    elif result["gbif_description"]:
        result["best_description"] = result["gbif_description"]
        result["best_description_source"] = "GBIF"
    else:
        result["best_description"] = ""
        result["best_description_source"] = ""
        if result["fetch_status"] == "ok":
            result["fetch_status"] = "no_description_found"

    result["rag_text"] = build_rag_text(species, result)
    return result


# --- I/O ------------------------------------------------------------

def unique_species_from_observations(
    path: Path,
) -> list[dict[str, str]]:
    """Read observations.jsonl and return one record per unique
    ``scientific_name``.

    Keeps the first-seen ``common_name`` / ``slug`` per species and
    records ``n_observations`` / ``n_photos`` (each observation can
    flatten to multiple photo records). Output is sorted by scientific
    name for deterministic re-runs.
    """
    by_sci: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sci = (rec.get("scientific_name") or "").strip()
            if not sci:
                continue
            entry = by_sci.get(sci)
            if entry is None:
                entry = {
                    "scientific_name": sci,
                    "common_name": rec.get("common_name") or "",
                    "slug": rec.get("slug") or "",
                    "rank": rec.get("rank") or "",
                    "_obs_ids": set(),
                    "n_photos": 0,
                }
                by_sci[sci] = entry
            obs_id = rec.get("observation_id")
            if obs_id is not None:
                entry["_obs_ids"].add(obs_id)
            entry["n_photos"] += 1
    out: list[dict[str, str]] = []
    for sci in sorted(by_sci):
        entry = by_sci[sci]
        out.append({
            "scientific_name": entry["scientific_name"],
            "common_name": entry["common_name"],
            "slug": entry["slug"],
            "rank": entry["rank"],
            "n_observations": len(entry["_obs_ids"]),
            "n_photos": entry["n_photos"],
        })
    return out


def load_resume_state(
    enriched_path: Path,
    docs_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Load existing enriched + docs JSONL so an interrupted run can
    resume. Resume key is ``scientific_name``. A row counts as done
    only if it appears in BOTH files (singletons are dropped so the
    next pass re-fetches them).
    """
    enriched: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    enr_ids: set[str] = set()
    doc_ids: set[str] = set()

    if enriched_path.exists():
        try:
            with enriched_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sci = (rec.get("scientific_name") or "").strip()
                    if not sci:
                        continue
                    enriched.append(rec)
                    enr_ids.add(sci)
        except Exception as e:  # noqa: BLE001
            print(
                f"[resume] WARN: failed to read {enriched_path}: {e}",
                file=sys.stderr,
            )
            enriched, enr_ids = [], set()

    if docs_path.exists():
        try:
            with docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sci = (doc.get("scientific_name") or "").strip()
                    if not sci:
                        continue
                    docs.append(doc)
                    doc_ids.add(sci)
        except Exception as e:  # noqa: BLE001
            print(
                f"[resume] WARN: failed to read {docs_path}: {e}",
                file=sys.stderr,
            )
            docs, doc_ids = [], set()

    done_ids = enr_ids & doc_ids
    if done_ids != enr_ids or done_ids != doc_ids:
        only_enr = enr_ids - done_ids
        only_docs = doc_ids - done_ids
        if only_enr:
            print(
                f"[resume] dropping {len(only_enr)} enriched rows "
                f"missing from docs",
                file=sys.stderr,
            )
            enriched = [
                r for r in enriched
                if (r.get("scientific_name") or "").strip() in done_ids
            ]
        if only_docs:
            print(
                f"[resume] dropping {len(only_docs)} docs rows "
                f"missing from enriched",
                file=sys.stderr,
            )
            docs = [
                d for d in docs
                if (d.get("scientific_name") or "").strip() in done_ids
            ]
    return enriched, docs, done_ids


def filter_resume_to_species(
    enriched: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    done_ids: set[str],
    species_list: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    current_ids = {s["scientific_name"] for s in species_list}
    if not current_ids:
        return [], [], set()
    stale_ids = done_ids - current_ids
    if stale_ids:
        print(
            f"[resume] dropping {len(stale_ids)} rows not present in "
            f"current input",
            file=sys.stderr,
        )
    return (
        [
            r for r in enriched
            if (r.get("scientific_name") or "").strip() in current_ids
        ],
        [
            d for d in docs
            if (d.get("scientific_name") or "").strip() in current_ids
        ],
        done_ids & current_ids,
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for rec in records:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_input = default_input_path()

    ap.add_argument(
        "--input", type=Path, default=default_input,
        help=f"Path to observations.jsonl from na_plantae_fetch.py. "
             f"Default: {default_input}.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir for species_enriched.jsonl + species_rag_docs.jsonl "
             "+ .species_cache/. Default: same dir as --input.",
    )
    ap.add_argument(
        "--enriched-name", default="species_enriched.jsonl",
        help="Filename for the enriched per-species JSONL. Default: "
             "species_enriched.jsonl.",
    )
    ap.add_argument(
        "--docs-name", default="species_rag_docs.jsonl",
        help="Filename for the RAG-ready docs JSONL. Default: "
             "species_rag_docs.jsonl.",
    )
    ap.add_argument(
        "--cache-dir-name", default=".species_cache",
        help="HTTP cache dir (relative to --output-dir). Default: "
             ".species_cache.",
    )
    ap.add_argument(
        "--max-species", type=int, default=0,
        help="If > 0, process only the first N unique species (for "
             "smoke tests). Default: 0 (all).",
    )
    ap.add_argument(
        "--sleep", type=float, default=0.25,
        help="Sleep seconds after each uncached HTTP request. "
             "Default: 0.25.",
    )
    ap.add_argument(
        "--timeout", type=float, default=30.0,
        help="HTTP timeout in seconds. Default: 30.",
    )
    ap.add_argument(
        "--refresh-cache", action="store_true",
        help="Ignore existing cached JSON and refetch.",
    )
    ap.add_argument(
        "--wiki-language", default="en",
        help="Wikipedia language code. Default: en.",
    )
    ap.add_argument(
        "--user-agent", default=DEFAULT_USER_AGENT,
        help="Set a descriptive User-Agent (contact info recommended "
             "for GBIF / Wikipedia rate-limit good citizenship).",
    )
    ap.add_argument(
        "--kingdom", default="Plantae",
        help="GBIF kingdom filter (reduces cross-kingdom homonyms). "
             "Pass 'none' to disable. Default: 'Plantae'.",
    )
    ap.add_argument(
        "--wiki-query-suffix", default="auto",
        help="Extra term appended to Wikipedia full-text search. "
             "'auto' (default) -> 'plant' when --kingdom=Plantae, "
             "blank when --kingdom=none.",
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing enriched + docs files; start from scratch.",
    )
    ap.add_argument(
        "--english-only", dest="english_only",
        action="store_true", default=True,
        help="(Default) Hard-filter GBIF descriptions + common names to "
             "English; non-English entries dropped.",
    )
    ap.add_argument(
        "--no-english-only", dest="english_only", action="store_false",
        help="Disable the English filter (use any-language "
             "longest-text behaviour).",
    )
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: --input not found: {args.input}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / args.enriched_name
    docs_path = output_dir / args.docs_name
    cache_dir = output_dir / args.cache_dir_name

    kingdom_arg: str | None = args.kingdom
    if kingdom_arg and kingdom_arg.lower() in {"none", "any", "all", ""}:
        kingdom_arg = None

    wiki_suffix: str = args.wiki_query_suffix
    if wiki_suffix == "auto":
        wiki_suffix = "plant" if kingdom_arg == "Plantae" else ""

    print(
        f"== na_plantae enrichment ==\n"
        f"  input        : {args.input}\n"
        f"  output dir   : {output_dir}\n"
        f"  enriched out : {enriched_path.name}\n"
        f"  docs out     : {docs_path.name}\n"
        f"  cache        : {cache_dir}\n"
        f"  kingdom      : {kingdom_arg or '<none>'}\n"
        f"  wiki suffix  : {wiki_suffix!r}\n"
        f"  english only : {args.english_only}",
        flush=True,
    )

    print(f"Step 1: load unique species from {args.input.name} ...", flush=True)
    species_list = unique_species_from_observations(args.input)
    print(f"  -> {len(species_list)} unique species", flush=True)
    if args.max_species and args.max_species > 0:
        species_list = species_list[: args.max_species]
        print(f"  -> capped to first {len(species_list)}", flush=True)

    if args.no_resume:
        enriched_rows: list[dict[str, Any]] = []
        docs: list[dict[str, Any]] = []
        done_ids: set[str] = set()
    else:
        enriched_rows, docs, done_ids = load_resume_state(
            enriched_path, docs_path
        )
        enriched_rows, docs, done_ids = filter_resume_to_species(
            enriched_rows, docs, done_ids, species_list
        )
        if done_ids:
            print(
                f"[resume] {len(done_ids)} species already enriched; "
                f"will skip",
                file=sys.stderr,
            )

    pending = [
        s for s in species_list
        if s["scientific_name"] not in done_ids
    ]
    skipped = len(species_list) - len(pending)
    if skipped and not args.no_resume:
        print(
            f"[resume] skipping {skipped} already-enriched species",
            file=sys.stderr,
        )

    client = HttpClient(
        cache_dir=cache_dir,
        sleep=args.sleep,
        timeout=args.timeout,
        user_agent=args.user_agent,
        refresh=args.refresh_cache,
    )

    total = len(pending)
    print(
        f"Step 2: enrich {total} species (GBIF match + 4 extensions + "
        f"Wikipedia summary each) ...",
        flush=True,
    )
    for i, species in enumerate(pending, start=1):
        sci = species["scientific_name"]
        print(f"[{i}/{total}] {sci}", file=sys.stderr)
        enriched = enrich_species(
            client,
            species,
            wiki_lang=args.wiki_language,
            kingdom=kingdom_arg,
            wiki_query_suffix=wiki_suffix,
            english_only=args.english_only,
        )
        row = dict(species)
        row.update(enriched)
        enriched_rows.append(row)

        doc_key = f"{species.get('slug') or 'species'}:{_short_hash(sci)}"
        docs.append({
            "id": f"na_plantae:{doc_key}",
            "scientific_name": sci,
            "common_name": species.get("common_name"),
            "slug": species.get("slug"),
            "n_observations": species.get("n_observations"),
            "n_photos": species.get("n_photos"),
            "text": enriched.get("rag_text", ""),
            "metadata": {
                "gbif_usage_key": enriched.get("gbif_usage_key"),
                "gbif_url": enriched.get("gbif_url"),
                "wikipedia_url": enriched.get("wikipedia_url"),
                "powo_search_url": enriched.get("powo_search_url"),
                "best_description_source": enriched.get(
                    "best_description_source"
                ),
                "fetch_status": enriched.get("fetch_status"),
            },
        })

        # Per-row flush: an interrupted fetch loses at most the
        # in-flight species. ~950 species * ~10 kB output rewritten
        # per row -> ~10 MB of redundant disk writes, negligible vs.
        # network round-trips between rows.
        _write_jsonl(enriched_path, enriched_rows)
        _write_jsonl(docs_path, docs)

    _write_jsonl(enriched_path, enriched_rows)
    _write_jsonl(docs_path, docs)

    status_counts: dict[str, int] = {}
    for row in enriched_rows:
        s = row.get("fetch_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print("=== Summary ===", file=sys.stderr)
    print(
        json.dumps(
            {
                "species": len(enriched_rows),
                "status_counts": status_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    print(str(enriched_path))
    print(str(docs_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
