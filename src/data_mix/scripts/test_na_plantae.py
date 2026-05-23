#!/usr/bin/env python3
"""Quick NA-Plantae-ID eyeball test against a SFT'd Gemma 4 E2B MLX checkpoint.

Mirrors ``src/finetune/src/evaluate.py``'s eval contract: the model
was trained with a v4 conditional-FT camera-state gate
(``data.prompt_prefixes={camera_on: "[camera=on] ", camera_off:
"[camera=off] "}``). Every image record gets ``[camera=on]`` prepended
to the first user turn at train time, so deploy-time prompts MUST
inject the same marker or species_match collapses to ~0.

Input
-----
Auto-detects two layouts under ``--images-dir``:

  - **subfolder layout** (na_plantae default): ``<slug>/<NN>.<ext>``
    (e.g. ``red_maple/003.jpg``). Slug = subfolder name. Used as-is
    for matching, also normalised to ``red-maple`` and ``Red Maple``.
  - **flat layout**: ``<slug>-<N>.<ext>`` directly under
    ``--images-dir`` (e.g. ``red-maple-3.jpg``). Useful for
    hand-curated subsets.

Both slugs ``red_maple`` and ``red-maple`` are accepted; underscores
and hyphens are interchangeable for the match check.

If ``--descriptions`` is given (yaml at
``assets/na_plantae/descriptions.yaml``), the matcher additionally
accepts the species' ``common_name`` and ``species`` (Latin name)
substrings — gives less noisy precision/recall on long natural-
language replies.

Output
------
- Per-image prediction printed live as the run progresses.
- ``<results-dir>/<model-tag>__<timestamp>.json`` with full
  per-image records (prompt, response, match flag, wall time).

Usage
-----
    # Local MLX dir
    python src/data_mix/scripts/test_na_plantae.py \\
        --model-path <local-mlx-dir> \\
        --model-tag r16-a16-step20000_g64_noaudio

    # HF model id (downloads to HF cache on first run)
    python src/data_mix/scripts/test_na_plantae.py \\
        --model-path <repo-owner>/gemma-4-E2B/r16-a16-nokl-step20000_mlx_g64_noaudio \\
        --model-tag r16-a16-step20000_g64_noaudio

Compare two checkpoints by running twice with different ``--model-tag``;
both JSONs land in ``<results-dir>`` for side-by-side review.

Camera-state prefix is hard-coded to ``[camera=on] `` (the value the
production v4 checkpoints were trained with). Override with
``--camera-prefix`` if you want a different gate string or empty
(``--camera-prefix ""`` for pre-v4 checkpoints).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".gif")

# v4 contract — must match training-time ``data.prompt_prefixes.camera_on``
DEFAULT_CAMERA_PREFIX = "[camera=on] "

# A neutral "what is this plant?" prompt — matches the most common
# question in ``src/finetune/src/prepare_plantnet.py:QUESTION_TEMPLATES``
# (and ``src/data_mix/src/prepare_na_plantae.py:QUESTION_TEMPLATES``).
DEFAULT_PROMPT = "What plant is this?"


def slug_to_label(slug: str) -> str:
    """Turn ``red-maple`` → ``Red Maple`` for display + match check."""
    return " ".join(w.capitalize() for w in slug.split("-"))


def parse_filename(name: str) -> tuple[str, str, str]:
    """Split ``red-maple-3.jpg`` → (``red-maple``, ``Red Maple``, ``3``).

    Trailing ``-<digits>`` is the per-species index. Everything before is
    the species slug.
    """
    stem = Path(name).stem  # red-maple-3
    m = re.match(r"^(.+)-(\d+)$", stem)
    if not m:
        # No -N suffix; treat whole stem as the slug.
        return stem, slug_to_label(stem), "0"
    slug, idx = m.group(1), m.group(2)
    return slug, slug_to_label(slug), idx


def discover_images(images_dir: Path) -> list[tuple[Path, str]]:
    """Return [(image_path, slug), ...] auto-detecting subfolder vs flat layout.

    Subfolder layout takes precedence: if any direct subdir contains
    image files, ALL images come from subdir traversal and the slug
    is the subdir name. Otherwise fall back to flat ``<slug>-<N>.<ext>``
    parsing of files at the top level.
    """
    if not images_dir.is_dir():
        return []
    # Detect subfolder layout — any subdir with at least one image file?
    subdir_images: list[tuple[Path, str]] = []
    has_subdirs_with_images = False
    for sub in sorted(images_dir.iterdir()):
        if not sub.is_dir():
            continue
        files = [
            p for p in sorted(sub.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        if files:
            has_subdirs_with_images = True
            for p in files:
                subdir_images.append((p, sub.name))
    if has_subdirs_with_images:
        return subdir_images
    # Fall back to flat layout.
    flat: list[tuple[Path, str]] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            slug, _, _ = parse_filename(p.name)
            flat.append((p, slug))
    return flat


def normalise_slug(s: str) -> str:
    """Underscores / hyphens / spaces collapse to a single canonical form."""
    return s.lower().replace("_", "-").replace(" ", "-")


def load_descriptions_if_any(path: Path | None) -> dict[str, dict]:
    """Optional yaml of species → {common_name, species, ...}. Returns
    a dict keyed by normalised slug (hyphenated). Empty if path is None
    or the file is missing."""
    if path is None or not Path(path).exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "WARNING: PyYAML not installed; --descriptions ignored",
            flush=True,
        )
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, dict] = {}
    for t in raw.get("trees", []):
        slug = normalise_slug(t["slug"])
        out[slug] = t
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model-path",
        required=True,
        help="Local MLX dir, or HF repo id like '<repo-owner>/gemma-4-E2B/...'",
    )
    ap.add_argument(
        "--model-tag",
        required=True,
        help="Short label for the JSON output file (no slashes).",
    )
    # script path: <repo>/src/data_mix/scripts/test_na_plantae.py
    #   parents[3] -> <repo>
    #   parents[3].parent -> <repo>'s parent (== external data root parent)
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _EXTERNAL = _REPO_ROOT.parent / "data"
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=_EXTERNAL / "inaturalist_na_plantae" / "test",
        help=(
            "Directory of test images. Subfolder layout "
            "(<slug>/<N>.jpg) auto-detected; falls back to flat "
            "<slug>-<N>.<ext>. Default points at the held-out test "
            "split produced by na_plantae_fetch.py: "
            "<repo>/../data/inaturalist_na_plantae/test/."
        ),
    )
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=_REPO_ROOT / "assets" / "na_plantae" / "descriptions.yaml",
        help=(
            "Optional yaml of species -> {common_name, species, ...} "
            "for richer match scoring. Default: "
            "<repo>/assets/na_plantae/descriptions.yaml"
        ),
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=_EXTERNAL / "run_results" / "na_plantae_eyeball",
        help="Where to write per-run JSON outputs. Default: "
             "<repo>/../data/run_results/na_plantae_eyeball/.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of images to test (after sorting). "
             "Use 50 to mirror the 'add 50 trees to eval' setup.",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"User-turn question. Default: {DEFAULT_PROMPT!r}",
    )
    ap.add_argument(
        "--camera-prefix",
        default=DEFAULT_CAMERA_PREFIX,
        help=(
            f"v4 conditional-FT prefix prepended to the user turn. "
            f"Default: {DEFAULT_CAMERA_PREFIX!r}. Use empty string for "
            "pre-v4 checkpoints."
        ),
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 = greedy (deterministic).",
    )
    args = ap.parse_args(argv)

    images = discover_images(args.images_dir)
    if not images:
        print(f"ERROR: no images in {args.images_dir}", file=sys.stderr)
        return 2
    if args.limit is not None:
        # Stratify across species so a small --limit doesn't all come
        # from one folder. Round-robin from each slug bucket.
        from collections import defaultdict
        by_slug = defaultdict(list)
        for p, slug in images:
            by_slug[slug].append((p, slug))
        rr: list[tuple[Path, str]] = []
        slugs = sorted(by_slug)
        idx = 0
        while len(rr) < args.limit and any(by_slug.values()):
            slug = slugs[idx % len(slugs)]
            bucket = by_slug[slug]
            if bucket:
                rr.append(bucket.pop(0))
            idx += 1
            if idx > 1_000_000:  # safety
                break
        images = rr
    print(
        f"Discovered {len(images)} image(s) in {args.images_dir}",
        file=sys.stderr,
    )
    descriptions = load_descriptions_if_any(args.descriptions)
    if descriptions:
        print(
            f"Loaded {len(descriptions)} species description(s) from {args.descriptions}",
            file=sys.stderr,
        )

    # Lazy heavy imports — mlx_vlm pulls in MLX + transformers.
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"Loading model: {args.model_path}", file=sys.stderr)
    t_load = time.perf_counter()
    model, processor = load(args.model_path)
    print(
        f"  loaded in {time.perf_counter() - t_load:.1f}s",
        file=sys.stderr,
    )

    user_text = f"{args.camera_prefix}{args.prompt}"
    print(f"User turn: {user_text!r}", file=sys.stderr)

    per_sample: list[dict] = []
    matches = 0
    total_wall = 0.0

    for i, (img_path, raw_slug) in enumerate(images, 1):
        # Subfolder mode supplies raw_slug from the directory name;
        # flat mode parses it from the filename.
        slug = normalise_slug(raw_slug)
        _, label, idx = parse_filename(img_path.name) if "-" in img_path.stem else (slug, slug_to_label(slug), "0")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        # ``apply_chat_template`` collapses messages to the raw prompt
        # string that ``generate`` expects. mlx_vlm 0.4.x: pass the
        # messages list directly (it handles the image content blocks).
        prompt_str = apply_chat_template(processor, messages)

        t0 = time.perf_counter()
        try:
            result = generate(
                model,
                processor,
                prompt_str,
                image=str(img_path),
                max_tokens=args.max_tokens,
                verbose=False,
                temperature=args.temperature,
            )
            response = result.text if hasattr(result, "text") else str(result)
        except Exception as e:  # noqa: BLE001
            response = f"<ERROR: {type(e).__name__}: {e}>"
        wall = time.perf_counter() - t0
        total_wall += wall

        # Substring match (case-insensitive) using whatever signals we
        # have for this species. Description-driven check (if loaded)
        # adds common_name + Latin species name aliases.
        resp_low = response.lower()
        candidates = {
            slug,
            slug.replace("-", " "),
            slug.replace("-", "_"),
            label.lower(),
        }
        desc = descriptions.get(slug) or descriptions.get(slug.replace("-", "_"))
        if desc:
            candidates.add(desc.get("common_name", "").lower())
            candidates.add(desc.get("species", "").lower())
        match = any(c and c in resp_low for c in candidates)
        if match:
            matches += 1

        print(
            f"[{i:2d}/{len(images)}] {img_path.name:30s}  "
            f"ref={label!r:25s}  match={match}  wall={wall:.1f}s\n"
            f"    → {response[:200]!r}",
        )

        per_sample.append(
            {
                "image": str(img_path),
                "slug": slug,
                "raw_slug": raw_slug,
                "ref_label": label,
                "idx": idx,
                "prompt": user_text,
                "response": response,
                "match": match,
                "wall_s": round(wall, 2),
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_tag = re.sub(r"[^A-Za-z0-9._-]", "_", args.model_tag)
    out_path = args.results_dir / f"{safe_tag}__{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "model_tag": args.model_tag,
                "images_dir": str(args.images_dir),
                "prompt": args.prompt,
                "camera_prefix": args.camera_prefix,
                "user_text": user_text,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "n_images": len(images),
                "n_matches": matches,
                "match_rate": matches / len(images) if images else 0.0,
                "total_wall_s": round(total_wall, 1),
                "per_sample": per_sample,
            },
            indent=2,
        )
    )

    # Per-species breakdown.
    by_slug: dict[str, list[dict]] = {}
    for s in per_sample:
        by_slug.setdefault(s["slug"], []).append(s)
    print()
    print(f"=== Summary ({args.model_tag}) ===")
    print(f"Overall: {matches}/{len(images)} = {100*matches/len(images):.1f}%")
    print(f"Per-species:")
    for slug in sorted(by_slug):
        rows = by_slug[slug]
        m = sum(1 for r in rows if r["match"])
        print(f"  {slug:25s} {m}/{len(rows)}")
    print(f"Wall: {total_wall:.0f}s total ({total_wall/len(images):.1f}s/image)")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
