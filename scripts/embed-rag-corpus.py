#!/usr/bin/env python3
"""Embed the RAG PoC corpus into float16 vectors using sentence-transformers.

Reads each `*.jsonl` in `rag-poc/`, runs every chunk's `text` through
`all-MiniLM-L6-v2`, and writes a sibling `*.embeddings.f16` binary blob
aligned by line index. Updates `manifest.json` with embedding metadata.

Output format per subject:
    rag-poc/<subject>.jsonl              (existing — chunks)
    rag-poc/<subject>.embeddings.f16     (new — np.float16 array, [N, 384])

The blob is raw little-endian float16 with no header — the iOS side
mmaps it and reshapes by `(chunkCount, embeddingDim)` from manifest.
This is the simplest possible interchange format; if we need richer
metadata later we can switch to safetensors.

Usage
-----
    pip install -r scripts/rag-requirements.txt
    python3 scripts/embed-rag-corpus.py
    python3 scripts/embed-rag-corpus.py --rag-dir rag-poc --model sentence-transformers/all-MiniLM-L6-v2

Notes
-----
- Embeddings are L2-normalized at write time so cosine similarity at
  query time is just a dot product.
- Float16 instead of float32 halves disk size with negligible quality
  loss for retrieval (we're ranking, not training).
- 90 chunks total; full embedding runs in well under a minute on Apple
  Silicon, ~10 seconds on a recent CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(f"Missing dependency: {exc}", file=sys.stderr)
    print("Install with:  pip install -r scripts/rag-requirements.txt",
          file=sys.stderr)
    sys.exit(1)


def load_chunks(jsonl_path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts, preserving line order."""
    chunks: list[dict] = []
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{jsonl_path.name}:{lineno} — invalid JSON: {e}"
                ) from e
    return chunks


def embed_subject(
    jsonl_path: Path,
    model: SentenceTransformer,
    dim: int,
) -> int:
    """Embed one subject's chunks and write the f16 blob alongside the JSONL.

    Returns the chunk count for manifest update.
    """
    chunks = load_chunks(jsonl_path)
    if not chunks:
        print(f"  {jsonl_path.name}: empty, skipping")
        return 0

    texts = [c["text"] for c in chunks]
    print(f"  {jsonl_path.name}: embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine sim == dot product downstream
    )
    if embeddings.shape != (len(chunks), dim):
        raise RuntimeError(
            f"shape mismatch on {jsonl_path.name}: "
            f"got {embeddings.shape}, expected ({len(chunks)}, {dim})"
        )

    embeddings_f16 = embeddings.astype(np.float16)

    out_path = jsonl_path.with_suffix(".embeddings.f16")
    embeddings_f16.tofile(out_path)
    print(
        f"    -> {out_path.name}  "
        f"({out_path.stat().st_size:,} bytes, dtype={embeddings_f16.dtype})"
    )
    return len(chunks)


def update_manifest(
    rag_dir: Path,
    model_name: str,
    dim: int,
    chunk_counts: dict[str, int],
) -> None:
    """Update manifest.json with embedding metadata for each subject."""
    manifest_path = rag_dir / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    for subj in manifest["subjects"]:
        sid = subj["id"]
        subj["embeddingDim"] = dim
        subj["embeddingModel"] = model_name
        subj["embeddingsFile"] = subj["file"].replace(".jsonl", ".embeddings.f16")
        subj["indexFormat"] = "flat_f16"  # brute-force search; no ANN index for PoC
        if sid in chunk_counts:
            subj["chunkCount"] = chunk_counts[sid]

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nUpdated {manifest_path.relative_to(manifest_path.parent.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rag-dir",
        default="rag-poc",
        help="Directory containing subject *.jsonl files (default: rag-poc)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformer model id "
             "(default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    if not rag_dir.is_dir():
        print(f"Error: {rag_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dimension: {dim}\n")

    jsonl_files = sorted(rag_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No *.jsonl files found in {rag_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(jsonl_files)} subject file(s) in {rag_dir}/:")
    chunk_counts: dict[str, int] = {}
    total = 0
    for jsonl_path in jsonl_files:
        n = embed_subject(jsonl_path, model, dim)
        # Subject id is the filename stem (e.g. "geology.jsonl" -> "geology").
        chunk_counts[jsonl_path.stem] = n
        total += n

    update_manifest(rag_dir, args.model, dim, chunk_counts)
    print(
        f"\nDone. Embedded {total} chunks across {len(jsonl_files)} subjects "
        f"({dim}-dim float16)."
    )


if __name__ == "__main__":
    main()
