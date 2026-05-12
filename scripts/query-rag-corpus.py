#!/usr/bin/env python3
"""CLI smoke-test for the RAG PoC retrieval.

Loads one subject's chunks + embeddings, embeds your query with the
same model, and prints the top-k matches by cosine similarity.

This is the same retrieval the iOS RAGService will do — useful for
validating retrieval quality offline before wiring up the device side.

Usage
-----
    python3 scripts/query-rag-corpus.py "why is this rock orange?" --subject geology
    python3 scripts/query-rag-corpus.py "tree with three leaf shapes" --subject plants --k 3
    python3 scripts/query-rag-corpus.py "why does the canyon echo?" --subject physics
    python3 scripts/query-rag-corpus.py "what does Thoreau say about time?" --subject english
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


def load_subject(rag_dir: Path, subject: str) -> tuple[list[dict], np.ndarray]:
    """Load chunks + embeddings for one subject. Returns (chunks, embeddings_f32)."""
    chunks_path = rag_dir / f"{subject}.jsonl"
    embeddings_path = rag_dir / f"{subject}.embeddings.f16"

    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} missing")
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"{embeddings_path} missing — run scripts/embed-rag-corpus.py first"
        )

    chunks: list[dict] = []
    with chunks_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    raw = np.fromfile(embeddings_path, dtype=np.float16)
    if raw.size % len(chunks) != 0:
        raise ValueError(
            f"embedding blob size {raw.size} not divisible by chunk count {len(chunks)}"
        )
    dim = raw.size // len(chunks)
    embeddings = raw.reshape(len(chunks), dim).astype(np.float32)
    return chunks, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument(
        "--subject",
        required=True,
        choices=["geology", "plants", "physics", "english"],
        help="Which subject corpus to search",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k results (default: 3)")
    parser.add_argument("--rag-dir", default="rag-poc")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Must match what was used to embed the corpus",
    )
    parser.add_argument(
        "--show-summary-only",
        action="store_true",
        help="Print only the chunk summary, not full text (matches iOS summary-first retrieval)",
    )
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    try:
        chunks, embeddings = load_subject(rag_dir, args.subject)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(args.model)
    qvec = model.encode([args.query], normalize_embeddings=True)[0].astype(np.float32)

    # Cosine similarity = dot product since both sides are L2-normalized.
    scores = embeddings @ qvec
    top_idx = np.argsort(-scores)[: args.k]

    print(f"\nQuery:    {args.query}")
    print(f"Subject:  {args.subject}")
    print(f"Corpus:   {len(chunks)} chunks\n")
    print("─" * 72)
    for rank, idx in enumerate(top_idx, start=1):
        chunk = chunks[idx]
        body = chunk["summary"] if args.show_summary_only else chunk["text"]
        print(f"#{rank}  score={scores[idx]:+.4f}  id={chunk['id']}")
        print(f"     title: {chunk['title']}")
        if chunk.get("region") and chunk["region"] != "general":
            print(f"     region: {chunk['region']}")
        print(f"\n     {body}\n")
        print("─" * 72)


if __name__ == "__main__":
    main()
