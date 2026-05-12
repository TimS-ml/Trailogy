# RAG PoC dataset

Hand-authored seed corpus for prototyping the RAG retrieval architecture.
Scope: hackathon proof-of-concept, not production content.

## Files

| File | Subject | Chunks | Notes |
|---|---|---|---|
| `geology.jsonl` | Geology | 25 | Heavy on western-PA / sandstone gorge content; general rock + earth-science framing |
| `plants.jsonl` | Plants | 25 | Eastern hardwood forest species; trail-relevant flora |
| `physics.jsonl` | Physics | 20 | Subset relevant to outdoor phenomena (acoustics, optics, thermodynamics, fluid mechanics) |
| `english.jsonl` | English | 20 | Analytical summaries of nature-writing themes (Thoreau, Muir, Burroughs, Emerson, Whitman) |
| `manifest.json` | — | — | Subject manifest with version, chunk counts |

## Schema

One JSON object per line:

```json
{
  "id": "geo-001",
  "subject": "geology",
  "title": "What is sandstone?",
  "text": "Sandstone is a sedimentary rock made of compressed sand-sized mineral grains...",
  "summary": "Sedimentary rock of compressed sand grains; common in PA gorges.",
  "tags": ["sandstone", "sedimentary", "rock_types"],
  "region": "general",
  "source": "Hand-authored, public-domain knowledge"
}
```

| Field | Purpose |
|---|---|
| `id` | Stable identifier, `{subject_prefix}-{NNN}` |
| `subject` | One of `geology`, `plants`, `physics`, `english` |
| `title` | Short human-readable label, used in citations |
| `text` | The chunk body (target ~100-150 words; embed this) |
| `summary` | One-sentence compression for the "summary-first" retrieval strategy in the ADR |
| `tags` | Lexical hints; useful if we layer BM25 over embeddings later |
| `region` | `general` or `western_pa` — lets us boost local content for our trails |
| `source` | Provenance string |

## Provenance and accuracy disclaimer

**These chunks were hand-authored from general public-domain knowledge.**
They are not pulled from a vetted educational source and have not been
fact-checked by a subject-matter expert. Specific dates (e.g.
"Mississippian limestone ~350 Mya") are plausible-range claims, not
formally cited.

For production, replace each chunk with text extracted from:

- **Geology** — USGS Open-File Reports (public domain), OpenStax Geology (CC-BY)
- **Plants** — USDA PLANTS database (public domain), Wikipedia species pages (CC-BY-SA)
- **Physics** — OpenStax College Physics (CC-BY)
- **English** — Project Gutenberg / Standard Ebooks (public domain Thoreau, Muir, Burroughs, etc.)

The English chunks here are **analytical summaries**, not direct quotes.
For richer English retrieval, swap in actual public-domain passages from
Project Gutenberg with proper attribution.

## Token budget

Each chunk targets **~150 tokens** (~100-150 words). At retrieval time,
the architecture in ADR-001 calls for k=1 retrieval with a token budget
of ~200 tokens — most chunks fit comfortably in that budget without
truncation.

The `summary` field is ~30-50 tokens, supporting the
"summary-first / full-chunk on demand" two-stage retrieval strategy.

## Embedding pipeline (next step)

This corpus has no embeddings yet. To produce them:

1. Pick the embedding model: `sentence-transformers/all-MiniLM-L6-v2`
   (384-dim, ~80 MB, Core ML-convertible)
2. Run a Python script: load each JSONL → `embeddings.f16` blob aligned
   by line index
3. Build a USearch HNSW index over the embeddings
4. Bundle: `chunks.jsonl` + `embeddings.f16` + `index.usearch` + `manifest.json`
5. Drop into `Documents/RAG/<subject>/` on the iOS side

See `docs/ADR-001-rag-architecture.md` (when written) for the full
deployment story.

## Counts

```
geology.jsonl   25 chunks  ~3,400 words  ~5,000 tokens
plants.jsonl    25 chunks  ~3,400 words  ~5,000 tokens
physics.jsonl   20 chunks  ~2,700 words  ~4,000 tokens
english.jsonl   20 chunks  ~2,700 words  ~4,000 tokens
                ─────────  ─────────────  ──────────────
total           90 chunks  ~12,200 words  ~18,000 tokens
```

Tiny. Fits in app bundle if needed for a no-download-required demo.
