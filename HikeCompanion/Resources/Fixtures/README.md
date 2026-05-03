# Fixtures/

Pre-tokenized JSON inputs for the Swift validator. **This directory is
intentionally empty in git** — contents are produced by running:

```
bash scripts/prepare-fixtures.sh
```

That script invokes the upstream `prepare_swift_bench_inputs.py` which writes:

- `3s.json` — "The quick brown fox jumps over the dog."
- `7s.json` — short paragraph
- `15s.json` — medium paragraph
- `30s.json` — long paragraph
- `hnsf_weights.json` — learned linear weights for the hn-nsf source module

Each input JSON has the shape (matches `BenchTypes.swift::BenchInput`):

```json
{
  "key": "3s",
  "text": "...",
  "voice": "af_heart",
  "speed": 1.0,
  "input_ids": [...],
  "attention_mask": [...],
  "ref_s": [...],
  "num_tokens": N,
  "canonical_duration_s": 3.0
}
```

The text-to-tokens step is done in Python because the upstream Kokoro
phonemizer (misaki) lives in Python. There is no Swift G2P in this repo.
