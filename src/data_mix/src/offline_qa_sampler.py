"""Sample the offline-AI persona Q&A corpus into the 'offline_qa' bucket.

The corpus lives at ``hikeCompanion/assets/data_offline_qa/offline_qa.json``
and is a tiny (~42 entries) hand-curated list of ``{"question", "answer"}``
pairs that teach the model the "I'm an offline AI" persona — graceful
in-character refusals for prompts like "are you ChatGPT?", "Google this
for me", "what's the weather?", etc.

Design rationale:

- **No oversampling.** With only ~42 records, repeating entries to
  inflate the bucket would teach the model the *exact phrasing* rather
  than the persona. We include each entry exactly once across train+val.
- **Sits OUTSIDE the main 45/30/15/10 ratio.** The mix orchestrator
  appends the offline_qa bucket on top of the budget (so a "50K mix"
  becomes 50,038 train records — 0.08 % drift, negligible).
- **No prompt prefix at training time** (config dispatch keeps
  ``offline_qa`` out of ``data.prompt_prefixes``). The persona should
  be the *default* unprefixed behaviour: when a user asks "are you
  online?" they don't add a task tag, but we still want the offline
  AI persona to fire. By leaving the records unprefixed, the
  unconditional (no-gate) output distribution learns the persona while
  the prefixed gates handle plant-ID + refusal modes.

Schema:

    {
        "image": None,
        "source": "offline_qa",
        "conversations": [
            {"role": "user",      "content": <question>},
            {"role": "assistant", "content": <answer>},
        ]
    }
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

from data_mix.src.schema import validate_record


def load_offline_qa_records(json_path: str | Path) -> List[dict]:
    """Read the JSON file, validate each entry, return v2-schema records.

    Raises ``FileNotFoundError`` if the file doesn't exist (silently
    skipping would let a typo in a config produce empty
    offline_qa contribution that looks fine in logs).
    Raises ``ValueError`` for any malformed entry (missing question /
    answer / empty strings) — better to fail loud at prep time than
    silently train on a partial corpus.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"offline_qa source file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"{p}: expected a JSON list at top level, got {type(raw).__name__}"
        )

    out: List[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{p}[{i}]: expected dict, got {type(entry).__name__}")
        q = entry.get("question")
        a = entry.get("answer")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"{p}[{i}]: 'question' must be a non-empty string")
        if not isinstance(a, str) or not a.strip():
            raise ValueError(f"{p}[{i}]: 'answer' must be a non-empty string")
        rec = {
            "image": None,
            "source": "offline_qa",
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        }
        # Tripwire: schema must hold for every record so the downstream
        # mix orchestrator can trust the source.
        validate_record(rec)
        out.append(rec)
    return out


def sample_offline_qa_records(
    json_path: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Load the corpus + carve a deterministic train/val split.

    val_count = ``floor(N * val_ratio)`` (no min/max clamp at the top
    end — at typical N=42 and val_ratio=0.1 this gives val_count=4).
    For N=1 the function returns ``([the_only_record], [])`` so a
    degenerate corpus still trains; for val_ratio=0.0 the entire
    corpus goes to train (matches ``class_stratified_split`` behaviour).

    Determinism: the split shuffles the loaded record list with the
    supplied seed before slicing, so the same ``(json_path, val_ratio,
    seed)`` triple always produces the same train/val partition.
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    records = load_offline_qa_records(json_path)
    n = len(records)

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    if val_ratio == 0.0 or n < 2:
        return shuffled, []

    n_val = int(n * val_ratio)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
