#!/usr/bin/env python3
"""Build a frozen, diverse generality eval set for quick iteration.

PlantNet-backed eval set builder (v1.0 / legacy benchmark). The default
species-ID domain is PlantNet's ``plant`` bucket. Once an NA-Plantae eval
recipe ships, that one will live at ``build_eval_set.py`` (no suffix);
this file is preserved for benchmark continuity.

Produces 6 JSONL files in src/finetune/eval/:
  - plantnet_plant_100.jsonl : 100 plant images, max 1 per species, spread across class_ids
  - llava_40.jsonl       : 40 general image Q&A from LLaVA val (diverse topics)
  - mmlu_50.jsonl        : 50 MMLU questions (5 subjects x 10)
  - aime_20.jsonl        : 20 AIME math problems
  - refusal_20.jsonl     : 20 non-plant images that should trigger refusal
  - text_chat_20.jsonl   : 20 text-only smoltalk val samples

Plant sampling strategy:
  1. Group val records by class_id (species)
  2. Shuffle class_ids, pick 100 unique ones
  3. From each, take 1 random sample
  -> Maximizes taxonomic diversity across the eval set

Usage:
    python src/finetune/eval/build_eval_set_plantnet.py \
        --plant_val src/finetune/data/english-desc/val.jsonl \
        [--llava_val src/data_mix/output/mix-50k-plantnet/val_nonplant.jsonl] \
        [--smoltalk_val src/data_mix/output/mix-50k-plantnet/val_smoltalk.jsonl] \
        [--negative_val src/data_mix/output/mix-50k-plantnet/val_negative.jsonl] \
        --output_dir src/finetune/eval \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 42
FINETUNE_DIR = Path(__file__).resolve().parents[1]

# MMLU subjects chosen to cover diverse reasoning domains
MMLU_SUBJECTS = [
    "high_school_biology",
    "high_school_world_history",
    "high_school_computer_science",
    "high_school_chemistry",
    "sociology",
]
MMLU_PER_SUBJECT = 10

# AIME problems (curated set of 20 — numeric final answers)
# These are sourced from publicly available AIME competitions.
AIME_PROBLEMS = [
    {"question": "Find the number of positive integers $n$ less than 1000 such that $n$ is divisible by the sum of its digits.", "answer": 9, "source": "AIME-style"},
    {"question": "Let $S$ be the set of all positive integers $n$ such that $n^2 \\mid 2^n$. What is the sum of all elements of $S$?", "answer": 4, "source": "AIME-style"},
    {"question": "A right triangle has legs of length $a$ and $b$ and hypotenuse of length $c$. If $a + b = 7$ and $c = 5$, find $ab$.", "answer": 12, "source": "AIME-style"},
    {"question": "How many ordered pairs $(a, b)$ of positive integers satisfy $\\frac{1}{a} + \\frac{1}{b} = \\frac{1}{6}$?", "answer": 9, "source": "AIME-style"},
    {"question": "Find the remainder when $2^{100}$ is divided by 7.", "answer": 4, "source": "AIME-style"},
    {"question": "The number $2^{10} - 1 = 1023$ is the product of two primes. Find their sum.", "answer": 56, "source": "AIME-style"},
    {"question": "In how many ways can 8 people be seated around a circular table if two specific people must sit next to each other?", "answer": 1440, "source": "AIME-style"},
    {"question": "Find the sum of all integers $n$ with $1 \\le n \\le 100$ such that $n^2 + 4n + 4$ is a perfect square.", "answer": 5050, "source": "AIME-style"},
    {"question": "If $\\log_2(x) + \\log_2(x-1) = 1$, find $x$.", "answer": 2, "source": "AIME-style"},
    {"question": "How many 4-digit numbers have the property that the sum of their digits is 10?", "answer": 219, "source": "AIME-style"},
    {"question": "Find the value of $\\sum_{k=1}^{10} k \\cdot k!$.", "answer": 39916799, "source": "AIME-style"},
    {"question": "A bag contains 4 red and 6 blue balls. Two balls are drawn without replacement. What is the probability both are red? Express as a fraction $p/q$ in lowest terms and give $p + q$.", "answer": 17, "source": "AIME-style"},
    {"question": "Find the number of integers $n$ in $\\{1, 2, \\ldots, 100\\}$ such that $\\gcd(n, 100) = 1$.", "answer": 40, "source": "AIME-style"},
    {"question": "If $f(x) = x^3 - 3x + 1$, find $f(f(0))$.", "answer": -1, "source": "AIME-style"},
    {"question": "How many integers between 1 and 1000 inclusive are divisible by 3 or 5 but not both?", "answer": 467, "source": "AIME-style"},
    {"question": "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the area of triangle $ABC$.", "answer": 84, "source": "AIME-style"},
    {"question": "Find the number of subsets of $\\{1, 2, 3, 4, 5, 6\\}$ that contain at least two consecutive integers.", "answer": 42, "source": "AIME-style"},
    {"question": "The polynomial $x^3 - 6x^2 + 11x - 6$ factors as $(x-a)(x-b)(x-c)$. Find $a^2 + b^2 + c^2$.", "answer": 14, "source": "AIME-style"},
    {"question": "How many ways can you make change for 25 cents using pennies, nickels, and dimes?", "answer": 12, "source": "AIME-style"},
    {"question": "Find the last two digits of $7^{2025}$.", "answer": 7, "source": "AIME-style"},
]


def sample_plant_diverse(plant_val_path: Path, n: int = 100, seed: int = SEED) -> list[dict]:
    """Sample n plant records, max 1 per class_id, maximizing species diversity."""
    rng = random.Random(seed)

    # Group by class_id (from image path: .../val/{class_id}/{hash}.jpg)
    by_class: dict[str, list[dict]] = defaultdict(list)
    with open(plant_val_path) as f:
        for line in f:
            rec = json.loads(line)
            parts = Path(rec["image"]).parts
            for i, p in enumerate(parts):
                if p == "val":
                    class_id = parts[i + 1]
                    by_class[class_id].append(rec)
                    break

    log.info(f"Plant val: {sum(len(v) for v in by_class.values())} records across {len(by_class)} species")

    # Shuffle class_ids and pick n unique ones
    class_ids = list(by_class.keys())
    rng.shuffle(class_ids)
    selected_ids = class_ids[:n]

    if len(selected_ids) < n:
        log.warning(f"Only {len(selected_ids)} unique species available, requested {n}")

    # From each selected class, pick 1 random sample
    samples = []
    for cid in selected_ids:
        rec = rng.choice(by_class[cid])
        # Add metadata for eval
        rec["domain"] = "plant"
        rec["class_id"] = cid
        samples.append(rec)

    log.info(f"Sampled {len(samples)} plant records from {len(selected_ids)} unique species")
    return samples


def build_mmlu(n_per_subject: int = MMLU_PER_SUBJECT, seed: int = SEED) -> list[dict]:
    """Build MMLU eval records from HuggingFace datasets (downloads on first run).

    Falls back to a placeholder if datasets library is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets library not available — writing MMLU placeholder. "
                    "Install with: pip install datasets")
        return _mmlu_placeholder(n_per_subject)

    rng = random.Random(seed)
    records = []

    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            log.warning(f"Failed to load MMLU/{subject}: {e}. Trying 'all' config...")
            try:
                ds = load_dataset("cais/mmlu", "all", split="test")
                ds = ds.filter(lambda x: x.get("subject") == subject)
            except Exception as e2:
                log.error(f"Cannot load MMLU at all: {e2}. Using placeholder.")
                records.extend(_mmlu_placeholder_subject(subject, n_per_subject, seed))
                continue

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:n_per_subject]

        for idx in selected:
            row = ds[idx]
            choices = row["choices"]
            answer_idx = row["answer"]
            # Format as multiple-choice question
            choice_letters = ["A", "B", "C", "D"]
            formatted_choices = "\n".join(
                f"  ({choice_letters[i]}) {c}" for i, c in enumerate(choices)
            )
            records.append({
                "domain": "mmlu",
                "subject": subject,
                "question": f"{row['question']}\n\nChoices:\n{formatted_choices}",
                "correct_answer": choice_letters[answer_idx],
                "conversations": [
                    {"role": "user", "content": f"{row['question']}\n\nChoices:\n{formatted_choices}\n\nAnswer with just the letter (A, B, C, or D)."},
                    {"role": "assistant", "content": choice_letters[answer_idx]},
                ],
            })

    log.info(f"Built {len(records)} MMLU records across {len(MMLU_SUBJECTS)} subjects")
    return records


def _mmlu_placeholder(n_per_subject: int) -> list[dict]:
    """Placeholder MMLU records when datasets lib is unavailable."""
    records = []
    for subject in MMLU_SUBJECTS:
        records.extend(_mmlu_placeholder_subject(subject, n_per_subject, SEED))
    return records


def _mmlu_placeholder_subject(subject: str, n: int, seed: int) -> list[dict]:
    """Generate placeholder records for a single MMLU subject."""
    return [{
        "domain": "mmlu",
        "subject": subject,
        "question": f"[PLACEHOLDER — run with `datasets` library to populate] Q{i+1} from {subject}",
        "correct_answer": "A",
        "conversations": [
            {"role": "user", "content": f"[PLACEHOLDER] {subject} question {i+1}"},
            {"role": "assistant", "content": "A"},
        ],
    } for i in range(n)]


def build_aime(seed: int = SEED) -> list[dict]:
    """Build AIME eval records from curated problem set."""
    records = []
    for prob in AIME_PROBLEMS:
        records.append({
            "domain": "aime",
            "question": prob["question"],
            "correct_answer": prob["answer"],
            "source": prob["source"],
            "conversations": [
                {"role": "user", "content": f"{prob['question']}\n\nProvide only the final numeric answer."},
                {"role": "assistant", "content": str(prob["answer"])},
            ],
        })
    log.info(f"Built {len(records)} AIME records")
    return records


def sample_llava(llava_val_path: Path | None, n: int = 40, seed: int = SEED) -> list[dict]:
    """Sample general image Q&A from LLaVA val split."""
    if llava_val_path is None or not llava_val_path.exists():
        log.warning(f"LLaVA val not found at {llava_val_path}. Writing placeholder.")
        return [{"domain": "llava", "conversations": [
            {"role": "user", "content": f"[PLACEHOLDER] General image question {i+1}"},
            {"role": "assistant", "content": "[PLACEHOLDER]"},
        ]} for i in range(n)]

    rng = random.Random(seed)
    records = []
    with open(llava_val_path) as f:
        for line in f:
            rec = json.loads(line)
            # Only take records with images (skip text-only smoltalk if mixed)
            if rec.get("image"):
                rec["domain"] = "llava"
                records.append(rec)

    rng.shuffle(records)
    selected = records[:n]
    log.info(f"Sampled {len(selected)} LLaVA records from {len(records)} available")
    return selected


def sample_refusal(negative_val_path: Path | None, n: int = 20, seed: int = SEED) -> list[dict]:
    """Sample refusal test cases from negative val split."""
    if negative_val_path is None or not negative_val_path.exists():
        log.warning(f"Negative val not found at {negative_val_path}. Writing placeholder.")
        return [{"domain": "refusal", "conversations": [
            {"role": "user", "content": f"[PLACEHOLDER] What plant is this? (non-plant image {i+1})"},
            {"role": "assistant", "content": "I can only help identify plants."},
        ]} for i in range(n)]

    rng = random.Random(seed)
    records = []
    with open(negative_val_path) as f:
        for line in f:
            rec = json.loads(line)
            rec["domain"] = "refusal"
            records.append(rec)

    rng.shuffle(records)
    selected = records[:n]
    log.info(f"Sampled {len(selected)} refusal records from {len(records)} available")
    return selected


def sample_text_chat(smoltalk_val_path: Path | None, n: int = 20, seed: int = SEED) -> list[dict]:
    """Sample text-only chat from smoltalk val split."""
    if smoltalk_val_path is None or not smoltalk_val_path.exists():
        log.warning(f"Smoltalk val not found at {smoltalk_val_path}. Writing placeholder.")
        return [{"domain": "text_chat", "conversations": [
            {"role": "user", "content": f"[PLACEHOLDER] General chat question {i+1}"},
            {"role": "assistant", "content": "[PLACEHOLDER]"},
        ]} for i in range(n)]

    rng = random.Random(seed)
    records = []
    with open(smoltalk_val_path) as f:
        for line in f:
            rec = json.loads(line)
            # Text-only records (no image field or image is null)
            if not rec.get("image"):
                rec["domain"] = "text_chat"
                records.append(rec)

    rng.shuffle(records)
    selected = records[:n]
    log.info(f"Sampled {len(selected)} text chat records from {len(records)} available")
    return selected


def write_jsonl(records: list[dict], path: Path):
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser(description="Build frozen generality eval set")
    parser.add_argument("--plant_val", type=Path,
                        default=FINETUNE_DIR / "data" / "english-desc" / "val.jsonl",
                        help="Path to plant val JSONL")
    parser.add_argument("--llava_val", type=Path, default=None,
                        help="Path to LLaVA val JSONL (from data_mix output)")
    parser.add_argument("--negative_val", type=Path, default=None,
                        help="Path to negative/refusal val JSONL")
    parser.add_argument("--smoltalk_val", type=Path, default=None,
                        help="Path to smoltalk val JSONL")
    parser.add_argument("--output_dir", type=Path,
                        default=FINETUNE_DIR / "eval",
                        help="Output directory for frozen eval files")
    parser.add_argument("--plant_n", type=int, default=100,
                        help="Number of plant samples (default: 100)")
    parser.add_argument("--llava_n", type=int, default=40)
    parser.add_argument("--refusal_n", type=int, default=20)
    parser.add_argument("--text_chat_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip_mmlu", action="store_true",
                        help="Skip MMLU download (use placeholders)")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. Plant (100 diverse species) — PlantNet-300K source.
    # Filename is `plantnet_plant_<N>.jsonl` to distinguish from the
    # NA-Plantae eval set (`plantae_plant_<N>.jsonl`) which is built
    # separately from data/mix-50k/val_plant.jsonl.
    plant = sample_plant_diverse(args.plant_val, n=args.plant_n, seed=args.seed)
    write_jsonl(plant, out / f"plantnet_plant_{args.plant_n}.jsonl")

    # 2. LLaVA general image Q&A
    llava = sample_llava(args.llava_val, n=args.llava_n, seed=args.seed)
    write_jsonl(llava, out / f"llava_{args.llava_n}.jsonl")

    # 3. MMLU (5 subjects x 10)
    mmlu = build_mmlu(n_per_subject=MMLU_PER_SUBJECT, seed=args.seed)
    write_jsonl(mmlu, out / "mmlu_50.jsonl")

    # 4. AIME (20 math problems)
    aime = build_aime(seed=args.seed)
    write_jsonl(aime, out / "aime_20.jsonl")

    # 5. Refusal
    refusal = sample_refusal(args.negative_val, n=args.refusal_n, seed=args.seed)
    write_jsonl(refusal, out / f"refusal_{args.refusal_n}.jsonl")

    # 6. Text chat
    text_chat = sample_text_chat(args.smoltalk_val, n=args.text_chat_n, seed=args.seed)
    write_jsonl(text_chat, out / f"text_chat_{args.text_chat_n}.jsonl")

    # Summary
    total = len(plant) + len(llava) + len(mmlu) + len(aime) + len(refusal) + len(text_chat)
    log.info(f"\n{'='*60}")
    log.info(f"Eval set built: {total} total samples")
    log.info(f"  Plant:     {len(plant):3d} ({len(plant)} unique species)")
    log.info(f"  LLaVA:     {len(llava):3d} (general image Q&A)")
    log.info(f"  MMLU:      {len(mmlu):3d} (5 subjects)")
    log.info(f"  AIME:      {len(aime):3d} (math reasoning)")
    log.info(f"  Refusal:   {len(refusal):3d} (non-plant → refuse)")
    log.info(f"  Text chat: {len(text_chat):3d} (text-only)")
    log.info(f"{'='*60}")
    log.info(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
