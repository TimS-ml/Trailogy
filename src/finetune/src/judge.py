#!/usr/bin/env python3
"""LLM-as-judge evaluation using Qwen VL.

Reads per-sample results from evaluate.py, sends each (image, prediction,
reference) triple to a Qwen VL judge model, and scores on three axes:
**accuracy**, **richness**, **hallucination**.

The species_match metric from evaluate.py is carried through as-is.

Prerequisites (on top of requirements.txt):

    pip install qwen-vl-utils

Usage:

    # Judge from existing eval results JSON (default: Qwen2.5-VL-72B on all GPUs)
    python -m src.judge \
        --results results/run_eval.json \
        --output results/run_judged.json

    # Use a smaller judge for fast iteration
    python -m src.judge \
        --results results/run_eval.json \
        --judge_model Qwen/Qwen2.5-VL-7B-Instruct \
        --output results/run_judged.json

    # Limit to first 50 samples for a quick smoke test
    python -m src.judge \
        --results results/run_eval.json \
        --max_samples 50 \
        --output results/run_judged.json
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_torch = None
_AutoProcessor = None
_AutoModelCls = None
_Image = None


def _import_deps():
    global _torch, _AutoProcessor, _AutoModelCls, _Image

    if _torch is not None:
        return

    import torch
    from transformers import AutoProcessor
    from PIL import Image

    # Try Qwen-specific class first (better defaults), fall back to generic.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        auto_cls = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForImageTextToText
        auto_cls = AutoModelForImageTextToText

    _torch = torch
    _AutoProcessor = AutoProcessor
    _AutoModelCls = auto_cls
    _Image = Image


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert botanist and a strict evaluator. You will be given:
- An image of a plant
- A reference answer (ground truth identification and description)
- A model's predicted answer

Score the prediction on three axes. Use the image and reference to judge.

Return ONLY a JSON object with exactly these fields:
{
  "accuracy": <int 1-5>,
  "accuracy_reason": "<one sentence>",
  "richness": <int 1-5>,
  "richness_reason": "<one sentence>",
  "hallucination": <int 1-5>,
  "hallucination_reason": "<one sentence>"
}

Scoring rubric:

**accuracy** (is the identification correct?):
  5 = Species name exactly matches reference (English or Latin)
  4 = Correct genus or very close common name (e.g. "oak" vs "white oak")
  3 = Correct family but wrong species
  2 = Same broad category (e.g. "tree" vs specific tree species)
  1 = Completely wrong identification

**richness** (how informative is the response?):
  5 = Name + multiple relevant facts (habitat, appearance, uses, ecology)
  4 = Name + one useful fact or description
  3 = Name only, no additional information
  2 = Vague response with minimal useful content
  1 = Empty, refusal, or completely uninformative

**hallucination** (are claims grounded in reality?):
  5 = All claims are factually correct and consistent with the image
  4 = Minor inaccuracy that doesn't affect the core identification
  3 = One clearly fabricated or wrong factual claim
  2 = Multiple fabricated claims or confidently wrong identification
  1 = Mostly fabricated content or contradicts what the image shows

Return ONLY the JSON object, no markdown fences, no extra text."""


def build_judge_prompt(reference: str, prediction: str) -> str:
    """Build the user-turn text for the judge."""
    return (
        f"**Reference answer (ground truth):**\n{reference}\n\n"
        f"**Model's prediction:**\n{prediction}\n\n"
        "Score the prediction according to the rubric."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_judge_model(model_id: str, dtype: str = "bfloat16"):
    """Load a Qwen VL judge model across all available GPUs."""
    _import_deps()

    log.info("Loading judge model: %s (dtype=%s, device_map=auto)", model_id, dtype)
    torch_dtype = getattr(_torch, dtype)

    model = _AutoModelCls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = _AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Log device placement
    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        log.info("Model spread across devices: %s", sorted(devices))

    return model, processor


# ---------------------------------------------------------------------------
# Judge inference
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(text: str) -> dict | None:
    """Extract and validate the JSON scores from judge output."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        obj = json.loads(text)
        if _validate_scores(obj):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object in text
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group())
            if _validate_scores(obj):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _validate_scores(obj: dict) -> bool:
    """Check that all three scores are ints in [1, 5]."""
    for key in ("accuracy", "richness", "hallucination"):
        val = obj.get(key)
        if not isinstance(val, int) or val < 1 or val > 5:
            return False
    return True


def judge_one_sample(
    model,
    processor,
    image_path: str | None,
    reference: str,
    prediction: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    retries: int = 2,
) -> dict:
    """Score a single prediction. Returns parsed scores dict or error dict."""
    _import_deps()

    # Build messages
    user_content = []
    image = None
    if image_path:
        try:
            image = _Image.open(image_path).convert("RGB")
            user_content.append({"type": "image", "image": image})
        except Exception as exc:
            log.warning("Could not open image %s: %s", image_path, exc)

    user_content.append({"type": "text", "text": build_judge_prompt(reference, prediction)})

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(retries + 1):
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        proc_kwargs = {"text": [prompt_text], "return_tensors": "pt", "padding": True}
        if image is not None:
            proc_kwargs["images"] = [image]

        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with _torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            input_len = inputs["input_ids"].shape[-1]

        raw_response = processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        scores = parse_judge_response(raw_response)
        if scores is not None:
            return scores

        if attempt < retries:
            log.warning(
                "Judge returned unparseable response (attempt %d/%d): %s",
                attempt + 1, retries + 1, raw_response[:200],
            )

    # All retries failed — return error with the raw text
    log.warning("Judge failed to return valid JSON after %d attempts.", retries + 1)
    return {
        "accuracy": None,
        "richness": None,
        "hallucination": None,
        "judge_error": raw_response[:500],
    }


# ---------------------------------------------------------------------------
# Batch judging
# ---------------------------------------------------------------------------


def judge_batch(
    model,
    processor,
    samples: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[dict]:
    """Judge all samples, returning enriched result dicts."""
    judged = []

    for idx, sample in enumerate(samples):
        image_path = sample.get("image")
        reference = sample.get("reference", "")
        prediction = sample.get("prediction", "")

        t0 = time.monotonic()
        scores = judge_one_sample(
            model, processor,
            image_path, reference, prediction,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.monotonic() - t0

        # Merge scores into the existing sample dict
        result = dict(sample)
        result["judge_accuracy"] = scores.get("accuracy")
        result["judge_richness"] = scores.get("richness")
        result["judge_hallucination"] = scores.get("hallucination")
        result["judge_accuracy_reason"] = scores.get("accuracy_reason", "")
        result["judge_richness_reason"] = scores.get("richness_reason", "")
        result["judge_hallucination_reason"] = scores.get("hallucination_reason", "")
        result["judge_elapsed_s"] = round(elapsed, 2)
        if "judge_error" in scores:
            result["judge_error"] = scores["judge_error"]

        judged.append(result)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            # Running averages
            valid = [r for r in judged if r["judge_accuracy"] is not None]
            if valid:
                avg_acc = statistics.mean(r["judge_accuracy"] for r in valid)
                avg_rich = statistics.mean(r["judge_richness"] for r in valid)
                avg_hall = statistics.mean(r["judge_hallucination"] for r in valid)
                log.info(
                    "[%d/%d] acc=%.2f rich=%.2f hall=%.2f (%.1fs)",
                    idx + 1, len(samples), avg_acc, avg_rich, avg_hall, elapsed,
                )
            else:
                log.info("[%d/%d] (no valid scores yet, %.1fs)", idx + 1, len(samples), elapsed)

    return judged


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_judge_summary(results: list[dict]) -> dict:
    """Print aggregate judge scores and return summary dict."""
    valid = [r for r in results if r.get("judge_accuracy") is not None]
    failed = len(results) - len(valid)

    if not valid:
        log.warning("No valid judge scores to summarise.")
        return {"num_samples": len(results), "num_failed": failed}

    def _stats(key):
        vals = [r[key] for r in valid]
        return {
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "min": min(vals),
            "max": max(vals),
            "distribution": {i: vals.count(i) for i in range(1, 6)},
        }

    acc_stats = _stats("judge_accuracy")
    rich_stats = _stats("judge_richness")
    hall_stats = _stats("judge_hallucination")

    # Species match from evaluate.py (carried through)
    species_matches = [r for r in results if r.get("species_match") is not None]
    species_match_rate = None
    if species_matches:
        species_match_rate = round(
            sum(1 for r in species_matches if r["species_match"]) / len(species_matches), 4
        )

    judge_times = [r["judge_elapsed_s"] for r in valid]

    agg = {
        "num_samples": len(results),
        "num_judged": len(valid),
        "num_failed": failed,
        "species_match_rate": species_match_rate,
        "accuracy": acc_stats,
        "richness": rich_stats,
        "hallucination": hall_stats,
        "judge_time_per_sample_s": round(statistics.mean(judge_times), 2),
    }

    header = "=" * 60
    print(f"\n{header}")
    print("  LLM JUDGE SUMMARY")
    print(header)
    print(f"  Samples judged        : {len(valid)} / {len(results)}")
    if failed:
        print(f"  Parse failures        : {failed}")
    if species_match_rate is not None:
        print(f"  Species match (regex) : {species_match_rate:.2%}")
    print()
    print(f"  Accuracy   mean={acc_stats['mean']:.2f}  median={acc_stats['median']:.1f}  "
          f"dist={acc_stats['distribution']}")
    print(f"  Richness   mean={rich_stats['mean']:.2f}  median={rich_stats['median']:.1f}  "
          f"dist={rich_stats['distribution']}")
    print(f"  Hallucin.  mean={hall_stats['mean']:.2f}  median={hall_stats['median']:.1f}  "
          f"dist={hall_stats['distribution']}")
    print()
    print(f"  Avg judge time        : {agg['judge_time_per_sample_s']:.2f}s / sample")
    print(header)
    print()

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-judge evaluation. Reads evaluate.py results JSON and "
            "scores each prediction on accuracy, richness, and hallucination "
            "using a Qwen VL judge model."
        ),
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help=(
            "Path to evaluate.py output JSON (results/<run>_eval.json). "
            "Must contain a 'results' array with image, reference, prediction."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write judged results JSON. Default: <results>_judged.json.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"HuggingFace model ID for the judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16"),
        help="Torch dtype for the judge model (default: bfloat16).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens for judge response (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of samples to judge (default: all).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load eval results
    results_path = Path(args.results)
    if not results_path.exists():
        log.error("Results file not found: %s", results_path)
        sys.exit(1)

    with open(results_path) as f:
        eval_data = json.load(f)

    samples = eval_data.get("results", [])
    if not samples:
        log.error("No 'results' array found in %s", results_path)
        sys.exit(1)

    log.info("Loaded %d samples from %s", len(samples), results_path)

    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        log.info("Limited to %d samples.", args.max_samples)

    # Load judge model
    model, processor = load_judge_model(args.judge_model, args.dtype)

    # Run judging
    judged = judge_batch(model, processor, samples, max_new_tokens=args.max_new_tokens)
    agg = print_judge_summary(judged)

    # Save
    output_path = args.output
    if output_path is None:
        stem = results_path.stem.replace("_eval", "")
        output_path = str(results_path.parent / f"{stem}_judged.json")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "judge_config": {
            "judge_model": args.judge_model,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "source_results": str(results_path),
        },
        "eval_config": eval_data.get("config", {}),
        "summary": agg,
        "results": judged,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("Judged results saved to %s", out_path)


if __name__ == "__main__":
    main()
