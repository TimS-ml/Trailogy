#!/usr/bin/env python3
"""Evaluate a finetuned Gemma 4 E2B model on a JSONL test set.

Supports two modes:

  1. **Batch evaluation** (default):
     Loads a JSONL test file, generates responses for each sample, and
     computes ROUGE-L, exact species match rate, and response length stats.

  2. **Interactive mode** (--interactive):
     Loads the model once and accepts user questions from stdin.  Optionally
     attach an image with --image_path.

Usage examples:

    # Batch evaluation
    python src/evaluate.py \
        --base_model google/gemma-4-e2b-it \
        --adapter_path outputs/hike-gemma4-lora \
        --test_file data/test.jsonl \
        --output_file results/eval_results.json

    # Interactive
    python src/evaluate.py \
        --base_model google/gemma-4-e2b-it \
        --adapter_path outputs/hike-gemma4-lora \
        --interactive
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:  # `python -m src.evaluate`
    from .data import build_vision_messages
except ImportError:  # `python src/evaluate.py`
    from data import build_vision_messages  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_MAX_EVAL_SAMPLES = 300

# ---------------------------------------------------------------------------
# Lazy imports — heavy deps are only pulled in when actually needed so
# argparse --help stays fast.
# ---------------------------------------------------------------------------

_torch = None
_AutoModelForCausalLM = None
_AutoModelForImageTextToText = None
_AutoProcessor = None
_BitsAndBytesConfig = None
_PeftModel = None
_Image = None


def _import_deps():
    """Import heavy dependencies once."""
    global _torch, _AutoModelForCausalLM, _AutoModelForImageTextToText, _AutoProcessor
    global _BitsAndBytesConfig, _PeftModel, _Image

    if _torch is not None:
        return

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import PeftModel
    from PIL import Image

    _torch = torch
    _AutoModelForCausalLM = AutoModelForCausalLM
    _AutoModelForImageTextToText = AutoModelForImageTextToText
    _AutoProcessor = AutoProcessor
    _BitsAndBytesConfig = BitsAndBytesConfig
    _PeftModel = PeftModel
    _Image = Image


_FastModel = None


def _import_unsloth():
    """Import unsloth FastModel (optional — used when --use_unsloth is set)."""
    global _FastModel
    if _FastModel is not None:
        return
    from unsloth import FastModel
    _FastModel = FastModel


def _apply_train_chat_template(processor):
    """Normalize eval processors to the exact gemma-4 template used in training."""
    from unsloth.chat_templates import get_chat_template

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return get_chat_template(processor, chat_template="gemma-4")

    templated = get_chat_template(tokenizer, chat_template="gemma-4")
    if templated is not tokenizer:
        processor.tokenizer = templated
    return processor


# ---------------------------------------------------------------------------
# ROUGE-L (simple, dependency-free implementation)
# ---------------------------------------------------------------------------


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of token lists *a* and *b*."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP: only keep two rows.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between *prediction* and *reference*."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Species extraction for exact-match scoring
# ---------------------------------------------------------------------------

# Heuristic patterns for extracting species names from model output.
# Covers:
#   * Trained short format with Latin scientific names
#     ("This is Tsuga canadensis. ..."), and
#   * Trained format with English common names
#     ("This is Eastern Hemlock. ...", "You've spotted Sugar Maple. ...",
#      "Looks like White Oak to me. ..."), and
#   * Verbose base-model format ("the plant is **{species}**").
#
# The capture group is non-greedy and uses a lookahead terminator so it
# stops at sentence boundaries (`.`, `!`, `?`, `,`, newline) and at the
# template-specific tail `" to me"`. The trigger phrase alternation is
# case-insensitive via the `(?i:...)` inline flag, but the capture group
# itself remains case-sensitive so we can distinguish Title Case English
# names from Latin binomials downstream if needed.

# Pattern 1: Explicit identification phrases.
_SPECIES_PHRASE_RE = re.compile(
    # Trigger phrase (case-insensitive). Each variant matches the
    # ANSWER_TEMPLATES used in prepare_plantnet.py and
    # prepare_plantnet_enriched.py. The em-dash variant tolerates ASCII
    # dash, em-dash, en-dash, and any non-alphanumeric run between
    # "Good eye" and "this is".
    r"(?i:(?:"
    r"This is|That's|You're looking at|That looks like|Looks like|"
    r"This appears to be|appears to be|looking at is|identified as|"
    r"plant is|species of|specimen of|type of|You've spotted|"
    r"Good eye[^A-Za-z0-9]*this is"
    r"))\s+"
    # Capture: non-greedy run of word/space chars, no sentence punctuation.
    # `*` is allowed in the consumed run only via the surrounding `\**`
    # markdown wrappers, not inside the capture itself.
    r"\**([^.!?,*\n]+?)\**"
    # Terminator: sentence punctuation, newline, end of string, or the
    # template-specific " to me" tail used by ANSWER_TEMPLATES.
    r"(?=\s*(?:[.!?,\n]|to me\b|$))",
)

# Pattern 2: Markdown bold species name — **Genus species** or **Common Name**.
_BOLD_SPECIES_RE = re.compile(
    r"\*\*([A-Z][a-z]+(?: [A-Za-z]+)*)\*\*",
)

# Pattern 3: Italicised binomial — *Genus species* (with optional author).
_ITALIC_BINOMIAL_RE = re.compile(
    r"\*([A-Z][a-z]+ [a-z]+(?:\s+[A-Z][a-z.]*)?)\*",
)


def extract_species(text: str) -> str:
    """Best-effort extraction of a species name from model output.

    Tries multiple patterns in order of specificity and returns the
    first match, lowercased.  Falls back to the first sentence if
    nothing matches.
    """
    # Try phrase-based extraction first.
    m = _SPECIES_PHRASE_RE.search(text)
    if m:
        return m.group(1).strip(" *").lower()

    # Try italic binomial (often more precise).
    m = _ITALIC_BINOMIAL_RE.search(text)
    if m:
        return m.group(1).strip().lower()

    # Try markdown bold name.
    m = _BOLD_SPECIES_RE.search(text)
    if m:
        return m.group(1).strip().lower()

    # Fallback: first sentence, stripped.
    first_sentence = text.split(".")[0].strip() if "." in text else text.strip()
    return first_sentence.lower()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    base_model: str,
    adapter_path: str | None,
    quantize: bool,
    device_map: str = "auto",
    use_unsloth: bool = False,
):
    """Load base model (optionally quantised) and merge LoRA adapter.

    When *use_unsloth* is True, load via unsloth's FastModel which
    handles 4-bit quantization exclusions for vision/audio towers
    correctly (avoids bitsandbytes shape assertions).
    """
    _import_deps()

    if use_unsloth:
        _import_unsloth()
        log.info(
            "Loading via unsloth FastModel: %s (load_in_4bit=%s)",
            base_model, quantize,
        )
        # Previously this hard-coded load_in_4bit=True regardless of the
        # `quantize` arg, silently overriding --no-quantize. Now respects
        # the flag → defaults to bf16 inference, matching training dtype.
        model, tokenizer = _FastModel.from_pretrained(
            model_name=base_model,
            max_seq_length=2048,
            load_in_4bit=quantize,
        )
        if adapter_path:
            log.info("Loading LoRA adapter from %s", adapter_path)
            from peft import PeftModel as _PM
            model = _PM.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            log.info("Adapter merged.")
        model.eval()
        processor = _AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor = _apply_train_chat_template(processor)
        return model, processor

    log.info("Loading base model: %s (quantize=%s)", base_model, quantize)

    kwargs: dict = {
        "device_map": device_map,
        "torch_dtype": _torch.bfloat16,
        "trust_remote_code": True,
    }

    if quantize:
        kwargs["quantization_config"] = _BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = _AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)

    if adapter_path:
        log.info("Loading LoRA adapter from %s", adapter_path)
        model = _PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        log.info("Adapter merged.")

    model.eval()

    processor = _AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor = _apply_train_chat_template(processor)

    return model, processor


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def generate_response(
    model,
    processor,
    question: str | None = None,
    image_path: str | None = None,
    max_new_tokens: int = 256,
    messages: list[dict] | None = None,
) -> str:
    """Generate a single response from the model."""
    _import_deps()

    image = None
    prompt_messages: list[dict]
    if messages is None:
        if question is None:
            raise ValueError("question is required when messages is not provided")
        prompt_messages = [{"role": "user", "content": []}]
        if image_path:
            image = _Image.open(image_path).convert("RGB")
            prompt_messages[0]["content"].append({"type": "image"})

        # Strip the <image> token if present — the processor adds it from the
        # image content block above.
        clean_question = question.replace("<image>", "").strip()
        prompt_messages[0]["content"].append({"type": "text", "text": clean_question})
    else:
        # The training data stores image blocks as {type: image, image: path}.
        # The chat template only needs {type: image}; the actual PIL image is
        # passed separately to the processor. Preserve every text turn so eval
        # sees the same conversation prefix as train.
        prompt_messages = []
        resolved_image_path = image_path
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                prompt_messages.append({"role": msg.get("role"), "content": content})
                continue
            out_blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    if resolved_image_path is None and block.get("image"):
                        resolved_image_path = block["image"]
                    out_blocks.append({"type": "image"})
                else:
                    out_blocks.append(block)
            prompt_messages.append({"role": msg.get("role"), "content": out_blocks})
        if resolved_image_path:
            image = _Image.open(resolved_image_path).convert("RGB")

    # Get the chat-formatted text, then tokenize with the processor so
    # image pixel values are included.
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    proc_kwargs = {"text": prompt_text, "return_tensors": "pt"}
    if image is not None:
        proc_kwargs["images"] = image

    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with _torch.inference_mode():
        # Greedy decoding for eval — eliminates sampling entropy so
        # the same (adapter, image, prompt) tuple produces the same
        # string across runs. Without do_sample=False the model's
        # generation_config.json defaults take over (Gemma 4 ships
        # do_sample=True), which means species_match drifts ~5 pts
        # run-to-run on plant_100. We had this bug invisibly under
        # the adapter-loading bug; once that's fixed the sampling
        # noise becomes the next determinism leak.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
        )
        input_len = inputs["input_ids"].shape[-1]

    generated_ids = output_ids[0, input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def load_test_data(path: str, require_image: bool = False) -> list[dict]:
    """Load JSONL test file.

    When require_image=True, mirror finetune.py's training/validation loader
    and drop text-only records before eval. Mixed image/text batches are not
    supported by the Gemma 4 VLM processor, and species metrics are meaningless
    for hiking-QA text-only records.
    """
    records = []
    n_dropped_no_image = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", i, exc)
                continue
            if require_image and not rec.get("image"):
                n_dropped_no_image += 1
                continue
            records.append(rec)
    if n_dropped_no_image:
        log.warning(
            "Dropped %d text-only record(s) (no 'image' field) from %s "
            "because require_image=True.",
            n_dropped_no_image, path,
        )
    log.info("Loaded %d test samples from %s", len(records), path)
    return records


def _build_eval_prompt(
    sample: dict,
    prompt_prefixes: Optional[Dict[str, str]] = None,
) -> tuple[list[dict], str, str]:
    """Return (prompt_messages, last_user_text, final_assistant_reference).

    ``prompt_prefixes`` (v4): forwarded to ``build_vision_messages`` for
    camera-state input-gate injection. Must match the prefixes used at
    training time, otherwise the eval prompt has no gate and the model
    falls back to base-like behaviour on the eval set — looks like a
    regression but is actually a prompt mismatch.
    """
    conversations = sample.get("conversations", [])
    target_idx = None
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("role") == "assistant":
            target_idx = i
            break
    if target_idx is None:
        prompt_conversations = conversations
        reference = ""
    else:
        prompt_conversations = conversations[:target_idx]
        reference = conversations[target_idx].get("content", "")

    user_msg = ""
    for turn in prompt_conversations:
        if turn.get("role") == "user":
            user_msg = turn.get("content", "")

    if not prompt_conversations:
        return [], user_msg, reference

    # Preserve the ``image`` field so the camera-state dispatch inside
    # build_vision_messages picks the right (camera_on / camera_off)
    # branch. ``source`` is no longer needed for prefix dispatch but
    # would be harmless if forwarded — we drop it for clarity.
    record = {"conversations": prompt_conversations}
    if sample.get("image"):
        record["image"] = sample["image"]
    messages = build_vision_messages(record, prompt_prefixes=prompt_prefixes)
    return messages["messages"], user_msg, reference


def evaluate_batch(
    model,
    processor,
    test_data: list[dict],
    batch_size: int,
    max_new_tokens: int = 256,
    prompt_prefixes: Optional[Dict[str, str]] = None,
) -> list[dict]:
    """Run generation on each test sample and compute per-sample metrics.

    NOTE: batch_size is accepted for CLI compatibility, but generation is
    currently done sample-by-sample because VLM inputs vary in image size
    and cannot be trivially batched without padding logic.

    ``prompt_prefixes`` (v4): must match the training-time setting or
    the eval prompts won't trigger the model's camera-state gate. Passed
    in by the CLI entry point from ``cfg.data.prompt_prefixes``.
    """
    results: list[dict] = []

    for idx, sample in enumerate(test_data):
        image_path = sample.get("image")
        messages, user_msg, reference = _build_eval_prompt(
            sample, prompt_prefixes=prompt_prefixes
        )

        if not user_msg:
            log.warning("Sample %d has no user message; skipping.", idx)
            continue

        t0 = time.monotonic()
        prediction = generate_response(
            model,
            processor,
            user_msg,
            image_path,
            max_new_tokens=max_new_tokens,
            messages=messages,
        )
        elapsed = time.monotonic() - t0

        # Metrics
        rl = rouge_l(prediction, reference)
        pred_species = extract_species(prediction)
        ref_species = extract_species(reference)
        species_match = pred_species == ref_species

        result = {
            "index": idx,
            "question": user_msg,
            "reference": reference,
            "prediction": prediction,
            "image": image_path,
            "rouge_l": round(rl, 4),
            "species_match": species_match,
            "pred_species": pred_species,
            "ref_species": ref_species,
            "pred_length": len(prediction),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_data):
            log.info(
                "[%d/%d] ROUGE-L=%.3f species_match=%s (%.1fs)",
                idx + 1,
                len(test_data),
                rl,
                species_match,
                elapsed,
            )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> dict:
    """Print a summary table and return aggregate metrics."""
    if not results:
        log.warning("No results to summarise.")
        return {}

    rouge_scores = [r["rouge_l"] for r in results]
    species_matches = [r["species_match"] for r in results]
    lengths = [r["pred_length"] for r in results]
    times = [r["elapsed_s"] for r in results]

    agg = {
        "num_samples": len(results),
        "rouge_l_mean": round(statistics.mean(rouge_scores), 4),
        "rouge_l_median": round(statistics.median(rouge_scores), 4),
        "rouge_l_min": round(min(rouge_scores), 4),
        "rouge_l_max": round(max(rouge_scores), 4),
        "species_match_rate": round(sum(species_matches) / len(species_matches), 4),
        "species_matches": sum(species_matches),
        "species_total": len(species_matches),
        "response_length_mean": round(statistics.mean(lengths), 1),
        "response_length_median": round(statistics.median(lengths), 1),
        "response_length_min": min(lengths),
        "response_length_max": max(lengths),
        "time_per_sample_mean_s": round(statistics.mean(times), 2),
    }

    header = "=" * 60
    print(f"\n{header}")
    print("  EVALUATION SUMMARY")
    print(header)
    print(f"  Samples evaluated     : {agg['num_samples']}")
    print()
    print(f"  ROUGE-L  mean         : {agg['rouge_l_mean']:.4f}")
    print(f"  ROUGE-L  median       : {agg['rouge_l_median']:.4f}")
    print(f"  ROUGE-L  min / max    : {agg['rouge_l_min']:.4f} / {agg['rouge_l_max']:.4f}")
    print()
    print(
        f"  Species match rate    : {agg['species_match_rate']:.2%}"
        f"  ({agg['species_matches']}/{agg['species_total']})"
    )
    print()
    print(f"  Response length mean  : {agg['response_length_mean']:.0f} chars")
    print(f"  Response length median: {agg['response_length_median']:.0f} chars")
    print(f"  Response length range : {agg['response_length_min']}–{agg['response_length_max']} chars")
    print()
    print(f"  Avg time per sample   : {agg['time_per_sample_mean_s']:.2f}s")
    print(header)
    print()

    return agg


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive_mode(
    model,
    processor,
    image_path: str | None,
    max_new_tokens: int = 256,
) -> None:
    """Interactive REPL: accept questions from stdin, generate answers."""
    print()
    print("=" * 60)
    print("  INTERACTIVE MODE")
    if image_path:
        print(f"  Image: {image_path}")
    print("  Type a question and press Enter.  Ctrl-D / 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        t0 = time.monotonic()
        response = generate_response(
            model,
            processor,
            question,
            image_path,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.monotonic() - t0
        print(f"model> {response}")
        print(f"  ({elapsed:.1f}s)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned Gemma 4 model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the same FinetuneConfig YAML used at training time. "
            "When set, every other arg below defaults to a value derived "
            "from that config (base_model, adapter_path, test_file, "
            "run_name, eval.*). Explicit CLI flags still take precedence."
        ),
    )
    # NB: all config-overridable args default to None so the helper
    # `apply_config_defaults` can detect 'user did not specify' robustly.
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=(
            "HuggingFace model ID or local path for the base model. "
            "Default (no --config): google/gemma-4-e2b-it. "
            "With --config: cfg.model.base_model."
        ),
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help=(
            "Path to the LoRA adapter directory (omit to evaluate base model only). "
            "With --config: <cfg.training.output_dir>/final-adapter."
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help=(
            "Path to JSONL test file (required for batch eval). "
            "With --config: cfg.data.val_file."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write detailed JSON results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Batch size hint (currently samples are processed one at a time). "
            "Default (no --config): 1. With --config: cfg.eval.batch_size."
        ),
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Load base model with 4-bit quantization for inference. "
            "Default (no --config): off (bf16, matches training dtype). "
            "With --config: cfg.eval.load_in_4bit (default false). "
            "This is post-training quantization for memory, NOT QLoRA "
            "(no gradients) — the no-QLoRA policy only governs training. "
            "Pass --no-quantize to force off, --quantize to force on."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of new tokens to generate per sample. "
            "Default (no --config): 256. With --config: cfg.eval.max_new_tokens."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive mode (ignore --test_file).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Image to attach in interactive mode.",
    )
    parser.add_argument(
        "--use_unsloth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Load model via unsloth FastModel (handles 4-bit vision tower correctly). "
            "Default (no --config): off. With --config: cfg.eval.use_unsloth. "
            "Pass --no-use_unsloth to force off."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Run name for output file naming. When set, output_file "
            "defaults to results/{run_name}_eval.json. Should match "
            "the training run_name for easy cross-referencing. "
            "With --config: cfg.training.run_name."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            f"Limit the number of evaluation samples (default: {DEFAULT_MAX_EVAL_SAMPLES}). "
            "With --config: cfg.eval.max_eval_samples."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Config-driven default population
# ---------------------------------------------------------------------------


def apply_config_defaults(args: argparse.Namespace, cfg) -> None:
    """Fill None-valued args fields from a FinetuneConfig in-place.

    Only fields the user didn't set on the CLI (= still None) are
    overwritten. Explicit CLI values always win.

    `cfg` is typed loosely (FinetuneConfig) to avoid a circular import
    at module load time; it is duck-typed through attribute access.
    """
    if args.base_model is None:
        args.base_model = cfg.model.base_model
    if args.run_name is None:
        args.run_name = cfg.training.run_name
    if args.adapter_path is None:
        output_dir = Path("outputs") / args.run_name if args.run_name else Path(cfg.training.output_dir)
        args.adapter_path = str(output_dir / "final-adapter")
    if args.test_file is None:
        args.test_file = cfg.data.val_file
    if args.max_eval_samples is None:
        args.max_eval_samples = cfg.eval.max_eval_samples
    if args.max_new_tokens is None:
        args.max_new_tokens = cfg.eval.max_new_tokens
    if args.use_unsloth is None:
        args.use_unsloth = cfg.eval.use_unsloth
    if args.batch_size is None:
        args.batch_size = cfg.eval.batch_size
    if args.quantize is None:
        args.quantize = cfg.eval.load_in_4bit


def apply_fallback_defaults(
    args: argparse.Namespace,
    *,
    preserve_max_eval_samples_none: bool = False,
) -> None:
    """Fill remaining None values with the legacy hardcoded defaults.

    Used when --config is NOT provided, so existing direct-CLI invocations
    keep working identically (use_unsloth off, max_new_tokens=256, etc.).

    When --config is provided, `eval.max_eval_samples: null` intentionally
    means full-set eval, so callers can preserve that None instead of applying
    the routine 300-sample fallback.
    """
    if args.base_model is None:
        args.base_model = "google/gemma-4-e2b-it"
    if args.max_eval_samples is None and not preserve_max_eval_samples_none:
        args.max_eval_samples = DEFAULT_MAX_EVAL_SAMPLES
    if args.max_new_tokens is None:
        args.max_new_tokens = 256
    if args.use_unsloth is None:
        args.use_unsloth = False
    if args.batch_size is None:
        args.batch_size = 1
    if args.quantize is None:
        # Default off — matches training dtype (bf16). User can opt in
        # via --quantize on the CLI or `eval.load_in_4bit: true` in YAML.
        args.quantize = False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Optional config-driven defaults. Explicit CLI values still win.
    has_config = args.config is not None
    if has_config:
        # Lazy import: config has no torch dep but keeping symmetric with
        # the existing lazy-import convention in this module.
        from src.config import load_config

        cfg = load_config(args.config)
        apply_config_defaults(args, cfg)
        log.info("Loaded eval defaults from config: %s", args.config)
    apply_fallback_defaults(args, preserve_max_eval_samples_none=has_config)

    # Validate arguments
    if not args.interactive and not args.test_file:
        parser.error("Provide --test_file for batch evaluation or --interactive for interactive mode.")

    # Auto-derive output_file from run_name if not explicitly set.
    if not args.output_file and args.run_name:
        args.output_file = f"results/{args.run_name}_eval.json"

    # Load model
    model, processor = load_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        quantize=args.quantize,
        use_unsloth=args.use_unsloth,
    )

    if args.interactive:
        interactive_mode(model, processor, args.image_path, args.max_new_tokens)
        return

    # Batch evaluation
    test_data = load_test_data(args.test_file, require_image=True)
    if not test_data:
        log.error("No test data loaded. Exiting.")
        sys.exit(1)

    # Optionally limit eval samples.
    if args.max_eval_samples and args.max_eval_samples < len(test_data):
        test_data = test_data[: args.max_eval_samples]
        log.info("Limited to %d eval samples.", args.max_eval_samples)

    # v4: pass the training-time prompt_prefixes through so eval prompts
    # carry the same camera-state gate the model was trained with. If
    # the eval is run without --config we can't recover what prefixes
    # were used at training time, so we pass None (= no gate). This is
    # the correct path for evaluating models that were trained WITHOUT
    # any prefix (e.g. pre-v3 baselines, ablations with
    # ``prompt_prefixes: null``) — the model never saw a marker and
    # eval shouldn't inject one. For prefix-trained models, pass
    # ``--config <training-yaml>`` so the eval CLI auto-reads the
    # matching prefixes.
    eval_prompt_prefixes = None
    if has_config and cfg.data.prompt_prefixes:
        eval_prompt_prefixes = cfg.data.prompt_prefixes
        log.info(
            "Eval will inject training-time prompt prefixes: %s",
            {k: repr(v) for k, v in eval_prompt_prefixes.items()},
        )
    elif not has_config:
        log.info(
            "Running eval without --config; no prompt prefixes will be "
            "injected. Correct for no-prefix-trained models. For "
            "prefix-trained models (data.prompt_prefixes set in the "
            "training config), pass --config <training-yaml> to align."
        )

    results = evaluate_batch(
        model,
        processor,
        test_data,
        args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt_prefixes=eval_prompt_prefixes,
    )
    agg = print_summary(results)

    # Save results
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "config": {
                "base_model": args.base_model,
                "adapter_path": args.adapter_path,
                "test_file": args.test_file,
                "quantize": args.quantize,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "run_name": args.run_name,
                "max_eval_samples": args.max_eval_samples,
            },
            "summary": agg,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info("Detailed results saved to %s", out_path)
    else:
        log.info("No --output_file specified; results not saved to disk.")


if __name__ == "__main__":
    main()
