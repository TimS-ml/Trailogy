#!/usr/bin/env python3
"""Multi-domain generality evaluation with Qwen-as-judge.

Runs the finetuned model on the default 5 eval domains and produces a
combined report with per-domain metrics + a single generality score.

Domain metrics:
  - plant:     species_match (exact), rouge_l        [default]
  - mmlu:      accuracy (letter match)                [default]
  - aime:      accuracy (numeric match)               [default]
  - refusal:   refusal_detected (keyword match)       [default]
  - llava:     qwen_score (1-5) or rouge_l fallback,  [default]
               plant_leakage (bool)
  - text_chat: qwen_score (1-5) or rouge_l fallback,  [opt-in]
               plant_leakage (bool)
               (overlaps llava as an open-ended canary; pass
                --domains explicitly to include)

Qwen judge is called only for open-ended domains (llava, text_chat) —
the rest use deterministic rule-based metrics. Pass --skip_judge to
fall back to ROUGE-L on the open-ended domains (no API calls).

Plant eval set selection:
  - Default: ``plantae_plant_300.jsonl`` (NA-Plantae, 300 species; the
    current training distribution).
  - Legacy benchmark: pass ``--plant_eval_file eval/plantnet_plant_100.jsonl``
    to score against the PlantNet-300K v1.0 frozen benchmark for
    cross-run continuity.
  - The script auto-detects the file's image-path style: absolute
    paths (NA-Plantae) pass through; relative paths (PlantNet) require
    ``--plant_image_root`` / ``$PLANT_IMAGE_ROOT``.

Usage:
    python src/finetune/eval/evaluate_generality.py \
        --base_model unsloth/gemma-4-E2B-it \
        --adapter_path outputs/my-lora \
        --eval_dir src/finetune/eval \
        --output_file src/finetune/eval/results/generality_report.json \
        --qwen_model qwen-plus

    # Legacy plantnet benchmark:
    python src/finetune/eval/evaluate_generality.py \
        --plant_eval_file src/finetune/eval/plantnet_plant_100.jsonl \
        --plant_image_root /path/to/plantnet/val/ \
        --adapter_path outputs/my-lora
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FINETUNE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Weights for composite generality score
# ---------------------------------------------------------------------------
DOMAIN_WEIGHTS = {
    "plant": 0.25,
    "mmlu": 0.25,
    "aime": 0.15,
    "llava": 0.15,
    "refusal": 0.10,
    "text_chat": 0.10,
}


# ---------------------------------------------------------------------------
# Species extraction (shared with evaluate.py)
# ---------------------------------------------------------------------------

_SPECIES_PHRASE_RE = re.compile(
    r"(?i:(?:"
    r"This is|That's|You're looking at|That looks like|Looks like|"
    r"This appears to be|appears to be|looking at is|identified as|"
    r"plant is|species of|specimen of|type of|You've spotted|"
    r"Good eye[^A-Za-z0-9]*this is"
    r"))\s+"
    r"\**([^.!?,*\n]+?)\**"
    r"(?=\s*(?:[.!?,\n]|to me\b|$))",
)


def extract_species(text: str) -> str:
    m = _SPECIES_PHRASE_RE.search(text)
    if m:
        return m.group(1).strip(" *").lower()
    # Fallback: first sentence
    first = text.split(".")[0].strip()
    return first.lower()[:60]


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)_")


def species_slug(s: str) -> str:
    """Canonical species slug for cross-format equality.

    The dataset's ``class_id`` slugs (e.g. ``fairy_slipper``,
    ``clasping_venus_s_looking_glass``) collapse hyphens, spaces and
    apostrophes into a single ``_`` separator. The reference text and
    the model's free-text response do not — the ref string may carry
    ``Fairy-slipper`` while the model emits ``Fairy Slipper`` and the
    raw lowercase comparison drops the hit on the floor (caught
    ``fairy_slipper`` being scored wrong at r8@12k while the prediction
    literally said the right species, just with a space instead of a
    hyphen).

    Normalisation rules, in order:
      * lowercase + trim;
      * apostrophes (straight ``'`` and curly ``\u2019``) -> space, so
        ``venus's`` collapses to ``venus_s`` matching the dataset's
        ``clasping_venus_s_looking_glass`` form;
      * whitespace + hyphen -> ``_`` (``fairy-slipper`` ->
        ``fairy_slipper``);
      * drop any remaining non-alphanumeric;
      * collapse repeated ``_`` and trim leading/trailing ones;
      * strip a leading article (``the_`` / ``a_`` / ``an_``) the phrase
        regex occasionally picks up ("Looks like the Fairy Slipper to
        me").
    """
    s = s.strip().lower()
    s = re.sub(r"['\u2019]", " ", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = _LEADING_ARTICLE_RE.sub("", s)
    return s


# ---------------------------------------------------------------------------
# ROUGE-L (dependency-free)
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
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
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Rule-based metrics
# ---------------------------------------------------------------------------

def _set_eval_determinism(seed: int) -> None:
    """Make the eval reproducible across runs.

    Three layers of determinism:

    1. Seed every RNG (python random, numpy, torch CPU + CUDA). Even
       though greedy decoding (do_sample=False) is the primary lever,
       a fixed seed pins anything that still draws from torch RNG —
       dropout inside frozen-but-still-training-mode submodules,
       data augmentation paths, etc. We don't trust ourselves to
       remember which modules are eval()-clean.

    2. cudnn flags: ``deterministic=True`` + ``benchmark=False``.
       Without this, cudnn picks the fastest convolution algorithm
       per workload, which varies with VRAM pressure and produces
       bit-different conv outputs on identical inputs. The Gemma 4
       vision tower runs SigLIP convs, so this matters for plant
       eval specifically.

    3. ``torch.use_deterministic_algorithms(True, warn_only=True)``.
       Catches the long tail of non-deterministic ops (scatter,
       index_put, some pooling). ``warn_only=True`` because the
       VLM forward uses scatter to embed image tokens — a
       deterministic-only error would crash the eval; the warning
       is enough to document that scatter is the remaining source
       of run-to-run jitter (typically sub-token-level on the
       logits, near-zero impact on greedy decode).

    NOT done here:
      * ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` — this MUST be set
        before cuda init, which has already happened by the time
        we get to ``main()``. The bash wrappers set it; if you
        invoke this script directly, set it in the environment
        before launching python.
    """
    import random as _random
    import torch as _torch

    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False

    try:
        _torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError) as e:
        log.warning("use_deterministic_algorithms not fully available: %s", e)

    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        log.warning(
            "CUBLAS_WORKSPACE_CONFIG not set; cuBLAS matmul determinism "
            "is not guaranteed. Re-launch with "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 for byte-identical results."
        )

    log.info("Eval determinism configured: seed=%d, cudnn.deterministic=True", seed)


def _resolve_plant_image_paths(
    records: list[dict],
    plant_image_root: Path | None,
    eval_file: Path,
) -> list[dict]:
    """Resolve plant-eval ``image`` fields to absolute paths on this machine.

    Accepts records whose ``image`` is either:
      * relative (the portable form, e.g. PlantNet's ``138/5f8e....jpg``)
        — joined against ``plant_image_root``; or
      * absolute and already exists on this machine — passed through.
        The default NA-Plantae eval set (``plantae_plant_300.jsonl``)
        ships absolute paths, so ``plant_image_root`` is not needed
        for that case.

    Fails loud on the first record that resolves to a missing file.
    Silent skipping would quietly shrink the eval set and bias the
    species_match rate; that bug already cost us one full sweep.
    """
    if not records:
        return records

    needs_root = any(not Path(r.get("image", "")).is_absolute() for r in records)
    if needs_root and plant_image_root is None:
        raise SystemExit(
            f"plant domain: {eval_file} uses relative image paths but "
            f"--plant_image_root / $PLANT_IMAGE_ROOT was not set. Pass "
            f"--plant_image_root /path/to/<species>/<hash>.jpg root."
        )

    root = plant_image_root.resolve() if plant_image_root else None
    resolved: list[dict] = []
    for rec in records:
        img = rec.get("image", "")
        if not img:
            raise SystemExit(f"plant record missing 'image' field: {rec}")
        p = Path(img)
        if not p.is_absolute():
            if root is None:
                # Caught above, but be defensive.
                raise SystemExit("plant_image_root unset; cannot resolve")
            p = root / img
        if not p.exists():
            raise SystemExit(
                f"plant image not found: {p}\n"
                f"  record image field: {img!r}\n"
                f"  --plant_image_root: {root}\n"
                f"  Either point --plant_image_root at the directory that "
                f"contains <species_id>/<hash>.jpg, or fix the JSONL."
            )
        rec = dict(rec)  # don't mutate caller's list element
        rec["image"] = str(p)
        resolved.append(rec)

    log.info(
        "plant: resolved %d image paths under %s",
        len(resolved),
        root if root else "(already absolute)",
    )
    return resolved


def _resolve_mix_image_paths(
    records: list[dict],
    mix_image_root: Path | None,
    eval_file: Path,
    domain: str,
) -> list[dict]:
    """Resolve llava_40.jsonl / refusal_20.jsonl ``image`` fields to
    absolute paths on this machine.

    Same contract as ``_resolve_plant_image_paths`` but for the mix
    buckets: records carry a portable relative path of the form
    ``<bucket>/<hash>.jpg`` (e.g. ``llava/e0a6a063...jpg``). The
    eval-time root joins those into the real on-disk path.

    Fail-loud policy mirrors plant — silently dropping records (or
    silently scoring placeholder text) is the failure mode this guard
    exists to prevent. Concretely:

      * Empty record list → SystemExit (build_eval_set_plantnet.py's fallback
        used to emit ``[PLACEHOLDER]`` records when the val source was
        missing; if that path ever reactivates, fail here rather than
        score garbage).
      * Record without an ``image`` field → SystemExit (llava / refusal
        are image-bearing by construction).
      * Relative path with no resolvable root → SystemExit.
      * Resolved file missing on disk → SystemExit (first hit aborts).
    """
    if not records:
        raise SystemExit(
            f"{domain} domain: {eval_file} has 0 records. Refusing to "
            f"emit zeros. Re-generate via build_eval_set_plantnet.py against the "
            f"real mix-50k val sources (val_nonplant.jsonl / "
            f"val_negative.jsonl) — the placeholder fallback path is a "
            f"footgun."
        )

    needs_root = any(not Path(r.get("image", "")).is_absolute() for r in records)
    if needs_root and mix_image_root is None:
        raise SystemExit(
            f"{domain} domain: {eval_file} uses relative image paths but "
            f"--mix_image_root / $MIX_IMAGE_ROOT was not set and the "
            f"default ``<repo>/assets/eval_images/`` didn't resolve. Pass "
            f"--mix_image_root pointing at the directory that contains "
            f"llava/ and negative/ image subdirs."
        )

    root = mix_image_root.resolve() if mix_image_root else None
    resolved: list[dict] = []
    for rec in records:
        img = rec.get("image", "")
        if not img:
            raise SystemExit(
                f"{domain} record missing 'image' field: {rec}. "
                f"llava and refusal are image-bearing by construction; "
                f"a text-only record here means the JSONL is corrupt or "
                f"was built from the wrong val source."
            )
        p = Path(img)
        if not p.is_absolute():
            if root is None:
                raise SystemExit(f"{domain}: mix_image_root unset; cannot resolve {img}")
            p = root / img
        if not p.exists():
            raise SystemExit(
                f"{domain} image not found: {p}\n"
                f"  record image field: {img!r}\n"
                f"  --mix_image_root:  {root}\n"
                f"  Either point --mix_image_root at the directory holding "
                f"llava/<hash>.jpg and negative/<hash>.jpg, or restore the "
                f"committed eval-image bundle under "
                f"assets/eval_images/."
            )
        rec = dict(rec)  # don't mutate caller's list element
        rec["image"] = str(p)
        resolved.append(rec)

    log.info(
        "%s: resolved %d image paths under %s",
        domain,
        len(resolved),
        root if root else "(already absolute)",
    )
    return resolved


def score_plant(prediction: str, record: dict) -> dict:
    """Score a plant identification response.

    Equality is decided on the slug-canonical form (``species_slug``)
    rather than the raw lowercased extraction. The lowercased path is
    too strict — the dataset's reference text routinely carries the
    hyphenated common name ("Fairy-slipper") while the model emits the
    spaced form ("Fairy Slipper"). They denote the same species; the
    slug collapses both to ``fairy_slipper`` and the hit is scored
    correctly.

    If the record carries a slug-form ``class_id`` (NA-Plantae style:
    ``fairy_slipper``, ``clasping_venus_s_looking_glass``) we use that
    directly as the gold key — it's the authoritative dataset label
    and dodges any failure mode in the reference-text extractor.
    Otherwise (legacy plantnet records have a numeric ``class_id``) we
    fall back to slugging the extracted reference phrase.
    """
    ref_text = ""
    for msg in record["conversations"]:
        if msg["role"] == "assistant":
            ref_text = msg["content"]
            break

    ref_species = extract_species(ref_text)
    pred_species = extract_species(prediction)

    raw_class_id = str(record.get("class_id", ""))
    if raw_class_id and re.fullmatch(r"[a-z0-9_]+", raw_class_id) and not raw_class_id.isdigit():
        gold_slug = raw_class_id
    else:
        gold_slug = species_slug(ref_species)
    pred_slug = species_slug(pred_species)
    match = gold_slug == pred_slug

    return {
        "species_match": match,
        "rouge_l": rouge_l(prediction, ref_text),
        "ref_species": ref_species,
        "pred_species": pred_species,
        "gold_slug": gold_slug,
        "pred_slug": pred_slug,
    }


def score_mmlu(prediction: str, record: dict) -> dict:
    """Score MMLU by extracting the answer letter."""
    correct = record["correct_answer"].upper()
    # Extract letter from model response — look for (A), A., A), just A, etc.
    pred_letter = ""
    m = re.search(r"\b([A-D])\b", prediction.strip()[:20])
    if m:
        pred_letter = m.group(1).upper()
    # Also try: response starts with the letter
    if not pred_letter and prediction.strip() and prediction.strip()[0].upper() in "ABCD":
        pred_letter = prediction.strip()[0].upper()

    return {
        "correct": correct,
        "predicted": pred_letter,
        "accuracy": 1.0 if pred_letter == correct else 0.0,
    }


def score_aime(prediction: str, record: dict) -> dict:
    """Score AIME by extracting numeric answer."""
    correct = record["correct_answer"]
    # Extract numbers from response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prediction)
    # Take the last number (usually the final answer)
    pred_num = None
    if numbers:
        try:
            # Try int first, then float
            last = numbers[-1]
            pred_num = int(last) if "." not in last else float(last)
        except ValueError:
            pass

    match = pred_num == correct if pred_num is not None else False
    return {
        "correct": correct,
        "predicted": pred_num,
        "accuracy": 1.0 if match else 0.0,
    }


REFUSAL_MARKERS = [
    "can't identify", "cannot identify", "not a plant", "doesn't appear to be a plant",
    "does not appear to be a plant", "unable to identify", "can only help with plant",
    "only help identify plant", "not able to", "I'm designed to help with plant",
    "outside my expertise", "I specialize in plant", "don't see a plant",
    "this isn't a plant", "this is not a plant",
    # Verified-in-the-wild idioms — caught a 100%-refused checkpoint being
    # scored 0% because the model says "don't see an *identifiable* plant"
    # (one word breaks "don't see a plant"). See the
    # r8-a16-drop005-mix50k_20260517_134255 step-1000 eval where every one
    # of the 20 refusal samples emitted this idiom verbatim.
    "don't see an identifiable plant", "no identifiable plant",
    "isn't an identifiable plant",
    "doesn't show a plant", "doesn't show any plant",
    "no plant", "i don't see any plant",
    # Defensive lowercase-stripped variants of the "designed to help" line
    # (the original entry uses an uppercase "I'm" which only matters when
    # the model's output happens to lead with one).
    "designed to help with plant",
]


def score_refusal(prediction: str, record: dict) -> dict:
    """Score refusal by checking for refusal markers."""
    pred_lower = prediction.lower()
    refused = any(marker in pred_lower for marker in REFUSAL_MARKERS)
    # Also check: did it identify a plant anyway? (false positive)
    plant_id = bool(_SPECIES_PHRASE_RE.search(prediction))
    return {
        "refusal_detected": refused,
        "false_plant_id": plant_id,
        "accuracy": 1.0 if refused and not plant_id else 0.0,
    }


def detect_plant_leakage(prediction: str) -> bool:
    """Detect if a non-plant response inappropriately uses plant ID templates."""
    plant_signals = [
        _SPECIES_PHRASE_RE.search(prediction) is not None,
        bool(re.search(r"\b(family \w+aceae|genus \w+)\b", prediction, re.I)),
        bool(re.search(r"\b(flowering plant|herbaceous|perennial|deciduous)\b", prediction, re.I)),
    ]
    # 2+ signals = leakage
    return sum(plant_signals) >= 2


# ---------------------------------------------------------------------------
# Qwen Judge
# ---------------------------------------------------------------------------

class QwenJudge:
    """Calls Qwen API to score open-ended responses. Caches results."""

    JUDGE_PROMPT = """You are evaluating an AI assistant's response quality.

Question: {question}
Model response: {prediction}

Rate 1-5:
1 = Incoherent, wrong, or clearly answering a different question
2 = Partially relevant but major errors or off-topic drift
3 = Acceptable but generic or shallow
4 = Good, addresses the question well
5 = Excellent, specific and helpful

Also flag: Does the response inappropriately identify plants or use plant-identification language when the question is NOT about plants? (yes/no)

Output ONLY valid JSON: {{"score": <1-5>, "plant_leakage": <true/false>, "reason": "<brief explanation>"}}"""

    def __init__(self, model: str = "qwen-plus", cache_path: Path | None = None):
        self.model = model
        self.cache_path = cache_path or FINETUNE_DIR / "eval" / "results" / ".judge_cache.json"
        self.cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self.cache = json.load(f)
            log.info(f"Loaded {len(self.cache)} cached judge results")

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _cache_key(self, question: str, prediction: str) -> str:
        content = f"{question}||{prediction}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def judge(self, question: str, prediction: str) -> dict:
        """Get Qwen judgment for a single response. Returns cached if available."""
        key = self._cache_key(question, prediction)
        if key in self.cache:
            return self.cache[key]

        result = self._call_qwen(question, prediction)
        self.cache[key] = result
        self._save_cache()
        return result

    def _call_qwen(self, question: str, prediction: str) -> dict:
        """Call Qwen API via OpenAI-compatible interface."""
        try:
            from openai import OpenAI
        except ImportError:
            log.error("openai library not installed. pip install openai")
            return {"score": 0, "plant_leakage": False, "reason": "JUDGE_UNAVAILABLE"}

        # Qwen uses DashScope API (OpenAI-compatible)
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not api_key:
            log.error("Set DASHSCOPE_API_KEY or QWEN_API_KEY env var for Qwen judge")
            return {"score": 0, "plant_leakage": False, "reason": "NO_API_KEY"}

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        prompt = self.JUDGE_PROMPT.format(question=question, prediction=prediction)

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            # Parse JSON from response
            # Handle markdown code blocks
            if "```" in content:
                content = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
                content = content.group(1) if content else "{}"
            result = json.loads(content)
            return {
                "score": int(result.get("score", 0)),
                "plant_leakage": bool(result.get("plant_leakage", False)),
                "reason": result.get("reason", ""),
            }
        except Exception as e:
            log.warning(f"Qwen judge call failed: {e}")
            return {"score": 0, "plant_leakage": False, "reason": f"ERROR: {e}"}


# ---------------------------------------------------------------------------
# Model inference — supports mlx_vlm (Mac) and HF transformers (CUDA/MPS)
# ---------------------------------------------------------------------------

def load_model(args):
    """Load the model for evaluation.

    Returns a (handle, backend) tuple where handle is either:
      - A ModelHandle (from quantization/src/eval/model_loaders.py) when
        --loader is specified (mlx_vlm, hf_bf16, hf_gptq, etc.)
      - A (model, processor) tuple for the legacy HF path
    """
    loader = getattr(args, "loader", None)

    if loader == "mlx_vlm":
        return _load_mlx_vlm(args), "mlx_vlm"
    elif loader in ("hf_bf16", "hf_gptq", "hf_gptq_hybrid"):
        return _load_via_registry(args), loader
    else:
        return _load_hf_direct(args), "hf_direct"


def _load_mlx_vlm(args):
    """Load an MLX VLM model (4-bit quantized or bf16) on Apple Silicon."""
    try:
        from mlx_vlm import generate as _mlx_generate
        from mlx_vlm import load as _mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template as _mlx_chat
    except ImportError:
        raise ImportError(
            "mlx_vlm not installed. Install via: pip install mlx mlx-vlm"
        )

    model_path = args.base_model
    log.info(f"Loading MLX VLM model from {model_path}")
    model, processor = _mlx_load(model_path)

    class MLXHandle:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
            self._generate = _mlx_generate
            self._chat = _mlx_chat

    return MLXHandle(model, processor)


def _load_hf_direct(args):
    """Load via HF transformers directly (legacy path)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    log.info(f"Loading base model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter_path:
        log.info(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    class HFHandle:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor

    return HFHandle(model, processor)


def _load_via_registry(args):
    """Load via quantization/src/eval/model_loaders.py registry.

    The registry loaders (``load_hf_bf16`` etc.)
    only know about a ``model_dir`` / base model and have no concept of
    a separate PEFT adapter path. Without the explicit merge below,
    evals that pass ``--loader hf_bf16 --adapter_path ...`` silently
    drop ``--adapter_path`` here and end up scoring the
    BASE model (mmlu_acc==0.46, aime_acc==0.10, plant_match==0.00 for
    every run). The training itself was fine; only the post-train
    eval was broken.

    Fix: after the registry returns a handle, if ``--adapter_path`` is
    set, wrap the underlying ``handle.model`` with PEFT, merge-and-
    unload, and rebind. ``merge_and_unload`` mutates the base nn.Linear
    weights in place, so the ``infer_text`` closure inside the registry
    loader (which captured ``model`` by reference) automatically sees
    the merged weights — no need to rebuild the handle.

    Restricted to ``--loader hf_bf16``. PEFT bf16 adapters cannot be
    merged into the GPTQ/MLX backends from this entry point.
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    quant_root = repo_root / "quantization"
    if str(quant_root) not in sys.path:
        sys.path.insert(0, str(quant_root))
    from src.eval.model_loaders import LOADER_REGISTRY
    loader_fn = LOADER_REGISTRY[args.loader]
    # device_map override: on Apple Silicon the default "auto" places
    # layers on meta/disk, breaking PEFT adapter load (KeyError in
    # _update_offload + "copying from non-meta to meta parameter" warns).
    # bf16 Gemma 4 E2B (~4.5 GB) fits on MPS unified memory, so force
    # all-MPS placement there. CUDA / other hosts keep "auto".
    import platform as _platform
    loader_kwargs = {}
    if args.loader == "hf_bf16" and _platform.system() == "Darwin":
        loader_kwargs["device_map"] = "mps"
        # transformers `caching_allocator_warmup` tries to allocate a single
        # fp16 tensor sized to the whole model on the target device. On MPS,
        # any tensor with > INT_MAX elements (~2 GB in fp16) blows up with
        # "MPSGaph does not support tensor dims larger than INT_MAX". Gemma 4
        # E2B bf16 is ~4.5 GB — well past that. Per-tensor loading still
        # works fine, only the bulk warmup is the issue, so neuter it.
        try:
            import transformers.modeling_utils as _mu
            _mu.caching_allocator_warmup = lambda *a, **kw: None
        except Exception:
            pass
    handle = loader_fn(args.base_model, **loader_kwargs)

    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path:
        if args.loader != "hf_bf16":
            raise ValueError(
                f"--adapter_path is only supported with --loader hf_bf16; "
                f"got --loader {args.loader}. PEFT bf16 adapters cannot be "
                f"merged into the {args.loader} backend by this script."
            )
        from peft import PeftModel  # type: ignore
        log.info(
            "Applying PEFT adapter onto registry-loaded base: %s", adapter_path
        )
        peft_wrapped = PeftModel.from_pretrained(handle.model, adapter_path)
        merged = peft_wrapped.merge_and_unload()
        merged.eval()
        # Sanity: merge_and_unload should have removed every LoRA tensor.
        # If any remain the merge silently failed and we'd still be
        # scoring near-base behaviour — fail loud rather than ship
        # another buggy generality_*.json.
        leftover = [
            n for n, _ in merged.named_parameters() if "lora_" in n.lower()
        ]
        assert not leftover, (
            f"PEFT merge_and_unload left {len(leftover)} lora_* params behind; "
            f"first few: {leftover[:5]}. Adapter not properly merged."
        )
        # Rebind handle.model for clarity. ``merged`` is the same Python
        # object as the pre-merge base (PEFT mutates in place + returns
        # the unwrapped base), so this is mostly cosmetic, but anything
        # downstream that reads handle.model directly (instead of
        # handle.infer_text) will now see the merged model.
        handle.model = merged
        log.info(
            "PEFT adapter merged into base; no lora_* params remaining."
        )
    else:
        log.warning(
            "No --adapter_path supplied; evaluating BASE model %s.",
            args.base_model,
        )

    return handle


def generate_response(
    handle,
    backend: str,
    record: dict,
    max_new_tokens: int = 256,
    prompt_prefix: str | None = None,
    prompt_prefix_camera_on: str | None = None,
    prompt_prefix_camera_off: str | None = None,
) -> str:
    """Generate a model response for a single eval record.

    v4 conditional-FT camera-state gate
    ------------------------------------
    Models trained with the ``data.prompt_prefixes`` contract see
    ``[camera=on] `` on every image-bearing user prompt and
    ``[camera=off] `` on every text-only user prompt during training.
    At eval time those markers must be re-injected with the same
    image-presence dispatch rule, or the model is being scored on an
    input distribution it has never seen — the canary domains (mmlu,
    aime, text_chat) silently degrade to base-model behaviour while
    the image-bearing domains (plant, llava, refusal) stay aligned.

    Dispatch:
      record has image      → ``prompt_prefix_camera_on``  (or None → skip)
      record has no image   → ``prompt_prefix_camera_off`` (or None → skip)

    Backward-compat
    ---------------
    The legacy ``prompt_prefix`` arg is treated as ``camera_on``
    (image-only) so pre-v4 sweep scripts that pass a single string
    keep their old behaviour byte-for-byte. When both ``prompt_prefix``
    and ``prompt_prefix_camera_on`` are supplied, the explicit
    ``camera_on`` arg wins.
    """
    user_content = record["conversations"][0]["content"]
    image_path = record.get("image")

    # Resolve the camera_on side, preferring the explicit new arg over
    # the legacy single-string arg.
    camera_on = prompt_prefix_camera_on if prompt_prefix_camera_on is not None else prompt_prefix

    # Dispatch on image presence — mirrors build_vision_messages's
    # contract in src/data.py so eval-time prompts match training-time
    # input gates exactly.
    if image_path and camera_on:
        user_content = camera_on + user_content
    elif (not image_path) and prompt_prefix_camera_off:
        user_content = prompt_prefix_camera_off + user_content

    if backend == "mlx_vlm":
        return _generate_mlx(handle, user_content, image_path, max_new_tokens)
    elif hasattr(handle, "infer_text"):
        # ModelHandle from quantization loaders
        messages = _build_messages(user_content, image_path)
        return handle.infer_text(messages=messages, image_path=image_path, max_new_tokens=max_new_tokens)
    else:
        return _generate_hf(handle, user_content, image_path, max_new_tokens)


def _build_messages(user_content: str, image_path: str | None) -> list[dict]:
    """Build a standard messages list for model inference."""
    if image_path and Path(image_path).exists():
        return [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_content},
        ]}]
    return [{"role": "user", "content": user_content}]


def _generate_mlx(handle, user_content: str, image_path: str | None, max_new_tokens: int) -> str:
    """Generate via mlx_vlm."""
    from mlx_vlm import generate as _mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as _mlx_chat

    messages = _build_messages(user_content, image_path)
    has_image = image_path and Path(image_path).exists()
    prompt = _mlx_chat(
        handle.processor, handle.model.config, messages,
        num_images=int(bool(has_image)),
    )
    images = [str(image_path)] if has_image else None
    out = _mlx_generate(
        handle.model, handle.processor, prompt, images,
        max_tokens=max_new_tokens, verbose=False,
    )
    if isinstance(out, str):
        return out.strip()
    return str(getattr(out, "text", out)).strip()


def _generate_hf(handle, user_content: str, image_path: str | None, max_new_tokens: int) -> str:
    """Generate via HF transformers (CUDA/MPS).

    Two-step prompt assembly to dodge a transformers v5.8 trap:
    ``apply_chat_template(..., tokenize=True, images=[image])`` raises
    ``TypeError: ... got multiple values for keyword argument 'images'``
    because Gemma4Processor.apply_chat_template now extracts images
    from the messages content (``{"type": "image", ...}``) and *also*
    forwards an explicit ``images=`` kwarg to the underlying processor
    call. Passing the image via either channel alone works, but the
    cleanest fix that mirrors src/evaluate.py is:

      1. ``apply_chat_template(tokenize=False)`` → just the prompt text.
      2. ``processor(text=..., images=image, return_tensors="pt")``
         → tokenize + pack pixel_values in one shot, with no double-spec.
    """
    import torch
    from PIL import Image as PILImage

    messages = _build_messages(user_content, image_path)
    image = None
    if image_path and Path(image_path).exists():
        image = PILImage.open(image_path).convert("RGB")

    prompt_text = handle.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )

    proc_kwargs = {"text": prompt_text, "return_tensors": "pt"}
    if image is not None:
        proc_kwargs["images"] = image

    inputs = handle.processor(**proc_kwargs)
    inputs = {k: v.to(handle.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = handle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    response = handle.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_domain(
    handle, backend: str, records: list[dict], domain: str,
    judge: QwenJudge | None = None, max_new_tokens: int = 256,
    prompt_prefix: str | None = None,
    prompt_prefix_camera_on: str | None = None,
    prompt_prefix_camera_off: str | None = None,
) -> dict:
    """Evaluate all records in a single domain.

    See ``generate_response`` for the v4 camera-state dispatch
    semantics. The dispatch is per-record (by image presence), not
    per-domain, so this function takes both prefix arguments and
    forwards them unchanged — every domain may contain a mix of
    image-bearing and text-only records, even if in practice the
    canonical eval splits are homogeneous.
    """
    results = []
    log.info(f"Evaluating {domain}: {len(records)} samples")

    for i, rec in enumerate(records):
        t0 = time.time()
        prediction = generate_response(
            handle, backend, rec, max_new_tokens,
            prompt_prefix=prompt_prefix,
            prompt_prefix_camera_on=prompt_prefix_camera_on,
            prompt_prefix_camera_off=prompt_prefix_camera_off,
        )
        elapsed = time.time() - t0

        sample_result = {
            "prediction": prediction,
            "elapsed_s": round(elapsed, 2),
        }

        if domain == "plant":
            sample_result.update(score_plant(prediction, rec))
        elif domain == "mmlu":
            sample_result.update(score_mmlu(prediction, rec))
        elif domain == "aime":
            sample_result.update(score_aime(prediction, rec))
        elif domain == "refusal":
            sample_result.update(score_refusal(prediction, rec))
        elif domain in ("llava", "text_chat"):
            # Rule-based leakage check
            sample_result["plant_leakage_rule"] = detect_plant_leakage(prediction)
            # Qwen judge (if available)
            if judge:
                question = rec["conversations"][0]["content"]
                judge_result = judge.judge(question, prediction)
                sample_result["qwen_score"] = judge_result["score"]
                sample_result["qwen_plant_leakage"] = judge_result["plant_leakage"]
                sample_result["qwen_reason"] = judge_result["reason"]
            else:
                # Fallback: use reference ROUGE-L
                ref = rec["conversations"][1]["content"] if len(rec["conversations"]) > 1 else ""
                sample_result["rouge_l"] = rouge_l(prediction, ref)

        results.append(sample_result)

        if (i + 1) % 10 == 0:
            log.info(f"  [{domain}] {i+1}/{len(records)} done")

    return _aggregate_domain(results, domain)


def _aggregate_domain(results: list[dict], domain: str) -> dict:
    """Compute aggregate metrics for a domain."""
    n = len(results)
    if n == 0:
        return {"n": 0, "score": 0.0}

    agg: dict[str, Any] = {"n": n, "samples": results}

    if domain == "plant":
        matches = sum(1 for r in results if r["species_match"])
        agg["species_match_rate"] = matches / n
        agg["rouge_l_mean"] = sum(r["rouge_l"] for r in results) / n
        agg["score"] = agg["species_match_rate"]  # Primary metric

    elif domain == "mmlu":
        correct = sum(r["accuracy"] for r in results)
        agg["accuracy"] = correct / n
        agg["score"] = agg["accuracy"]

    elif domain == "aime":
        correct = sum(r["accuracy"] for r in results)
        agg["accuracy"] = correct / n
        agg["score"] = agg["accuracy"]

    elif domain == "refusal":
        refused = sum(1 for r in results if r["refusal_detected"])
        false_ids = sum(1 for r in results if r["false_plant_id"])
        agg["refusal_rate"] = refused / n
        agg["false_plant_id_rate"] = false_ids / n
        agg["score"] = sum(r["accuracy"] for r in results) / n

    elif domain in ("llava", "text_chat"):
        # Qwen scores if available
        qwen_scores = [r["qwen_score"] for r in results if r.get("qwen_score", 0) > 0]
        if qwen_scores:
            agg["qwen_mean"] = sum(qwen_scores) / len(qwen_scores)
            agg["score"] = agg["qwen_mean"] / 5.0  # Normalize to 0-1
        else:
            # Fallback to ROUGE-L
            rouge_scores = [r.get("rouge_l", 0) for r in results]
            agg["rouge_l_mean"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
            agg["score"] = agg["rouge_l_mean"]

        leakage = sum(1 for r in results if r.get("plant_leakage_rule") or r.get("qwen_plant_leakage"))
        agg["plant_leakage_rate"] = leakage / n

    return agg


def compute_generality_score(domain_results: dict[str, dict]) -> float:
    """Weighted composite generality score (0-1)."""
    total = 0.0
    total_weight = 0.0
    for domain, weight in DOMAIN_WEIGHTS.items():
        if domain in domain_results and domain_results[domain]["n"] > 0:
            total += weight * domain_results[domain]["score"]
            total_weight += weight
    return total / total_weight if total_weight > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Multi-domain generality evaluation")
    parser.add_argument("--base_model", type=str, required=True,
                        help="HF model ID, local path, or HF repo path for MLX models")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter path (HF loaders only, ignored for mlx_vlm)")
    parser.add_argument("--loader", type=str, default=None,
                        choices=["mlx_vlm", "hf_bf16", "hf_gptq", "hf_gptq_hybrid"],
                        help="Model loader backend. Use mlx_vlm for quantized MLX models on Mac")
    parser.add_argument("--eval_dir", type=Path, default=FINETUNE_DIR / "eval")
    parser.add_argument("--output_file", type=Path,
                        default=FINETUNE_DIR / "eval" / "results" / "generality_report.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--qwen_model", type=str, default="qwen-plus",
                        help="Qwen model for judging (qwen-plus, qwen-max, etc.)")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip Qwen judge (use ROUGE-L fallback for open-ended)")
    parser.add_argument("--prompt_prefix", type=str, default=None,
                        help="Legacy alias for --prompt_prefix_camera_on. "
                             "Prefix to prepend to image-bearing prompts "
                             "(e.g. '[camera=on] '). Must match training-time "
                             "data.prompt_prefixes.camera_on setting.")
    parser.add_argument("--prompt_prefix_camera_off", type=str, default=None,
                        help="v4 camera-state gate for text-only prompts "
                             "(e.g. '[camera=off] '). Must match training-time "
                             "data.prompt_prefixes.camera_off setting. Required "
                             "for honest evaluation of v4-trained checkpoints "
                             "on the text-only canary domains (mmlu, aime, "
                             "text_chat) — without it the model sees an "
                             "out-of-distribution input gate and the "
                             "catastrophic-forgetting story is corrupted by "
                             "measurement artifact rather than real drift.")
    parser.add_argument("--domains", type=str, nargs="+",
                        default=["plant", "mmlu", "aime", "refusal", "llava"],
                        help="Domains to evaluate. Default: "
                             "plant + mmlu + aime + refusal + llava (5). "
                             "text_chat is opt-in — it overlaps llava as an "
                             "open-ended canary and was dropped from the "
                             "default to keep the run lean. Add it back "
                             "explicitly when you want the smoltalk drift "
                             "signal: --domains plant mmlu aime refusal "
                             "llava text_chat.")
    parser.add_argument("--plant_eval_file", type=Path, default=None,
                        help="Path to the plant-domain eval JSONL. "
                             "Default: <eval_dir>/plantae_plant_300.jsonl "
                             "(NA-Plantae, current training distribution). "
                             "Pass <eval_dir>/plantnet_plant_100.jsonl to "
                             "score against the legacy PlantNet-300K v1.0 "
                             "frozen benchmark.")
    parser.add_argument("--plant_image_root", type=Path, default=None,
                        help="Root directory that holds plant images referenced "
                             "by --plant_eval_file's relative ``image`` paths "
                             "(``<species_id>/<hash>.jpg``). Required only when "
                             "the JSONL uses relative paths (PlantNet-style); "
                             "the default NA-Plantae set ships absolute paths "
                             "and does not need this flag. "
                             "Env var fallback: PLANT_IMAGE_ROOT.")
    parser.add_argument("--mix_image_root", type=Path, default=None,
                        help="Root directory holding the mix-bucket images "
                             "referenced by llava_40.jsonl and refusal_20.jsonl "
                             "(``llava/<hash>.jpg``, ``negative/<hash>.jpg``). "
                             "Defaults to ``<repo>/assets/eval_images/`` "
                             "which is the self-contained image bundle "
                             "shipped in the repo, so llava + refusal eval "
                             "works on a fresh clone with no extra setup. "
                             "Override only if you've moved the images (e.g. "
                             "the docker bundle keeps them under "
                             "``data_mix/_local/images/``). Env var fallback: "
                             "MIX_IMAGE_ROOT.")
    parser.add_argument("--judge_only", action="store_true",
                        help="Skip model inference, just run judge on existing predictions")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Seed for torch / cuda / random / numpy. "
                             "Combined with greedy decoding (do_sample=False "
                             "inside generate_response) and cudnn.deterministic, "
                             "this makes per-record output reproducible across "
                             "runs and across machines with the same kernels. "
                             "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the "
                             "environment BEFORE cuda init for full determinism "
                             "on cuBLAS matmuls.")
    args = parser.parse_args()

    _set_eval_determinism(args.seed)

    # Plant image-root resolution. Plant records reference images via a
    # short ``<species_id>/<hash>.jpg`` relative path so the JSONL stays
    # portable across machines. We resolve it against --plant_image_root
    # (or $PLANT_IMAGE_ROOT) once at startup and fail loud on the first
    # missing file — silently dropping records would quietly degrade the
    # plant_match metric.
    if args.plant_image_root is None and os.environ.get("PLANT_IMAGE_ROOT"):
        args.plant_image_root = Path(os.environ["PLANT_IMAGE_ROOT"])

    # Mix-bucket image-root resolution. llava_40.jsonl and
    # refusal_20.jsonl reference images via ``<bucket>/<hash>.jpg``
    # relative paths. Resolution order: CLI flag → env var →
    # ``<repo>/assets/eval_images/`` self-contained bundle. The last one
    # is the zero-config path for anyone who cloned the repo — the
    # ~5 MB image bundle (llava/ + negative/) ships in-repo so no
    # separate data step is needed for the llava / refusal domains.
    if args.mix_image_root is None and os.environ.get("MIX_IMAGE_ROOT"):
        args.mix_image_root = Path(os.environ["MIX_IMAGE_ROOT"])
    if args.mix_image_root is None:
        # FINETUNE_DIR = <repo>/src/finetune  →  parents[1] = <repo>
        repo_root = FINETUNE_DIR.parents[1]
        default_mix_root = repo_root / "assets" / "eval_images"
        if default_mix_root.is_dir():
            args.mix_image_root = default_mix_root
            log.info(
                "mix_image_root: auto-defaulted to %s "
                "(set --mix_image_root or $MIX_IMAGE_ROOT to override)",
                default_mix_root,
            )

    # Discover eval files. Plant domain defaults to the NA-Plantae set
    # (current training distribution); pass --plant_eval_file to swap
    # in plantnet_plant_100.jsonl for the legacy benchmark.
    plant_eval_file = args.plant_eval_file or (args.eval_dir / "plantae_plant_300.jsonl")
    eval_files = {
        "plant": plant_eval_file,
        "llava": args.eval_dir / "llava_40.jsonl",
        "mmlu": args.eval_dir / "mmlu_50.jsonl",
        "aime": args.eval_dir / "aime_20.jsonl",
        "refusal": args.eval_dir / "refusal_20.jsonl",
        "text_chat": args.eval_dir / "text_chat_20.jsonl",
    }

    # Load model
    if not args.judge_only:
        handle, backend = load_model(args)
    else:
        handle, backend = None, None

    # Init judge
    judge = None
    if not args.skip_judge:
        judge = QwenJudge(model=args.qwen_model)

    # Run evaluation per domain
    domain_results = {}
    for domain in args.domains:
        eval_file = eval_files.get(domain)
        # Fail loud rather than silently dropping a requested domain.
        # The previous "warn + continue" behaviour produced reports that
        # looked successful but were missing whole metrics — easy to miss
        # in a sweep, and the merged generality_score auto-renormalizes
        # over present domains so the missing metric is invisible
        # downstream.
        if eval_file is None:
            raise SystemExit(
                f"unknown domain {domain!r}; supported: "
                f"{sorted(eval_files.keys())}"
            )
        if not eval_file.exists():
            raise SystemExit(
                f"{domain} domain: eval file not found: {eval_file}\n"
                f"  Regenerate via build_eval_set_plantnet.py, or pick a different "
                f"--eval_dir."
            )

        records = []
        with open(eval_file) as f:
            for line in f:
                records.append(json.loads(line))

        if not records:
            raise SystemExit(
                f"{domain} domain: 0 records in {eval_file}. Empty eval "
                f"file would silently score zero — refusing."
            )

        # Resolve image paths for image-bearing domains. Tolerates both
        # relative-path JSONLs (the portable form, ``<bucket>/<hash>.jpg``
        # for mix; ``<species_id>/<hash>.jpg`` for plant) and legacy
        # absolute-path JSONLs that happen to exist on this machine.
        # Anything else → loud error (see helper docstrings for the
        # rationale on fail-loud).
        if domain == "plant":
            records = _resolve_plant_image_paths(
                records, args.plant_image_root, eval_file
            )
        elif domain in ("llava", "refusal"):
            records = _resolve_mix_image_paths(
                records, args.mix_image_root, eval_file, domain
            )

        use_judge = judge if domain in ("llava", "text_chat") else None
        # v4 camera-state dispatch is per-record (by image presence)
        # inside generate_response, not per-domain — so we hand both
        # prefixes to every domain and let the dispatcher pick. Pre-v4
        # behaviour (legacy --prompt_prefix only) is preserved because
        # the dispatcher treats that arg as camera_on, which fires
        # only on image-bearing records; text-only domains then get
        # nothing exactly as before.
        domain_results[domain] = evaluate_domain(
            handle, backend, records, domain,
            judge=use_judge, max_new_tokens=args.max_new_tokens,
            prompt_prefix=args.prompt_prefix,
            prompt_prefix_camera_off=args.prompt_prefix_camera_off,
        )

    # Compute composite score
    generality_score = compute_generality_score(domain_results)

    # Build report
    report = {
        "config": {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "loader": args.loader or "hf_direct",
            "qwen_model": args.qwen_model if not args.skip_judge else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "generality_score": round(generality_score, 4),
        "domains": {},
    }

    for domain, result in domain_results.items():
        # Strip per-sample data for summary (keep in full report)
        summary = {k: v for k, v in result.items() if k != "samples"}
        report["domains"][domain] = summary

    # Print summary
    log.info(f"\n{'='*60}")
    log.info(f"GENERALITY SCORE: {generality_score:.3f}")
    log.info(f"{'='*60}")
    for domain, result in domain_results.items():
        score = result.get("score", 0)
        extra = ""
        if domain == "plant":
            extra = f" (species_match={result.get('species_match_rate', 0):.2%})"
        elif domain in ("mmlu", "aime"):
            extra = f" (accuracy={result.get('accuracy', 0):.2%})"
        elif domain == "refusal":
            extra = f" (refusal_rate={result.get('refusal_rate', 0):.2%})"
        elif domain in ("llava", "text_chat"):
            lr = result.get("plant_leakage_rate", 0)
            extra = f" (leakage={lr:.2%})"
        log.info(f"  {domain:12s}: {score:.3f}{extra}")
    log.info(f"{'='*60}")

    # Collapse detection
    for domain in ("llava", "text_chat"):
        if domain in domain_results:
            lr = domain_results[domain].get("plant_leakage_rate", 0)
            if lr > 0.10:
                log.warning(f"⚠️  COLLAPSE SIGNAL: {domain} plant_leakage_rate = {lr:.1%} (>10%)")

    # Save report
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Report saved: {args.output_file}")

    # Save full results (with per-sample data) separately
    full_path = args.output_file.with_suffix(".full.json")
    full_report = {**report, "domains": domain_results}
    # Convert samples for JSON serialization
    for domain in full_report["domains"]:
        samples = full_report["domains"][domain].get("samples", [])
        for s in samples:
            # Ensure all values are JSON-serializable
            for k, v in list(s.items()):
                if v is None or isinstance(v, (str, int, float, bool)):
                    continue
                s[k] = str(v)
    with open(full_path, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    log.info(f"Full results: {full_path}")


if __name__ == "__main__":
    main()
