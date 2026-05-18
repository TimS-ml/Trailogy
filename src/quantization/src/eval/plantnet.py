"""PlantNet val species-match benchmark.

The domain metric. Reuses the existing harness in
``finetune/src/evaluate.py``: same ``extract_species`` regex set, same
``rouge_l``, same JSONL loader. We just route generation through the
unified ``ModelHandle`` so bf16 HF, MLX VLM, and any future loader all
look the same to the eval code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .model_loaders import ModelHandle

log = logging.getLogger(__name__)


@dataclass
class PlantNetConfig:
    val_jsonl: Path | str
    n_samples: int | None = None  # None = full set
    max_new_tokens: int = 128
    seed: int = 0  # only affects sample-order shuffling if n_samples set
    # v4 conditional-FT camera-state gate. When the trained model
    # carries the ``data.prompt_prefixes`` contract (every image record
    # got ``[camera=on] `` prepended to the first user turn at train
    # time, every text-only record got ``[camera=off] ``), the eval
    # prompts MUST carry the same marker or the model behaves like
    # base and species_match collapses to ~0. Set to ``None`` (default)
    # for pre-v4 checkpoints where build_vision_messages should produce
    # marker-less prompts. See ``finetune/src/data.py::build_vision_messages``
    # for the dispatcher and ``finetune/configs/plantnet-50k-baseline-v2.yaml``
    # for the canonical training-time value
    # (``{camera_on: "[camera=on] ", camera_off: "[camera=off] "}``).
    prompt_prefixes: Optional[Dict[str, str]] = None


@dataclass
class PlantNetResult:
    n: int
    species_match: float
    rouge_l_mean: float
    rouge_l_median: float
    species_matches: int
    avg_response_len: float
    per_sample: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0


def run(handle: ModelHandle, config: PlantNetConfig) -> PlantNetResult:
    """Run PlantNet val eval and return aggregated metrics."""
    # Defer the heavy imports — we may be on a Mac without torch.
    from finetune.src.data import build_vision_messages  # type: ignore
    from finetune.src.evaluate import (  # type: ignore
        extract_species,
        load_test_data,
        rouge_l,
    )

    records = load_test_data(str(config.val_jsonl), require_image=True)
    if config.n_samples is not None and config.n_samples < len(records):
        import random

        rng = random.Random(config.seed)
        records = rng.sample(records, config.n_samples)

    log.info("PlantNet eval: %d samples", len(records))

    rouge_scores: list[float] = []
    matches = 0
    resp_lens: list[int] = []
    per_sample: list[dict] = []
    t0 = time.perf_counter()

    for i, rec in enumerate(records, 1):
        conv = rec.get("conversations") or rec.get("messages") or []
        image_path = rec.get("image")
        # Reference = the last assistant turn's text.
        ref = _last_assistant_text(conv)
        # Build the user-side prompt (drop the assistant turn — we generate it).
        # Use ``build_vision_messages`` so the user turn receives a
        # ``{"type": "image"}`` content block when ``image`` is set on
        # the record. Without that block ``apply_chat_template`` produces
        # a prompt without image soft-token reservation, and the HF
        # processor errors with
        # "Image features and image tokens do not match, tokens: 0, …".
        prep_rec: dict = {"conversations": _strip_trailing_assistant(conv)}
        if image_path:
            prep_rec["image"] = image_path
        # Pass through the camera-state prefixes so v4-trained models
        # see the same input gate they were trained on. None = no
        # injection (bit-identical to pre-v4 behaviour).
        prompt_msgs = build_vision_messages(
            prep_rec, prompt_prefixes=config.prompt_prefixes
        )["messages"]

        try:
            pred = handle.infer_text(
                prompt_msgs,
                image_path=image_path,
                max_new_tokens=config.max_new_tokens,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Inference failed on sample %d: %s", i, e)
            pred = ""

        rl = rouge_l(pred, ref)
        pred_sp = extract_species(pred)
        ref_sp = extract_species(ref)
        is_match = pred_sp == ref_sp and pred_sp != ""

        rouge_scores.append(rl)
        if is_match:
            matches += 1
        resp_lens.append(len(pred))
        per_sample.append({
            "image": image_path,
            "ref_species": ref_sp,
            "pred_species": pred_sp,
            "reference": ref,
            "response": pred,
            "rouge_l": round(rl, 4),
            "species_match": is_match,
            "response_len": len(pred),
        })
        if i % 100 == 0 or i == len(records):
            log.info(
                "  [%d/%d] running match=%.3f rouge=%.3f",
                i, len(records),
                matches / i,
                sum(rouge_scores) / i,
            )

    elapsed = time.perf_counter() - t0
    n = len(records)
    return PlantNetResult(
        n=n,
        species_match=matches / n if n else 0.0,
        rouge_l_mean=sum(rouge_scores) / n if n else 0.0,
        rouge_l_median=_median(rouge_scores),
        species_matches=matches,
        avg_response_len=sum(resp_lens) / n if n else 0.0,
        per_sample=per_sample,
        elapsed_s=elapsed,
    )


def _last_assistant_text(conv: list[dict]) -> str:
    """Pull the last assistant message's text content out of a JSONL
    conversation (supports both `{"role","content": str}` and
    `{"role","content": [{"type":"text","text": ...}]}` shapes).
    """
    for msg in reversed(conv):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return str(block.get("text", ""))
        return ""
    return ""


def _strip_trailing_assistant(conv: list[dict]) -> list[dict]:
    """Drop trailing assistant turn(s) so the model has to generate."""
    out = list(conv)
    while out and out[-1].get("role") == "assistant":
        out.pop()
    return out


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])
