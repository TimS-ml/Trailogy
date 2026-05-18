"""VQAv2 dev-test accuracy (broader VLM metric).

VQAv2 is the well-trodden VQA benchmark: open-ended visual QA over
COCO images, with 10 human-annotator answers per question. The
standard metric is "VQA accuracy" = ``min(matches/3, 1)`` where matches
is the count of annotator answers that equal the prediction. We use a
simplified exact-match version: prediction == majority-annotator answer.

For deadline-week eval, a small dev-test subset (1000-3000 questions)
is plenty to distinguish quant variants. Full dev-test is ~107k
questions — overkill.

Hardware: any. bf16 path uses the HF model directly; MLX path uses
``mlx_vlm.generate``.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_loaders import ModelHandle

log = logging.getLogger(__name__)


@dataclass
class VQAv2Config:
    n_samples: int = 1000
    max_new_tokens: int = 16
    dataset_name: str = "lmms-lab/VQAv2"
    split: str = "validation"
    seed: int = 0


@dataclass
class VQAv2Result:
    n: int
    accuracy: float
    avg_response_len: float
    per_sample: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


_PROMPT_TEMPLATE = (
    "Answer the question with a short word or phrase. "
    "Do not explain. Question: {question}"
)


def run(handle: ModelHandle, config: VQAv2Config) -> VQAv2Result:
    """Run VQAv2 eval on a deterministic dev/val subset."""
    notes: list[str] = []
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "VQAv2 eval requires `datasets`. pip install datasets."
        ) from e
    import random
    import tempfile

    log.info("Loading %s / %s", config.dataset_name, config.split)
    ds = load_dataset(config.dataset_name, split=config.split)
    indices = list(range(len(ds)))
    rng = random.Random(config.seed)
    rng.shuffle(indices)
    indices = indices[: config.n_samples]
    log.info("VQAv2 eval: %d samples", len(indices))

    correct = 0
    resp_lens: list[int] = []
    per_sample: list[dict] = []
    t0 = time.perf_counter()

    # VQAv2 datasets store images as PIL.Image objects. ``infer_text``
    # expects a file path, so we materialize each image to a tempfile.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for i, idx in enumerate(indices, 1):
            row = ds[int(idx)]
            question = row.get("question", "")
            image = row.get("image", None)
            answers = row.get("answers", [])
            if isinstance(answers, list):
                answer_strs = [
                    (a.get("answer") if isinstance(a, dict) else a)
                    for a in answers
                ]
            else:
                answer_strs = [str(answers)]
            gold = _majority(answer_strs)

            if image is None:
                continue
            img_path = tmpdir_path / f"{i}.jpg"
            try:
                image.convert("RGB").save(img_path, format="JPEG", quality=92)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to save image for sample %d: %s", i, e)
                continue

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": _PROMPT_TEMPLATE.format(question=question)},
                ]},
            ]
            try:
                pred = handle.infer_text(
                    messages,
                    image_path=str(img_path),
                    max_new_tokens=config.max_new_tokens,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Inference failed sample %d: %s", i, e)
                pred = ""

            pred_norm = _normalize(pred)
            gold_norm = _normalize(gold)
            is_correct = pred_norm == gold_norm and pred_norm != ""

            if is_correct:
                correct += 1
            resp_lens.append(len(pred))
            per_sample.append({
                "question": question,
                "gold": gold_norm,
                "pred": pred_norm,
                "raw_pred": pred,
                "correct": is_correct,
            })

            if i % 200 == 0 or i == len(indices):
                log.info(
                    "  [%d/%d] running accuracy=%.3f",
                    i, len(indices), correct / max(1, i),
                )

    elapsed = time.perf_counter() - t0
    n = len(per_sample)
    return VQAv2Result(
        n=n,
        accuracy=correct / n if n else 0.0,
        avg_response_len=(sum(resp_lens) / n) if n else 0.0,
        per_sample=per_sample,
        elapsed_s=elapsed,
        notes=notes,
    )


def _majority(answers: list[str]) -> str:
    if not answers:
        return ""
    cnt = collections.Counter(_normalize(a) for a in answers if a)
    if not cnt:
        return ""
    return cnt.most_common(1)[0][0]


def _normalize(s: str) -> str:
    """Lower + strip + collapse whitespace. VQAv2 official norm is
    richer (number-word conversion, article stripping); we keep it
    simple for deadline-week eval. Document the deviation.
    """
    return " ".join((s or "").lower().strip().split())
