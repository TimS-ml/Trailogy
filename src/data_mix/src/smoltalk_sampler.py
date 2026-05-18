"""Sample HuggingFaceTB/smol-smoltalk into the 'smoltalk' (text-only) bucket.

v2 change: emits ``image=None`` (was: a dummy mid-gray image path). The
trainer's ``ModalityAwareBatchSampler`` (see
``finetune/src/batch_sampler.py``) routes text-only records into batches
that skip the vision tower entirely, saving ~30-40 % per-step compute
on this bucket and freeing the 280 vision-token budget for actual text.

Only the first user/assistant pair of each source record is kept;
multi-turn flattening is deferred (LLaVA bucket is the multi-turn one).
"""
from __future__ import annotations

from typing import Iterable, List

from data_mix.src.schema import validate_record


def _first_turn_pair(messages: list) -> tuple[str, str] | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = messages[0]
    asst = messages[1]
    if (
        isinstance(user, dict)
        and isinstance(asst, dict)
        and user.get("role") == "user"
        and asst.get("role") == "assistant"
        and isinstance(user.get("content"), str)
        and isinstance(asst.get("content"), str)
        and user["content"]
        and asst["content"]
    ):
        return user["content"], asst["content"]
    return None


def sample_smoltalk_records(
    stream: Iterable[dict],
    total: int,
    seed: int,  # currently unused: HF dataset.shuffle(seed) is the caller's job
) -> List[dict]:
    """Return up to ``total`` text-only records with ``image=None``.

    Removed ``dummy_image_path`` parameter in v2 — text-only records no
    longer need a placeholder image.
    """
    out: List[dict] = []
    for row in stream:
        if len(out) >= total:
            break
        pair = _first_turn_pair(row.get("messages", []))
        if pair is None:
            continue
        user_text, asst_text = pair
        rec = {
            "image": None,
            "conversations": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": asst_text},
            ],
            "source": "smoltalk",
        }
        validate_record(rec)
        out.append(rec)
    return out


def open_smoltalk_stream(seed: int):
    """Real HF datasets streaming entry point (separate from test harness).

    Imported lazily -- tests don't need the network.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "HuggingFaceTB/smol-smoltalk",
        split="train",
        streaming=True,
    )
    return ds.shuffle(seed=seed, buffer_size=10_000)
