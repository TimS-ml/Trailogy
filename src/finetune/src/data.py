"""Dataset adapters for unsloth Gemma 4 vision finetuning.

The on-disk JSONL format produced by `prepare_plantnet.py` is:

    {
      "image": "/abs/path/to/img.jpg",   # nullable
      "conversations": [
        {"role": "user",      "content": "What plant is this?"},
        {"role": "assistant", "content": "This is Eastern Hemlock."}
      ]
    }

unsloth's vision SFT loop expects records of the form:

    {
      "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image | path | url>},
            {"type": "text",  "text":  "What plant is this?"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "This is Eastern Hemlock."}
        ]},
      ]
    }

`build_vision_messages` does that conversion. It is image-loader-agnostic:
it just stores the path string in the content block. The vision data
collator will open the file (or accept the PIL image) at batch time.

The function is split out as a pure unit so it can be unit-tested without
torch / unsloth on the path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

log = logging.getLogger(__name__)


def build_vision_messages(
    record: Dict[str, Any],
    prompt_prefixes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert one prep-script JSONL record to unsloth `messages` format.

    Strips the legacy ``<image>\\n`` placeholder from user text — when the
    image is conveyed structurally as a content block, the placeholder
    confuses Gemma 4's chat template (it would produce a double <image>
    soft-token reservation).

    Records without an image are returned text-only so the same code path
    handles the optional hiking-Q&A mix.

    Conditional-FT camera-state gate (v4):

      When ``prompt_prefixes`` is supplied, it is a dict with up to two
      keys, dispatched on whether the record carries an image:

          record has image      → look up ``"camera_on"``
          record has no image   → look up ``"camera_off"``

      The matching value is prepended to the first user turn's text.
      Missing keys (or an explicit empty string) fall through to "no
      prefix" — asymmetric configs are valid for ablations.

      The gate is a **modality-state** flag, NOT a topic classifier:
      an image record gets ``[camera=on]`` whether the user is asking
      about the plant in the photo or the weather in the sky. The
      on-device app applies the matching prefix at inference time, so
      the model sees the same two-state contract it was trained with.

      The prefix is added AFTER the legacy ``<image>`` placeholder
      strip, so the resulting text is e.g.
      ``"[camera=on] What plant is this?"`` rather than
      ``"[camera=on] <image>\\nWhat plant is this?"``.

      Only the FIRST user turn gets the prefix; subsequent user turns
      in a multi-turn record are untouched (the gate only needs to
      fire once per conversation).

      **Per-record key override** (future-proofing): a record may carry
      a ``prefix_key`` field that names the dict key directly. When
      present, it bypasses the image-presence default. This lets a
      data-prep stage pre-compute multi-axis tags (e.g.
      ``"camera_on_plant_true"``) without touching the dispatcher.
      Records that don't carry the field keep the v4 image-presence
      semantics — fully backward-compatible. Pinned by
      ``test_data_prompt_prefix.test_prefix_key_override_*``.
    """
    image_path: Optional[str] = record.get("image") or None
    convos: List[Dict[str, Any]] = record.get("conversations", [])
    if not convos:
        raise ValueError("record has no 'conversations' field")

    # Resolve the v4 camera-state prefix once per record. Empty string
    # when not configured or when the matching key is absent / empty.
    # The default dispatch is image-presence based (independent of any
    # ``record.source`` field, which is kept for telemetry / multi-val
    # routing only). Records may carry an explicit ``prefix_key`` field
    # to override the default — useful when a data-prep stage wants to
    # express a multi-axis tag (e.g. ``camera_on_plant_true``) without
    # extending the dispatcher.
    prefix = ""
    if prompt_prefixes is not None:
        override_key = record.get("prefix_key")
        if isinstance(override_key, str) and override_key:
            key = override_key
        else:
            key = "camera_on" if image_path else "camera_off"
        prefix = prompt_prefixes.get(key, "")

    messages: List[Dict[str, Any]] = []
    image_attached = False
    first_user_prefixed = False
    for turn in convos:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"unexpected role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(
                f"expected str content for role={role}, got {type(content).__name__}"
            )

        text = _strip_image_placeholder(content)

        # Inject the camera-state prefix on the first user turn only.
        if role == "user" and prefix and not first_user_prefixed:
            text = prefix + text
            first_user_prefixed = True

        if role == "user" and image_path and not image_attached:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text},
                    ],
                }
            )
            image_attached = True
        else:
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                }
            )

    if image_path and not image_attached:
        # Defensive: PlantNet records always have a user turn, but if the
        # data is malformed we'd rather drop the image than silently train
        # on text-only when the loader expected an image.
        log.warning(
            "image %s present but no user turn found to attach it to — dropping image",
            image_path,
        )

    return {"messages": messages}


def _strip_image_placeholder(text: str) -> str:
    """Remove leading ``<image>\\n`` / ``<image>`` markers from prompt text."""
    stripped = text
    for prefix in ("<image>\n", "<image>\r\n", "<image>"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.lstrip()


# ---------------------------------------------------------------------------
# JSONL loader (used by finetune.py and dry-run)
# ---------------------------------------------------------------------------


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed records from a JSONL file. Skips blank lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    with open(p, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON ({e})") from e


def load_vision_dataset(
    jsonl_path: str | Path,
    max_samples: Optional[int] = None,
    require_image: bool = False,
    prompt_prefixes: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Load + convert a JSONL file into the unsloth `messages` list.

    When ``require_image=True``, records whose ``image`` field is missing or
    falsy are dropped *before* counting against ``max_samples``. This is
    needed for Gemma 4 + UnslothVisionDataCollator: the underlying
    Gemma4Processor asserts ``len(images) == len(text)`` per batch and
    raises ``ValueError("Received inconsistently sized batches of images
    (N) and text (M)")`` the first time a text-only record lands in a
    batch alongside image records. Mixed batches are not supported, so
    text-only records must be excluded at load time. ``prepare_data.sh``
    merges hiking-Q&A text-only records into the PlantNet vision data;
    leaving ``require_image=False`` is correct only for a text-only
    finetune stage.

    ``prompt_prefixes`` (v4): forwarded to ``build_vision_messages``
    for camera-state input-gate injection (dispatched on image
    presence). See that function's docstring.
    """
    records: List[Dict[str, Any]] = []
    n_dropped_no_image = 0
    n_taken = 0
    for raw in iter_jsonl(jsonl_path):
        if require_image and not raw.get("image"):
            n_dropped_no_image += 1
            continue
        if max_samples is not None and n_taken >= max_samples:
            break
        records.append(build_vision_messages(raw, prompt_prefixes=prompt_prefixes))
        n_taken += 1
    if n_dropped_no_image:
        log.warning(
            "Dropped %d text-only record(s) (no 'image' field) from %s "
            "because require_image=True. Mixed image/text batches are not "
            "supported by Gemma4Processor.",
            n_dropped_no_image, jsonl_path,
        )
    log.info("Loaded %d vision-format records from %s", len(records), jsonl_path)
    return records


def summarize_dataset(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Quick stats over a converted dataset (for dry-run/CLI logging)."""
    n_total = 0
    n_with_image = 0
    n_user_turns = 0
    n_asst_turns = 0
    for rec in records:
        n_total += 1
        for msg in rec["messages"]:
            if msg["role"] == "user":
                n_user_turns += 1
                for block in msg["content"]:
                    if block.get("type") == "image":
                        n_with_image += 1
                        break
            elif msg["role"] == "assistant":
                n_asst_turns += 1
    return {
        "records": n_total,
        "with_image": n_with_image,
        "user_turns": n_user_turns,
        "assistant_turns": n_asst_turns,
    }


# ---------------------------------------------------------------------------
# v2 additions: multi-val loader + modality helpers + modality-aware collator
# ---------------------------------------------------------------------------

def load_vision_dataset_dict(
    val_files: Dict[str, str | Path],
    max_samples_per_key: Optional[int] = None,
    prompt_prefixes: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load multiple JSONL files into a named dict of dataset partitions.

    Used by the trainer's multi-eval-dataset feature so the trainer logs
    ``eval_<key>_loss`` per modality:

        eval_dataset = load_vision_dataset_dict({
            "plant":    "data/mix-100k/val_plant.jsonl",
            "nonplant": "data/mix-100k/val_nonplant.jsonl",
            "negative": "data/mix-100k/val_negative.jsonl",
        })

    Records with ``image=None`` are NOT dropped (require_image=False) —
    smoltalk text-only val records are valid in v2 and the trainer routes
    them via ModalityAwareBatchSampler to vision-skip batches.

    Raises ``FileNotFoundError`` if any input path is missing — silently
    skipping would let a typo in a config produce empty eval metrics
    that look fine in the logs.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, path in val_files.items():
        out[key] = load_vision_dataset(
            path,
            max_samples=max_samples_per_key,
            require_image=False,
            prompt_prefixes=prompt_prefixes,
        )
    return out


def record_has_image(record: Dict[str, Any]) -> bool:
    """Returns True iff the converted record has an image content block.

    Designed for ``ModalityAwareBatchSampler(has_image_fn=record_has_image)``.
    Checks the unsloth-format messages list (post-``build_vision_messages``)
    rather than the raw JSONL ``image`` field, so it works on the same
    in-memory representation the trainer sees.
    """
    msgs = record.get("messages") or []
    for msg in msgs:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content") or []:
            if block.get("type") == "image":
                return True
    return False


class TextOnlyChatCollator:
    """Tokenize raw ``messages`` records into a padded LM batch.

    The v2 ModalityAware path needs a text-only collator that matches
    the surface our records carry: each record is the post-
    ``build_vision_messages`` dict with a ``messages`` key, NOT
    pre-tokenized ``input_ids``. HF's stock
    ``DataCollatorForLanguageModeling`` assumes ``input_ids`` are
    already present (it just pads), so it raises::

        ValueError: You should supply an encoding ... that includes
        input_ids, but you provided ['messages', 'length']

    This collator closes the gap. For each batch it:

      1. Applies the chat template to each record's ``messages`` to
         get a flat string per record.
      2. Tokenizes + pads the strings via the wrapped tokenizer.
      3. Builds an LM-style ``labels`` tensor: ``labels = input_ids``
         with padded positions masked to ``-100`` (HF's ignore_index).

    Label policy: we DO NOT mask the user-turn tokens — the model
    trains on the full sequence including the prompt. This matches
    HF's basic causal-LM example and is fine for the small text-only
    buckets (smoltalk + offline_qa, ~15 % of the training mix). For
    a tighter SFT we'd add chat-role-aware masking that zeros the
    user turn loss; deferred until we see whether it materially
    moves eval_<bucket>_loss.

    Truncation: caller passes ``max_length`` (set from
    ``cfg.model.max_seq_length`` in production). Without an explicit
    cap, smoltalk's longer multi-turn records can produce 3-6 K token
    sequences, and the LM head ``[B*T, V]`` bf16 tensor at V=262 K
    blows up to multiple GB per batch — which already OOMed one run.
    """

    def __init__(self, processor, max_length: Optional[int] = 1024):
        self.processor = processor
        # HF processors expose .tokenizer; some are just bare tokenizers.
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length

    def __call__(self, batch):
        if not batch:
            raise ValueError("TextOnlyChatCollator received an empty batch")
        # Step 1: per-record chat template → flat string.
        texts: List[str] = []
        for rec in batch:
            msgs = rec.get("messages")
            if msgs is None:
                raise ValueError(
                    "TextOnlyChatCollator: record missing 'messages' field"
                )
            texts.append(
                self.processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        # Step 2: tokenize + pad. ``return_tensors="pt"`` gives a dict of
        # tensors. ``max_length`` caps each record's sequence (we pass
        # an explicit value rather than relying on the tokenizer's
        # ``model_max_length`` default — Gemma 4's default is 8192,
        # which would produce ``[B, 8192, V]`` outputs and OOM the GPU).
        enc = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        # Step 3: LM labels. labels = input_ids, with pad → -100.
        labels = enc["input_ids"].clone()
        attn = enc["attention_mask"]
        labels[attn == 0] = -100
        enc["labels"] = labels
        return enc


class ModalityAwareCollator:
    """Dispatches a batch to one of two collators based on modality.

    Pairs with ``ModalityAwareBatchSampler`` to implement the v2
    skip-vision-on-text-only optimization. Each batch is guaranteed
    homogeneous by the sampler; the collator picks the appropriate
    underlying collator and asserts homogeneity defensively in case a
    misconfigured trainer leaks a mixed batch.

    Parameters
    ----------
    vision_collator : callable
        Handles batches where every record has an image. Typically
        ``UnslothVisionDataCollator`` from the unsloth package.
    text_collator : callable
        Handles batches where no record has an image. Typically
        ``DataCollatorForLanguageModeling(tokenizer, mlm=False)`` from
        HuggingFace transformers.
    """

    def __init__(self, vision_collator, text_collator):
        self.vision_collator = vision_collator
        self.text_collator = text_collator

    def __call__(self, batch):
        if not batch:
            raise ValueError("ModalityAwareCollator received an empty batch")
        flags = [record_has_image(r) for r in batch]
        if all(flags):
            return self.vision_collator(batch)
        if not any(flags):
            return self.text_collator(batch)
        n_img = sum(flags)
        n_txt = len(flags) - n_img
        raise ValueError(
            f"ModalityAwareCollator got a mixed-modality batch "
            f"({n_img} image-having, {n_txt} text-only). "
            "ModalityAwareBatchSampler must enforce homogeneity — "
            "is the trainer's group_by_length re-mixing batches?"
        )
