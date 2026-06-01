"""Dataset + collator for HF VLM SFT (Qwen3.5 / InternVL3.5).

Reuses the project's JSONL format and `build_vision_messages` converter, then
renders each record through the model's own `AutoProcessor` chat template. The
collator masks the prompt so loss is computed only on the assistant turn.

Label masking with images is subtle: the chat template emits a single image
placeholder token, which the processor then expands to N tokens based on the
actual image resolution. So we cannot count prompt length by tokenizing text
alone. Instead we process the prompt (everything up to the assistant turn) with
the SAME image through the processor — the expansion is identical — and mask
that many leading tokens. Robust for the single-assistant-turn records that
dominate the corpus.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_images(messages: List[Dict[str, Any]]):
    """Replace image content blocks' path strings with loaded PIL images.

    Returns the (mutated) messages and the ordered list of PIL images.
    """
    from PIL import Image

    images = []
    for turn in messages:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image":
                src = block.get("image")
                if isinstance(src, str):
                    with Image.open(src) as im:
                        img = im.convert("RGB")
                    block["image"] = img
                    images.append(img)
                else:
                    images.append(src)
    return messages, images


class VlmSftDataset:
    """Holds raw JSONL records; conversion happens in the collator (needs the
    processor). Kept tiny so DataLoader workers pickle cheaply."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


class VlmSftCollator:
    """Collate JSONL records into a padded model batch with masked labels."""

    def __init__(self, processor, prompt_prefixes: Optional[Dict[str, str]] = None,
                 max_length: int = 1024, image_max_patches: Optional[int] = None,
                 image_max_pixels: Optional[int] = None):
        self.processor = processor
        self.prompt_prefixes = prompt_prefixes
        self.max_length = max_length
        self.tokenizer = getattr(processor, "tokenizer", processor)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        self.pad_token_id = pad if pad is not None else 0
        ip = getattr(processor, "image_processor", None)
        # Cap image tiles for tiling processors (InternVL) so one image stays
        # within max_length. No-op for processors without a tiling knob.
        if image_max_patches is not None and ip is not None and hasattr(ip, "max_patches"):
            ip.max_patches = image_max_patches
            log.info("capped image_processor.max_patches=%d", image_max_patches)
        # Cap pixels for resolution processors (Qwen3-VL's Qwen2VLImageProcessor)
        # → bounded image-token count. The knob is size.longest_edge (max
        # pixels after smart-resize). `size` is a SizeDict (attribute access,
        # not a plain dict). No-op for processors without it.
        if image_max_pixels is not None and ip is not None:
            applied = False
            size = getattr(ip, "size", None)
            if size is not None and hasattr(size, "longest_edge"):
                size.longest_edge = image_max_pixels
                if getattr(size, "shortest_edge", 0) and size.shortest_edge > image_max_pixels:
                    size.shortest_edge = image_max_pixels
                applied = True
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = image_max_pixels
                applied = True
            if applied:
                log.info("capped image max_pixels=%d (Qwen-style)", image_max_pixels)

    def _render_one(self, record: Dict[str, Any]):
        from src.data import build_vision_messages

        messages = build_vision_messages(record, self.prompt_prefixes)["messages"]
        messages, images = _load_images(messages)
        imgs = images if images else None

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
        full = self.processor(text=[full_text], images=imgs, return_tensors="pt")
        prompt = self.processor(text=[prompt_text], images=imgs, return_tensors="pt")
        prompt_len = int(prompt["input_ids"].shape[1])

        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full["labels"] = labels
        return full

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        feats = [self._render_one(r) for r in records]
        max_len = min(self.max_length, max(f["input_ids"].shape[1] for f in feats))
        pad_for = {"input_ids": self.pad_token_id, "attention_mask": 0,
                   "labels": -100, "token_type_ids": 0, "position_ids": 0}

        def _is_token_aligned(key: str) -> bool:
            # A per-token stream has shape [1, seq_len] matching that example's
            # input_ids (e.g. attention_mask, labels, token_type_ids). Image
            # features (pixel_values, image_grid_thw, ...) do not.
            for f in feats:
                if key not in f or not hasattr(f[key], "ndim"):
                    return False
                if f[key].ndim < 2 or f[key].shape[1] != f["input_ids"].shape[1]:
                    return False
            return True

        all_keys = set()
        for f in feats:
            all_keys.update(k for k, v in f.items() if v is not None)

        batch: Dict[str, Any] = {}
        for key in all_keys:
            if _is_token_aligned(key):
                rows = []
                pad_val = pad_for.get(key, 0)
                for f in feats:
                    t = f[key][0][:max_len]
                    if t.shape[0] < max_len:
                        pad = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=0)
                    rows.append(t)
                batch[key] = torch.stack(rows, dim=0)
            else:
                # Ragged image features: concatenate along dim 0 (flattened
                # layout expected by Qwen3-VL and InternVL forwards).
                tensors = [f[key] for f in feats if key in f and f[key] is not None]
                if tensors:
                    batch[key] = torch.cat(tensors, dim=0)
        return batch
