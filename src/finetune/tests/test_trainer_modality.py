"""Tests for the modality-aware trainer dataloader factory.

We exercise ``build_modality_aware_dataloader`` directly (with a mock
collator + simple list dataset) since instantiating SFTTrainer requires
unsloth + GPU. The SFTTrainer subclass itself is structurally simple
(it just calls this factory in ``get_train_dataloader``) so unit-testing
the factory covers the load-bearing logic.
"""
from __future__ import annotations

from typing import List

import pytest

from src.trainer_modality import build_modality_aware_dataloader


def _has_image(rec):
    return bool(rec.get("image"))


def _mk_dataset(n_image: int, n_text: int) -> List[dict]:
    out = []
    for i in range(n_image):
        out.append({"image": f"/img/{i}.jpg", "length": 10 + i, "row_id": f"img_{i}"})
    for j in range(n_text):
        out.append({"image": None, "length": 20 + j, "row_id": f"txt_{j}"})
    return out


def _identity_collator(batch):
    """Return the batch verbatim so we can inspect its contents."""
    return list(batch)


def test_dataloader_yields_homogeneous_batches():
    """End-to-end: ModalityAwareBatchSampler + DataLoader + collator
    must yield batches that are each homogeneous in image-presence."""
    dataset = _mk_dataset(n_image=12, n_text=8)
    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        collator=_identity_collator,
        length_fn=lambda r: r["length"],
        seed=0,
        num_workers=0,
        pin_memory=False,
    )
    for batch in loader:
        flags = [_has_image(rec) for rec in batch]
        assert all(flags) or not any(flags), (
            f"mixed batch leaked through dataloader: {flags}"
        )


def test_dataloader_covers_every_record_exactly_once_per_epoch():
    dataset = _mk_dataset(n_image=12, n_text=8)
    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        collator=_identity_collator,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )
    seen_ids: List[str] = []
    for batch in loader:
        seen_ids.extend(rec["row_id"] for rec in batch)
    assert sorted(seen_ids) == sorted(rec["row_id"] for rec in dataset)


def test_dataloader_collator_receives_homogeneous_batches():
    """The collator must never be called with a mixed batch — this is
    the contract that lets ModalityAwareCollator dispatch safely."""
    dataset = _mk_dataset(n_image=8, n_text=8)

    captured_batches: List[list] = []

    def capturing_collator(batch):
        captured_batches.append(list(batch))
        return list(batch)

    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        collator=capturing_collator,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )
    list(loader)  # exhaust the iterator

    for batch in captured_batches:
        flags = [_has_image(rec) for rec in batch]
        assert all(flags) or not any(flags), (
            f"collator received mixed batch: {flags}"
        )


def test_dataloader_works_with_all_text_dataset():
    """No image records at all — must still yield text-only batches
    without crashing."""
    dataset = _mk_dataset(n_image=0, n_text=10)
    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        collator=_identity_collator,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )
    batches = list(loader)
    # Expect 3 batches (4, 4, 2).
    assert len(batches) == 3
    for b in batches:
        assert all(rec["image"] is None for rec in b)


def test_dataloader_works_with_all_image_dataset():
    dataset = _mk_dataset(n_image=10, n_text=0)
    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        collator=_identity_collator,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert all(rec["image"] for rec in b)


# ---------------------------------------------------------------------------
# Integration: dataloader feeds ModalityAwareCollator without mixed batches
# ---------------------------------------------------------------------------

def test_dataloader_feeds_modality_aware_collator_without_assertion_error():
    """The two halves of the v2 pipeline (sampler + collator) must
    compose cleanly: sampler guarantees homogeneity, collator's defensive
    assertion never fires under normal use."""
    from src.data import ModalityAwareCollator

    class _MarkerVision:
        def __init__(self):
            self.n = 0
        def __call__(self, batch):
            self.n += 1
            return {"kind": "vision", "n": len(batch)}

    class _MarkerText:
        def __init__(self):
            self.n = 0
        def __call__(self, batch):
            self.n += 1
            return {"kind": "text", "n": len(batch)}

    # Need to give MarkerVision/Text an attribute that ModalityAwareCollator
    # checks via record_has_image — pass the underlying records through.
    # The simplest path: identity wrap so the dispatch sees the original
    # dataset record dicts.
    class _PassthroughVision:
        def __init__(self):
            self.batches = []
        def __call__(self, batch):
            self.batches.append(("vision", len(batch)))
            return list(batch)

    class _PassthroughText:
        def __init__(self):
            self.batches = []
        def __call__(self, batch):
            self.batches.append(("text", len(batch)))
            return list(batch)

    # Build a mock dataset whose records carry a `messages` field that
    # ModalityAwareCollator's record_has_image can introspect.
    def _img_rec(i):
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": f"/img/{i}.jpg"},
                    {"type": "text", "text": f"Q{i}"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            ],
            "row_id": f"img_{i}",
        }

    def _txt_rec(j):
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"Q{j}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            ],
            "row_id": f"txt_{j}",
        }

    dataset = [_img_rec(i) for i in range(8)] + [_txt_rec(j) for j in range(8)]

    vc = _PassthroughVision()
    tc = _PassthroughText()
    collator = ModalityAwareCollator(vision_collator=vc, text_collator=tc)

    loader = build_modality_aware_dataloader(
        dataset=dataset,
        batch_size=4,
        has_image_fn=lambda r: any(
            b.get("type") == "image"
            for m in r.get("messages", [])
            for b in m.get("content", [])
        ),
        collator=collator,
        seed=0,
        num_workers=0,
        pin_memory=False,
    )

    # Exhaust the loader; any mixed batch would raise inside the
    # ModalityAwareCollator's defensive check.
    batches = list(loader)
    assert len(batches) == 4  # 2 image batches + 2 text batches
    # Both collators were called.
    assert len(vc.batches) == 2  # 8 image records / batch=4
    assert len(tc.batches) == 2  # 8 text records / batch=4
