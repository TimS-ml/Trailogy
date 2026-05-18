"""Tests for ModalityAwareBatchSampler.

The sampler guarantees each yielded batch is HOMOGENEOUS in image-presence
(every record either has an image or none do). This unlocks the v2
"skip vision tower on text-only batches" optimization in finetune.py
without breaking the ``len(images) == len(text)`` invariant enforced by
Gemma4Processor + UnslothVisionDataCollator.
"""
from __future__ import annotations

import random
from typing import List

import pytest

from src.batch_sampler import ModalityAwareBatchSampler


def _mk_dataset(n_image: int, n_text: int, length_min: int = 10, length_max: int = 100):
    """Build a simple list-of-dict dataset. ``image`` key None for text-only,
    truthy string for image-having. ``length`` simulates token count."""
    rng = random.Random(42)
    records: List[dict] = []
    for i in range(n_image):
        records.append({
            "image": f"/path/img_{i}.jpg",
            "length": rng.randint(length_min, length_max),
            "row_id": f"img_{i}",
        })
    for j in range(n_text):
        records.append({
            "image": None,
            "length": rng.randint(length_min, length_max),
            "row_id": f"txt_{j}",
        })
    return records


def _has_image(rec):
    return bool(rec.get("image"))


# ---------------------------------------------------------------------------
# Homogeneity invariant — THE critical property
# ---------------------------------------------------------------------------

def test_every_batch_is_homogeneous_in_modality():
    """No batch may contain both image-having and text-only records.

    This is the foundational guarantee — without it, the data collator
    would face mixed-modality batches and Gemma4Processor would raise
    ``ValueError("Received inconsistently sized batches of images and text")``.
    """
    dataset = _mk_dataset(n_image=20, n_text=15)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset,
        batch_size=4,
        has_image_fn=_has_image,
        seed=0,
    )
    for batch in sampler:
        has_img_flags = [_has_image(dataset[i]) for i in batch]
        # All True or all False — never mixed.
        assert all(has_img_flags) or not any(has_img_flags), (
            f"mixed batch: {has_img_flags}"
        )


def test_every_index_appears_exactly_once_per_epoch():
    dataset = _mk_dataset(n_image=20, n_text=15)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset, batch_size=4, has_image_fn=_has_image, seed=0,
    )
    all_indices: List[int] = []
    for batch in sampler:
        all_indices.extend(batch)
    assert sorted(all_indices) == list(range(len(dataset)))


# ---------------------------------------------------------------------------
# Length grouping within a modality
# ---------------------------------------------------------------------------

def test_within_modality_batches_are_length_sorted():
    """When a length_fn is provided, lengths WITHIN each batch should be
    sorted (so each batch has similar-length sequences and padding is
    minimized — same goal as HF's group_by_length).

    Note: batch ORDER across batches is shuffled by design, so the flat
    concatenation across batches is NOT sorted — that's intentional, the
    trainer should see batches interleaved across the dataset rather
    than always processing short sequences before long ones.
    """
    dataset = _mk_dataset(n_image=12, n_text=0)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset,
        batch_size=3,
        has_image_fn=_has_image,
        length_fn=lambda r: r["length"],
        seed=0,
    )
    batches = list(sampler)
    for batch in batches:
        lengths = [dataset[i]["length"] for i in batch]
        assert lengths == sorted(lengths), (
            f"within-batch lengths not sorted: {lengths}"
        )


# ---------------------------------------------------------------------------
# Empty modality cases
# ---------------------------------------------------------------------------

def test_handles_all_image_dataset():
    dataset = _mk_dataset(n_image=10, n_text=0)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset, batch_size=3, has_image_fn=_has_image, seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 4  # 3 + 3 + 3 + 1 (drop_last=False default)
    # All batches must be image-modality.
    for b in batches:
        assert all(_has_image(dataset[i]) for i in b)


def test_handles_all_text_dataset():
    dataset = _mk_dataset(n_image=0, n_text=10)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset, batch_size=4, has_image_fn=_has_image, seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 3  # 4 + 4 + 2
    for b in batches:
        assert not any(_has_image(dataset[i]) for i in b)


# ---------------------------------------------------------------------------
# drop_last behavior
# ---------------------------------------------------------------------------

def test_drop_last_false_keeps_incomplete_final_batch():
    # 10 image records, batch_size=4 -> batches of size 4, 4, 2
    dataset = _mk_dataset(n_image=10, n_text=0)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset, batch_size=4, has_image_fn=_has_image,
        seed=0, drop_last=False,
    )
    batches = list(sampler)
    sizes = sorted(len(b) for b in batches)
    assert sizes == [2, 4, 4]


def test_drop_last_true_drops_incomplete_batches_per_modality():
    """drop_last must apply per-modality so we don't accidentally drop
    text-modality batches just because image-modality has a small tail."""
    # 10 image (full=4, full=4, tail=2 -> drop), 6 text (full=4, tail=2 -> drop)
    dataset = _mk_dataset(n_image=10, n_text=6)
    sampler = ModalityAwareBatchSampler(
        dataset=dataset, batch_size=4, has_image_fn=_has_image,
        seed=0, drop_last=True,
    )
    batches = list(sampler)
    # Expect: 2 image batches (4, 4) + 1 text batch (4) = 3 batches total.
    assert len(batches) == 3
    assert all(len(b) == 4 for b in batches)


# ---------------------------------------------------------------------------
# Determinism + epoch progression
# ---------------------------------------------------------------------------

def test_same_seed_same_batch_order():
    dataset = _mk_dataset(n_image=15, n_text=10)
    s1 = ModalityAwareBatchSampler(dataset, batch_size=4, has_image_fn=_has_image, seed=42)
    s2 = ModalityAwareBatchSampler(dataset, batch_size=4, has_image_fn=_has_image, seed=42)
    assert list(s1) == list(s2)


def test_different_seeds_produce_different_orders():
    dataset = _mk_dataset(n_image=15, n_text=10)
    s1 = ModalityAwareBatchSampler(dataset, batch_size=4, has_image_fn=_has_image, seed=0)
    s2 = ModalityAwareBatchSampler(dataset, batch_size=4, has_image_fn=_has_image, seed=999)
    # Same content but different order. Compare by full batch list equality.
    batches_a = list(s1)
    batches_b = list(s2)
    assert batches_a != batches_b


def test_epochs_yield_different_orders():
    """Sampling across epochs must differ — otherwise the trainer sees
    the same batch order every epoch, which is suboptimal for SGD."""
    dataset = _mk_dataset(n_image=15, n_text=10)
    sampler = ModalityAwareBatchSampler(
        dataset, batch_size=4, has_image_fn=_has_image, seed=7,
    )
    epoch1 = list(sampler)
    epoch2 = list(sampler)
    # Same content (every index exactly once) but different batch order.
    assert sorted(_flat(epoch1)) == sorted(_flat(epoch2))
    assert epoch1 != epoch2, "epoch 2 must reshuffle batch order"


def _flat(batches: List[List[int]]) -> List[int]:
    return [i for b in batches for i in b]


# ---------------------------------------------------------------------------
# __len__ accuracy
# ---------------------------------------------------------------------------

def test_len_matches_iter_count_drop_last_false():
    dataset = _mk_dataset(n_image=10, n_text=7)
    sampler = ModalityAwareBatchSampler(
        dataset, batch_size=4, has_image_fn=_has_image, seed=0, drop_last=False,
    )
    # 10 img -> 3 batches (4,4,2); 7 txt -> 2 batches (4,3); total 5
    assert len(sampler) == 5
    assert len(list(sampler)) == 5


def test_len_matches_iter_count_drop_last_true():
    dataset = _mk_dataset(n_image=10, n_text=7)
    sampler = ModalityAwareBatchSampler(
        dataset, batch_size=4, has_image_fn=_has_image, seed=0, drop_last=True,
    )
    # 10 img -> 2 batches (drop 2); 7 txt -> 1 batch (drop 3); total 3
    assert len(sampler) == 3
    assert len(list(sampler)) == 3
