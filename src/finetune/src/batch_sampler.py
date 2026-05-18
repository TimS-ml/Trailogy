"""Modality-aware batch sampler for mixed-modality SFT.

Guarantees every yielded batch is homogeneous in image-presence (either
all records have an image, or none do). This is the foundational
invariant that unlocks the v2 "skip vision tower on text-only batches"
optimization: when a batch is all-text, the data collator can emit a
batch without ``pixel_values`` and the model.forward skips the vision
encoder entirely (saving ~30-40 % per-step compute on the text-only
portion of the data).

Without homogeneity, ``Gemma4Processor`` raises::

    ValueError: Received inconsistently sized batches of images (N)
                and text (M)

…the first time a text-only record lands in a batch alongside image
records.

Design:

- At construction, partition dataset indices by modality (image vs text).
- Within each modality, sort indices by length (if ``length_fn`` given)
  so batches retain the ``group_by_length`` benefit.
- Build batches independently per modality, then concatenate the two
  batch lists and shuffle the COMBINED order so a training run sees
  modalities interleaved (not first all-image then all-text).
- Per-epoch reshuffle: the internal epoch counter advances on each
  ``__iter__`` call so the trainer sees different orders across epochs.

Compatibility with HF ``Trainer`` / ``SFTTrainer``: subclass the trainer
and override ``get_train_dataloader`` (or ``_get_train_sampler`` on
transformers 5.x+) to inject this sampler. When the modality-aware
sampler is active, the trainer's own ``group_by_length`` /
``train_sampling_strategy`` MUST be disabled to avoid double-sorting.
"""
from __future__ import annotations

import random
from typing import Callable, Iterator, List, Optional, Sequence


class ModalityAwareBatchSampler:
    """Yield batches that are homogeneous in image-presence.

    Parameters
    ----------
    dataset : Sequence
        Anything supporting ``len(dataset)`` and ``dataset[i]`` -> dict.
        The sampler reads modality + length from each record at init time;
        the actual record contents are touched only via ``has_image_fn``
        and ``length_fn``.
    batch_size : int
        Per-modality batch size. The same value is used for both image
        and text batches.
    has_image_fn : Callable[[dict], bool]
        Returns True iff the record has an image. Conventional check:
        ``lambda r: bool(r.get("image"))``.
    length_fn : Optional[Callable[[dict], int]]
        Returns a length-like score for sorting within a modality.
        When None, no length-sort is applied (insertion order is kept).
    seed : int
        Base RNG seed. Combined with the internal epoch counter for the
        per-epoch reshuffle.
    drop_last : bool
        If True, drop incomplete batches PER MODALITY. The image-modality
        tail and text-modality tail are dropped independently.
    """

    def __init__(
        self,
        dataset: Sequence,
        batch_size: int,
        has_image_fn: Callable[[dict], bool],
        length_fn: Optional[Callable[[dict], int]] = None,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.has_image_fn = has_image_fn
        self.length_fn = length_fn
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        # Partition indices by modality at construction time.
        self._image_indices: List[int] = []
        self._text_indices: List[int] = []
        for i in range(len(dataset)):
            rec = dataset[i]
            if has_image_fn(rec):
                self._image_indices.append(i)
            else:
                self._text_indices.append(i)

    def _build_batches_for_modality(self, indices: List[int]) -> List[List[int]]:
        if not indices:
            return []
        if self.length_fn is not None:
            indices = sorted(indices, key=lambda i: self.length_fn(self.dataset[i]))
        else:
            indices = list(indices)
        batches: List[List[int]] = [
            indices[k : k + self.batch_size]
            for k in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        img_batches = self._build_batches_for_modality(self._image_indices)
        txt_batches = self._build_batches_for_modality(self._text_indices)
        all_batches = img_batches + txt_batches
        rng.shuffle(all_batches)
        self._epoch += 1
        yield from all_batches

    def __len__(self) -> int:
        def _n(indices: List[int]) -> int:
            if not indices:
                return 0
            n_full, rem = divmod(len(indices), self.batch_size)
            if self.drop_last or rem == 0:
                return n_full
            return n_full + 1
        return _n(self._image_indices) + _n(self._text_indices)

    # --- introspection helpers (useful for trainer logging + assertions) ---

    @property
    def n_image_records(self) -> int:
        return len(self._image_indices)

    @property
    def n_text_records(self) -> int:
        return len(self._text_indices)
