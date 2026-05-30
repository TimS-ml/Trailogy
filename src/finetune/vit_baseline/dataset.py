"""ImageFolder dataset for the NA-Plantae ViT baseline.

On-disk layout (produced by the iNaturalist prep step)::

    <image_root>/train/<class_slug>/<id>.jpg
    <image_root>/val/<class_slug>/<id>.jpg

The class label is the directory slug (e.g. ``american_beech``). The train
split defines the canonical label space; val classes not present in train are
dropped (a few singleton val-only classes exist in the corpus) so the head's
output dimension is fixed and reproducible.

The scanning / class-map logic is split out as pure functions (filesystem +
stdlib only) so it can be unit-tested without torch / PIL on the path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_class_dirs(split_dir: str | Path) -> List[str]:
    """Sorted list of class-slug subdirectory names under ``split_dir``."""
    split = Path(split_dir)
    if not split.is_dir():
        raise FileNotFoundError(f"split dir not found: {split}")
    return sorted(p.name for p in split.iterdir() if p.is_dir())


def build_class_to_idx(train_dir: str | Path) -> Dict[str, int]:
    """Canonical, deterministic class→index map from the TRAIN split.

    Sorted by slug so the mapping is stable across machines and runs.
    """
    classes = list_class_dirs(train_dir)
    if not classes:
        raise ValueError(f"no class subdirs found under {train_dir}")
    return {name: i for i, name in enumerate(classes)}


def scan_split(
    split_dir: str | Path,
    class_to_idx: Dict[str, int],
    max_images_per_class: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Build a ``(image_path, label_idx)`` list for one split.

    Classes absent from ``class_to_idx`` (val-only singletons) are skipped and
    counted in a single summary log line rather than per-file spam.
    """
    split = Path(split_dir)
    samples: List[Tuple[str, int]] = []
    skipped_classes = 0
    for class_name in sorted(p.name for p in split.iterdir() if p.is_dir()):
        idx = class_to_idx.get(class_name)
        if idx is None:
            skipped_classes += 1
            continue
        files = sorted(
            f for f in (split / class_name).iterdir()
            if f.suffix.lower() in _IMG_EXTS and f.is_file()
        )
        if max_images_per_class is not None:
            files = files[:max_images_per_class]
        for f in files:
            samples.append((str(f), idx))
            if max_samples is not None and len(samples) >= max_samples:
                log.info("scan_split capped at max_samples=%d", max_samples)
                return samples
    if skipped_classes:
        log.info(
            "scan_split(%s): skipped %d class(es) not in the train label space",
            split.name, skipped_classes,
        )
    return samples


# ---------------------------------------------------------------------------
# torch Dataset (lazy torch / PIL import so the helpers above stay CPU-testable)
# ---------------------------------------------------------------------------


def build_datasets(
    image_root: str | Path,
    train_split: str,
    val_split: str,
    train_transform,
    val_transform,
    max_images_per_class: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    """Return ``(train_ds, val_ds, class_to_idx)``.

    Imports torch lazily so ``config.py``-style CPU tests don't pull torch.
    """
    from torch.utils.data import Dataset
    from PIL import Image

    root = Path(image_root)
    train_dir = root / train_split
    val_dir = root / val_split
    class_to_idx = build_class_to_idx(train_dir)

    train_samples = scan_split(
        train_dir, class_to_idx,
        max_images_per_class=max_images_per_class,
        max_samples=max_train_samples,
    )
    val_samples = scan_split(
        val_dir, class_to_idx,
        max_images_per_class=max_images_per_class,
        max_samples=max_val_samples,
    )
    log.info(
        "datasets: %d classes | train=%d imgs | val=%d imgs",
        len(class_to_idx), len(train_samples), len(val_samples),
    )

    class _PlantImageDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, label = self.samples[i]
            with Image.open(path) as im:
                img = im.convert("RGB")
            return self.transform(img), label

    return (
        _PlantImageDataset(train_samples, train_transform),
        _PlantImageDataset(val_samples, val_transform),
        class_to_idx,
    )


def build_transforms(backbone: str, image_size: int, pretrained: bool):
    """Build (train, val) transforms matched to the backbone's data config.

    Uses timm's data pipeline so normalisation stats / interpolation match the
    pretrained weights. Lazy import keeps the module importable without timm.
    """
    import timm
    from timm.data import create_transform, resolve_data_config

    # A throwaway model instance just to read its default data config (mean,
    # std, interpolation). Cheap relative to training.
    tmp = timm.create_model(backbone, pretrained=False, num_classes=0)
    data_cfg = resolve_data_config({"input_size": (3, image_size, image_size)}, model=tmp)
    del tmp

    train_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=True,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        interpolation=data_cfg["interpolation"],
        crop_pct=data_cfg.get("crop_pct", 0.9),
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
    )
    val_tf = create_transform(
        input_size=data_cfg["input_size"],
        is_training=False,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        interpolation=data_cfg["interpolation"],
        crop_pct=data_cfg.get("crop_pct", 0.9),
    )
    return train_tf, val_tf
