"""Online data augmentation for vision finetuning.

Applies random image transforms at collation time so each epoch sees a
different augmented version of the same training images.  This is "free"
data expansion — no disk copies needed.

The augmentation is applied to PIL Images AFTER they are loaded from disk
and BEFORE the HF processor normalises them into pixel tensors.

Design constraints:
  * No new dependencies — uses torchvision.transforms.v2 (ships with torch).
  * Deterministic when seeded via ``torch.manual_seed``.
  * Must preserve image size (960x672) — the SigLIP patch grid depends on it.
  * Controlled by a single ``data.augmentation`` bool in the YAML config.
"""

from __future__ import annotations

import logging
from typing import List

from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default augmentation pipeline
# ---------------------------------------------------------------------------


def build_default_augment_transform():
    """Build the default torchvision transform pipeline for plant images.

    All transforms preserve the input image size (no crop/resize).

    Returns a callable that takes a PIL Image and returns a PIL Image.
    """
    import torchvision.transforms.v2 as T

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, fill=0),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        # Gentle perspective warp — simulates different camera angles on
        # the trail.  distortion_scale=0.1 is subtle enough to keep the
        # plant recognisable.
        T.RandomPerspective(distortion_scale=0.1, p=0.3, fill=0),
    ])


# ---------------------------------------------------------------------------
# Module-level registry for augmentation transforms.
#
# UnslothVisionDataCollator uses __slots__, so we cannot set arbitrary
# instance attributes.  Instead we patch the CLASS-level method once and
# use this dict (keyed by ``id(collator_instance)``) to look up whether
# a given instance should augment.
# ---------------------------------------------------------------------------

_AUGMENT_REGISTRY: dict[int, object] = {}  # id(collator) -> transform callable


def _augment_images(transform, images: List[Image.Image]) -> List[Image.Image]:
    """Apply *transform* to each PIL Image in *images*."""
    out: List[Image.Image] = []
    for img in images:
        try:
            aug = transform(img)
            # torchvision v2 transforms on PIL input return PIL output
            # when no ToTensor/ToDtype is in the pipeline.  Belt-and-
            # braces: if we somehow get a tensor back, convert.
            if not isinstance(aug, Image.Image):
                import torchvision.transforms.v2.functional as F
                aug = F.to_pil_image(aug)
            out.append(aug)
        except Exception:
            log.warning(
                "Augmentation failed for image (type=%s, size=%s); "
                "using original",
                type(img).__name__,
                getattr(img, "size", "?"),
                exc_info=True,
            )
            out.append(img)
    return out


_CLASS_PATCHED = False


def _ensure_class_patched(cls) -> None:
    """Patch ``cls._extract_images_videos_for_example`` once at the class
    level so every call through a registered instance applies augmentation.
    """
    global _CLASS_PATCHED
    if _CLASS_PATCHED:
        return

    _orig = cls._extract_images_videos_for_example

    def _patched_extract(self, example, messages):
        images, videos, video_kwargs = _orig(self, example, messages)
        transform = _AUGMENT_REGISTRY.get(id(self))
        if transform is not None and images:
            images = _augment_images(transform, images)
        return images, videos, video_kwargs

    cls._extract_images_videos_for_example = _patched_extract
    _CLASS_PATCHED = True
    log.info(
        "Patched %s._extract_images_videos_for_example for online "
        "augmentation (class-level, registry-gated).",
        cls.__name__,
    )


def enable_augmentation(collator, transform=None) -> None:
    """Enable online augmentation for *collator*.

    Call this AFTER constructing the ``UnslothVisionDataCollator`` and
    BEFORE passing it to ``SFTTrainer``.  The collator instance itself
    is returned unchanged (SFTTrainer receives the same object), but its
    ``_extract_images_videos_for_example`` now applies random transforms
    to each PIL image before it reaches the HF processor.

    Usage::

        collator = UnslothVisionDataCollator(model, tokenizer, ...)
        enable_augmentation(collator)   # done — same object, now augments
        trainer = SFTTrainer(..., data_collator=collator)
    """
    _ensure_class_patched(type(collator))
    _transform = transform or build_default_augment_transform()
    _AUGMENT_REGISTRY[id(collator)] = _transform
    log.info(
        "Online augmentation ENABLED for collator id=%d (%s)",
        id(collator),
        type(_transform).__name__,
    )
