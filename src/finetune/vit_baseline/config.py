"""Typed configuration for the ViT classification baseline.

Loaded from a YAML file plus optional CLI overrides. Pure-Python; no torch /
timm import, so it can be exercised by pytest on a CPU-only box.

Mirrors the structure of ``src/finetune/src/config.py`` (defaults → YAML
overlay → CLI overrides) so the two pipelines feel the same.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    # ImageFolder root holding ``<image_root>/<train_split>/<class>/*.jpg`` and
    # the matching val split. Left empty so no absolute path is tracked: it is
    # resolved at runtime from the ``PLANT_IMAGE_ROOT`` env var (or the
    # ``--image-root`` CLI flag). See ``resolve_image_root``.
    image_root: str = ""
    train_split: str = "train"
    val_split: str = "val"
    image_size: int = 224
    num_workers: int = 8
    # Quick smoke-run knobs: cap images per class / total. None = use all.
    max_images_per_class: Optional[int] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class ModelConfig:
    # timm model name. Default is the SigLIP ViT-B/16 — same encoder family as
    # the Gemma 4 vision tower, so the baseline is a fair stand-in for "what
    # the on-device vision encoder alone can do". Any timm classification
    # backbone works (e.g. ``vit_base_patch16_224.augreg2_in21k_ft_in1k``,
    # ``eva02_base_patch14_448``, ``convnext_base``).
    backbone: str = "vit_base_patch16_siglip_224"
    pretrained: bool = True
    # Linear-probe switch: True freezes the backbone and trains only the head.
    # False (default) is full vision-tower SFT — matches the project intent of
    # tuning the vision tower end-to-end.
    freeze_backbone: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class TrainConfig:
    output_dir: str = "outputs/vit-baseline"
    run_name: Optional[str] = None  # auto: backbone + timestamp
    num_train_epochs: int = 15
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None  # overrides epochs when set (smoke runs)
    learning_rate: float = 3.0e-4
    # Backbone gets a smaller LR than the freshly-initialised head; mirrors the
    # LLaVA-style layered-LR intuition used in the LoRA pipeline. None = use
    # ``learning_rate`` for both.
    backbone_learning_rate: Optional[float] = 3.0e-5
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    # Project policy: bf16 only. NO 8-bit / 4-bit optimizers or weights, so
    # SFT bake-offs stay comparable. Set "float16"/"float32" to override.
    dtype: str = "bfloat16"
    optim: str = "adamw"
    clip_grad_norm: Optional[float] = 1.0
    seed: int = 3407
    log_interval: int = 20      # steps
    eval_interval: int = 1      # epochs
    save_best_only: bool = True
    # "none" | "wandb". Defaults off; the sweep wrapper sets it explicitly.
    report_to: str = "none"
    wandb_project: str = "trailogy-vit-baseline"
    torch_compile: bool = False


@dataclass
class VitBaselineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _merge_into_dataclass(dc: Any, src: Dict[str, Any]) -> None:
    """Deep-merge ``src`` dict into ``dc`` dataclass instance (warn-skip unknown)."""
    if not is_dataclass(dc):
        raise TypeError(f"Expected dataclass instance, got {type(dc).__name__}")
    valid_names = {f.name for f in fields(dc)}
    for key, value in src.items():
        if key not in valid_names:
            log.warning("Unknown config key ignored: %s", key)
            continue
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


# CLI flag → dotted dataclass path.
CLI_OVERRIDE_MAP: Dict[str, tuple[str, ...]] = {
    "image_root": ("data", "image_root"),
    "image_size": ("data", "image_size"),
    "num_workers": ("data", "num_workers"),
    "max_images_per_class": ("data", "max_images_per_class"),
    "max_train_samples": ("data", "max_train_samples"),
    "max_val_samples": ("data", "max_val_samples"),
    "backbone": ("model", "backbone"),
    "pretrained": ("model", "pretrained"),
    "freeze_backbone": ("model", "freeze_backbone"),
    "output_dir": ("train", "output_dir"),
    "run_name": ("train", "run_name"),
    "num_train_epochs": ("train", "num_train_epochs"),
    "per_device_train_batch_size": ("train", "per_device_train_batch_size"),
    "per_device_eval_batch_size": ("train", "per_device_eval_batch_size"),
    "gradient_accumulation_steps": ("train", "gradient_accumulation_steps"),
    "max_steps": ("train", "max_steps"),
    "learning_rate": ("train", "learning_rate"),
    "backbone_learning_rate": ("train", "backbone_learning_rate"),
    "weight_decay": ("train", "weight_decay"),
    "report_to": ("train", "report_to"),
    "seed": ("train", "seed"),
    "torch_compile": ("train", "torch_compile"),
}


def apply_cli_overrides(cfg: VitBaselineConfig, overrides: Dict[str, Any]) -> None:
    """Apply CLI flag overrides (dict of flag→value, None values ignored)."""
    for flag, value in overrides.items():
        if value is None:
            continue
        path = CLI_OVERRIDE_MAP.get(flag)
        if path is None:
            log.warning("CLI override '%s' not mapped; ignored", flag)
            continue
        target: Any = cfg
        for attr in path[:-1]:
            target = getattr(target, attr)
        setattr(target, path[-1], value)


def load_config(
    yaml_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> VitBaselineConfig:
    """Build a VitBaselineConfig: defaults → YAML overlay → CLI overrides."""
    cfg = VitBaselineConfig()
    if yaml_path is not None:
        import yaml  # lazy: pyyaml in requirements, keep import local

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Top-level YAML in {yaml_path} must be a mapping")
        _merge_into_dataclass(cfg, raw)
    if cli_overrides:
        apply_cli_overrides(cfg, cli_overrides)
    return cfg


def resolve_image_root(cfg: VitBaselineConfig) -> str:
    """Resolve the ImageFolder root: config value first, else ``PLANT_IMAGE_ROOT``.

    Kept out of tracked YAML so no host-specific absolute path is committed.
    Raises a clear error if neither is set.
    """
    root = cfg.data.image_root or os.environ.get("PLANT_IMAGE_ROOT", "")
    if not root:
        raise ValueError(
            "image_root is not set. Pass --image-root, set data.image_root in "
            "the YAML, or export PLANT_IMAGE_ROOT=/path/to/imagefolder "
            "(the dir containing the train/ and val/ class subdirs)."
        )
    return root


def validate_config(cfg: VitBaselineConfig) -> List[str]:
    """Return a list of human-readable problems. Empty list == valid.

    Enforces the project's bf16-only policy and basic sanity bounds.
    """
    errs: List[str] = []
    t = cfg.train

    # Project policy: no 8-bit / 4-bit anything in the default SFT bake-offs.
    if t.dtype not in ("bfloat16", "float16", "float32"):
        errs.append(f"train.dtype must be bfloat16/float16/float32, got {t.dtype!r}")
    bad_optim = {"adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "lion_8bit"}
    if t.optim.lower() in bad_optim:
        errs.append(
            f"train.optim={t.optim!r} is an 8-bit optimizer — forbidden by "
            "project policy (keep SFT bake-offs comparable)."
        )
    if t.per_device_train_batch_size < 1:
        errs.append("train.per_device_train_batch_size must be >= 1")
    if t.num_train_epochs < 1 and t.max_steps is None:
        errs.append("set train.num_train_epochs >= 1 or train.max_steps")
    if cfg.data.image_size < 32:
        errs.append("data.image_size looks too small (< 32)")
    if t.learning_rate <= 0:
        errs.append("train.learning_rate must be > 0")
    return errs
