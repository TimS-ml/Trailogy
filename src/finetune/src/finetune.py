#!/usr/bin/env python3
"""Unsloth-based LoRA finetune of Gemma 4 E2B for the hikeCompanion app.

This script mirrors the structure of unsloth's official Gemma-4 E4B
notebook (`gemma4-e4b-unsloth.py`) but:

  * targets E2B (matches the iOS runtime checkpoint),
  * trains on PlantNet-300K image/species pairs (vision frozen),
  * keeps **both** vision and audio towers frozen — vision via the
    `finetune_vision_layers=False` flag exposed by unsloth, audio via the
    explicit `freeze_vision_audio_towers` walker (unsloth has no audio
    flag yet).

Usage
-----
    # Real run (requires NVIDIA GPU + unsloth installed):
    python -m src.finetune --config configs/default.yaml

    # Dry run (Mac / CPU friendly): exercises every step EXCEPT the
    # `FastModel.from_pretrained` call. Verifies that data loads, that
    # the config validates, and that the chat-template formatter would
    # produce coherent text.
    python -m src.finetune --config configs/default.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Local package imports (the script lives in src/, so it is importable as
# `src.finetune` from the repo root).
try:  # `python -m src.finetune` from finetune/
    from .config import (
        FinetuneConfig,
        apply_cli_overrides,
        load_config,
        validate_config,
    )
    from .data import (
        build_vision_messages,
        iter_jsonl,
        load_vision_dataset,
        summarize_dataset,
    )
    from .freeze import (
        assert_frozen,
        freeze_vision_audio_towers,
        freeze_vision_audio_towers_keeping_projector,
        freeze_vision_audio_towers_keeping_projector_and_vision_layers,
    )
    from .projector import (
        PROJECTOR_CANDIDATE_TOKENS,
        ensure_projector_trainable,
        find_projector_module_names,
        find_projector_param_names,
    )
    from .vision_layers import (
        VISION_ENCODER_LAYERS_TOKEN,
        ensure_vision_layers_trainable,
        find_last_n_vision_layer_module_names,
        find_vision_encoder_layer_count,
        find_vision_layer_param_names,
    )
    from .augment import enable_augmentation
except ImportError:  # `python src/finetune.py` from finetune/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (  # type: ignore[no-redef]
        FinetuneConfig,
        apply_cli_overrides,
        load_config,
        validate_config,
    )
    from data import (  # type: ignore[no-redef]
        build_vision_messages,
        iter_jsonl,
        load_vision_dataset,
        summarize_dataset,
    )
    from freeze import (  # type: ignore[no-redef]
        assert_frozen,
        freeze_vision_audio_towers,
        freeze_vision_audio_towers_keeping_projector,
        freeze_vision_audio_towers_keeping_projector_and_vision_layers,
    )
    from projector import (  # type: ignore[no-redef]
        PROJECTOR_CANDIDATE_TOKENS,
        ensure_projector_trainable,
        find_projector_module_names,
        find_projector_param_names,
    )
    from vision_layers import (  # type: ignore[no-redef]
        VISION_ENCODER_LAYERS_TOKEN,
        ensure_vision_layers_trainable,
        find_last_n_vision_layer_module_names,
        find_vision_encoder_layer_count,
        find_vision_layer_param_names,
    )
    from augment import (  # type: ignore[no-redef]
        enable_augmentation,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("finetune")


def _estimate_total_optimizer_steps(
    cfg: FinetuneConfig,
    num_train_records: int,
) -> int:
    if cfg.training.max_steps and cfg.training.max_steps > 0:
        return cfg.training.max_steps
    epochs = cfg.training.num_train_epochs or 1
    effective_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    return max(1, math.ceil(num_train_records * epochs / effective_batch))


def _resolve_warmup_steps(
    cfg: FinetuneConfig,
    num_train_records: int,
) -> int:
    if cfg.training.warmup_ratio is None or cfg.training.warmup_ratio <= 0:
        return cfg.training.warmup_steps
    total_steps = _estimate_total_optimizer_steps(cfg, num_train_records)
    # Match transformers 5.x TrainingArguments.get_warmup_steps(): fractional
    # warmup uses ceil(total_steps * ratio), not round().
    return max(1, math.ceil(total_steps * cfg.training.warmup_ratio))


def _resolve_effective_tf32(torch_module: Any, requested: Optional[bool]) -> Optional[bool]:
    """Apply TF32 backend flags and return what should be passed to SFTConfig.

    Returning None means "omit the SFTConfig.tf32 kwarg". That matters for
    training.tf32=True on CPU/pre-Ampere: passing True through to
    TrainingArguments raises, so unsupported requests must be ignored before
    SFTConfig construction.
    """
    if requested is None:
        return None
    if not requested:
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
        return False
    if not torch_module.cuda.is_available():
        return None
    major, _ = torch_module.cuda.get_device_capability(0)
    if major < 8:
        return None
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    return True


def _is_projector_param_name(name: str) -> bool:
    """Check if a parameter name belongs to the **trainable** projector.

    Uses the same candidate/exclude token lists as ``projector.py`` but
    works on PEFT-wrapped names where the original path may be prefixed
    by ``base_model.model.``, ``modules_to_save.default.``, etc.

    Excludes PEFT's ``original_module`` frozen reference copy — only
    ``modules_to_save.{adapter}`` copies are trainable projector params.
    """
    # PEFT's frozen reference copy — never a trainable projector param.
    if ".original_module." in name:
        return False
    for tok in PROJECTOR_CANDIDATE_TOKENS:
        # Exclude tokens checked first to avoid false positives.
        excluded = False
        for ex in ("vision_tower.patch_embedder.", "vision_tower.encoder.",
                    "vision_tower.pooler.", "embed_audio.", "audio_tower."):
            if ex in name:
                excluded = True
                break
        if excluded:
            continue
        if tok in name:
            return True
    return False


# Regex anchored at a path-component boundary — same semantics as
# vision_layers._LAYER_INDEX_RE / freeze._VISION_LAYER_INDEX_RE /
# export_mlx._VISION_LAYER_IDX_RE. Duplicated to keep finetune.py free
# of cross-module imports for this hot-path check.
import re as _re  # noqa: E402 — local alias avoids shadowing top-level `re`
_VISION_LAYER_IDX_RE = _re.compile(
    r"(?:^|\.)vision_tower\.encoder\.layers\.(\d+)(?:\.|$)"
)


def _is_vision_layer_param_name(
    name: str, tuned_layer_indices: "set[int]"
) -> bool:
    """Check if `name` belongs to one of the tuned vision encoder layers.

    Used by the 3-group optimizer split in real_train when
    ``cfg.lora.tune_last_n_vision_layers > 0``. Order matters: this
    check is strictly MORE specific than ``_is_projector_param_name``
    and should run first when classifying a param into a group.

    Excludes PEFT's ``.original_module.`` frozen reference copy — only
    ``.modules_to_save.{adapter}.`` copies are trainable.
    """
    if ".original_module." in name:
        return False
    if not tuned_layer_indices:
        return False
    m = _VISION_LAYER_IDX_RE.search(name)
    if m is None:
        return False
    return int(m.group(1)) in tuned_layer_indices


def _assert_no_4bit_in_trainable_full_param_modules(model: Any) -> None:
    """Tripwire: every trainable projector / vision-layer param must NOT be 4-bit.

    Why: when the base model was loaded with ``load_in_4bit=True``, every
    ``Linear`` in the model — including ``vision_tower.*`` and
    ``embed_vision.*`` — is a ``bitsandbytes.nn.Linear4bit`` whose
    weight is a ``Params4bit`` (a 4-bit packed, NON-differentiable
    tensor). PEFT's ``modules_to_save`` creates a deep copy of the
    target module, but the copy is STILL ``Linear4bit`` — the projector
    / vision-layer arms of the optimizer would silently no-op (zero
    grad, params never change).

    The config validator emits a WARNING (not a hard reject) when
    ``load_in_4bit=True``, because the QLoRA-baseline comparison runs
    legitimately opt in. This tripwire is the hard guarantee: it runs
    after PEFT wrapping + freeze pass and fails loudly before training
    if any trainable projector / vision-layer param landed as
    ``Params4bit``. That happens when the base_model does NOT skip-
    quantize ``embed_vision`` / ``vision_tower`` / tuned vision-encoder
    layers — point ``base_model`` at a checkpoint whose
    ``quantization_config.llm_int8_skip_modules`` covers them (e.g.
    ``unsloth/gemma-4-E2B-it-unsloth-bnb-4bit``).

    Heuristic: walks ``model.named_parameters()``. For every trainable
    param whose name matches the projector token list or the
    ``vision_tower.encoder.layers.<i>.`` pattern (excluding PEFT's
    ``.original_module.`` frozen reference copy), assert the param is
    not a ``bitsandbytes.nn.Params4bit``. ``bitsandbytes`` is imported
    lazily — if it isn't installed (CPU box, dry-run env), there can
    be no ``Params4bit`` to find, so we log and return.

    Raises ``RuntimeError`` listing offenders if any are found.
    """
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError:
        log.info(
            "bitsandbytes not importable; skipping 4-bit-in-trainable "
            "tripwire (no Params4bit possible in this environment)."
        )
        return

    bad: List[str] = []
    inspected = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".original_module." in name:
            continue
        is_projector = _is_projector_param_name(name)
        is_vision_layer = _VISION_LAYER_IDX_RE.search(name) is not None
        if not (is_projector or is_vision_layer):
            continue
        inspected += 1
        if isinstance(p, bnb.nn.Params4bit):
            bad.append(
                f"{name} (dtype={p.dtype}, type={type(p).__name__})"
            )

    if bad:
        # Cap the printed list — the message is plenty diagnostic with
        # the first 20. The full count is in the lead sentence.
        shown = bad[:20]
        more = "" if len(bad) <= 20 else f"\n  ... and {len(bad) - 20} more"
        raise RuntimeError(
            f"4-bit-in-trainable tripwire FAILED: {len(bad)} trainable "
            "projector / vision-layer param(s) are still bitsandbytes "
            "Params4bit (NON-differentiable). PEFT's modules_to_save "
            "did not produce a full-precision copy because the base "
            "model was loaded in 4-bit. Set model.load_in_4bit=false in "
            "the config and re-run. Offending params:\n  "
            + "\n  ".join(shown) + more
        )

    log.info(
        "4-bit-in-trainable tripwire PASSED: inspected %d trainable "
        "projector/vision-layer param(s); none are bnb.Params4bit.",
        inspected,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsloth LoRA finetune of Gemma 4 E2B for hikeCompanion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Exercise the pipeline without calling FastModel.from_pretrained. "
            "Validates config, loads + converts data, prints stats. Runnable "
            "without a CUDA GPU."
        ),
    )
    # CLI overrides
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument(
        "--load_in_4bit", type=_parse_bool, default=None,
        help=(
            "Opt in to QLoRA (4-bit base + LoRA adapters). Project "
            "default is False (full bf16 LoRA). Only flip to True for "
            "the explicit QLoRA-baseline comparison configs; pair with "
            "a base_model whose quantization_config.llm_int8_skip_modules "
            "keeps embed_vision/vision_tower/audio_tower/embed_audio in "
            "bf16 (e.g. unsloth/gemma-4-E2B-it-unsloth-bnb-4bit) so "
            "modules_to_save projector tuning still backpropagates."
        ),
    )
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None,
                   help="Override training.run_name. Used by scripts/run/train.sh to "
                        "lock in the timestamped run name BEFORE training so tee'd "
                        "logs land at the same outputs/<run_name>/ path that "
                        "finetune.py + evaluate.py write artifacts to.")
    p.add_argument("--report_to", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--val_file", type=str, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )
    # Projector tuning (feature/lora-plus-projector)
    p.add_argument("--tune_projector", type=_parse_bool, default=None,
                   help="Unfreeze the vision-language projector for full-param "
                        "co-training with the language LoRA.")
    p.add_argument("--projector_learning_rate", type=float, default=None,
                   help="LR for the projector when --tune_projector is set. "
                        "Defaults to training.learning_rate / 10.")
    # Vision-tower last-N tuning
    # (feature/lora-plus-projector-plus-vision-tower)
    p.add_argument("--tune_last_n_vision_layers", type=int, default=None,
                   help="Unfreeze the last-N vision encoder layers as full "
                        "params via PEFT modules_to_save. Requires "
                        "--tune_projector. 0 = off.")
    p.add_argument("--vision_layers_learning_rate", type=float, default=None,
                   help="LR for the unfrozen vision encoder layers when "
                        "--tune_last_n_vision_layers > 0. Defaults to "
                        "training.learning_rate / 20.")
    # Data augmentation
    p.add_argument("--augmentation", type=_parse_bool, default=None,
                   help="Enable online random image augmentation at collation "
                        "time (horizontal flip, rotation, color jitter, "
                        "perspective). Default: false.")
    return p.parse_args(argv)


def _parse_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes", "on")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _generate_run_name(config_path: Optional[str]) -> str:
    """Generate run_name from config filename stem + timestamp."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_path:
        stem = Path(config_path).stem  # e.g. "plantnet-50k-r256-a8-lr2e4"
        return f"{stem}_{timestamp}"
    return f"run_{timestamp}"


def build_config(args: argparse.Namespace) -> FinetuneConfig:
    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k not in ("config", "dry_run") and v is not None
    }
    cfg = load_config(args.config, cli_overrides)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            log.error("config: %s", e)
        raise SystemExit(2)

    # Auto-generate run_name if not explicitly set in config/CLI.
    if not cfg.training.run_name:
        cfg.training.run_name = _generate_run_name(args.config)
    # Set output_dir to outputs/{run_name}/ so each run is isolated.
    cfg.training.output_dir = str(Path("outputs") / cfg.training.run_name)

    return cfg


def log_resolved_config(cfg: FinetuneConfig) -> None:
    log.info("=== Resolved configuration ===")
    for section_name in ("model", "lora", "training", "data"):
        section = getattr(cfg, section_name)
        for fname in section.__dataclass_fields__:
            log.info("  %s.%s = %s", section_name, fname, getattr(section, fname))


def load_datasets(
    cfg: FinetuneConfig,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]] | Dict[str, List[Dict[str, Any]]]]:
    """Load train + val records honoring v1 (single val_file) and
    v2 (val_files dict + modality_aware_sampler) paths.

    v1 path (default): single val_file, ``require_image=True`` so
    text-only records are filtered out (Gemma4Processor +
    UnslothVisionDataCollator reject mixed batches).

    v2 path (``modality_aware_sampler=True``): text-only records are
    KEPT (``require_image=False``); ModalityAwareBatchSampler routes
    them into vision-skip batches at train time.

    v2 path (``val_files`` dict set): returns a Dict[str, List[record]]
    so the caller can build a dict of eval datasets and pass it to
    SFTTrainer.eval_dataset for multi-eval-dataset reporting.
    """
    # In v2, text-only records are valid because the modality-aware
    # sampler + collator handle them natively.
    require_image = not cfg.training.modality_aware_sampler

    # v4: camera-state prefixes are dispatched at messages-build time
    # (keyed on image presence) so they're consistently applied across
    # train + val without touching the on-disk JSONL. None = no prefix
    # (v2 default behaviour).
    prompt_prefixes = cfg.data.prompt_prefixes
    if prompt_prefixes:
        log.info(
            "Camera-state prompt prefixes: %s "
            "(camera_on for records with images, camera_off for text-only; "
            "prepended to first user turn)",
            {k: repr(v) for k, v in prompt_prefixes.items()},
        )

    train_records = load_vision_dataset(
        cfg.data.train_file,
        max_samples=cfg.data.max_train_samples,
        require_image=require_image,
        prompt_prefixes=prompt_prefixes,
    )

    # v2 multi-val path: data.val_files is a Dict[str, str].
    if cfg.data.val_files:
        from .data import load_vision_dataset_dict

        # Resolve relative paths against repo root so the same yaml works
        # from any cwd, matching the existing val_file behaviour.
        resolved = {
            k: str(Path(p)) for k, p in cfg.data.val_files.items()
        }
        val_records_dict = load_vision_dataset_dict(
            resolved,
            max_samples_per_key=cfg.data.max_val_samples,
            prompt_prefixes=prompt_prefixes,
        )
        log.info(
            "Loaded %d multi-val partitions: %s",
            len(val_records_dict),
            {k: len(v) for k, v in val_records_dict.items()},
        )
        return train_records, val_records_dict

    # v1 single-val path.
    val_records: List[Dict[str, Any]] = []
    if cfg.data.val_file:
        val_path = Path(cfg.data.val_file)
        if val_path.exists():
            val_records = load_vision_dataset(
                val_path,
                max_samples=cfg.data.max_val_samples,
                require_image=require_image,
                prompt_prefixes=prompt_prefixes,
            )
        else:
            log.warning("Validation file not found at %s — skipping eval", val_path)
    return train_records, val_records


def dry_run(cfg: FinetuneConfig) -> None:
    """Execute every step that does not require CUDA / unsloth."""
    log.info("=== DRY RUN ===")
    log_resolved_config(cfg)

    log.info("--- Loading + converting datasets ---")
    train_records, val_records = load_datasets(cfg)
    train_stats = summarize_dataset(train_records)
    log.info("train: %s", train_stats)
    if val_records:
        if isinstance(val_records, dict):
            for key, records in val_records.items():
                val_stats = summarize_dataset(records)
                log.info("val[%s]: %s", key, val_stats)
        else:
            val_stats = summarize_dataset(val_records)
            log.info("val:   %s", val_stats)

    # Preview the warmup_ratio resolution that real_train will perform.
    # Visible in dry-run so the operator can sanity-check the schedule
    # before consuming GPU time.
    if cfg.training.warmup_ratio is not None and cfg.training.warmup_ratio > 0:
        if cfg.training.max_steps and cfg.training.max_steps > 0:
            _total = _estimate_total_optimizer_steps(cfg, len(train_records))
            _src = "max_steps"
        else:
            _total = _estimate_total_optimizer_steps(cfg, len(train_records))
            _src = "ceil(records * epochs / effective_batch)"
        _wsteps = _resolve_warmup_steps(cfg, len(train_records))
        log.info(
            "--- warmup_ratio resolution ---\n"
            "  warmup_ratio                 = %.4f\n"
            "  total_optimizer_steps        = %d  (from %s)\n"
            "  effective warmup_steps       = %d  (overrides cfg=%d)",
            cfg.training.warmup_ratio, _total, _src,
            _wsteps, cfg.training.warmup_steps,
        )

    # Preview group_by_length / tf32 / save_total_limit toggles so they
    # show up in the dry-run log alongside the other plumbing decisions.
    _flags = []
    if cfg.training.group_by_length:
        _flags.append("group_by_length=True (will auto-populate `length` col)")
    if cfg.training.tf32 is not None:
        _flags.append(f"tf32={cfg.training.tf32}")
    if cfg.training.save_total_limit is not None:
        _flags.append(f"save_total_limit={cfg.training.save_total_limit}")
    if _flags:
        log.info("--- Trainer flags ---\n  %s", "\n  ".join(_flags))

    if train_records:
        sample = train_records[0]
        log.info("--- Sample record (first) ---")
        log.info(json.dumps(sample, indent=2)[:1000])

    if cfg.lora.tune_projector:
        projector_lr = cfg.lora.projector_learning_rate
        if projector_lr is None:
            projector_lr = cfg.training.learning_rate / 10.0
            lr_note = " (auto = training.learning_rate / 10)"
        else:
            lr_note = ""

        n_vision = cfg.lora.tune_last_n_vision_layers
        if n_vision > 0:
            vision_lr = cfg.lora.vision_layers_learning_rate
            if vision_lr is None:
                vision_lr = cfg.training.learning_rate / 20.0
                vision_lr_note = " (auto = training.learning_rate / 20)"
            else:
                vision_lr_note = ""
            log.info(
                "--- Freeze plan (Projector tuning ENABLED + "
                "Vision-tower tuning ENABLED) ---\n"
                "  vision_tower.{patch_embedder,pooler}        -> FROZEN\n"
                "  vision_tower.encoder.layers (last %d, n=%d)  -> "
                "TRAINABLE (full param via modules_to_save)\n"
                "  vision_tower.encoder.layers (earlier)        -> FROZEN\n"
                "  embed_vision.* (projector)                  -> TRAINABLE (full param)\n"
                "  audio_tower / embed_audio                   -> FROZEN\n"
                "  language layers                             -> trainable via LoRA "
                "(r=%d, alpha=%d, dropout=%.3f)\n"
                "  projector learning rate                     -> %.2e%s\n"
                "  vision_layers_learning_rate                  -> %.2e%s\n"
                "  projector candidate tokens (searched at real-train time):\n"
                "    %s\n"
                "  vision-encoder-layer token (searched at real-train time):\n"
                "    %s",
                n_vision, n_vision,
                cfg.lora.r, cfg.lora.lora_alpha, cfg.lora.lora_dropout,
                projector_lr, lr_note,
                vision_lr, vision_lr_note,
                ", ".join(PROJECTOR_CANDIDATE_TOKENS),
                VISION_ENCODER_LAYERS_TOKEN,
            )
        else:
            log.info(
                "--- Freeze plan (Projector tuning ENABLED) ---\n"
                "  vision_tower.{encoder,patch_embedder,pooler} -> FROZEN\n"
                "  embed_vision.* (projector / Gemma4MultimodalEmbedder)\n"
                "                                              -> TRAINABLE (full param)\n"
                "  audio_tower / embed_audio                   -> FROZEN\n"
                "  language layers                             -> trainable via LoRA "
                "(r=%d, alpha=%d, dropout=%.3f)\n"
                "  projector learning rate                     -> %.2e%s\n"
                "  projector candidate tokens (searched at real-train time):\n"
                "    %s",
                cfg.lora.r, cfg.lora.lora_alpha, cfg.lora.lora_dropout,
                projector_lr, lr_note,
                ", ".join(PROJECTOR_CANDIDATE_TOKENS),
            )
    else:
        log.info(
            "--- Freeze plan ---\n"
            "  vision_tower / embed_vision  -> frozen via finetune_vision_layers=False "
            "AND post-LoRA freeze pass\n"
            "  audio_tower  / embed_audio   -> frozen via post-LoRA freeze pass "
            "(unsloth has no audio flag yet)\n"
            "  language layers              -> trainable via LoRA "
            "(r=%d, alpha=%d, dropout=%.3f)",
            cfg.lora.r, cfg.lora.lora_alpha, cfg.lora.lora_dropout,
        )

    log.info(
        "Online data augmentation: %s",
        "ENABLED" if cfg.data.augmentation else "disabled (default)",
    )
    log.info("Dry run complete. Skipped: FastModel.from_pretrained, training loop.")


# ---------------------------------------------------------------------------
# Real training (CUDA only — imports unsloth lazily)
# ---------------------------------------------------------------------------


def _patch_sft_trainer_entropy(cls):
    """Wrap SFTTrainer.compute_loss to skip the entropy metric.

    Unsloth wraps ``outputs.logits`` as a lazy callable, which breaks
    trl's ``entropy_from_logits`` (it receives a function instead of a
    tensor).  We catch the TypeError and log NaN entropy so training can
    proceed.
    """
    _orig = cls.compute_loss

    def _patched(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            return _orig(self, model, inputs,
                         return_outputs=return_outputs,
                         num_items_in_batch=num_items_in_batch)
        except TypeError as exc:
            if "not subscriptable" in str(exc):
                # Fall back to the parent Trainer's compute_loss (which
                # skips entropy entirely).
                from transformers import Trainer
                return Trainer.compute_loss(
                    self, model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            raise

    cls.compute_loss = _patched
    return cls


def _patch_peft_embedder_wrappers(model) -> None:
    """Fix PEFT AuxiliaryTrainingWrapper forward() for Gemma 4 embedders.

    PEFT's ``AuxiliaryTrainingWrapper.forward(self, x, *args, **kwargs)``
    requires ``x`` as a positional argument. But HF's Gemma4Model calls
    ``self.embed_vision(inputs_embeds=last_hidden_state)`` with a keyword
    argument only. When the embedder is wrapped by ``modules_to_save``,
    the ``x`` parameter is never bound and the call raises::

        TypeError: AuxiliaryTrainingWrapper.forward() missing 1 required
                   positional argument: 'x'

    This function walks the model, finds any PEFT-wrapped module whose
    underlying original is a ``Gemma4MultimodalEmbedder``, and replaces
    its ``forward`` with a version that maps ``inputs_embeds=`` to the
    positional ``x``.
    """
    from peft.utils.other import AuxiliaryTrainingWrapper

    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, AuxiliaryTrainingWrapper):
            continue
        # Check if the wrapped module is a Gemma4MultimodalEmbedder
        orig = getattr(module, "original_module", None)
        if orig is None:
            continue
        cls_name = type(orig).__name__
        if "MultimodalEmbedder" not in cls_name:
            continue

        # Replace forward: accept inputs_embeds as keyword, forward as
        # positional to the original PEFT forward.
        _orig_forward = module.forward

        def _patched_forward(*args, inputs_embeds=None, _orig=_orig_forward, **kwargs):
            if inputs_embeds is not None and not args:
                return _orig(inputs_embeds, **kwargs)
            return _orig(*args, **kwargs)

        module.forward = _patched_forward
        patched += 1
        log.info(
            "Patched PEFT wrapper forward for %s (underlying: %s)",
            name, cls_name,
        )
    if patched:
        log.info("Patched %d PEFT embedder wrapper(s) for keyword-arg compat", patched)
    else:
        log.warning(
            "tune_projector=True but no PEFT AuxiliaryTrainingWrapper found "
            "for MultimodalEmbedder modules. The forward-signature patch was "
            "not applied — this may indicate PEFT did not wrap the projector."
        )


def _assert_save_matches_in_memory_state(
    adapter_dir: Path,
    in_memory_state: Dict[str, "torch.Tensor"],  # type: ignore[name-defined]
) -> None:
    """Tripwire: after save_pretrained, on-disk adapter must equal in-memory state.

    Targets the AGENTS.md "orphan tensors" bug class — a model trained on
    one transformers/PEFT version whose adapter contains tensor keys that
    don't appear in the current in-memory PEFT state (or vice versa). The
    diff is computed bytewise against ``adapter_model.safetensors`` (or
    its sharded equivalents) and the saved state dict PEFT itself would
    write at this moment.

    Raises ``RuntimeError`` with an actionable message referencing
    AGENTS.md whenever any key or any byte disagrees.
    """
    try:  # `python -m src.finetune`
        from .save_reload_check import (
            assert_no_diff,
            diff_in_memory_vs_disk,
        )
    except ImportError:  # `python src/finetune.py`
        from save_reload_check import (  # type: ignore[no-redef]
            assert_no_diff,
            diff_in_memory_vs_disk,
        )

    diff = diff_in_memory_vs_disk(in_memory_state, adapter_dir)
    assert_no_diff(diff, label="save (memory vs disk)")


def _assert_projector_tensors_present_if_tuned(
    adapter_dir: Path,
    tune_projector: bool,
) -> None:
    """Tripwire: fail fast if projector tensors are missing from the saved adapter.

    When ``tune_projector=True``, PEFT's ``save_pretrained`` should save the
    projector weights via the ``ModulesToSaveWrapper``. If the wrapper was
    silently dropped (unsloth regression, PEFT version mismatch, or our
    ``modules_to_save`` list was wrong), the adapter directory contains only
    LoRA tensors and no projector weights. Shipping this adapter would merge
    into a model with unchanged base projector — the entire projector-tuning
    run would be wasted.

    Reads only safetensors headers (no tensor data loaded).
    """
    if not tune_projector:
        return

    import json
    import struct

    def _safetensor_keys(path: Path) -> list[str]:
        with path.open("rb") as f:
            raw = f.read(8)
            header_len = struct.unpack("<Q", raw)[0]
            header = json.loads(f.read(header_len))
        return [k for k in header if k != "__metadata__"]

    has_projector = False
    for sf in sorted(adapter_dir.glob("*.safetensors")):
        for key in _safetensor_keys(sf):
            if _is_projector_param_name(key):
                has_projector = True
                log.info("Projector tensor found in adapter: %s", key)
                break
        if has_projector:
            break

    if not has_projector:
        raise RuntimeError(
            "tune_projector=True at training time, but no projector tensors "
            f"found in adapter directory {adapter_dir}. PEFT silently dropped "
            "modules_to_save — the projector was NOT saved. This training run "
            "is wasted; do not ship this adapter. Check that "
            "modules_to_save=['embed_vision'] was correctly passed to "
            "FastModel.get_peft_model and that PEFT version supports it."
        )
    log.info("Projector save tripwire passed: projector tensors present in adapter.")


def _assert_vision_layer_tensors_present_if_tuned(
    adapter_dir: Path,
    tuned_vision_layer_indices: Sequence[int],
) -> None:
    """Tripwire: fail fast if vision-encoder-layer tensors are missing from
    the saved adapter (parallel to the projector save tripwire).

    When ``cfg.lora.tune_last_n_vision_layers > 0``, PEFT's
    ``save_pretrained`` should save the last-N vision encoder layer
    weights via the ``ModulesToSaveWrapper`` (one per index). If the
    wrapper was silently dropped, the adapter directory contains LoRA
    (and possibly projector) tensors but no vision-layer tensors —
    shipping it would merge into a model with unchanged base
    vision-encoder weights, wasting the run.

    The check requires AT LEAST ONE tensor for EACH tuned layer index
    to be present.

    Reads only safetensors headers (no tensor data loaded).
    """
    if not tuned_vision_layer_indices:
        return

    import json
    import struct

    def _safetensor_keys(path: Path) -> list[str]:
        with path.open("rb") as f:
            raw = f.read(8)
            header_len = struct.unpack("<Q", raw)[0]
            header = json.loads(f.read(header_len))
        return [k for k in header if k != "__metadata__"]

    tuned_idx_set: set[int] = set(tuned_vision_layer_indices)
    found_idx: set[int] = set()
    for sf in sorted(adapter_dir.glob("*.safetensors")):
        for key in _safetensor_keys(sf):
            if ".original_module." in key:
                continue
            m = _VISION_LAYER_IDX_RE.search(key)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx in tuned_idx_set:
                found_idx.add(idx)
        if found_idx == tuned_idx_set:
            break

    missing = sorted(tuned_idx_set - found_idx)
    if missing:
        raise RuntimeError(
            f"tune_last_n_vision_layers={len(tuned_idx_set)} at training "
            f"time (indices {sorted(tuned_idx_set)}), but the saved adapter "
            f"directory {adapter_dir} is missing tensors for vision encoder "
            f"layer(s) {missing}. PEFT silently dropped modules_to_save for "
            f"these layers — the vision tower was NOT actually trained. This "
            f"training run is wasted; do not ship this adapter. Check that "
            f"modules_to_save included vision_tower.encoder.layers.{{i}} "
            f"entries when calling FastModel.get_peft_model."
        )
    log.info(
        "Vision-layer save tripwire passed: tensors present for all %d "
        "tuned indices %s.",
        len(tuned_idx_set), sorted(tuned_idx_set),
    )


def _snapshot_trainable_params(
    model,
    tuned_vision_layer_indices: Sequence[int] = (),
) -> Dict[str, set]:
    """Snapshot the names of currently-trainable params, grouped.

    Returns a dict with keys ``"vision"``, ``"projector"``, ``"lora_other"``,
    each mapped to a set of parameter names. Used by
    ``_assert_trainable_set_unchanged_post_train`` to catch the case
    where a checkpoint resume (or any other code path) silently flips
    ``requires_grad=False`` on params that were trainable before
    training started.

    Same 3-way classification order as the optimizer builder: vision
    layers first (most specific), then projector, then everything else.
    Frozen params are not included.
    """
    tuned_idx_set: set = set(tuned_vision_layer_indices)
    groups: Dict[str, set] = {"vision": set(), "projector": set(), "lora_other": set()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if tuned_idx_set and _is_vision_layer_param_name(name, tuned_idx_set):
            groups["vision"].add(name)
        elif _is_projector_param_name(name):
            groups["projector"].add(name)
        else:
            groups["lora_other"].add(name)
    return groups


def _assert_trainable_set_unchanged_post_train(
    pre: Dict[str, set],
    post: Dict[str, set],
) -> None:
    """Raise if the post-train trainable-param set differs from the pre-train set.

    Defense-in-depth for ``--resume_from_checkpoint``: HF Trainer's
    resume path reloads PEFT state via a different code branch than
    the initial training launch. If the resume silently fails to
    re-apply ``modules_to_save`` (PEFT/unsloth regression), the
    projector and/or tuned vision layers can become frozen mid-run.
    The save-side tripwire in ``_assert_projector_tensors_present_if_tuned``
    catches the symptom at save time, but only after wasted steps.

    This pre/post comparison catches the same regression immediately
    after ``trainer.train()`` returns, so the operator learns at the
    moment of corruption rather than at adapter save.

    Both ``pre`` and ``post`` are the return values of
    ``_snapshot_trainable_params``.
    """
    for group in ("vision", "projector", "lora_other"):
        pre_names = pre.get(group, set())
        post_names = post.get(group, set())
        lost = pre_names - post_names
        gained = post_names - pre_names
        if lost or gained:
            lost_sample = ", ".join(sorted(lost)[:3])
            gained_sample = ", ".join(sorted(gained)[:3])
            raise RuntimeError(
                f"Trainable-param set for group '{group}' changed during "
                f"training. Before: {len(pre_names)} params. After: "
                f"{len(post_names)} params. "
                f"Lost ({len(lost)}): {lost_sample}"
                + (" ..." if len(lost) > 3 else "")
                + f". Gained ({len(gained)}): {gained_sample}"
                + (" ..." if len(gained) > 3 else "")
                + ". This typically indicates a resume-from-checkpoint "
                "regression where PEFT silently dropped modules_to_save "
                "for the projector or vision layers. The training results "
                "from this run are NOT trustworthy — investigate before "
                "shipping the adapter."
            )


def real_train(cfg: FinetuneConfig, resume_from_checkpoint: Optional[str] = None) -> None:
    """Full unsloth-based training run. Requires CUDA."""
    # Lazy imports — keep the dry-run / unit-test path importable on Mac.
    import torch  # noqa: F401  (presence check)

    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available. Real training requires an NVIDIA GPU. "
            "Use --dry-run on Mac to validate the pipeline."
        )
        raise SystemExit(1)

    import random

    import numpy as np

    # v3: Unsloth 2024.11+ drops ``outputs.logits`` from the model
    # forward by default ("logits are empty" NotImplementedError) to
    # save memory in CE-only training. The KL penalty needs the raw
    # logits to compute KL(student ‖ teacher); the L2 anchor does not.
    # When KL is enabled we set ``UNSLOTH_RETURN_LOGITS=1`` BEFORE
    # importing unsloth so the env var is read at import time. When
    # KL is disabled we LEAVE the env unset so unsloth keeps its
    # ``EmptyLogits`` memory-saver — at bs=16, V=262K, T=1024 that's
    # ~8 GB per step saved.
    if cfg.regularization.kl_enabled:
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    # Wandb env priming. HF Trainer auto-integrates wandb whenever
    # ``report_to`` contains ``"wandb"`` AND the ``wandb`` package is
    # importable; no extra code path is needed. We only set defaults
    # for env vars the operator hasn't already pinned:
    #
    #   WANDB_PROJECT — defaults to "hikecompanion-finetune" so all
    #                   runs land in one project unless overridden.
    #   WANDB_MODE    — left untouched. On an offline 4090 box set
    #                   ``WANDB_MODE=offline`` before launching, then
    #                   ``wandb sync outputs/<run_name>/wandb/`` later.
    #
    # The wandb run name is taken from ``cfg.training.run_name`` (which
    # is the basename of ``output_dir`` by construction). Override at
    # launch time with ``--run_name <name>``.
    if "wandb" in (cfg.training.report_to or ""):
        os.environ.setdefault("WANDB_PROJECT", "hikecompanion-finetune")
        log.info(
            "wandb enabled: project=%s run=%s mode=%s "
            "(set WANDB_MODE=offline on air-gapped boxes; sync later)",
            os.environ.get("WANDB_PROJECT"),
            cfg.training.run_name,
            os.environ.get("WANDB_MODE", "online"),
        )

    from unsloth import FastModel  # type: ignore[import-not-found]
    from unsloth.chat_templates import (  # type: ignore[import-not-found]
        get_chat_template,
    )
    from unsloth.trainer import (  # type: ignore[import-not-found]
        UnslothVisionDataCollator,
    )
    from datasets import Dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    # Global seed: SFTConfig.seed only controls the HF Trainer's internal
    # shuffling / sampler. torch, numpy, and stdlib random are NOT seeded
    # by HF Trainer.set_seed until training_loop entry. Any randomness
    # between here and there (dropout init, data augmentation, DataLoader
    # worker seeds) uses unseeded defaults. Set them explicitly.
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Global seed set to %d (torch + numpy + random)", seed)

    # TF32 toggle: HF's TrainingArguments.tf32 also flips these flags when
    # the SFTConfig is constructed, but we apply it BEFORE
    # FastModel.from_pretrained so the model's eager init / unsloth
    # compile-time kernels also see the new matmul precision. On Ampere+
    # this is a small free perf win for FP32 ops (optimizer math, residual
    # FP32 paths); bf16 matmul is unaffected. None = leave defaults.
    _effective_tf32 = _resolve_effective_tf32(torch, cfg.training.tf32)
    if cfg.training.tf32 is True and _effective_tf32 is True:
        log.info("TF32 matmul enabled (Ampere+ tensor cores).")
    elif cfg.training.tf32 is True:
        log.warning("training.tf32=True requested but unsupported here — ignored.")
    elif cfg.training.tf32 is False:
        log.info("TF32 matmul explicitly disabled.")

    # Patch before instantiation so the first training_step doesn't crash.
    _patch_sft_trainer_entropy(SFTTrainer)

    log_resolved_config(cfg)

    # 1. Load the base model.
    log.info("Loading base model: %s", cfg.model.base_model)
    # cfg.model.dtype is stored as an Optional[str] ("bfloat16", "float16",
    # "float32", or None). FastModel.from_pretrained asserts the value is a
    # torch.dtype object (or None for auto-detect), so convert here.
    _DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    _dtype_str = cfg.model.dtype
    if _dtype_str is not None:
        if _dtype_str not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype {_dtype_str!r}. Must be one of "
                f"{list(_DTYPE_MAP)} or null (auto-detect)."
            )
        _torch_dtype = _DTYPE_MAP[_dtype_str]
    else:
        _torch_dtype = None
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.base_model,
        dtype=_torch_dtype,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        full_finetuning=cfg.model.full_finetuning,
    )

    # 1.5 Projector identification (always run for visibility, even when
    #     tune_projector is False — logs the projector layout so the GPU
    #     operator can confirm the model has the expected structure).
    projector_param_names: List[str] = []
    projector_module_names: List[str] = []
    if cfg.lora.tune_projector:
        try:
            projector_param_names = find_projector_param_names(model)
            projector_module_names = find_projector_module_names(model)
        except RuntimeError as exc:
            log.error(
                "tune_projector=True but no projector params/modules found in "
                "the loaded base model. Aborting. Error: %s", exc,
            )
            raise SystemExit(1)
        log.info(
            "Projector tuning ENABLED. Identified %d projector param(s) under "
            "%d module name(s) for full-param training: modules=%s",
            len(projector_param_names),
            len(projector_module_names),
            projector_module_names,
        )

    # 1.6 Vision-tower last-N layer identification
    #     (feature/lora-plus-projector-plus-vision-tower).
    vision_layer_param_names: List[str] = []
    vision_layer_module_names: List[str] = []
    tuned_vision_layer_indices: List[int] = []
    if cfg.lora.tune_last_n_vision_layers > 0:
        try:
            total_vision_layers = find_vision_encoder_layer_count(model)
            n = cfg.lora.tune_last_n_vision_layers
            if n > total_vision_layers:
                log.error(
                    "tune_last_n_vision_layers=%d exceeds the model's "
                    "vision encoder layer count (%d). Aborting.",
                    n, total_vision_layers,
                )
                raise SystemExit(1)
            tuned_vision_layer_indices = list(
                range(total_vision_layers - n, total_vision_layers)
            )
            vision_layer_module_names = find_last_n_vision_layer_module_names(
                model, n
            )
            vision_layer_param_names = find_vision_layer_param_names(model, n)
        except (RuntimeError, ValueError) as exc:
            log.error(
                "tune_last_n_vision_layers=%d but vision encoder layers could "
                "not be located. Aborting. Error: %s",
                cfg.lora.tune_last_n_vision_layers, exc,
            )
            raise SystemExit(1)
        log.info(
            "Vision-tower tuning ENABLED. Total vision encoder layers: %d. "
            "Unfreezing last %d (indices %s) as full params via "
            "modules_to_save. Modules: %s. Params: %d.",
            total_vision_layers, n,
            tuned_vision_layer_indices, vision_layer_module_names,
            len(vision_layer_param_names),
        )

    # 2. Apply LoRA. finetune_vision_layers=False keeps vision frozen at the
    #    LoRA-injection level (no adapters in vision tower). When
    #    tune_projector=True, we additionally pass modules_to_save so PEFT
    #    keeps the projector as full trainable params (not LoRA-adapted).
    adapter_label = "DoRA" if cfg.lora.use_dora else "LoRA"
    log.info("Attaching %s adapters (r=%d, alpha=%d)", adapter_label, cfg.lora.r, cfg.lora.lora_alpha)
    peft_kwargs: Dict[str, Any] = dict(
        finetune_vision_layers=cfg.lora.finetune_vision_layers,    # False
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        random_state=cfg.lora.random_state,
        use_dora=cfg.lora.use_dora,
    )
    if cfg.lora.tune_projector:
        # Stack projector + (optional) vision-layer modules under
        # modules_to_save. PEFT matches each entry as a suffix of the
        # module's full path — both 'embed_vision' and
        # 'vision_tower.encoder.layers.14' resolve unambiguously under
        # any wrapping depth.
        peft_kwargs["modules_to_save"] = (
            projector_module_names + vision_layer_module_names
        )
    model = FastModel.get_peft_model(model, **peft_kwargs)

    # 2.5 Fix PEFT wrapper forward signature for Gemma 4 multimodal
    #     embedders. PEFT's AuxiliaryTrainingWrapper.forward(self, x, ...)
    #     requires `x` as a positional arg, but HF's Gemma4Model calls
    #     self.embed_vision(inputs_embeds=last_hidden_state) with keyword
    #     only. Monkey-patch any wrapped embedder to map the keyword arg
    #     to the positional slot.
    if cfg.lora.tune_projector:
        _patch_peft_embedder_wrappers(model)

    # 3. Manual freeze pass: belt-and-braces for vision, ESSENTIAL for audio
    #    (unsloth has no `finetune_audio_layers` flag yet). When projector
    #    tuning is on, the freeze pass keeps the projector params trainable.
    #    When vision-tower tuning is also on, it additionally keeps the
    #    last-N vision encoder layers trainable.
    if cfg.lora.tune_projector:
        if cfg.lora.tune_last_n_vision_layers > 0:
            report = freeze_vision_audio_towers_keeping_projector_and_vision_layers(
                model,
                projector_param_names=projector_param_names,
                tuned_vision_layer_indices=tuned_vision_layer_indices,
            )
        else:
            report = freeze_vision_audio_towers_keeping_projector(
                model, projector_param_names=projector_param_names
            )
        # Belt-and-braces: if FastModel.get_peft_model silently dropped
        # modules_to_save (an unsloth regression we want to detect), the
        # projector / vision-layer params are still requires_grad=False
        # after the freeze pass. Manually re-enable them and log a WARNING.
        flipped = ensure_projector_trainable(model, projector_param_names)
        if flipped:
            log.warning(
                "FastModel.get_peft_model did not honor modules_to_save "
                "for the projector; manually re-enabled requires_grad on "
                "%d projector param(s). Unsloth regression — file upstream.",
                flipped,
            )
        if cfg.lora.tune_last_n_vision_layers > 0:
            vision_flipped = ensure_vision_layers_trainable(
                model, set(tuned_vision_layer_indices)
            )
            if vision_flipped:
                log.warning(
                    "FastModel.get_peft_model did not honor modules_to_save "
                    "for the vision encoder layers; manually re-enabled "
                    "requires_grad on %d vision-layer param(s). Unsloth "
                    "regression — file upstream.",
                    vision_flipped,
                )
        log.info("%s", report)
        assert_frozen(
            model,
            allowlist=projector_param_names,
            tuned_vision_layer_indices=tuned_vision_layer_indices,
        )
        # Last-line tripwire: confirm every trainable projector /
        # vision-layer param is differentiable (not bnb.Params4bit).
        # Under bf16 LoRA (project default) this is a sanity check.
        # Under QLoRA (load_in_4bit=true, used for the explicit baseline-2
        # comparison runs), this is THE guarantee that the chosen
        # base_model actually skip-quantizes the projector / vision
        # tower — fails fast if not.
        _assert_no_4bit_in_trainable_full_param_modules(model)
    else:
        report = freeze_vision_audio_towers(model)
        log.info("%s", report)
        assert_frozen(model)  # tripwire — fail fast if anything slipped through

    # 4. Apply the gemma-4 chat template.
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    # 5. Build datasets in the {messages: [...]} format.
    train_records, val_records = load_datasets(cfg)
    if not train_records:
        log.error("No training records loaded — aborting")
        raise SystemExit(1)
    train_ds = Dataset.from_list(train_records)

    # val_records is either:
    #   v1: List[record] -> build a single Dataset
    #   v2: Dict[str, List[record]] -> build a Dict[str, Dataset] for
    #       SFTTrainer's multi-eval-dataset feature.
    val_ds: Any
    if isinstance(val_records, dict):
        val_ds = {
            key: Dataset.from_list(recs) if recs else None
            for key, recs in val_records.items()
        }
        # Drop empty partitions so SFTTrainer doesn't log empty eval.
        val_ds = {k: v for k, v in val_ds.items() if v is not None}
        if not val_ds:
            val_ds = None
            log.warning("All val_files partitions empty — eval disabled.")
        else:
            log.info(
                "Multi-eval partitions: %s",
                {k: len(v) for k, v in val_ds.items()},
            )
    else:
        val_ds = Dataset.from_list(val_records) if val_records else None

    # group_by_length: HF's LengthGroupedSampler requires either a
    # `length` column or `input_ids` in each row. Our records carry the
    # unsloth `messages` format with raw text blocks and a path string
    # for the image, so neither is present and the sampler would crash
    # with a KeyError at the start of training.
    #
    # Workaround: precompute an approximate length per record from the
    # total character count of all text blocks in `messages`. Every
    # image is processed at 960x672 -> 2520 vision tokens -> 280 pooled
    # tokens (a constant), so it doesn't contribute to length variance
    # across the batch — only the text part matters for pad-waste
    # reduction. Char count is a fine proxy for token count for
    # ordering purposes (Pearson > 0.95 in our corpora).
    if cfg.training.group_by_length:
        def _approx_length(rec: Dict[str, Any]) -> Dict[str, int]:
            total = 0
            for msg in rec.get("messages", []) or []:
                content = msg.get("content")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total += len(block.get("text", ""))
            return {"length": total}

        train_ds = train_ds.map(_approx_length)
        if isinstance(val_ds, dict):
            val_ds = {k: v.map(_approx_length) for k, v in val_ds.items()}
        elif val_ds is not None:
            val_ds = val_ds.map(_approx_length)
        log.info(
            "group_by_length enabled: populated `length` column "
            "(train min/median/max = %d / %d / %d chars)",
            min(train_ds["length"]),
            sorted(train_ds["length"])[len(train_ds) // 2],
            max(train_ds["length"]),
        )

    # v2: modality_aware_sampler also benefits from the `length` column
    # (within-modality length sort). Populate it unconditionally when
    # the flag is set, even if group_by_length is False — the sampler
    # uses length_fn=lambda r: r.get("length", 0) and falls back to 0
    # if absent, but actually populating the column gives proper sort.
    if cfg.training.modality_aware_sampler and not cfg.training.group_by_length:
        def _approx_length_v2(rec: Dict[str, Any]) -> Dict[str, int]:
            total = 0
            for msg in rec.get("messages", []) or []:
                content = msg.get("content")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total += len(block.get("text", ""))
            return {"length": total}

        train_ds = train_ds.map(_approx_length_v2)
        if isinstance(val_ds, dict):
            val_ds = {k: v.map(_approx_length_v2) for k, v in val_ds.items()}
        elif val_ds is not None:
            val_ds = val_ds.map(_approx_length_v2)
        log.info(
            "modality_aware_sampler enabled: populated `length` column "
            "on train_ds (and val_ds) for within-modality length sort"
        )

    # 6. SFTTrainer with the vision data collator.
    # Enable train_on_responses_only at the collator level so loss is
    # computed only on assistant turns, not on the user prompt template.
    # Without this, the model memorizes prompt patterns like "What plant
    # is this?" and parrots species-ID answers to any similarly-phrased
    # question. The collator's native support works on already-tokenized
    # batches — no dataset pre-tokenization needed (unlike the top-level
    # train_on_responses_only() function which chokes on multi-modal
    # content blocks).
    #
    # Gemma 4 chat template markers:
    #   instruction: "<|turn>user\n"
    #   response:    "<|turn>model\n"
    collator = UnslothVisionDataCollator(
        model, tokenizer,
        train_on_responses_only=True,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    # 6b. Online data augmentation (opt-in via data.augmentation=true).
    if cfg.data.augmentation:
        enable_augmentation(collator)
        log.info("Online data augmentation ENABLED")
    else:
        log.info("Online data augmentation disabled (default)")

    # 6c. v2: wrap collator with ModalityAwareCollator so text-only
    # batches skip the vision tower. The underlying text collator is
    # TextOnlyChatCollator (our own — see src/data.py), which applies
    # the chat template + tokenizes + pads + builds LM labels in one
    # pass. Originally we wired HF's DataCollatorForLanguageModeling
    # but that expects records to ALREADY carry input_ids (e.g. from
    # a dataset.map(tokenize) step), while our records carry raw
    # ``messages`` dicts — running real training on text-only batches
    # failed with "you provided ['messages', 'length']". The custom
    # collator closes that gap.
    if cfg.training.modality_aware_sampler:
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator  # type: ignore[import-not-found]
        from .data import ModalityAwareCollator, TextOnlyChatCollator

        # In FastModel's API the second return is named "tokenizer" but
        # is actually a multimodal processor for VLM checkpoints. Pass
        # it as the ``processor`` arg of TextOnlyChatCollator; the
        # collator pulls ``.tokenizer`` off it for the token-id encode.
        # max_length matches cfg.model.max_seq_length so text-only
        # batches don't blow past the LM head's [B*T, V] memory budget
        # at V=262 K (Gemma 4 vocab).
        text_collator = TextOnlyChatCollator(
            processor=tokenizer,
            max_length=cfg.model.max_seq_length,
        )

        # Unsloth's auto-patched trainer __init__ injects a "pad_check"
        # block (see unsloth/models/rl.py:989) that REPLACES any
        # collator that isn't an instance of UnslothVisionDataCollator
        # with a plain TransformersDataCollatorForLanguageModeling
        # whenever the processor has .tokenizer but no top-level .pad.
        # That's our exact case (Gemma 4 processor). Without the
        # bypass below, ModalityAwareCollator silently gets swapped out
        # and text-only batches fail at fetch time with
        # "you provided ['messages', 'length']" because the LM
        # collator needs pre-tokenized input_ids.
        #
        # Workaround: subclass UnslothVisionDataCollator so the
        # isinstance() check returns True. We deliberately skip its
        # heavyweight __init__ (which needs (model, processor) and
        # runs side-effect-y patch-size detection) — the actual
        # dispatch lives in our own __call__. UnslothVisionDataCollator
        # declares __slots__, but a subclass without __slots__ carries
        # a __dict__, so our instance attrs work.
        class _UnslothCompatModalityAwareCollator(UnslothVisionDataCollator):  # type: ignore[misc, valid-type]
            def __init__(self, vision_collator, text_collator):
                object.__init__(self)
                self.vision_collator = vision_collator
                self.text_collator = text_collator

            __call__ = ModalityAwareCollator.__call__

        # The ModalityAwareBatchSampler (built inside ModalityAwareSFTTrainer
        # below) guarantees each batch is homogeneous, so the dispatcher's
        # defensive assertion never fires under normal use.
        collator = _UnslothCompatModalityAwareCollator(
            vision_collator=collator,
            text_collator=text_collator,
        )
        log.info(
            "modality_aware_sampler enabled: collator wrapped in "
            "ModalityAwareCollator(vision=UnslothVisionDataCollator, "
            "text=TextOnlyChatCollator) "
            "[subclassing UnslothVisionDataCollator to bypass unsloth's "
            "pad_check collator-replacement]"
        )
    # warmup_ratio resolution: HF's TrainingArguments still accepts
    # `warmup_ratio` but emits a DeprecationWarning ("will be removed
    # in v5.2") in transformers 5.x. To stay forward-compatible we
    # resolve the ratio to an integer warmup_steps ourselves and only
    # pass warmup_steps to SFTConfig. This matches HF's own internal
    # math: warmup_steps = ceil(warmup_ratio * total_optimizer_steps).
    _effective_warmup_steps = _resolve_warmup_steps(cfg, len(train_records))
    if cfg.training.warmup_ratio is not None and cfg.training.warmup_ratio > 0:
        _total_steps = _estimate_total_optimizer_steps(cfg, len(train_records))
        log.info(
            "warmup_ratio=%.4f resolved to %d steps "
            "(total optimizer steps ~ %d, overrides warmup_steps=%d)",
            cfg.training.warmup_ratio,
            _effective_warmup_steps,
            _total_steps,
            cfg.training.warmup_steps,
        )

    # Optional new fields are added conditionally so we don't pass
    # explicit Nones to SFTConfig (which would override its own defaults
    # in some HF versions).
    _sft_kwargs: Dict[str, Any] = dict(
        output_dir=cfg.training.output_dir,
        # HF Trainer uses args.run_name as the wandb run name (and as
        # the tensorboard subdir name). Without this passthrough, HF
        # falls back to a synthesized run name that drifts from our
        # ``outputs/<run_name>/`` directory layout. Wandb defaults to
        # the basename of output_dir (which equals run_name by
        # construction; see build_config) — pass a CLI ``--run_name``
        # to override.
        run_name=cfg.training.run_name,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=_effective_warmup_steps,
        num_train_epochs=cfg.training.num_train_epochs or 1,
        max_steps=cfg.training.max_steps if cfg.training.max_steps else -1,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        optim=cfg.training.optim,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        seed=cfg.training.seed,
        save_steps=cfg.training.save_steps,
        report_to=cfg.training.report_to,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        # required for vision SFT — collator builds batches itself
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    # warmup_ratio is intentionally NOT forwarded to SFTConfig — see the
    # resolution block above that converted it to warmup_steps.
    if cfg.training.save_total_limit is not None:
        _sft_kwargs["save_total_limit"] = cfg.training.save_total_limit
    if _effective_tf32 is not None:
        # SFTConfig accepts tf32; also kept in sync with the
        # torch.backends toggle we set above so HF won't flip it back.
        _sft_kwargs["tf32"] = _effective_tf32
    if cfg.training.group_by_length:
        # transformers 5.x replaced `group_by_length: bool` with
        # `train_sampling_strategy: str`. The `length` column was
        # populated above on train_ds (and val_ds if present).
        _sft_kwargs["train_sampling_strategy"] = "group_by_length"

    # v2: mid-training eval (eval_strategy=steps, eval_steps=N).
    # Forward to SFTConfig only when explicitly set so we don't override
    # HF's defaults (which match the v1 'no' behaviour).
    if cfg.training.eval_strategy != "no":
        _sft_kwargs["eval_strategy"] = cfg.training.eval_strategy
    if cfg.training.eval_steps is not None:
        _sft_kwargs["eval_steps"] = cfg.training.eval_steps
    if cfg.training.per_device_eval_batch_size is not None:
        _sft_kwargs["per_device_eval_batch_size"] = cfg.training.per_device_eval_batch_size

    sft_args = SFTConfig(**_sft_kwargs)
    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        args=sft_args,
    )

    # When tuning the projector (and optionally the last-N vision encoder
    # layers), build a custom optimizer with 2 or 3 param groups so the
    # full-param parts train at lower LRs than the low-rank LoRA.
    #
    #   LoRA params       -> cfg.training.learning_rate
    #   Projector params  -> cfg.lora.projector_learning_rate
    #                        (or training.lr / 10 — LLaVA-style ratio)
    #   Vision-layer params (last-N, optional)
    #                     -> cfg.lora.vision_layers_learning_rate
    #                        (or training.lr / 20 — pretrained SigLIP is
    #                        more fragile than the projector adaptation)
    #
    # The scheduler is left to SFTTrainer by passing (optimizer, None);
    # HF Trainer constructs it from sft_args.lr_scheduler_type against
    # the optimizer we provide.
    if cfg.lora.tune_projector:
        projector_lr = cfg.lora.projector_learning_rate
        if projector_lr is None:
            projector_lr = cfg.training.learning_rate / 10.0

        vision_lr: Optional[float] = None
        if cfg.lora.tune_last_n_vision_layers > 0:
            vision_lr = cfg.lora.vision_layers_learning_rate
            if vision_lr is None:
                vision_lr = cfg.training.learning_rate / 20.0

        # 3-way classification of trainable params. Order matters: the
        # vision-layer check is most specific and runs first. Then
        # projector (which excludes vision_tower.encoder.* anyway), then
        # everything else (LoRA + miscellaneous trainable).
        tuned_idx_set: set[int] = set(tuned_vision_layer_indices)
        lora_params: List[Any] = []
        proj_params: List[Any] = []
        vision_params: List[Any] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if tuned_idx_set and _is_vision_layer_param_name(name, tuned_idx_set):
                vision_params.append(param)
            elif _is_projector_param_name(name):
                proj_params.append(param)
            else:
                lora_params.append(param)

        if tuned_idx_set:
            log.info(
                "Optimizer param groups: %d LoRA @ lr=%.2e, %d projector @ lr=%.2e, "
                "%d vision-layer @ lr=%.2e",
                len(lora_params), cfg.training.learning_rate,
                len(proj_params), projector_lr,
                len(vision_params), vision_lr,
            )
        else:
            log.info(
                "Optimizer param groups: %d LoRA @ lr=%.2e, %d projector @ lr=%.2e",
                len(lora_params), cfg.training.learning_rate,
                len(proj_params), projector_lr,
            )

        # Honor cfg.training.optim. The non-projector path passes this
        # string straight to SFTConfig/TrainingArguments which builds the
        # optimizer via HF's `optim` enum — but here we own optimizer
        # construction directly, so we must dispatch the class ourselves.
        # Previously this path was hardcoded to bitsandbytes AdamW8bit,
        # which (a) violated the project's no-8bit/4bit policy and (b)
        # silently ignored whatever the YAML said.
        #
        # Supported values (kept in sync with HF's OptimizerNames):
        #   "adamw_torch"        -> torch.optim.AdamW          (eager)
        #   "adamw_torch_fused"  -> torch.optim.AdamW(fused=True)
        #                          ~5-15% faster on Ampere/Ada (4090) at
        #                          no memory cost. Default for new
        #                          configs.
        #   "adamw_8bit"         -> bitsandbytes AdamW8bit
        #                          REJECTED by project policy (8-bit
        #                          optimizer state is still 8-bit
        #                          quantization). Kept here only so an
        #                          explicit user override surfaces a
        #                          clear error instead of silent fallback.
        import torch
        optim_name = cfg.training.optim
        opt_kwargs: Dict[str, Any] = {}
        if optim_name == "adamw_torch":
            opt_cls = torch.optim.AdamW
        elif optim_name == "adamw_torch_fused":
            opt_cls = torch.optim.AdamW
            opt_kwargs["fused"] = True
        elif optim_name == "adamw_8bit":
            raise ValueError(
                "training.optim='adamw_8bit' is rejected by project policy "
                "(no 8-bit / 4-bit quantization, see AGENTS.md). Use "
                "'adamw_torch_fused' (recommended) or 'adamw_torch'."
            )
        else:
            raise ValueError(
                f"Unsupported training.optim={optim_name!r} for the "
                "projector / vision-layer multi-group path. Supported: "
                "'adamw_torch', 'adamw_torch_fused'."
            )

        param_groups: List[Dict[str, Any]] = [
            {"params": lora_params, "lr": cfg.training.learning_rate},
            {"params": proj_params, "lr": projector_lr},
        ]
        if tuned_idx_set:
            param_groups.append({"params": vision_params, "lr": vision_lr})

        optimizer = opt_cls(
            param_groups,
            weight_decay=cfg.training.weight_decay,
            **opt_kwargs,
        )
        log.info(
            "Multi-group optimizer: %s (%s)",
            optim_name,
            "fused" if opt_kwargs.get("fused") else "eager",
        )
        # (optimizer, None): HF Trainer builds the scheduler from sft_args
        # against this optimizer.
        trainer_kwargs["optimizers"] = (optimizer, None)

    # v3: build the regularization state BEFORE constructing the trainer
    # subclass — the WeightL2Anchor snapshots trainable params at
    # construction time, and we want that snapshot to reflect the
    # post-PEFT-wrap state (LoRA delta zeros, projector/vision-layer
    # full pretrained values), NOT some intermediate state.
    from .regularization import build_regularizers

    reg_state = build_regularizers(cfg.regularization, model=model)
    if reg_state.enabled:
        log.info(
            "Regularization ENABLED: KL=%s (weight=%.4g, T=%.2g), "
            "L2=%s (weight=%.4g, anchored=%d params)",
            "ON" if reg_state.kl is not None else "off",
            reg_state.kl_weight,
            cfg.regularization.kl_temperature,
            "ON" if reg_state.l2 is not None else "off",
            reg_state.l2_weight,
            reg_state.l2.num_anchored_params() if reg_state.l2 is not None else 0,
        )
    else:
        log.info("Regularization disabled (v2 default behaviour).")

    if cfg.training.modality_aware_sampler:
        # v2: swap in the ModalityAware subclass that overrides
        # get_train_dataloader to use ModalityAwareBatchSampler.
        # v3: same subclass also overrides compute_loss when
        # reg_state.enabled is True (no-op fast-path otherwise).
        from .trainer_modality import make_modality_aware_sft_trainer_class

        ModalityAwareSFTTrainer = make_modality_aware_sft_trainer_class(
            seed=cfg.training.seed,
            regularization_state=reg_state,
        )
        trainer = ModalityAwareSFTTrainer(**trainer_kwargs)
        log.info(
            "modality_aware_sampler enabled: using ModalityAwareSFTTrainer "
            "(every train batch is homogeneous in image-presence; "
            "text-only batches skip the vision tower forward pass)"
        )
    else:
        if reg_state.enabled:
            # The regularizers are wired through ModalityAwareSFTTrainer.
            # Without that subclass we fall back to plain SFTTrainer +
            # CE only, which silently drops the KL / L2 terms — fail
            # loudly so the operator notices the config mismatch.
            raise RuntimeError(
                "regularization.* is enabled but training.modality_aware_sampler "
                "is False. The v3 regularization hooks live on "
                "ModalityAwareSFTTrainer; falling back to plain SFTTrainer "
                "would silently drop them. Either set "
                "training.modality_aware_sampler=true, or disable both "
                "regularization.kl_enabled and regularization.l2_enabled."
            )
        trainer = SFTTrainer(**trainer_kwargs)

    # 6c. Attach the JSONL metrics callback. Streams every trainer.log()
    #     emit as one JSON line to {output_dir}/metrics.jsonl — crash-safe
    #     local source-of-truth for the loss / eval / regularizer curves,
    #     independent of report_to. Stacks cleanly with wandb / tensorboard:
    #     the callback never mutates the logs dict that HF forwards to
    #     external trackers.
    try:  # `python -m src.finetune`
        from .metrics_callback import JsonlMetricsCallback
    except ImportError:  # `python src/finetune.py`
        from metrics_callback import JsonlMetricsCallback  # type: ignore[no-redef]
    trainer.add_callback(JsonlMetricsCallback(output_dir=cfg.training.output_dir))

    # 7. Train.
    log.info("run_name: %s", cfg.training.run_name)
    log.info("output_dir: %s", cfg.training.output_dir)
    log.info("Starting training: %d records", len(train_records))

    # Snapshot the trainable-param set BEFORE training so we can detect
    # mid-run requires_grad regressions (typically caused by
    # resume-from-checkpoint silently dropping modules_to_save wrappers).
    # This complements the save-side tripwires: those catch missing
    # tensors at save time, this catches the silent freeze the moment
    # training returns. Cheap and pure-Python — no GPU memory cost.
    pre_train_trainable: Dict[str, set] = _snapshot_trainable_params(
        model, tuned_vision_layer_indices=tuned_vision_layer_indices
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    log.info("Training complete (loss=%.4f)", train_result.training_loss)

    # 7b. Post-train requires_grad consistency check. Fires loud BEFORE
    #     save_pretrained so the operator sees the failure immediately
    #     after the corrupted training run, not at the save tripwire.
    post_train_trainable: Dict[str, set] = _snapshot_trainable_params(
        model, tuned_vision_layer_indices=tuned_vision_layer_indices
    )
    _assert_trainable_set_unchanged_post_train(
        pre_train_trainable, post_train_trainable
    )
    log.info(
        "Post-train trainable-set check passed: %d vision, %d projector, "
        "%d lora_other params remain trainable.",
        len(post_train_trainable["vision"]),
        len(post_train_trainable["projector"]),
        len(post_train_trainable["lora_other"]),
    )

    # 8. Save adapter.
    out = Path(cfg.training.output_dir) / "final-adapter"
    out.mkdir(parents=True, exist_ok=True)

    # Snapshot in-memory PEFT state BEFORE save_pretrained so the
    # tripwire below can diff "what was in memory" against "what hit
    # disk" — catching the AGENTS.md orphan-tensor regression at the
    # earliest possible point. extract_savable_state mirrors PEFT's own
    # internal save path (get_peft_model_state_dict), so the snapshot is
    # exactly the set of tensors save_pretrained intended to write.
    try:  # `python -m src.finetune`
        from .save_reload_check import extract_savable_state
    except ImportError:  # `python src/finetune.py`
        from save_reload_check import extract_savable_state  # type: ignore[no-redef]
    _pre_save_state = extract_savable_state(model)
    log.info(
        "Pre-save PEFT state snapshot: %d tensors (will be diffed against "
        "the saved adapter directory).", len(_pre_save_state),
    )

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    log.info("Saved LoRA adapter to %s", out)

    # 8b. AGENTS.md orphan-tensor tripwire — fires the moment disk
    #     disagrees with memory. Cheap (a single safetensors header
    #     parse + byte-equality check). Catches the bug class where an
    #     HF/PEFT/unsloth version drift silently drops k_proj/v_proj
    #     LoRA tensors at save or reload time.
    _assert_save_matches_in_memory_state(out, _pre_save_state)
    log.info(
        "Save tripwire passed: %d tensors written to disk match in-memory "
        "PEFT state byte-for-byte.", len(_pre_save_state),
    )

    # 9. Tripwire: verify projector tensors were actually saved if
    #    tune_projector was on. PEFT's save_pretrained only includes
    #    projector weights when they live inside a ModulesToSaveWrapper.
    #    If the wrapper was silently dropped (unsloth regression, PEFT
    #    version mismatch), the adapter dir has no projector tensors and
    #    the entire run is wasted — the export step would merge a
    #    projector-less adapter and ship unchanged base projector weights.
    if cfg.lora.tune_projector:
        _assert_projector_tensors_present_if_tuned(out, cfg.lora.tune_projector)

    # 9b. Parallel tripwire for vision-encoder-layer tensors when
    #     tune_last_n_vision_layers > 0. Same failure-mode story: if
    #     PEFT silently dropped the modules_to_save wrapper for these
    #     layers, the adapter contains no vision-layer tensors and the
    #     export step would merge a vision-untrained model.
    if cfg.lora.tune_last_n_vision_layers > 0:
        _assert_vision_layer_tensors_present_if_tuned(
            out, tuned_vision_layer_indices
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    if args.dry_run:
        dry_run(cfg)
        return 0

    real_train(cfg, resume_from_checkpoint=args.resume_from_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
