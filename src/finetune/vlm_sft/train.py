"""HF VLM SFT driver for Qwen3.5 / InternVL3.5 vision-tower finetuning.

    python -m vlm_sft.train --config configs/local_sweep/<name>.yaml

Reuses the project config schema (`src.config`) and JSONL/messages format. The
tuning mode is read from the LoRA-config flags; the vision tower gets its own
(lower) LR group via a Trainer subclass.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("vlm_sft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3.5 / InternVL3.5 vision-tower SFT")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p.add_argument("--max-train-samples", dest="max_train_samples", type=int, default=None)
    p.add_argument("--report-to", dest="report_to", type=str, default=None)
    return p.parse_args()


def _build_eval_dataset(cfg, max_val_samples):
    """Concatenate the per-bucket val files into one eval set (val loss)."""
    from vlm_sft.data import VlmSftDataset, load_jsonl

    records = []
    if cfg.data.val_files:
        for path in cfg.data.val_files.values():
            if path and Path(path).exists():
                records.extend(load_jsonl(path))
    elif cfg.data.val_file and Path(cfg.data.val_file).exists():
        records.extend(load_jsonl(cfg.data.val_file))
    if not records:
        return None
    if max_val_samples:
        records = records[:max_val_samples]
    return VlmSftDataset(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from src.config import load_config
    from vlm_sft import data as data_mod
    from vlm_sft import modeling as model_mod

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.report_to is not None:
        overrides["report_to"] = args.report_to
    cfg = load_config(args.config, overrides or None)
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = args.max_train_samples

    run_name = args.run_name or cfg.training.run_name or (
        f"{Path(args.config).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir = str(Path(cfg.training.output_dir))
    log.info("run: %s | base_model=%s", run_name, cfg.model.base_model)

    import torch
    from transformers import Trainer, TrainingArguments

    # --- model + mode ----------------------------------------------------
    model, processor = model_mod.load_model_and_processor(cfg.model.base_model, cfg.model.dtype)
    model, info = model_mod.apply_mode(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        random_state=cfg.lora.random_state,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # needed for grad-checkpointing + frozen embeds

    # --- data ------------------------------------------------------------
    train_records = data_mod.load_jsonl(cfg.data.train_file)
    if cfg.data.max_train_samples:
        train_records = train_records[: cfg.data.max_train_samples]
    train_ds = data_mod.VlmSftDataset(train_records)
    eval_ds = _build_eval_dataset(cfg, cfg.data.max_val_samples)
    collator = data_mod.VlmSftCollator(
        processor,
        prompt_prefixes=cfg.data.prompt_prefixes,
        max_length=cfg.model.max_seq_length,
        image_max_patches=cfg.model.image_max_patches,
    )
    log.info("data: train=%d eval=%s", len(train_ds), len(eval_ds) if eval_ds else 0)

    # --- training args ---------------------------------------------------
    vision_lr = cfg.lora.vision_layers_learning_rate or cfg.training.learning_rate
    eval_strategy = cfg.training.eval_strategy if eval_ds is not None else "no"
    targs = TrainingArguments(
        output_dir=out_dir,
        run_name=run_name,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs or 1,
        max_steps=cfg.training.max_steps or -1,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        warmup_ratio=cfg.training.warmup_ratio or 0.0,
        logging_steps=cfg.training.logging_steps,
        optim=cfg.training.optim,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        seed=cfg.training.seed,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        report_to=cfg.training.report_to,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        tf32=bool(cfg.training.tf32),
        bf16=cfg.model.dtype == "bfloat16",
        fp16=cfg.model.dtype == "float16",
        eval_strategy=eval_strategy,
        eval_steps=cfg.training.eval_steps,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size or 1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # --- Trainer with a vision-LR param group ----------------------------
    class _VlmTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                groups = model_mod.split_param_groups(
                    self.model, cfg.training.learning_rate, vision_lr,
                    cfg.training.weight_decay,
                )
                self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
            return self.optimizer

    trainer = _VlmTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    log.info("starting training (vision_lr=%.1e, lora_lr=%.1e)", vision_lr, cfg.training.learning_rate)
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    log.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
