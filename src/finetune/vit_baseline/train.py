"""Train the NA-Plantae ViT classification baseline.

    python -m src.finetune.vit_baseline.train --config <yaml> [overrides]

Pure-vision baseline: a ViT backbone + linear head trained on the iNaturalist
NA-Plantae ImageFolder, language model removed. Reports top-1 / top-5 species
accuracy on the val split — the number to compare against the VLM's plant
score so we can see how much the LLM adds on top of the vision encoder.

bf16-only by project policy (no 8-bit / 4-bit) to stay comparable with the
LoRA SFT bake-offs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

# Make the package importable both as ``-m src.finetune.vit_baseline.train``
# and when run with the finetune dir on sys.path.
try:
    from . import config as cfg_mod
    from . import dataset as ds_mod
    from . import model as model_mod
except ImportError:  # pragma: no cover - direct-script fallback
    from finetune.vit_baseline import config as cfg_mod  # type: ignore
    from finetune.vit_baseline import dataset as ds_mod  # type: ignore
    from finetune.vit_baseline import model as model_mod  # type: ignore

log = logging.getLogger("vit_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NA-Plantae ViT classification baseline")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    # Common overrides (full set lives in config.CLI_OVERRIDE_MAP).
    p.add_argument("--image-root", dest="image_root", type=str, default=None)
    p.add_argument("--image-size", dest="image_size", type=int, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=None)
    p.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--epochs", dest="num_train_epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="per_device_train_batch_size", type=int, default=None)
    p.add_argument("--lr", dest="learning_rate", type=float, default=None)
    p.add_argument("--backbone-lr", dest="backbone_learning_rate", type=float, default=None)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    p.add_argument("--max-images-per-class", dest="max_images_per_class", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--report-to", dest="report_to", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> cfg_mod.VitBaselineConfig:
    overrides = {
        "image_root": args.image_root,
        "image_size": args.image_size,
        "backbone": args.backbone,
        "freeze_backbone": args.freeze_backbone,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "max_steps": args.max_steps,
        "max_images_per_class": args.max_images_per_class,
        "num_workers": args.num_workers,
        "report_to": args.report_to,
        "seed": args.seed,
    }
    cfg = cfg_mod.load_config(args.config, overrides)
    errs = cfg_mod.validate_config(cfg)
    if errs:
        raise SystemExit("invalid config:\n  - " + "\n  - ".join(errs))
    return cfg


def evaluate(model, loader, device, autocast_dtype):
    import torch

    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(images)
            _, pred5 = logits.topk(5, dim=1)
            correct = pred5.eq(labels.view(-1, 1))
            top1 += correct[:, 0].sum().item()
            top5 += correct.any(dim=1).sum().item()
            total += labels.size(0)
    model.train()
    return {
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "n": total,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = _build_config(args)

    import torch
    from torch.utils.data import DataLoader

    image_root = cfg_mod.resolve_image_root(cfg)
    run_name = cfg.train.run_name or (
        f"{cfg.model.backbone.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir = Path(cfg.train.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("run: %s", run_name)
    log.info("output dir: %s", out_dir)

    torch.manual_seed(cfg.train.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for ViT baseline training")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    autocast_dtype = dtype_map[cfg.train.dtype]

    # --- data -------------------------------------------------------------
    train_tf, val_tf = ds_mod.build_transforms(
        cfg.model.backbone, cfg.data.image_size, cfg.model.pretrained
    )
    train_ds, val_ds, class_to_idx = ds_mod.build_datasets(
        image_root,
        cfg.data.train_split,
        cfg.data.val_split,
        train_tf,
        val_tf,
        max_images_per_class=cfg.data.max_images_per_class,
        max_train_samples=cfg.data.max_train_samples,
        max_val_samples=cfg.data.max_val_samples,
    )
    num_classes = len(class_to_idx)
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.per_device_train_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.per_device_eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    # --- model ------------------------------------------------------------
    model = model_mod.build_model(
        cfg.model.backbone,
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)
    if cfg.train.torch_compile:
        model = torch.compile(model)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("model %s: %.1fM trainable / %.1fM total params (%d classes)",
             cfg.model.backbone, n_trainable / 1e6, n_total / 1e6, num_classes)

    # --- optim / sched ----------------------------------------------------
    backbone_lr = (
        cfg.train.backbone_learning_rate
        if cfg.train.backbone_learning_rate is not None
        else cfg.train.learning_rate
    )
    param_groups = model_mod.split_param_groups(
        model, cfg.train.learning_rate, backbone_lr, cfg.train.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(train_loader) // cfg.train.gradient_accumulation_steps)
    if cfg.train.max_steps is not None:
        total_steps = cfg.train.max_steps
    else:
        total_steps = steps_per_epoch * cfg.train.num_train_epochs
    warmup_steps = int(cfg.train.warmup_epochs * steps_per_epoch)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

    use_wandb = cfg.train.report_to == "wandb"
    if use_wandb:
        import wandb

        wandb.init(project=cfg.train.wandb_project, name=run_name,
                   config={"backbone": cfg.model.backbone, "num_classes": num_classes})

    # --- train loop -------------------------------------------------------
    global_step = 0
    best_top1 = 0.0
    accum = cfg.train.gradient_accumulation_steps
    t0 = time.time()
    model.train()
    stop = False
    for epoch in range(cfg.train.num_train_epochs):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(images)
                loss = criterion(logits, labels) / accum
            loss.backward()
            running += loss.item() * accum

            if (i + 1) % accum == 0:
                if cfg.train.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        cfg.train.clip_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.train.log_interval == 0:
                    avg = running / (cfg.train.log_interval * accum)
                    rate = global_step / (time.time() - t0)
                    cur_lr = scheduler.get_last_lr()[0]
                    log.info(
                        "epoch %d step %d/%d loss %.4f lr %.2e %.1f it/s",
                        epoch, global_step, total_steps, avg, cur_lr, rate,
                    )
                    if use_wandb:
                        wandb.log({"train/loss": avg, "train/lr": cur_lr}, step=global_step)
                    running = 0.0

                if cfg.train.max_steps is not None and global_step >= cfg.train.max_steps:
                    stop = True
                    break

        # epoch-end eval
        if (epoch + 1) % cfg.train.eval_interval == 0 or stop or epoch + 1 == cfg.train.num_train_epochs:
            metrics = evaluate(model, val_loader, device, autocast_dtype)
            log.info("[eval] epoch %d top1 %.4f top5 %.4f (n=%d)",
                     epoch, metrics["top1"], metrics["top5"], metrics["n"])
            if use_wandb:
                wandb.log({"val/top1": metrics["top1"], "val/top5": metrics["top5"]},
                          step=global_step)
            is_best = metrics["top1"] > best_top1
            if is_best:
                best_top1 = metrics["top1"]
                _save_checkpoint(model, out_dir / "best.pt", class_to_idx, cfg, metrics, epoch)
            if not cfg.train.save_best_only:
                _save_checkpoint(model, out_dir / f"epoch{epoch}.pt", class_to_idx, cfg, metrics, epoch)
        if stop:
            break

    summary = {
        "run_name": run_name,
        "backbone": cfg.model.backbone,
        "num_classes": num_classes,
        "best_top1": best_top1,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "total_steps": global_step,
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("done. best val top1 = %.4f | summary -> %s", best_top1, out_dir / "summary.json")
    if use_wandb:
        wandb.finish()


def _save_checkpoint(model, path, class_to_idx, cfg, metrics, epoch) -> None:
    import torch

    state = model.state_dict()
    torch.save(
        {
            "model_state_dict": state,
            "class_to_idx": class_to_idx,
            "backbone": cfg.model.backbone,
            "image_size": cfg.data.image_size,
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )
    log.info("saved checkpoint -> %s (top1 %.4f)", path, metrics["top1"])


if __name__ == "__main__":
    main()
