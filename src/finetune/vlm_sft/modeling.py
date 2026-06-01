"""Model loading + tuning-mode setup for HF VLM SFT.

Loads any `AutoModelForImageTextToText` backbone, identifies its vision-side
parameters by name, and wires up one of three tuning modes from the existing
LoRA-config flags (see package docstring).

Vision-side = the vision encoder plus its connector/projector (Qwen's
``visual.*`` incl. ``merger``; InternVL's ``vision_tower.*`` +
``multi_modal_projector.*``). The language model is everything else.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# Substrings that mark a parameter as vision-side (encoder + projector/merger).
_VISION_MARKERS = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
)

# Linear projections we adapt with LoRA, restricted to the language model.
_LLM_LORA_REGEX = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def load_model_and_processor(base_model: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model, processor


def is_vision_param(name: str) -> bool:
    if "language_model" in name:
        return False
    return any(m in name for m in _VISION_MARKERS)


def vision_param_names(model) -> List[str]:
    return [n for n, _ in model.named_parameters() if is_vision_param(n)]


def vision_module_markers(model) -> List[str]:
    """Markers that correspond to an actual vision-tower root module.

    Used as PEFT ``modules_to_save`` so the (fully-trained) vision tower is
    saved alongside the LoRA adapter — otherwise PEFT's adapter-only save
    silently drops the trained vision weights.
    """
    present = []
    names = [n for n, _ in model.named_modules()]
    for m in _VISION_MARKERS:
        if any(n == m or n.endswith("." + m) for n in names):
            present.append(m)
    return present


def apply_mode(model, *, finetune_vision_layers: bool, finetune_language_layers: bool,
               lora_r: int, lora_alpha: int, lora_dropout: float, random_state: int = 3407):
    """Configure trainable params for the requested mode.

    Returns ``(model, info)``. ``model`` may be PEFT-wrapped (LoRA modes).
    """
    info: Dict[str, object] = {}

    # Start from a clean slate: freeze everything.
    for p in model.parameters():
        p.requires_grad_(False)

    vnames = set(vision_param_names(model))
    info["n_vision_params"] = len(vnames)
    if not vnames and finetune_vision_layers:
        raise RuntimeError(
            "finetune_vision_layers=True but no vision-side params matched "
            f"{_VISION_MARKERS}; check the model's module names."
        )

    if finetune_language_layers:
        if lora_r <= 0:
            raise ValueError("finetune_language_layers=True requires lora r > 0")
        from peft import LoraConfig, get_peft_model

        # When co-training the vision tower, register it as modules_to_save so
        # PEFT keeps it trainable AND saves it with the adapter (adapter-only
        # save would otherwise drop the trained vision weights). PEFT then
        # manages requires_grad for those modules, so no manual unfreeze needed.
        modules_to_save = vision_module_markers(model) if finetune_vision_layers else None
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=_LLM_LORA_REGEX,
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        info["lora"] = True
        info["modules_to_save"] = modules_to_save
    elif finetune_vision_layers:
        # vision-only mode (no LoRA): unfreeze the vision tower directly.
        unfrozen = 0
        for n, p in model.named_parameters():
            if is_vision_param(n):
                p.requires_grad_(True)
                unfrozen += 1
        info["n_vision_unfrozen"] = unfrozen

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info["trainable_params"] = trainable
    info["total_params"] = total
    log.info(
        "mode: vision=%s lora=%s | trainable %.1fM / %.1fM (%.3f%%) | vision params matched=%d",
        finetune_vision_layers, finetune_language_layers,
        trainable / 1e6, total / 1e6, 100 * trainable / max(total, 1), len(vnames),
    )
    return model, info


def split_param_groups(model, base_lr: float, vision_lr: float, weight_decay: float) -> List[Dict]:
    """LoRA/other trainable params at ``base_lr``; vision-side at ``vision_lr``.

    Norm/bias kept out of weight decay.
    """
    groups: Dict[str, Dict] = {
        "base_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "base_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "vis_decay": {"params": [], "lr": vision_lr, "weight_decay": weight_decay},
        "vis_nodecay": {"params": [], "lr": vision_lr, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_vis = is_vision_param(name)
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        key = ("vis" if is_vis else "base") + ("_nodecay" if no_decay else "_decay")
        groups[key]["params"].append(p)
    return [g for g in groups.values() if g["params"]]
