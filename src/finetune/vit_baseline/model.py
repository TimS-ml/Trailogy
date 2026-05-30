"""ViT backbone + linear classification head for the plant-ID baseline.

Thin wrapper over ``timm.create_model``: timm already attaches a single linear
classifier on top of the pooled features when ``num_classes`` is set, which is
exactly the "ViT + linear head" baseline we want. The only extras here are the
linear-probe freeze switch and the two-group (head vs backbone) parameter split
used by the optimiser.
"""

from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    freeze_backbone: bool = False,
):
    """Create a timm classification model and (optionally) freeze the trunk.

    Returns the ``nn.Module``. The classifier head is left trainable in all
    cases; ``freeze_backbone=True`` gives a linear-probe baseline.
    """
    import timm

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if freeze_backbone:
        head_params = set(_head_param_names(model))
        frozen = 0
        for name, p in model.named_parameters():
            if name not in head_params:
                p.requires_grad_(False)
                frozen += 1
        log.info("linear-probe: froze %d backbone params, head stays trainable", frozen)

    return model


def _head_param_names(model) -> List[str]:
    """Parameter names belonging to the classifier head.

    timm exposes the head module name via ``get_classifier`` / the standard
    ``head`` / ``fc`` attribute; we resolve it by matching the classifier
    submodule's parameters rather than hard-coding a name (varies by arch).
    """
    classifier = model.get_classifier()
    head_ids = {id(p) for p in classifier.parameters()}
    return [name for name, p in model.named_parameters() if id(p) in head_ids]


def split_param_groups(
    model,
    base_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> List[Dict]:
    """Two LR groups: classifier head at ``base_lr``, trunk at ``backbone_lr``.

    Frozen params (requires_grad=False) are dropped so the optimiser never
    sees them. Norm/bias params are kept out of weight decay.
    """
    head_params = set(_head_param_names(model))

    groups: Dict[str, Dict] = {
        "head_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "bb_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "bb_nodecay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name in head_params
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        key = ("head" if is_head else "bb") + ("_nodecay" if no_decay else "_decay")
        groups[key]["params"].append(p)

    return [g for g in groups.values() if g["params"]]
