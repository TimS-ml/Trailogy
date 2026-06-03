"""Tests for vlm_sft: vision-param detection, mode config, param groups.

Pure / CPU only — no model download or GPU. The collator's processor path is
exercised by the smoke run, not unit tests (it needs the real processors).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_sft.data import filter_image_only
from vlm_sft.modeling import is_vision_param, split_param_groups

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "local_sweep"


def test_filter_image_only_drops_camera_off() -> None:
    recs = [
        {"image": "/a.jpg", "conversations": []},
        {"image": None, "conversations": []},     # camera=off (text-only)
        {"conversations": []},                      # camera=off (no image key)
        {"image": "/b.jpg", "conversations": []},
        {"image": "", "conversations": []},        # empty path == camera=off
    ]
    kept = filter_image_only(recs)
    assert [r["image"] for r in kept] == ["/a.jpg", "/b.jpg"]


def test_six_configs_image_only_default_on() -> None:
    from src.config import load_config

    for name in [
        "vtower-nokl-local-30k-v3-internvl.yaml",
        "r64-a64-nokl-vtower-local-30k-v3-internvl.yaml",
        "r64-a64-nokl-local-30k-v3-internvl.yaml",
        "vtower-nokl-local-30k-v3-qwen.yaml",
        "r64-a64-nokl-vtower-local-30k-v3-qwen.yaml",
        "r64-a64-nokl-local-30k-v3-qwen.yaml",
    ]:
        cfg = load_config(CONFIG_DIR / name)
        assert cfg.data.image_only is True
        # effective batch preserved at 16 for bake-off comparability
        assert (cfg.training.per_device_train_batch_size
                * cfg.training.gradient_accumulation_steps) == 16
        # save every 1k step, keep-all (save_total_limit null -> None)
        assert cfg.training.save_steps == 1000
        assert cfg.training.save_total_limit is None


def test_is_vision_param_markers() -> None:
    assert is_vision_param("model.visual.blocks.0.attn.q_proj.weight")
    assert is_vision_param("vision_tower.encoder.layers.3.mlp.fc1.weight")
    assert is_vision_param("model.multi_modal_projector.linear_1.weight")
    # language model is never vision, even if it somehow contains a marker word
    assert not is_vision_param("model.language_model.layers.0.self_attn.q_proj.weight")
    assert not is_vision_param("language_model.embed_tokens.weight")


def test_split_param_groups_separates_vision_and_lr() -> None:
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(4, 4)               # vision (2d weight + bias)
            self.language_model = nn.Linear(4, 4)       # base
            self.norm = nn.LayerNorm(4)                 # base, no-decay

    m = Tiny()
    groups = split_param_groups(m, base_lr=2e-4, vision_lr=1e-5, weight_decay=0.05)
    by_lr = {}
    for g in groups:
        by_lr.setdefault(g["lr"], 0)
        by_lr[g["lr"]] += sum(p.numel() for p in g["params"])
    assert 1e-5 in by_lr and 2e-4 in by_lr           # both LR groups present
    # vision weight+bias both at vision_lr
    vis_params = sum(p.numel() for g in groups if g["lr"] == 1e-5 for p in g["params"])
    assert vis_params == 4 * 4 + 4


@pytest.mark.parametrize(
    "name,vision,lang,r",
    [
        ("vtower-nokl-local-30k-v3-qwen.yaml", True, False, 0),
        ("r64-a64-nokl-vtower-local-30k-v3-qwen.yaml", True, True, 64),
        ("r64-a64-nokl-local-30k-v3-qwen.yaml", False, True, 64),
        ("vtower-nokl-local-30k-v3-internvl.yaml", True, False, 0),
        ("r64-a64-nokl-vtower-local-30k-v3-internvl.yaml", True, True, 64),
        ("r64-a64-nokl-local-30k-v3-internvl.yaml", False, True, 64),
    ],
)
def test_six_configs_mode_flags(name, vision, lang, r) -> None:
    from src.config import load_config

    cfg = load_config(CONFIG_DIR / name)
    assert cfg.lora.finetune_vision_layers is vision
    assert cfg.lora.finetune_language_layers is lang
    assert cfg.lora.r == r
    assert cfg.data.train_file.endswith("mix-50k-v3/train.jsonl")
    # InternVL configs must cap tiles so one image fits max_seq_length.
    if "internvl" in name:
        assert cfg.model.image_max_patches == 1
        assert cfg.model.base_model == "OpenGVLab/InternVL3_5-4B-HF"
    else:
        assert cfg.model.base_model == "Qwen/Qwen3.5-4B"
