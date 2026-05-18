"""Tests for src.export_mlx — processor patch + vision-tower presence guards.

These tests cover the post-MLX-conversion contract that the iOS app
depends on:

  1. processor_config.json must end up with `size: {height: 960, width: 672}`
     at the top level (mlx-swift-lm's Gemma4ProcessorConfiguration decoder
     reads it there).
  2. The MLX safetensors must contain vision_tower / embed_vision keys.
     If the merge or convert step accidentally produces a language-only
     model, we want a hard failure at export time, not at first iOS run.

We synthesize tiny safetensors files by hand here (no torch / mlx
required) so the tests run on any Python install.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.export_mlx import (
    TRAINED_VISION_SIZE,
    _mlx_dir_has_vision_weights,
    patch_processor_config_for_mlx_swift,
)


# ---------------------------------------------------------------------------
# Tiny safetensors helper — write a header-only file with the keys we care
# about. The data section is empty because we only inspect headers.
# ---------------------------------------------------------------------------


def _write_safetensors_with_keys(path: Path, keys: list[str]) -> None:
    """Write a syntactically valid safetensors file with empty tensors.

    Each key gets a zero-length F32 tensor (data_offsets [0, 0]). The
    header fully describes the file even with no data section, which is
    enough for `_mlx_dir_has_vision_weights` to walk.
    """
    header: dict = {}
    for key in keys:
        header[key] = {
            "dtype": "F32",
            "shape": [0],
            "data_offsets": [0, 0],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad
    with path.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)


# ---------------------------------------------------------------------------
# patch_processor_config_for_mlx_swift
# ---------------------------------------------------------------------------


def test_patch_adds_size_when_missing(tmp_path: Path) -> None:
    """Upstream HF processor_config.json has no top-level `size` — we add it."""
    pcfg = tmp_path / "processor_config.json"
    pcfg.write_text(json.dumps({
        "processor_class": "Gemma4Processor",
        "image_processor": {
            "image_processor_type": "Gemma4ImageProcessor",
            "do_normalize": False,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
            "patch_size": 16,
            "max_soft_tokens": 280,
            "pooling_kernel_size": 3,
        },
    }))

    changed = patch_processor_config_for_mlx_swift(tmp_path)
    assert changed is True

    out = json.loads(pcfg.read_text())
    # Top-level size present and correct
    assert out["size"] == TRAINED_VISION_SIZE == {"height": 960, "width": 672}
    # Hoisted normalization fields
    assert out["do_normalize"] is False
    assert out["image_mean"] == [0.0, 0.0, 0.0]
    assert out["image_std"] == [1.0, 1.0, 1.0]
    # Nested size also patched
    assert out["image_processor"]["size"] == TRAINED_VISION_SIZE
    # Architectural fields untouched
    assert out["image_processor"]["max_soft_tokens"] == 280
    assert out["image_processor"]["pooling_kernel_size"] == 3
    assert out["image_processor"]["patch_size"] == 16


def test_patch_overrides_wrong_size(tmp_path: Path) -> None:
    """If upstream ships size: 224x224 (mlx-community case), we override it."""
    pcfg = tmp_path / "processor_config.json"
    pcfg.write_text(json.dumps({
        "processor_class": "Gemma4Processor",
        "size": {"height": 224, "width": 224},
        "image_processor": {
            "size": {"height": 224, "width": 224},
            "max_soft_tokens": 280,
            "pooling_kernel_size": 3,
            "patch_size": 16,
        },
    }))

    changed = patch_processor_config_for_mlx_swift(tmp_path)
    assert changed is True

    out = json.loads(pcfg.read_text())
    assert out["size"] == TRAINED_VISION_SIZE
    assert out["image_processor"]["size"] == TRAINED_VISION_SIZE


def test_patch_idempotent(tmp_path: Path) -> None:
    """Running the patch twice on already-correct config is a no-op."""
    pcfg = tmp_path / "processor_config.json"
    pcfg.write_text(json.dumps({
        "size": dict(TRAINED_VISION_SIZE),
        "do_normalize": False,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "image_processor": {
            "size": dict(TRAINED_VISION_SIZE),
            "do_normalize": False,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
        },
    }))
    assert patch_processor_config_for_mlx_swift(tmp_path) is False
    assert patch_processor_config_for_mlx_swift(tmp_path) is False


def test_patch_does_not_touch_architectural_fields(tmp_path: Path) -> None:
    """We never modify trained fields (max_soft_tokens, pooling_kernel_size,
    patch_size, image_seq_length). They define what the model was trained
    on and the iOS pooler depends on them being exactly 280 / 3 / 16 / 280."""
    pcfg = tmp_path / "processor_config.json"
    pcfg.write_text(json.dumps({
        "image_seq_length": 280,
        "image_processor": {
            "image_seq_length": 280,
            "max_soft_tokens": 280,
            "pooling_kernel_size": 3,
            "patch_size": 16,
            "rescale_factor": 0.00392156862745098,
        },
    }))

    patch_processor_config_for_mlx_swift(tmp_path)
    out = json.loads(pcfg.read_text())

    assert out["image_seq_length"] == 280
    ip = out["image_processor"]
    assert ip["image_seq_length"] == 280
    assert ip["max_soft_tokens"] == 280
    assert ip["pooling_kernel_size"] == 3
    assert ip["patch_size"] == 16
    assert ip["rescale_factor"] == 0.00392156862745098


def test_patch_missing_file_returns_false(tmp_path: Path) -> None:
    """No processor_config.json — return False, do not crash."""
    assert patch_processor_config_for_mlx_swift(tmp_path) is False


# ---------------------------------------------------------------------------
# _mlx_dir_has_vision_weights
# ---------------------------------------------------------------------------


def test_vision_weights_detected_in_single_safetensors(tmp_path: Path) -> None:
    _write_safetensors_with_keys(
        tmp_path / "model.safetensors",
        keys=[
            "language_model.layers.0.self_attn.q_proj.weight",
            "vision_tower.encoder.layer.0.attn.q_proj.weight",
            "embed_vision.weight",
        ],
    )
    assert _mlx_dir_has_vision_weights(tmp_path) is True


def test_vision_weights_missing_returns_false(tmp_path: Path) -> None:
    """Language-only checkpoint (the AutoModelForCausalLM / mlx_lm fault mode)."""
    _write_safetensors_with_keys(
        tmp_path / "model.safetensors",
        keys=[
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.layers.0.mlp.gate_proj.weight",
            "language_model.norm.weight",
        ],
    )
    assert _mlx_dir_has_vision_weights(tmp_path) is False


def test_vision_weights_detected_across_shards(tmp_path: Path) -> None:
    """Sharded checkpoint with vision weights in shard 2 of 2."""
    _write_safetensors_with_keys(
        tmp_path / "model-00001-of-00002.safetensors",
        keys=["language_model.layers.0.self_attn.q_proj.weight"],
    )
    _write_safetensors_with_keys(
        tmp_path / "model-00002-of-00002.safetensors",
        keys=["vision_tower.encoder.layer.5.attn.q_proj.weight"],
    )
    assert _mlx_dir_has_vision_weights(tmp_path) is True


def test_no_safetensors_returns_false(tmp_path: Path) -> None:
    assert _mlx_dir_has_vision_weights(tmp_path) is False


def test_embed_vision_alone_is_sufficient(tmp_path: Path) -> None:
    """Some shardings put embed_vision separate from vision_tower."""
    _write_safetensors_with_keys(
        tmp_path / "model.safetensors",
        keys=["embed_vision.weight"],
    )
    assert _mlx_dir_has_vision_weights(tmp_path) is True


def test_lookalike_keys_are_not_false_positives(tmp_path: Path) -> None:
    """Keys with 'vision' in them but not the actual tower must not match."""
    _write_safetensors_with_keys(
        tmp_path / "model.safetensors",
        keys=[
            "language_model.layers.0.self_attn.q_proj.weight",
            # Hypothetical hostile names — none are the real tower.
            "config.use_vision_features.flag",
            "lm_head.vision_aware.weight",
        ],
    )
    # The substring "vision_tower." is what we look for — these don't
    # contain it (no trailing dot after "vision_tower"), so the check
    # correctly returns False.
    assert _mlx_dir_has_vision_weights(tmp_path) is False
