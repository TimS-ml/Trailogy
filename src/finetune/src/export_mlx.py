#!/usr/bin/env python3
"""Export a finetuned Gemma 4 E2B LoRA model to MLX format for the iOS app.

Pipeline:
  1. Merge LoRA adapter into the **multimodal** base model
     (Gemma4ForConditionalGeneration), preserving vision_tower + embed_vision.
  2. Save merged model in safetensors format.
  3. Convert to MLX with quantization via **mlx_vlm.convert**
     (NOT mlx_lm.convert — that one only handles language-only models and
     would silently drop the vision tower weights).
  4. Patch the exported processor_config.json so mlx-swift-lm sees the
     trained `size: 960x672` (without this patch, mlx-swift-lm falls back
     to 800x800 and the trained kernel-3 vision pooler degenerates).
  5. Optionally strip audio tower weights (same logic as
     hikeCompanion/scripts/strip-gemma-audio.py).

The final output directory can be copied directly into the iOS bundle:
    cp -R <output_dir>/* <hikeCompanion>/HikeCompanion/Resources/Models/Gemma/

Usage:
    # On the NVIDIA training box (merge + save safetensors):
    python src/export_mlx.py \
        --base_model unsloth/gemma-4-E2B-it \
        --adapter_path outputs/hike-gemma4-lora \
        --output_dir exports/gemma4-mlx \
        --merge_only

    # On a Mac with mlx / mlx_vlm installed (convert + quantize):
    python src/export_mlx.py \
        --merged_dir exports/gemma4-merged \
        --output_dir exports/gemma4-mlx \
        --quantize_bits 4

    # Full pipeline (if running on a Mac with both torch and mlx):
    python src/export_mlx.py \
        --base_model unsloth/gemma-4-E2B-it \
        --adapter_path outputs/hike-gemma4-lora \
        --output_dir exports/gemma4-mlx \
        --quantize_bits 4 \
        --strip_audio
"""

import argparse
import json
import logging
import os
import re
import shutil
import struct
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio tower stripping (adapted from hikeCompanion/scripts/strip-gemma-audio.py)
# ---------------------------------------------------------------------------

AUDIO_PREFIXES = ("audio_tower", "embed_audio")
COPY_CHUNK = 8 * 1024 * 1024  # 8 MB
HEADER_ALIGN = 8


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n} B"


def _read_safetensors_header(fin) -> tuple[int, dict]:
    """Read safetensors header. Returns (data_start_offset, header_dict)."""
    raw = fin.read(8)
    if len(raw) != 8:
        raise ValueError("File too short to be safetensors")
    header_len = struct.unpack("<Q", raw)[0]
    header_bytes = fin.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError("Truncated safetensors header")
    return 8 + header_len, json.loads(header_bytes)


def strip_audio_tower(safetensors_path: Path) -> bool:
    """Strip audio_tower / embed_audio weights from a safetensors file in-place.

    Returns True if any weights were stripped, False if none were found.
    """
    if not safetensors_path.exists():
        log.warning("safetensors file not found: %s", safetensors_path)
        return False

    orig_size = safetensors_path.stat().st_size
    log.info("Checking %s (%s) for audio tower weights...", safetensors_path.name, fmt_bytes(orig_size))

    with safetensors_path.open("rb") as fin:
        data_start, header = _read_safetensors_header(fin)

    metadata = header.pop("__metadata__", None)
    audio_keys = [k for k in header if any(k.startswith(p) for p in AUDIO_PREFIXES)]

    if not audio_keys:
        log.info("No audio tower weights found — nothing to strip.")
        return False

    log.info("Found %d audio key(s) to strip, keeping %d tensor(s).", len(audio_keys), len(header) - len(audio_keys))

    # Build new header
    new_header: dict = {}
    if metadata is not None:
        new_header["__metadata__"] = metadata

    kept_plan: list[tuple[str, int, int]] = []  # (key, orig_start, orig_end)
    new_offset = 0
    for key, meta in header.items():
        if key in audio_keys:
            continue
        orig_start, orig_end = meta["data_offsets"]
        size = orig_end - orig_start
        new_header[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [new_offset, new_offset + size],
        }
        kept_plan.append((key, orig_start, orig_end))
        new_offset += size

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (-len(new_header_bytes)) % HEADER_ALIGN
    if pad:
        new_header_bytes += b" " * pad

    # Write stripped file alongside, then replace
    tmp_path = safetensors_path.with_suffix(".stripped.tmp")
    try:
        with safetensors_path.open("rb") as fin, tmp_path.open("wb") as fout:
            fout.write(struct.pack("<Q", len(new_header_bytes)))
            fout.write(new_header_bytes)
            for i, (key, orig_start, orig_end) in enumerate(kept_plan, 1):
                fin.seek(data_start + orig_start)
                remaining = orig_end - orig_start
                while remaining > 0:
                    chunk = fin.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"Unexpected EOF copying tensor {key}")
                    fout.write(chunk)
                    remaining -= len(chunk)
                if i % 200 == 0 or i == len(kept_plan):
                    log.info("  [%d/%d] tensors written", i, len(kept_plan))

        # Swap
        safetensors_path.unlink()
        tmp_path.rename(safetensors_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    new_size = safetensors_path.stat().st_size
    log.info(
        "Stripped audio tower: %s -> %s (saved %s)",
        fmt_bytes(orig_size),
        fmt_bytes(new_size),
        fmt_bytes(orig_size - new_size),
    )

    # Update sidecar index if present
    index_path = safetensors_path.parent / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        removed = sum(1 for k in audio_keys if wm.pop(k, None) is not None)
        idx["weight_map"] = wm
        if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
            del idx["metadata"]["total_size"]
        with open(index_path, "w") as f:
            json.dump(idx, f, indent=2)
        log.info("Removed %d key(s) from %s", removed, index_path.name)

    return True


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA adapter into base model
# ---------------------------------------------------------------------------


def merge_adapter(base_model: str, adapter_path: str, output_dir: str) -> Path:
    """Load the **multimodal** base model + LoRA adapter, merge, save to disk.

    Critical: must use ``AutoModelForImageTextToText`` (not
    ``AutoModelForCausalLM``). Gemma 4 is ``Gemma4ForConditionalGeneration``;
    the CausalLM auto-class only loads the language sub-module and silently
    drops vision_tower / embed_vision. Without this distinction the merged
    checkpoint is language-only and the iOS VLM path has no visual weights
    to deploy.
    """
    log.info("=== Step 1: Merging LoRA adapter (multimodal) ===")

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
    except ImportError as exc:
        log.error(
            "torch / transformers / peft are required for merging.\n"
            "Install: pip install 'transformers>=5.5.0' peft torch\n"
            "Error: %s",
            exc,
        )
        sys.exit(1)

    merged_dir = Path(output_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading multimodal base model: %s", base_model)
    # AutoModelForImageTextToText resolves to Gemma4ForConditionalGeneration
    # for Gemma 4 checkpoints, preserving language_model + vision_tower +
    # embed_vision (and audio_tower; we strip that later).
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Merge on CPU to avoid GPU memory issues
    )

    # Sanity check: confirm vision tower survived the load. This is the
    # exact fault mode that AutoModelForCausalLM was producing.
    if not _model_has_vision_tower(model):
        log.error(
            "Loaded base model has no vision_tower / embed_vision parameters. "
            "This means AutoModelForImageTextToText resolved to a language-only "
            "class for `%s`. Aborting before producing a broken merge.",
            base_model,
        )
        sys.exit(1)

    # Snapshot projector params from the BASE model BEFORE adapter load,
    # so the tripwire can byte-diff against them after merge_and_unload.
    # This is cheap (a few MB at bf16) and only matters for projector-tuned
    # adapters; the snapshot is dropped immediately after the check.
    _base_projector_snapshot = _model_param_data_bytes(
        model,
        (
            "embed_vision.",
            "vision_tower.embedding_projection.",
            "vision_tower.projection.",
        ),
    )

    # Snapshot vision-encoder-layer params for any tuned layer indices
    # the adapter directory advertises (feature/lora-plus-projector-plus-
    # vision-tower). Empty dict for LoRA-only / projector-only adapters,
    # in which case the snapshot is a cheap no-op.
    _adapter_vision_layer_indices = sorted(
        _adapter_has_vision_layer_tensors(adapter_path).keys()
    )
    _base_vision_layer_snapshot: dict[int, dict[str, bytes]] = (
        _model_param_data_bytes_by_vision_layer(
            model, _adapter_vision_layer_indices
        )
        if _adapter_vision_layer_indices
        else {}
    )

    log.info("Loading adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    log.info("Merging adapter weights...")
    model = model.merge_and_unload()

    # Tripwire (feature/lora-plus-projector): if the adapter dir contains
    # projector tensors (modules_to_save was used at training time), the
    # merged model's projector params MUST differ from the base. Identical
    # tensors mean PEFT silently failed to restore modules_to_save weights.
    # Construct a minimal "base view" object that exposes the snapshot via
    # the named_parameters() shape the tripwire expects.
    class _SnapshotView:  # noqa: D401 — local helper
        def __init__(self, snapshot: dict[str, bytes]) -> None:
            self._snap = snapshot

        def named_parameters(self):
            class _Param:
                def __init__(self, data: bytes) -> None:
                    self.data = data

            for n, b in self._snap.items():
                yield n, _Param(b)

    _assert_projector_changed_if_tuned(
        _SnapshotView(_base_projector_snapshot), model, adapter_path
    )

    # Tripwire (feature/lora-plus-projector-plus-vision-tower): if the
    # adapter dir contains vision-encoder-layer tensors, the merged
    # model's params under those layer indices MUST differ from the
    # base. Identical tensors mean PEFT silently dropped the
    # modules_to_save wrapper for those layers — ship-stopper.
    if _adapter_vision_layer_indices:
        # Flatten the snapshot into a named_parameters()-like view so the
        # tripwire can iterate it the same way as the projector path.
        class _VisionLayerSnapshotView:  # noqa: D401
            def __init__(self, snapshot: dict[int, dict[str, bytes]]) -> None:
                self._snap = snapshot

            def named_parameters(self):
                class _Param:
                    def __init__(self, data: bytes) -> None:
                        self.data = data

                for _idx, by_name in self._snap.items():
                    for n, b in by_name.items():
                        yield n, _Param(b)

        _assert_vision_layers_changed_if_tuned(
            _VisionLayerSnapshotView(_base_vision_layer_snapshot),
            model,
            adapter_path,
        )

    log.info("Saving merged model to %s", merged_dir)
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Also copy processor / tokenizer files
    log.info("Saving processor / tokenizer...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(merged_dir)

    merged_size = sum(f.stat().st_size for f in merged_dir.rglob("*") if f.is_file())
    log.info("Merged model saved: %s total", fmt_bytes(merged_size))

    return merged_dir


def _model_has_vision_tower(model) -> bool:
    """Return True if the loaded HF model exposes vision_tower / embed_vision.

    Used as a tripwire after `from_pretrained` to detect the
    AutoModelForCausalLM-style silent vision drop.
    """
    for name, _ in model.named_parameters():
        if "vision_tower." in name or "embed_vision." in name:
            return True
    return False


# ---------------------------------------------------------------------------
# Projector-changed tripwire (feature/lora-plus-projector)
# ---------------------------------------------------------------------------


def _adapter_has_projector_tensors(adapter_dir: "Path | None") -> list[str]:
    """Return projector tensor key names found in the adapter directory.

    The projector is identified by substring tokens shared with
    ``src.projector.PROJECTOR_CANDIDATE_TOKENS`` (locally redefined here
    to keep export_mlx.py free of an extra import — the tokens are
    short and stable). Encoder paths are explicitly excluded so a
    future ``vision_tower.embedding_projection.*`` layout doesn't
    accidentally match encoder weights.

    Returns an empty list for LoRA-only adapters (no projector tensors)
    or missing/empty adapter dirs. The caller treats an empty return
    as "skip the check".
    """
    if adapter_dir is None:
        return []
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        return []

    candidate_tokens = (
        "embed_vision.",
        "vision_tower.embedding_projection.",
        "vision_tower.projection.",
        "multi_modal_projector.vision.",
        "multi_modal_projector.image.",
    )
    exclude_tokens = (
        "vision_tower.patch_embedder.",
        "vision_tower.encoder.",
        "vision_tower.pooler.",
        "embed_audio.",
        "audio_tower.",
    )

    def _matches(name: str, tokens: tuple[str, ...]) -> bool:
        for t in tokens:
            if name.startswith(t) or ("." + t) in name:
                return True
        return False

    found: list[str] = []
    for sf in sorted(adapter_path.glob("*.safetensors")):
        try:
            with sf.open("rb") as fin:
                _data_start, header = _read_safetensors_header(fin)
        except Exception as exc:
            log.warning("Could not read safetensors header for %s: %s", sf, exc)
            continue
        for key in header:
            if key == "__metadata__":
                continue
            if _matches(key, exclude_tokens):
                continue
            if _matches(key, candidate_tokens):
                found.append(key)
    return found


def _model_param_data_bytes(model, name_substrings: tuple[str, ...]) -> dict[str, bytes]:
    """Collect raw byte representations of parameters whose name contains
    any of `name_substrings`. Used for byte-equality comparison between
    base and merged models in the projector tripwire.

    For real torch.nn.Parameter objects, we read `.data.cpu().contiguous()`
    and dump via .numpy().tobytes(). For unit-test fakes that store a
    pre-baked `data` attribute (bytes), we use that directly.
    """
    out: dict[str, bytes] = {}
    for name, param in model.named_parameters():
        if not any(s in name for s in name_substrings):
            continue
        # Fake (test) path: param has .data as bytes already.
        data_attr = getattr(param, "data", None)
        if isinstance(data_attr, (bytes, bytearray)):
            out[name] = bytes(data_attr)
            continue
        # Real torch path: param.data is a Tensor.
        try:
            out[name] = _tensor_to_comparable_bytes(param.data)
        except Exception as exc:  # pragma: no cover — only triggers for exotic dtypes
            log.warning("Could not snapshot bytes for projector param %s: %s", name, exc)
    return out


def _tensor_to_comparable_bytes(tensor) -> bytes:
    """Return stable CPU bytes for tensor equality checks.

    PyTorch bf16 tensors cannot be converted to NumPy directly
    (``TypeError: Got unsupported ScalarType BFloat16``), but the export
    merge path loads the base model as bf16. Cast only for the comparison
    representation; the model tensors themselves are not mutated.
    """
    t = tensor.detach().to("cpu").contiguous()
    try:
        return t.numpy().tobytes()
    except TypeError as exc:
        if "BFloat16" not in str(exc) and "bfloat16" not in str(exc):
            raise
        return t.float().contiguous().numpy().tobytes()


def _assert_projector_changed_if_tuned(
    base_model,
    merged_model,
    adapter_dir: "Path | str | None",
) -> None:
    """Tripwire: detect silently-dropped modules_to_save weights at export.

    If the adapter dir contains projector tensors (= ``modules_to_save``
    fired during training and PEFT serialized full projector weights),
    then the projector tensors of ``merged_model`` MUST differ from
    those of ``base_model``. Identical tensors mean PEFT silently lost
    the modules_to_save weights at load time — a ship-stopper, because
    the merged model is no better than a LoRA-only run.

    No-ops in three cases:
      * ``adapter_dir`` is None or doesn't exist.
      * The adapter contains no projector tensors (LoRA-only flow).
      * The merged/base model has no projector params to compare against
        (logged warning; ``_model_has_vision_tower`` already handles
        the more catastrophic 'no vision at all' case).
    """
    adapter_keys = _adapter_has_projector_tensors(
        Path(adapter_dir) if adapter_dir is not None else None
    )
    if not adapter_keys:
        log.debug(
            "Projector-changed tripwire: adapter has no projector tensors "
            "(LoRA-only adapter or merge from --merged_dir). Skipping check."
        )
        return

    log.info(
        "Projector-changed tripwire: adapter contains %d projector tensor(s) "
        "(modules_to_save was used during training). Verifying they were "
        "applied to the merged model.",
        len(adapter_keys),
    )

    name_substrings = (
        "embed_vision.",
        "vision_tower.embedding_projection.",
        "vision_tower.projection.",
    )
    base_bytes = _model_param_data_bytes(base_model, name_substrings)
    merged_bytes = _model_param_data_bytes(merged_model, name_substrings)

    if not base_bytes or not merged_bytes:
        log.warning(
            "Projector-changed tripwire: could not locate projector params "
            "in base (%d) or merged (%d) model for comparison — skipping. "
            "If this happens during a real export, investigate the merge step.",
            len(base_bytes),
            len(merged_bytes),
        )
        return

    unchanged: list[str] = []
    compared = 0
    for name, b_bytes in base_bytes.items():
        if name not in merged_bytes:
            continue
        compared += 1
        if merged_bytes[name] == b_bytes:
            unchanged.append(name)

    if compared == 0:
        log.warning(
            "Projector-changed tripwire: no projector param names matched "
            "between base and merged models. Skipping byte-diff."
        )
        return

    if len(unchanged) == compared:
        sample = ", ".join(unchanged[:5])
        raise RuntimeError(
            f"Projector-changed tripwire FIRED: all {compared} projector "
            f"parameter(s) in the merged model are byte-identical to the "
            f"base model, but the LoRA adapter directory contains "
            f"{len(adapter_keys)} projector tensor(s) (modules_to_save was "
            f"used during training). PEFT silently failed to restore the "
            f"full-param projector weights at load time. Identical params: "
            f"{sample}"
            + (" ..." if len(unchanged) > 5 else "")
        )

    log.info(
        "Projector-changed tripwire passed: %d/%d projector params differ "
        "from base (as expected after full-param tuning).",
        compared - len(unchanged),
        compared,
    )


# ---------------------------------------------------------------------------
# Vision-layer-changed tripwire
# (feature/lora-plus-projector-plus-vision-tower)
# ---------------------------------------------------------------------------


# Regex anchored to a path-component boundary, matching the matcher in
# src/vision_layers.py and src/freeze.py. Captures the layer index.
_VISION_LAYER_IDX_RE = re.compile(
    r"(?:^|\.)vision_tower\.encoder\.layers\.(\d+)(?:\.|$)"
)


def _adapter_has_vision_layer_tensors(
    adapter_dir: "Path | str | None",
) -> dict[int, list[str]]:
    """Detect vision-encoder-layer tensors saved in the adapter directory.

    Returns a ``{layer_index: [tensor_keys]}`` mapping. An empty dict
    means the adapter is LoRA-only or projector-only — in both cases
    the caller treats the tripwire as a no-op.

    Excludes PEFT's ``.original_module.`` frozen reference copies — only
    ``.modules_to_save.{adapter}.`` copies count as "tuned".

    Reads only safetensors headers (no tensor data loaded).
    """
    if adapter_dir is None:
        return {}
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        return {}

    found: dict[int, list[str]] = {}
    for sf in sorted(adapter_path.glob("*.safetensors")):
        try:
            with sf.open("rb") as fin:
                _data_start, header = _read_safetensors_header(fin)
        except Exception as exc:
            log.warning("Could not read safetensors header for %s: %s", sf, exc)
            continue
        for key in header:
            if key == "__metadata__":
                continue
            # Skip PEFT's frozen reference copy.
            if ".original_module." in key:
                continue
            m = _VISION_LAYER_IDX_RE.search(key)
            if m is None:
                continue
            idx = int(m.group(1))
            found.setdefault(idx, []).append(key)
    return found


def _assert_vision_layers_changed_if_tuned(
    base_model,
    merged_model,
    adapter_dir: "Path | str | None",
) -> None:
    """Tripwire: detect silently-dropped vision-layer modules_to_save weights.

    Parallel to ``_assert_projector_changed_if_tuned``. If the adapter
    dir contains vision-encoder-layer tensors
    (= ``tune_last_n_vision_layers > 0`` and PEFT serialized the
    ``modules_to_save`` wrapper for those layers), then for EACH tuned
    layer index the merged model's params under that layer MUST differ
    byte-for-byte from the base model's. If every comparable param of
    even one tuned layer is identical, fire — PEFT silently lost the
    full-param vision-layer weights and the export is wasted.

    No-ops for:
      * ``adapter_dir`` is None or missing.
      * The adapter contains no vision-layer tensors (LoRA-only or
        projector-only).
      * Neither base nor merged model has comparable params under those
        indices (logged warning).
    """
    by_idx = _adapter_has_vision_layer_tensors(adapter_dir)
    if not by_idx:
        log.debug(
            "Vision-layer-changed tripwire: adapter has no vision-encoder-layer "
            "tensors (LoRA-only / projector-only adapter or merge from "
            "--merged_dir). Skipping check."
        )
        return

    tuned_indices = sorted(by_idx.keys())
    log.info(
        "Vision-layer-changed tripwire: adapter contains tensors for layer "
        "indices %s (%d total tensors). Verifying merged model weights changed.",
        tuned_indices,
        sum(len(v) for v in by_idx.values()),
    )

    # Snapshot bytes for any param whose name resolves to a tuned index.
    base_bytes = _model_param_data_bytes_by_vision_layer(base_model, tuned_indices)
    merged_bytes = _model_param_data_bytes_by_vision_layer(merged_model, tuned_indices)

    if not base_bytes or not merged_bytes:
        log.warning(
            "Vision-layer-changed tripwire: could not locate vision encoder "
            "layer params in base (%d) or merged (%d) model for comparison — "
            "skipping. If this happens during a real export, investigate the "
            "merge step.",
            sum(len(v) for v in base_bytes.values()),
            sum(len(v) for v in merged_bytes.values()),
        )
        return

    # Per-tuned-index: if EVERY comparable param is byte-identical, fire.
    # A single changed param within a layer is enough to count as "tuned"
    # (PEFT may have partially restored).
    unchanged_layers: list[int] = []
    for idx in tuned_indices:
        base_for_idx = base_bytes.get(idx, {})
        merged_for_idx = merged_bytes.get(idx, {})
        common = set(base_for_idx) & set(merged_for_idx)
        if not common:
            continue
        if all(base_for_idx[n] == merged_for_idx[n] for n in common):
            unchanged_layers.append(idx)

    if unchanged_layers:
        raise RuntimeError(
            f"Vision-layer-changed tripwire FIRED: vision encoder layers "
            f"{unchanged_layers} have byte-identical params in merged vs "
            f"base model, but the LoRA adapter directory contains "
            f"modules_to_save tensors for those layers "
            f"(tune_last_n_vision_layers was on). PEFT silently failed to "
            f"restore the full-param vision-layer weights at load time. "
            f"DO NOT SHIP — re-train or investigate."
        )

    total_compared = sum(
        len(set(base_bytes.get(idx, {})) & set(merged_bytes.get(idx, {})))
        for idx in tuned_indices
    )
    log.info(
        "Vision-layer-changed tripwire passed: all %d tuned layer(s) show "
        "changed params (compared %d total tensors).",
        len(tuned_indices),
        total_compared,
    )


def _model_param_data_bytes_by_vision_layer(
    model,
    tuned_indices: "list[int] | tuple[int, ...]",
) -> dict[int, dict[str, bytes]]:
    """Collect raw bytes of model params grouped by vision encoder layer index.

    For each param whose name contains ``vision_tower.encoder.layers.{i}.``
    at a path boundary with `i` in `tuned_indices` (and NOT inside
    ``.original_module.``), record the bytes under ``out[i][name]``.

    Same data-extraction rules as ``_model_param_data_bytes``: supports
    both real torch.nn.Parameter (``.data`` is a Tensor) and the test
    fake (``.data`` is bytes).
    """
    tuned_set = set(tuned_indices)
    out: dict[int, dict[str, bytes]] = {}
    for name, param in model.named_parameters():
        if ".original_module." in name:
            continue
        m = _VISION_LAYER_IDX_RE.search(name)
        if m is None:
            continue
        idx = int(m.group(1))
        if idx not in tuned_set:
            continue
        data_attr = getattr(param, "data", None)
        if isinstance(data_attr, (bytes, bytearray)):
            out.setdefault(idx, {})[name] = bytes(data_attr)
            continue
        try:
            out.setdefault(idx, {})[name] = _tensor_to_comparable_bytes(param.data)
        except Exception as exc:  # pragma: no cover
            log.warning(
                "Could not snapshot bytes for vision-layer param %s: %s",
                name, exc,
            )
    return out


# ---------------------------------------------------------------------------
# Step 2 & 3: Convert to MLX format with quantization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trained vision input shape (matches scripts/fetch-gemma.sh in the iOS app).
#
# Gemma 4's trained vision pooler is a kernel=3 stride pooler over the
# (60, 42) (or (42, 60)) bucket grid that exactly fills the max_patches=2520
# budget (60×42=2520 → 20×14=280 cleanly-pooled soft tokens). At 224×224 the
# pooler degenerates to identity (14×14=196 raw + 84 zero outputs) and the
# language model sees out-of-distribution inputs.
#
# mlx-swift-lm's Gemma4Processor.preprocess reads the top-level `size` field
# from processor_config.json. If `size` is missing, it falls back to 800×800
# (also wrong). We therefore force the trained shape into the exported
# processor config after MLX conversion. This MUST stay in sync with
# the app model-fetch script's trained-size patch.
# ---------------------------------------------------------------------------
TRAINED_VISION_SIZE = {"height": 960, "width": 672}


def convert_to_mlx(merged_dir: Path, output_dir: str, quantize_bits: int) -> Path:
    """Convert a HuggingFace **multimodal** Gemma 4 directory to MLX format.

    Uses ``mlx_vlm.convert`` (NOT ``mlx_lm.convert``). mlx_lm is a
    language-only converter and silently drops vision_tower / embed_vision
    weights — the same fault mode as AutoModelForCausalLM at load time.
    For VLM checkpoints we must use mlx_vlm so the vision sub-graph is
    preserved end-to-end.
    """
    log.info("=== Step 2–3: Converting to MLX VLM (quantize=%d-bit) ===", quantize_bits)

    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        log.error(
            "mlx_vlm is required for MLX VLM conversion.\n"
            "This step must run on a Mac with Apple Silicon.\n"
            "Install: pip install mlx mlx-vlm\n\n"
            "DO NOT use mlx_lm.convert here — it is a language-only converter\n"
            "and will silently drop the vision_tower / embed_vision weights,\n"
            "producing a checkpoint that the iOS VLM path cannot serve.\n\n"
            "If you're on the training box (NVIDIA GPU), use --merge_only to\n"
            "save the merged safetensors, then copy them to a Mac and run:\n"
            "    python src/export_mlx.py --merged_dir <path> --output_dir <path> --quantize_bits %d",
            quantize_bits,
        )
        sys.exit(1)

    mlx_dir = Path(output_dir) / "mlx"
    mlx_dir.mkdir(parents=True, exist_ok=True)

    log.info("Input:  %s", merged_dir)
    log.info("Output: %s", mlx_dir)

    # mlx_vlm exposes a CLI compatible with mlx_lm's convert flags, plus a
    # vision-aware sanitize step. Invoking via subprocess for the same
    # reasons as before — the python API is not version-stable.
    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_vlm.convert",
        "--hf-path", str(merged_dir),
        "--mlx-path", str(mlx_dir),
        "-q",
        "--q-bits", str(quantize_bits),
    ]
    log.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("mlx_vlm.convert failed:\nstdout: %s\nstderr: %s", result.stdout, result.stderr)
        sys.exit(1)

    if result.stdout:
        log.info("mlx_vlm output:\n%s", result.stdout.strip())

    # Sanity check: confirm the converted safetensors actually contain
    # vision_tower keys. If they don't, something silently downgraded the
    # model to language-only — fail loudly rather than ship a broken bundle.
    if not _mlx_dir_has_vision_weights(mlx_dir):
        log.error(
            "Converted MLX directory at %s contains NO vision_tower / "
            "embed_vision weights. The conversion silently dropped the "
            "vision sub-graph. Likely causes: merged base was language-only "
            "(check Bug 1 in export pipeline), or mlx_vlm version mismatch.",
            mlx_dir,
        )
        sys.exit(1)

    # Patch processor_config.json so mlx-swift-lm reads the trained 960x672
    # shape. mlx_vlm.convert preserves the upstream HF processor config
    # verbatim; that config has no top-level `size` field, so the iOS
    # runtime would fall back to 800x800. See TRAINED_VISION_SIZE above.
    patch_processor_config_for_mlx_swift(mlx_dir)

    mlx_size = sum(f.stat().st_size for f in mlx_dir.rglob("*") if f.is_file())
    log.info("MLX model saved: %s total", fmt_bytes(mlx_size))

    return mlx_dir


def _mlx_dir_has_vision_weights(mlx_dir: Path) -> bool:
    """Return True if any safetensors file in `mlx_dir` exposes a vision key.

    Reads only safetensors headers — does not load tensor data. Robust to
    sharded checkpoints (model-00001-of-N.safetensors).
    """
    safetensors_files = sorted(mlx_dir.glob("*.safetensors"))
    if not safetensors_files:
        return False
    for sf in safetensors_files:
        try:
            with sf.open("rb") as fin:
                _data_start, header = _read_safetensors_header(fin)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not read safetensors header for %s: %s", sf, exc)
            continue
        for key in header:
            if key == "__metadata__":
                continue
            if "vision_tower." in key or "embed_vision." in key:
                return True
    return False


def patch_processor_config_for_mlx_swift(mlx_dir: Path) -> bool:
    """Force the trained vision input shape into processor_config.json.

    Hoists `do_normalize` / `image_mean` / `image_std` from the nested
    `image_processor` block to the top level (mlx-swift-lm's
    `Gemma4ProcessorConfiguration` decoder reads them from the top level
    only) and overrides the top-level + nested `size` to 960x672.

    This MUST mirror the app model-fetch script's trained-size patch.

    Returns True if the file was modified, False if no changes were needed
    or the file does not exist.
    """
    pcfg_path = mlx_dir / "processor_config.json"
    if not pcfg_path.exists():
        log.warning(
            "No processor_config.json at %s — skipping mlx-swift-lm size patch. "
            "iOS will fall back to 800x800 unless you patch this manually.",
            pcfg_path,
        )
        return False

    with pcfg_path.open() as f:
        cfg = json.load(f)

    image_processor = cfg.get("image_processor", {}) or {}
    patch = {
        "do_normalize": image_processor.get("do_normalize", False),
        "image_mean": image_processor.get("image_mean", [0.0, 0.0, 0.0]),
        "image_std": image_processor.get("image_std", [1.0, 1.0, 1.0]),
        "size": dict(TRAINED_VISION_SIZE),
    }
    changed: list[str] = []
    for k, v in patch.items():
        if cfg.get(k) != v:
            cfg[k] = v
            changed.append(k)
    if image_processor and image_processor.get("size") != TRAINED_VISION_SIZE:
        image_processor["size"] = dict(TRAINED_VISION_SIZE)
        cfg["image_processor"] = image_processor
        changed.append("image_processor.size")

    if changed:
        with pcfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        log.info(
            "Patched %s for mlx-swift-lm: %s",
            pcfg_path.name,
            ", ".join(changed),
        )
        return True

    log.info("processor_config.json already matches trained shape — no patch needed.")
    return False


# ---------------------------------------------------------------------------
# Step 4: Optional audio tower stripping
# ---------------------------------------------------------------------------


def strip_audio_from_mlx_dir(mlx_dir: Path) -> None:
    """Strip audio tower weights from all safetensors files in the MLX output."""
    log.info("=== Step 4: Stripping audio tower weights ===")

    safetensors_files = list(mlx_dir.glob("*.safetensors"))
    if not safetensors_files:
        log.warning("No safetensors files found in %s", mlx_dir)
        return

    stripped_any = False
    for sf in safetensors_files:
        if strip_audio_tower(sf):
            stripped_any = True

    if not stripped_any:
        log.info("No audio tower weights found in any safetensors file.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export finetuned Gemma 4 to MLX format for iOS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Full pipeline (Mac with torch + mlx):
  python src/export_mlx.py \\
      --base_model google/gemma-4-e2b-it \\
      --adapter_path outputs/hike-gemma4-lora \\
      --output_dir exports/gemma4-mlx \\
      --strip_audio

  # Merge only (training box):
  python src/export_mlx.py \\
      --base_model google/gemma-4-e2b-it \\
      --adapter_path outputs/hike-gemma4-lora \\
      --output_dir exports/gemma4-merged \\
      --merge_only

  # MLX convert only (Mac, from pre-merged safetensors):
  python src/export_mlx.py \\
      --merged_dir exports/gemma4-merged/merged \\
      --output_dir exports/gemma4-mlx \\
      --quantize_bits 4

After export, copy the MLX directory into the iOS app bundle:
  cp -R exports/gemma4-mlx/mlx/* \\
      ../hikeCompanion/HikeCompanion/Resources/Models/Gemma/
""",
    )

    # Source model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/gemma-4-E2B-it",
        help=(
            "HuggingFace model ID or local path for the base model. "
            "MUST match the base used during training in configs/default.yaml; "
            "the LoRA adapter only merges cleanly against the same checkpoint."
        ),
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default=None,
        help="Path to an already-merged model directory (skip merge step).",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory for exported artifacts.",
    )

    # Quantization
    parser.add_argument(
        "--quantize_bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Quantization bit width for MLX conversion (default: 4).",
    )

    # Workflow control
    parser.add_argument(
        "--merge_only",
        action="store_true",
        default=False,
        help="Only merge the adapter (skip MLX conversion). Use this on the training box.",
    )
    parser.add_argument(
        "--strip_audio",
        action="store_true",
        default=False,
        help="Strip audio tower weights from the final safetensors.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the merged model directory
    merged_dir: Path | None = None

    if args.merged_dir:
        # Skip merge — user provided a pre-merged directory
        merged_dir = Path(args.merged_dir)
        if not merged_dir.is_dir():
            log.error("--merged_dir does not exist: %s", merged_dir)
            sys.exit(1)
        log.info("Using pre-merged model at %s (skipping merge step).", merged_dir)
    elif args.adapter_path:
        # Step 1: Merge
        merged_dir = merge_adapter(args.base_model, args.adapter_path, str(output_dir))
    else:
        parser.error("Provide either --adapter_path (to merge) or --merged_dir (pre-merged).")

    if args.merge_only:
        log.info("=== Done (merge only). Merged model at: %s ===", merged_dir)
        log.info(
            "Copy this directory to a Mac and run:\n"
            "  python src/export_mlx.py --merged_dir %s --output_dir <path> --quantize_bits %d",
            merged_dir,
            args.quantize_bits,
        )
        return

    # Steps 2–3: MLX conversion
    mlx_dir = convert_to_mlx(merged_dir, str(output_dir), args.quantize_bits)

    # Step 4: Optional audio stripping
    if args.strip_audio:
        strip_audio_from_mlx_dir(mlx_dir)

    # Final summary
    final_size = sum(f.stat().st_size for f in mlx_dir.rglob("*") if f.is_file())
    print()
    print("=" * 60)
    print("  EXPORT COMPLETE")
    print("=" * 60)
    print(f"  MLX model directory : {mlx_dir}")
    print(f"  Total size          : {fmt_bytes(final_size)}")
    print()
    print("  To deploy to the iOS app:")
    print(f"    cp -R {mlx_dir}/* \\")
    print("        ../hikeCompanion/HikeCompanion/Resources/Models/Gemma/")
    print()
    print("  Then regenerate the Xcode project:")
    print("    cd ../hikeCompanion && bash scripts/generate-project.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
