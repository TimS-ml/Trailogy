"""GPTQ — calibration-based post-training quantization.

GPTQ takes a small calibration set, runs forward through the model
capturing per-layer activations, then uses OBS-style Hessian inversion
to find the optimal 4-bit weight assignment per layer that minimizes
reconstruction error.

Backend: ``gptqmodel`` >= 7.0.0. Has native
``Gemma4ForConditionalGenerationGPTQ`` support — we don't need to
fall back to language-only quantization. End-to-end multimodal GPTQ
runs out of the box.

Hardware: NVIDIA CUDA. The bf16 base + activations fit in ~12 GB at
typical calibration shapes. Quantization itself uses
``offload_to_disk=True`` so peak VRAM stays bounded.

Output format: gptqmodel native (HF-compatible safetensors with
INT4 packed weights + scales/zeros + a ``quantize_config.json``
sidecar). NOT directly MLX-loadable — for the iOS deliverable we
need a separate bridge step (or evaluate as a separate variant in
the eval matrix without shipping it).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Calibration logic lives in a route-agnostic module so AWQ / dynamic_quant
# (and the future B.1 bridge) can reuse the same loaders + leak guards.
from src.common.calibration import (
    CalibrationDataLeakError,
    build_calibration_dataset as _build_calibration_dataset,
    load_plantnet_calibration as _load_plantnet_calibration,
    load_text_calibration as _load_text_calibration,
    reject_calibration_leak as _reject_calibration_leak,
)

log = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    # Quantization knobs (map to gptqmodel.QuantizeConfig)
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False  # Activation-order quant. Slower, slightly better accuracy.
    sym: bool = True
    true_sequential: bool = True
    lm_head: bool = False  # Keep lm_head at higher precision (Unsloth-UD style).

    # Hessian dampening for OBS inverse. The upstream gptqmodel 7.0
    # default is 0.05; we lower it to 0.01 because Gemma 4 E2B
    # KV-shared layers (15-34) ship narrower Hessians and the higher
    # damp washes out their structure. If Cholesky fails on a layer
    # gptqmodel auto-increments by ``damp_auto_increment`` per retry.
    # Both are exposed so the YAML config can record what was actually
    # used (the saved ``quantize_config.json`` only reports the post-
    # auto-increment value, not our starting choice).
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025

    # Calibration data
    n_calib_plantnet: int = 256
    n_calib_text: int = 256
    calib_seq_len: int = 1024
    calib_seed: int = 0

    backend: Literal["gptqmodel", "auto_gptq", "optimum"] = "gptqmodel"
    offload_to_disk: bool = True
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Calibration data builders
#
# Implementation moved to ``src.common.calibration``. Symbols
# re-exported above so existing callers / tests keep working without
# import-site changes.
# ---------------------------------------------------------------------------


def build_calibration_dataset(
    config: "GPTQConfig",
    plantnet_jsonl: Path | str | None,
    tokenizer: Any | None = None,
) -> list[dict]:
    """Concatenate domain (PlantNet) and general-text (WikiText) calibration.

    Thin wrapper around ``common.calibration.build_calibration_dataset``
    that unpacks fields from a ``GPTQConfig``. ``tokenizer`` is forwarded
    so ``apply_chat_template`` is used when available (else the loader
    falls back to a plain role-prefixed flatten).
    """
    return _build_calibration_dataset(
        n_calib_plantnet=config.n_calib_plantnet,
        n_calib_text=config.n_calib_text,
        calib_seq_len=config.calib_seq_len,
        calib_seed=config.calib_seed,
        plantnet_jsonl=plantnet_jsonl,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_lm_head(config: GPTQConfig, input_dir: Path) -> bool:
    """Check if lm_head=True is compatible with the model's config.

    Gemma 4 (and many other models) use ``tie_word_embeddings=True``,
    which means the lm_head shares weights with the embedding layer.
    gptqmodel raises ``NotImplementedError`` when attempting to quantize
    a tied lm_head. This helper detects the conflict upfront and
    downgrades to ``lm_head=False`` with a warning.

    Returns:
        The resolved lm_head value (True or False).
    """
    if not config.lm_head:
        return False

    config_path = Path(input_dir) / "config.json"
    if not config_path.is_file():
        return config.lm_head

    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return config.lm_head

    tied = cfg.get("tie_word_embeddings", False)
    if tied:
        log.warning(
            "lm_head=True requested but model has tie_word_embeddings=True. "
            "GPTQ cannot quantize a tied lm_head — downgrading to lm_head=False."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Quantization entry point
# ---------------------------------------------------------------------------


def quantize(
    input_dir: Path | str,
    output_dir: Path | str,
    config: GPTQConfig | None = None,
    plantnet_jsonl: Path | str | None = None,
) -> Path:
    """Run GPTQ on a bf16 multimodal Gemma 4 E2B and write the
    quantized checkpoint to ``output_dir``.

    Args:
        input_dir: Path to the bf16 merged model directory (or a HF
            repo id like ``unsloth/gemma-4-E2B-it``).
        output_dir: Where to write the GPTQ-quantized output.
        config: GPTQ knobs. Defaults to bits=4, group=128, desc_act=False.
        plantnet_jsonl: Optional path to PlantNet train.jsonl for
            domain-aware calibration. Eval/test sources are rejected.

    Returns:
        Path to the saved quantized model directory.
    """
    config = config or GPTQConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.backend != "gptqmodel":
        raise NotImplementedError(
            f"Only the 'gptqmodel' backend is implemented today (requested: {config.backend!r}). "
            "auto_gptq and optimum paths are stubs in 02-methods.md."
        )

    # Guard: lm_head=True is incompatible with tied embeddings.
    effective_lm_head = _resolve_lm_head(config, input_dir)

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as e:
        raise ImportError(
            "gptqmodel is required for GPTQ. Install via `pip install gptqmodel`. "
            "On this box you may also need `LD_LIBRARY_PATH=<env>/lib:$LD_LIBRARY_PATH` "
            "to resolve GLIBCXX. See AGENTS.md."
        ) from e

    # Load the tokenizer up front so calibration text is formatted with
    # the model's real chat template (Gemma 4's <start_of_turn>...). The
    # previous hard-coded "<|turn>{role}\n..." marker did not match the
    # tokenizer's expectations — calibration saw a different token
    # distribution than inference, biasing OBS toward irrelevant stats.
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(input_dir), trust_remote_code=True
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "Failed to load tokenizer for chat-template formatting (%s); "
            "calibration will fall back to raw role:content flattening.",
            e,
        )

    log.info("Building calibration dataset...")
    calib = build_calibration_dataset(
        config, plantnet_jsonl, tokenizer=tokenizer
    )
    if not calib:
        raise RuntimeError(
            "Calibration set is empty. Provide --plantnet_jsonl and/or ensure "
            "`datasets` is installed for WikiText fallback."
        )

    qcfg = QuantizeConfig(
        bits=config.bits,
        group_size=config.group_size,
        desc_act=config.desc_act,
        sym=config.sym,
        true_sequential=config.true_sequential,
        lm_head=effective_lm_head,
        damp_percent=config.damp_percent,
        damp_auto_increment=config.damp_auto_increment,
        offload_to_disk=config.offload_to_disk,
    )

    log.info("Loading bf16 model via gptqmodel.GPTQModel.load(%s)", input_dir)
    model = GPTQModel.load(
        str(input_dir),
        quantize_config=qcfg,
        trust_remote_code=True,
    )

    # gptqmodel.BaseQModel.quantize takes calibration as List[{"text": str}]
    # or List[{"input_ids": List[int]}]. We pass text and let it tokenize
    # via its own tokenizer.
    log.info("Running GPTQ quantization (bits=%d, group=%d, desc_act=%s)",
             config.bits, config.group_size, config.desc_act)
    model.quantize(
        calibration=calib,
        batch_size=config.batch_size,
    )

    log.info("Saving quantized model to %s", output_dir)
    model.save_quantized(str(output_dir))

    # gptqmodel.save_quantized writes config.json + the sharded
    # safetensors + tokenizer.json + tokenizer_config.json, but it does
    # NOT carry forward the multimodal `processor_config.json` (the
    # image preprocessor side-car). Without it, the eval step's
    # `AutoProcessor.from_pretrained(output_dir)` aborts with
    # "Can't load feature extractor for ...". Mirror the side-cars from
    # the bf16 input dir so the GPTQ output dir is self-contained.
    from src.common.model_io import copy_processor_assets
    copied = copy_processor_assets(input_dir, output_dir)
    if copied:
        log.info("Copied %d processor side-car(s) from %s: %s",
                 len(copied), input_dir, copied)
    else:
        log.warning(
            "No processor side-cars found in %s — downstream "
            "AutoProcessor.from_pretrained(%s) may fail.",
            input_dir, output_dir,
        )
    return output_dir


# ---------------------------------------------------------------------------
# Smoke check (no full model load)
# ---------------------------------------------------------------------------


def smoke_check(backend: str = "gptqmodel") -> dict:
    """Verify the chosen GPTQ backend can recognize Gemma 4 multimodal."""
    notes: list[str] = []
    result = {"backend_available": False, "architecture_recognized": False, "notes": ""}

    if backend == "gptqmodel":
        try:
            import gptqmodel
            from gptqmodel.models import MODEL_MAP

            result["backend_available"] = True
            notes.append(f"gptqmodel version: {gptqmodel.__version__}")
            if "gemma4" in MODEL_MAP:
                notes.append(
                    f"native Gemma 4 GPTQ support: {MODEL_MAP['gemma4'].__name__}"
                )
        except ImportError as e:
            notes.append(f"gptqmodel import failed: {e}")
    elif backend == "auto_gptq":
        try:
            import auto_gptq  # noqa: F401

            result["backend_available"] = True
            notes.append(f"auto_gptq version: {auto_gptq.__version__}")
        except ImportError as e:
            notes.append(f"auto_gptq import failed: {e}")
    elif backend == "optimum":
        try:
            from optimum.gptq import GPTQQuantizer  # noqa: F401

            result["backend_available"] = True
            notes.append("optimum.gptq available")
        except ImportError as e:
            notes.append(f"optimum.gptq import failed: {e}")
    else:
        notes.append(f"Unknown backend: {backend}")

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("unsloth/gemma-4-E2B-it", trust_remote_code=True)
        notes.append(f"HF config loads OK: architectures={cfg.architectures}")
        if cfg.architectures and "Gemma4" in cfg.architectures[0]:
            result["architecture_recognized"] = True
    except Exception as e:  # noqa: BLE001
        notes.append(f"HF config load failed: {e}")

    result["notes"] = " | ".join(notes)
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="GPTQ smoke/quant CLI")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--backend", default="gptqmodel", choices=("gptqmodel", "auto_gptq", "optimum")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.smoke:
        result = smoke_check(args.backend)
        for k, v in result.items():
            print(f"{k}: {v}")
        return (
            0
            if result["backend_available"] and result["architecture_recognized"]
            else 1
        )
    print(
        "Pass --smoke for the availability check. For full GPTQ runs, "
        "use `python -m scripts.run.quant --method gptq ...`."
    )
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
