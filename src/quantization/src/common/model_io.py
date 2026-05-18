"""Shared model I/O for quantization methods.

Most methods need the same input: a bf16 multimodal Gemma 4 E2B
checkpoint, optionally with a LoRA adapter merged in. This module
provides one well-tested loader that preserves vision_tower +
embed_vision (the trap that ``AutoModelForCausalLM`` walks into).

For the existing baseline path, see ``finetune/src/export_mlx.py``.
This module re-exposes its merge logic for reuse by quantization
methods that need access to the merged bf16 model in memory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


# All processor/tokenizer side-cars we mirror from a bf16 source dir
# into a PTQ output dir.
#
# Background: every method that writes its own model files via a third-
# party serializer (``gptqmodel.save_quantized``, bitsandbytes'
# ``save_pretrained`` under bnb quant, ``mlx_vlm.convert``) needs to
# carry these forward — none of those serializers preserve the full
# multimodal processor side-car set on their own.
#
# Empty files / files that don't exist in the source dir are silently
# skipped so the helper is safe to call unconditionally regardless of
# tokenizer flavor (e.g. ``tokenizer.model`` only exists when the slow
# SentencePiece tokenizer is shipped; Gemma 4 ships only the fast
# ``tokenizer.json``).
#
# **This is the canonical list for the entire quantization/ package.**
# ``scripts/merge_safetensors.py`` and ``scripts/splice_lm_into_multimodal.py``
# import this name (and may extend it with BPE-only files like
# ``vocab.json`` / ``merges.txt`` if the use case calls for it).
PROCESSOR_SIDECAR_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
)

# Back-compat private alias — kept so internal callers that imported the
# underscore name keep working. New code should use the public name.
_PROCESSOR_FILES_TO_COPY = PROCESSOR_SIDECAR_FILES


def copy_processor_assets(src_dir: Path | str, dst_dir: Path | str) -> list[str]:
    """Mirror processor / tokenizer side-cars from ``src_dir`` into
    ``dst_dir``. Returns the list of file basenames actually copied
    (for logging).

    PTQ methods MUST call this after their own ``save_*`` to keep
    downstream evals (``AutoProcessor.from_pretrained`` on the output
    dir) working. Without it, ``processor_config.json`` is missing
    from the quantized dir for multimodal Gemma 4 checkpoints and
    ``run_eval`` crashes with "Can't load feature extractor for ..."
    (observed 2026-05-13 on the first GPTQ w4g128 run).

    Args:
        src_dir: Directory containing the side-cars (usually the
            bf16 merged input dir).
        dst_dir: Where the quantized weights were just saved. Created
            by the caller's save step; this helper does NOT mkdir it.

    Returns:
        List of file basenames copied. Empty if ``src_dir`` had no
        relevant side-cars (no error).
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    copied: list[str] = []
    for name in PROCESSOR_SIDECAR_FILES:
        s = src / name
        if not s.is_file():
            continue
        shutil.copyfile(s, dst / name)
        copied.append(name)
    return copied


def load_bf16_multimodal(
    base_model: str,
    adapter_path: str | None = None,
    device: str = "cpu",
):
    """Load the multimodal Gemma 4 E2B as bf16 with adapter merged.

    Mirrors the import + merge pattern in
    ``finetune/src/export_mlx.py:merge_adapter``. We intentionally do
    NOT import torch / transformers at module top-level so that
    ``quantization/`` is importable on a Mac without CUDA (for shape /
    size analysis on existing safetensors files).

    Args:
        base_model: HF repo id or local path to the bf16 base.
        adapter_path: Path to a PEFT LoRA adapter directory. If None,
            loads the base only.
        device: "cpu" (safe default, ~20 GB RAM) or "cuda" (faster but
            needs ~12 GB free VRAM for the bf16 base alone).

    Returns:
        A ``Gemma4ForConditionalGeneration`` instance with adapter
        merged. ``vision_tower.*`` and ``embed_vision.*`` are preserved
        per the tripwire in ``_model_has_vision_tower``.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    torch_dtype = torch.bfloat16
    log.info(
        "Loading base model %s (dtype=bf16, device=%s)", base_model, device
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device,
    )

    if not _model_has_vision_tower(model):
        raise RuntimeError(
            f"Base model {base_model!r} has no vision_tower params. "
            "Did you accidentally load via AutoModelForCausalLM somewhere? "
            "See finetune/src/export_mlx.py:_model_has_vision_tower."
        )

    if adapter_path is None:
        log.info("No adapter path provided; returning base bf16 only.")
        return model

    from peft import PeftModel

    log.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    log.info("Merging adapter into base...")
    model = model.merge_and_unload()

    if not _model_has_vision_tower(model):
        raise RuntimeError(
            "Post-merge model lost vision_tower params. Typical cause: "
            "PEFT/transformers version mismatch silently drops adapter "
            "tensors at reload time (the 'orphan-tensor' bug). Re-pin "
            "package versions from finetune/ and retry."
        )

    return model


def _model_has_vision_tower(model) -> bool:
    """True iff any parameter name contains ``vision_tower.`` or
    ``embed_vision.``. Mirrors ``export_mlx.py:_model_has_vision_tower``.
    """
    for name, _ in model.named_parameters():
        if "vision_tower." in name or "embed_vision." in name:
            return True
    return False


def _load_processor(source: str | Path):
    """Load an ``AutoProcessor`` from a local dir or HF repo id.

    Isolated as a function so tests can mock it without going through
    ``transformers``' import chain.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(source), trust_remote_code=True)


def save_bf16_merged(
    model,
    output_dir: Path,
    processor_source: str | Path | None = None,
) -> Path:
    """Save the merged model in HF safetensors format. Output is
    consumable by ``mlx_vlm.convert`` and most quantization toolchains.

    Args:
        model: The merged ``Gemma4ForConditionalGeneration`` instance
            returned by ``load_bf16_multimodal``.
        output_dir: Where to write. Created if missing.
        processor_source: HF repo id or local path to load the
            ``AutoProcessor`` from. The processor's
            ``save_pretrained(output_dir)`` is called immediately after
            the model save, populating ``tokenizer.json``,
            ``tokenizer_config.json``, ``processor_config.json``, and
            ``chat_template.jinja``.

            If ``None``, falls back to copying processor files from
            ``model.config.name_or_path`` (the base model directory).
            If that also fails, a WARNING is logged.

    Returns:
        ``Path`` of ``output_dir`` (created).
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    if processor_source is not None:
        log.info("Saving processor side-cars from %s", processor_source)
        processor = _load_processor(processor_source)
        processor.save_pretrained(output_dir)
    else:
        # Fallback: copy processor files from the base model directory.
        # Uses the canonical sidecar list so the set stays in sync with
        # ``copy_processor_assets`` and the merge_safetensors / splice
        # tools.
        base_dir = Path(getattr(model.config, "name_or_path", ""))
        if base_dir.is_dir():
            copied = []
            for fname in PROCESSOR_SIDECAR_FILES:
                src = base_dir / fname
                dst = output_dir / fname
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied.append(fname)
            if copied:
                log.info("Copied processor side-cars from %s: %s", base_dir, ", ".join(copied))
            else:
                log.warning(
                    "save_bf16_merged: no processor_source given and no "
                    "side-cars found in %s. Downstream tools may fail.",
                    base_dir,
                )
        else:
            log.warning(
                "save_bf16_merged: no processor_source given. "
                "Output dir %s has only model weights — downstream tools "
                "(gptqmodel.GPTQModel.load, mlx_vlm.convert, "
                "AutoProcessor.from_pretrained) will fail. "
                "Pass processor_source=<adapter-or-base> to fix.",
                output_dir,
            )

    return output_dir
