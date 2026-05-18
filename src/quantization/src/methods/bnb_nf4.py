"""bitsandbytes NF4 — reference comparison only, NOT iOS-deployable.

NF4 is bitsandbytes' 4-bit format optimized for QLoRA-style training:
non-uniform quantization levels matching a fitted normal distribution.
MLX cannot load this format. We compute NF4 stats purely to put
"what does QLoRA-era SOTA quant cost on this model" into the eval
matrix.

Hardware: NVIDIA CUDA.

Useful because:
- One eval data point that the field considers "good 4-bit"
- Sanity check: if NF4 perplexity is wildly better than every MLX
  variant, the MLX recipe is the bottleneck, not 4-bit itself

Useless for shipping. Don't waste time tuning this.

Note on AGENTS.md [0]: this is a **post-training-quantization output
artifact**, not 4-bit training. The rule about avoiding 4/8-bit exists
to keep the bf16 SFT run free of memory-saver contamination (e.g.
adam_fp8 / QLoRA optimizers). Producing an NF4 PTQ checkpoint for the
eval matrix is in-scope for the quantization feature branch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# Shared with other PTQ methods (e.g. gptq). The bnb_nf4-specific
# rationale is: ``save_pretrained`` writes ``config.json`` + the
# quantized safetensors but does NOT carry forward the processor /
# tokenizer side-cars; we mirror them ourselves.
from src.common.model_io import (
    PROCESSOR_SIDECAR_FILES,
    copy_processor_assets,
)

# Back-compat alias kept for callers that imported the private name.
_PROCESSOR_FILES_TO_COPY = PROCESSOR_SIDECAR_FILES

log = logging.getLogger(__name__)


@dataclass
class BnBNF4Config:
    compute_dtype: str = "bfloat16"
    quant_storage_dtype: str = "uint8"
    double_quant: bool = True
    # Prefix-match skip list passed to
    # ``BitsAndBytesConfig.llm_int8_skip_modules``. transformers'
    # ``should_convert_module`` (in
    # ``transformers/quantizers/quantizers_utils.py``) does an anchored
    # prefix match (``re.match(f"{key}\\.", full_name)`` or
    # ``re.match(f"{key}", full_name)``) plus a suffix check
    # (``full_name.endswith(key)``). It is **NOT** a substring match —
    # passing ``"vision_tower"`` for the Gemma 4 multimodal architecture
    # is a silent no-op because every Linear inside the vision tower has
    # a full path like ``"model.vision_tower.encoder.layers.0.q_proj"``
    # which starts with ``"model."``, not ``"vision_tower."``.
    #
    # For convenience, ``quantize()`` re-maps a small set of well-known
    # logical names to their actual prefix paths before handing the list
    # to bnb. The supported logical names are:
    #
    #     vision_tower   → model.vision_tower
    #     embed_vision   → model.embed_vision
    #     audio_tower    → model.audio_tower
    #     embed_audio    → model.embed_audio
    #     language_model → model.language_model
    #     lm_head        → lm_head  (already a top-level child)
    #
    # Any string not in this map is passed through unchanged, so callers
    # can also supply raw module-path prefixes directly.
    #
    # Used for vision-collapse ablation experiments
    # (07-quantization/docs/06-bnb-nf4-vision-collapse.md): pass
    # ``["embed_vision"]`` to test whether the projector alone is the
    # culprit, or ``["vision_tower", "embed_vision"]`` to test the
    # union. The 0.1 % PlantNet baseline used skip_modules=None.
    skip_modules: list[str] | None = None


# Logical-name → real-prefix translation for Gemma 4
# ``ForConditionalGeneration``. See the docstring of
# ``BnBNF4Config.skip_modules`` for why this exists.
_SKIP_NAME_MAP = {
    "vision_tower": "model.vision_tower",
    "embed_vision": "model.embed_vision",
    "audio_tower": "model.audio_tower",
    "embed_audio": "model.embed_audio",
    "language_model": "model.language_model",
    "lm_head": "lm_head",
}


def _resolve_skip_names(names: list[str]) -> list[str]:
    """Translate user-supplied logical names to the actual full-path
    prefixes that transformers ``should_convert_module`` expects.

    Returned list preserves caller order. Unknown names pass through
    unchanged so raw paths still work.
    """
    return [_SKIP_NAME_MAP.get(n, n) for n in names]


def _build_bnb_quantization_config(config: BnBNF4Config):
    """Translate ``BnBNF4Config`` into a HF ``BitsAndBytesConfig``.

    Kept as a small pure function so the unit tests can validate the
    mapping without instantiating any model.
    """
    import torch
    from transformers import BitsAndBytesConfig

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "uint8": torch.uint8,
    }
    compute_dtype = dtype_map.get(config.compute_dtype)
    if compute_dtype is None:
        raise ValueError(
            f"Unsupported compute_dtype {config.compute_dtype!r}; "
            f"expected one of {sorted(dtype_map)}."
        )
    storage_dtype = dtype_map.get(config.quant_storage_dtype)
    if storage_dtype is None:
        raise ValueError(
            f"Unsupported quant_storage_dtype {config.quant_storage_dtype!r}; "
            f"expected one of {sorted(dtype_map)}."
        )

    kwargs = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_quant_storage=storage_dtype,
    )
    # ``llm_int8_skip_modules`` is the canonical bnb knob for "leave
    # these Linears at bf16". The flag name is historical (it
    # predates 4-bit support) but applies to both 8-bit and 4-bit.
    #
    # CRITICAL: when the caller supplies a skip list, transformers
    # disables its default ``lm_head`` auto-skip heuristic. Gemma 4
    # has ``tie_word_embeddings = True``, which means
    # ``lm_head.weight`` and ``embed_tokens.weight`` are the same
    # tensor. If ``lm_head`` is NF4-quantized while ``embed_tokens``
    # (Embedding, not touched by bnb) stays bf16, the
    # ``save_pretrained → from_pretrained`` round-trip loses
    # ``lm_head.weight.quant_state``, and every forward call emits the
    # UserWarning "FP4 quantization state not initialized" followed by
    # all-zero logits (i.e. silent garbage output).
    #
    # We always merge ``lm_head`` into the user's skip list — it's a
    # no-op for skip_modules=None (transformers handles it
    # automatically) and a correctness fix when the user supplies
    # their own list.
    if config.skip_modules:
        resolved = _resolve_skip_names(config.skip_modules)
        if "lm_head" not in resolved:
            resolved.append("lm_head")
        kwargs["llm_int8_skip_modules"] = resolved
    return BitsAndBytesConfig(**kwargs)





def quantize(
    input_dir: Path | str,
    output_dir: Path | str,
    config: BnBNF4Config | None = None,
) -> Path:
    """Load the bf16 multimodal checkpoint with NF4 4-bit quantization
    and persist it via ``save_pretrained``.

    Args:
        input_dir: Directory containing the merged bf16 model
            (output of ``src.common.model_io.save_bf16_merged``
            or the bf16 base model).
        output_dir: Where to write the NF4 checkpoint. Will be created.
        config: Optional ``BnBNF4Config`` (defaults are the QLoRA-paper
            recipe: NF4 + bf16 compute + double quantization).

    Returns:
        ``Path`` of the directory containing the NF4 safetensors,
        ``config.json`` (with ``quantization_config`` embedded), and
        the copied processor side-cars.

    Notes:
        - Requires CUDA. bitsandbytes silently falls back to fake-CPU
          quant on non-CUDA, which has shipped severe accuracy
          regressions in the past — we refuse to run rather than
          produce a misleading eval point.
        - The saved checkpoint is **not** MLX-loadable. It exists so
          the eval matrix has a non-MLX 4-bit reference column for
          PlantNet / WikiText PPL / VQAv2.
    """
    config = config or BnBNF4Config()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    # Heavy imports happen after the path check so typos fail fast.
    try:
        import bitsandbytes as _bnb  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bitsandbytes is not installed. Required for the bnb_nf4 "
            "method. Install via `pip install bitsandbytes`."
        ) from e

    import torch
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError(
            "bnb_nf4 requires CUDA. bitsandbytes' CPU fallback has shipped "
            "silent accuracy regressions; refusing to run a misleading "
            "eval point. Re-run on a CUDA host."
        )

    qc = _build_bnb_quantization_config(config)

    log.info(
        "Loading %s with NF4 quantization (compute_dtype=%s, double_quant=%s, skip_modules=%s)",
        input_dir,
        config.compute_dtype,
        config.double_quant,
        config.skip_modules,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(input_dir),
        quantization_config=qc,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving NF4 checkpoint to %s", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    copied = copy_processor_assets(input_dir, output_dir)
    if copied:
        log.info("Copied processor side-cars: %s", ", ".join(copied))
    else:
        log.warning(
            "No processor side-cars found alongside %s — eval / inference "
            "loaders may need ``--base_model_for_processor`` to resolve "
            "the tokenizer.",
            input_dir,
        )

    return output_dir
