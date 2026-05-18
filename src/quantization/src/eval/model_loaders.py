"""Model loaders for the eval harness.

Each loader produces a uniform ``ModelHandle`` (a small dataclass
holding the model object, the processor/tokenizer, and an
``infer_text(messages, image_path=None, max_new_tokens=...)`` callable).
Benchmarks consume the handle and don't care which loader produced it.

Three loaders today:

- ``load_hf_bf16``: Hugging Face ``Gemma4ForConditionalGeneration`` in
  bf16 on CUDA (or CPU). Used for the bf16 reference and for bnb_nf4.
- ``load_hf_gptq``: gptqmodel-native loader for GPTQ-quantized
  checkpoints. Avoids HF's auto-quantizer path that triggers Marlin
  JIT compilation (broken on torch < 2.11).
- ``load_mlx_vlm``: mlx_vlm-loaded INT4 model. Used for the
  mlx_vlm.convert outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


class InferTextFn(Protocol):
    def __call__(
        self,
        messages: list[dict],
        image_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> str: ...


@dataclass
class ModelHandle:
    """Uniform interface returned by every loader.

    Attributes:
        infer_text: Callable that takes a multimodal-messages list and
            an optional image path and returns the generated string.
        backend: "hf_bf16" / "mlx_vlm" / etc. — for logging only.
        model: Raw model object (kept so PPL eval can reach .forward()).
        processor: HF processor / mlx_vlm processor.
        device: "cuda" / "cpu" / "mps" (informational).
        model_dir: Path of the loaded checkpoint, for the JSON report.
    """

    infer_text: Callable[..., str]
    backend: str
    model: Any
    processor: Any
    device: str
    model_dir: Path


def load_hf_bf16(
    model_dir: Path | str,
    device_map: str = "auto",
    base_model_for_processor: str | None = None,
) -> ModelHandle:
    """Load a bf16 multimodal Gemma 4 E2B via HF transformers.

    Args:
        model_dir: Path to a directory with ``model.safetensors`` +
            ``config.json``. Typically the output of an SFT merge.
        device_map: Passed to ``from_pretrained``. ``"auto"`` lets HF
            place layers on GPU/CPU; ``"cuda"`` forces all on GPU
            (24 GB needed); ``"cpu"`` is the safest.
        base_model_for_processor: If the merged checkpoint lacks the
            processor configs (rare), fall back to this HF repo for
            the processor.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_dir = Path(model_dir)
    log.info("Loading HF bf16 model from %s (device_map=%s)", model_dir, device_map)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    proc_src = model_dir if (model_dir / "processor_config.json").exists() else (
        base_model_for_processor or model_dir
    )
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

    # Defer import — generate_response lives in the sister `finetune` package.
    from finetune.src.evaluate import generate_response  # type: ignore

    def infer_text(
        messages: list[dict], image_path: str | None = None, max_new_tokens: int = 128
    ) -> str:
        return generate_response(
            model=model,
            processor=processor,
            messages=messages,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )

    return ModelHandle(
        infer_text=infer_text,
        backend="hf_bf16",
        model=model,
        processor=processor,
        device=str(getattr(model, "device", "auto")),
        model_dir=model_dir,
    )


# Supported gptqmodel backend names for the hf_gptq loader. Listed here
# (rather than buried inside the loader) so the CLI / shell scripts can
# import the set without spinning up a CUDA context.
#
# Speed ranking (batch=1 decode on Ada/Ampere, w4g128 sym da=0) is roughly
#   marlin > exllama_v2 ≈ machete > triton > torch
# but actual numbers depend heavily on (a) the model shape (lm_head size,
# hidden dim, KV head layout) and (b) torch / triton versions. Treat the
# ranking as a starting point, not a guarantee — always smoke-time the
# kernel on a few samples before committing to a full eval run.
#
# Backend / desc_act compatibility (relevant when reloading da=1 checkpoints):
#
#   triton       — works for everything (sym/asym, any group, da=0 or da=1)
#   marlin       — sym only, group ∈ {-1, 32, 64, 128}, desc_act=False
#                  (Marlin classic; gptqmodel does NOT yet wire Marlin-v2
#                  which would support desc_act=True)
#   exllama_v2   — works for desc_act=True as well; the recommended
#                  fast path for da=1 variants
#   machete     — Hopper / Ada specific, sym only, desc_act=False
#   auto         — let gptqmodel pick; in 7.0.0 this falls back to triton
#                  on non-vLLM environments which means no speedup
GPTQ_BACKEND_CHOICES = (
    "triton",
    "auto",
    "marlin",
    "exllama_v2",
    "machete",
    "torch",
)


def load_hf_gptq(
    model_dir: Path | str,
    base_model_for_processor: str | None = None,
    backend: str = "triton",
) -> ModelHandle:
    """Load a GPTQ-quantized multimodal Gemma 4 E2B via gptqmodel.

    Uses ``GPTQModel.from_quantized`` which handles kernel selection
    internally. Defaults to the Triton backend (compatible with every
    knob combination but is also the slowest at batch=1 decode).

    Why this loader exposes the backend at all: the Triton kernel
    unpacks INT4 → bf16 in a separate kernel before each matmul, which
    in our eval pattern (per-image generative decode, batch=1, vision
    tower stays bf16) is ~3-5× slower than the bf16 reference. Faster
    fused kernels (Marlin, ExLlamaV2, Machete) exist and are drop-in
    via this parameter — same on-disk weights, different math kernel.

    See ``GPTQ_BACKEND_CHOICES`` for the supported names and their
    desc_act / group_size constraints.

    Args:
        model_dir: Path to a GPTQ-quantized checkpoint directory
            (contains ``quantize_config.json`` + packed safetensors).
        base_model_for_processor: If the quantized checkpoint lacks
            processor configs, fall back to this HF repo.
        backend: gptqmodel backend name. Default ``"triton"`` —
            slowest but most compatible. Use ``"marlin"`` for w4g128
            sym desc_act=False checkpoints (~2-3× speedup expected at
            batch=1 decode). Use ``"exllama_v2"`` if the checkpoint
            was quantized with ``desc_act=True`` (Marlin classic
            doesn't support act-order).
    """
    import torch

    try:
        from gptqmodel import GPTQModel
        from gptqmodel.utils.backend import BACKEND
    except ImportError as e:
        raise ImportError(
            "gptqmodel is required for the hf_gptq loader. "
            "Install via `pip install gptqmodel`."
        ) from e

    from transformers import AutoProcessor

    backend_map = {
        "triton": BACKEND.TRITON,
        "auto": BACKEND.AUTO,
        "marlin": BACKEND.MARLIN,
        "exllama_v2": BACKEND.EXLLAMA_V2,
        "machete": BACKEND.MACHETE,
        "torch": BACKEND.TORCH,
    }
    if backend not in backend_map:
        raise ValueError(
            f"Unknown gptqmodel backend: {backend!r}. "
            f"Expected one of {sorted(backend_map)}. "
            "If you need a new backend, extend GPTQ_BACKEND_CHOICES and backend_map."
        )
    gptq_backend = backend_map[backend]

    model_dir = Path(model_dir)
    log.info("Loading GPTQ model from %s via gptqmodel (backend=%s)", model_dir, backend)
    # Gemma4 has global_head_dim=512 which exceeds FlashAttention's
    # limit, and device_map="auto" breaks shared_kv_states passing
    # via accelerate hooks. Force SDPA + single-device placement.
    model = GPTQModel.from_quantized(
        str(model_dir),
        trust_remote_code=True,
        backend=gptq_backend,
        attn_implementation="sdpa",
        device="cuda:0",
    )
    model.eval()

    proc_src = model_dir if (model_dir / "processor_config.json").exists() else (
        base_model_for_processor or model_dir
    )
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

    from finetune.src.evaluate import generate_response  # type: ignore

    def infer_text(
        messages: list[dict], image_path: str | None = None, max_new_tokens: int = 128
    ) -> str:
        return generate_response(
            model=model,
            processor=processor,
            messages=messages,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )

    return ModelHandle(
        infer_text=infer_text,
        backend="hf_gptq",
        model=model,
        processor=processor,
        device=str(getattr(model, "device", "auto")),
        model_dir=model_dir,
    )


def load_mlx_vlm(model_dir: Path | str) -> ModelHandle:
    """Load an INT4 MLX checkpoint produced by ``mlx_vlm.convert``.

    Runs on Apple Silicon (metal backend) and on Linux + NVIDIA via
    the ``mlx-cuda-12`` package (see ``quantization/scripts/_env/_mlx_env.sh``
    for the CUDA 12 header staging that mlx-cuda's NVRTC needs to JIT
    kernels — required when the host system only ships CUDA 13 headers).
    """
    try:
        from mlx_vlm import generate as _mlx_generate
        from mlx_vlm import load as _mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template as _mlx_chat
    except ImportError as e:
        raise ImportError(
            "mlx_vlm not installed. Install via "
            "`pip install mlx mlx-vlm` (Apple Silicon: native metal) or "
            "`pip install mlx-cuda-12 mlx-vlm` (Linux + NVIDIA)."
        ) from e

    model_dir = Path(model_dir)
    log.info("Loading MLX VLM model from %s", model_dir)
    model, processor = _mlx_load(str(model_dir))

    def infer_text(
        messages: list[dict],
        image_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        # mlx_vlm's apply_chat_template expects a list of `{"role", "content"}` dicts;
        # content can be a string OR a list of blocks like HF. We flatten our
        # message format to mlx_vlm's expectation.
        prompt = _mlx_chat(processor, model.config, messages, num_images=int(bool(image_path)))
        images = [str(image_path)] if image_path else None
        out = _mlx_generate(
            model,
            processor,
            prompt,
            images,
            max_tokens=max_new_tokens,
            verbose=False,
        )
        # mlx_vlm.generate returns a `GenerationResult`-ish object on recent
        # versions; older versions returned a str. Handle both.
        if isinstance(out, str):
            return out.strip()
        return str(getattr(out, "text", out)).strip()

    return ModelHandle(
        infer_text=infer_text,
        backend="mlx_vlm",
        model=model,
        processor=processor,
        # mlx 0.31 picks GPU automatically: metal on Apple, CUDA on
        # Linux + nvidia (via mlx-cuda-12). Record what we actually got.
        device=str(_mx_default_device()),
        model_dir=model_dir,
    )


def _mx_default_device() -> str:
    """Return mlx's default device label (or 'unknown' on failure)."""
    try:
        import mlx.core as _mx
        return str(_mx.default_device())
    except Exception:  # noqa: BLE001
        return "unknown"


def load_hf_gptq_hybrid(
    model_dir: Path | str,
    base_model_for_processor: str | None = None,
    backend: str = "marlin",
) -> ModelHandle:
    """Load a GPTQ + torchao-packed-embedding hybrid checkpoint.

    Identical to :func:`load_hf_gptq` for the GPTQ side (Linear quant
    via ``gptqmodel.GPTQModel.from_quantized``), then post-load swaps
    the bf16 ``nn.Embedding`` modules listed in
    ``config.json["hybrid_quant"]["embeddings"]`` with
    :class:`~src.methods.gptq_torchao_hybrid.PackedQuantizedEmbedding`
    instances populated from the safetensors ``.qweight_packed`` /
    ``.scales`` / ``.zero_point`` tensors.

    Two key behaviors:

    1. ``GPTQModel.from_quantized`` will log "missing key
       embed_tokens.weight" and similar — these are expected and
       handled by the post-load patch. The corresponding randomly-
       initialized ``nn.Embedding`` modules get replaced before any
       forward pass.
    2. The packed tensors (``.qweight_packed`` etc.) will be logged
       as "unexpected" by HF's state-dict load and ignored. We re-read
       them via ``safetensors.safe_open`` inside
       ``load_hybrid_embeddings``.

    Args:
        model_dir: Path to a hybrid checkpoint (see
            ``src.methods.gptq_torchao_hybrid.quantize_hybrid``).
        base_model_for_processor: Same role as in :func:`load_hf_gptq`.
        backend: GPTQ kernel backend. Defaults to ``"marlin"`` because
            the only hybrid input we've produced so far is
            ``gptq_w4g64_da0`` (sym, ``desc_act=False``) which Marlin
            supports. Change for ``desc_act=True`` checkpoints
            (``exllama_v2``).
    """
    import torch

    try:
        from gptqmodel import GPTQModel
        from gptqmodel.utils.backend import BACKEND
    except ImportError as e:
        raise ImportError(
            "gptqmodel is required for the hf_gptq_hybrid loader."
        ) from e

    from transformers import AutoProcessor

    backend_map = {
        "triton": BACKEND.TRITON,
        "auto": BACKEND.AUTO,
        "marlin": BACKEND.MARLIN,
        "exllama_v2": BACKEND.EXLLAMA_V2,
        "machete": BACKEND.MACHETE,
        "torch": BACKEND.TORCH,
    }
    if backend not in backend_map:
        raise ValueError(
            f"Unknown gptqmodel backend: {backend!r}; expected one of {sorted(backend_map)}."
        )

    model_dir = Path(model_dir)
    log.info("Loading hybrid (GPTQ + packed-embed) model from %s (backend=%s)",
             model_dir, backend)
    # Same gemma4 placement caveats as load_hf_gptq (SDPA + single-device).
    wrapper = GPTQModel.from_quantized(
        str(model_dir),
        trust_remote_code=True,
        backend=backend_map[backend],
        attn_implementation="sdpa",
        device="cuda:0",
    )
    wrapper.eval()

    # GPTQModel returns a BaseQModel wrapper; the bare HF model is at
    # ``.model``. Our `load_hybrid_embeddings` walks dotted FQNs from
    # whatever object we pass it, and the saved FQNs are rooted at
    # the HF Gemma4ForConditionalGeneration instance (e.g.
    # "model.language_model.embed_tokens_per_layer"), so we pass
    # ``wrapper.model``.
    from src.methods.gptq_torchao_hybrid import load_hybrid_embeddings

    # Patch the embed modules in place. Buffers go to the same device
    # as the rest of the model (cuda:0 here per the from_quantized call).
    load_hybrid_embeddings(wrapper.model, model_dir, device="cuda:0")

    proc_src = model_dir if (model_dir / "processor_config.json").exists() else (
        base_model_for_processor or model_dir
    )
    processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

    from finetune.src.evaluate import generate_response  # type: ignore

    def infer_text(
        messages: list[dict], image_path: str | None = None, max_new_tokens: int = 128
    ) -> str:
        return generate_response(
            model=wrapper,
            processor=processor,
            messages=messages,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )

    return ModelHandle(
        infer_text=infer_text,
        backend="hf_gptq_hybrid",
        model=wrapper,
        processor=processor,
        device=str(getattr(wrapper, "device", "auto")),
        model_dir=model_dir,
    )


# Registry for use by CLI flags / configs.
LOADER_REGISTRY: dict[str, Callable[..., ModelHandle]] = {
    "hf_bf16": load_hf_bf16,
    "hf_gptq": load_hf_gptq,
    "hf_gptq_hybrid": load_hf_gptq_hybrid,
    "mlx_vlm": load_mlx_vlm,
}
