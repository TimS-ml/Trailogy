"""GPTQ + torchao hybrid quantization — CUDA-side PyTorch reference.

Purpose
-------
The standalone GPTQModel quantizer (``src.methods.gptq``) lands at
~7.45 GB on the multimodal SFT'd Gemma 4 E2B (B1-sft-results.md R2)
because GPTQ quantizes ``nn.Linear`` only. Gemma 4's per-layer-input
architecture carries a 4.70 GB ``embed_tokens_per_layer`` table
(`[262144, 8960]` bf16, see B1-torchao-vs-gptqmodel.md §1) and a
0.81 GB ``embed_tokens`` table — both are ``nn.Embedding``, both
remain bf16 in a pure-GPTQ output.

This module post-processes a GPTQModel output by:

1. Stripping ``audio_tower`` + ``embed_audio`` (iOS-unused).
2. Quantizing the two ``nn.Embedding`` tables to int4 group-affine,
   storing them as **true uint8 4-bit packed** (two nibbles per byte).
3. Writing a CUDA-loadable PyTorch artifact at ~2.77 GB.

Division of labour:

- **Quantization math** (selecting per-group scales / zero_points,
  rounding) — torchao ``IntxWeightOnlyConfig`` (stable public API).
  Validated across many models, well-debugged.
- **Storage** — we pack int4 values stored in torchao's int8 ``qdata``
  into uint8 (saves the 50% padding torchao leaves on CUDA). Plain
  safetensors, no torchao prototype glue.
- **Inference forward** — :class:`PackedQuantizedEmbedding`, ~30 LOC
  of gather-then-unpack-then-dequant. We control this so the
  checkpoint is independent of torchao versioning at load time.

NOT iOS-deployable
------------------
The output is a PyTorch CUDA artifact, not MLX. ``mlx_vlm.convert``
cannot consume GPTQ-packed int32 qweight nor our custom uint8 embed
pack — it expects bf16/fp16. The b1 hybrid path therefore serves as
a CUDA quality reference; iOS deploy goes through ``mlx_vlm.convert``
on the bf16 merged model (b2 route, see
``docs/quantization/B2-sft-results.md``). The full rationale + the
comparison table sit in ``docs/quantization/B1-torchao-vs-gptqmodel.md``.

Hardware
--------
NVIDIA CUDA. Loads the ~7 GB GPTQ-bf16 mixed source on CPU + does the
embedding quant pass on GPU (transient ~5 GB while torchao processes
``embed_tokens_per_layer``; freed after pack).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from src.common.model_io import copy_processor_assets

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module paths in the multimodal Gemma 4 HF safetensors
# ---------------------------------------------------------------------------

EMBED_TOKENS_KEY = "model.language_model.embed_tokens.weight"
EMBED_TOKENS_PER_LAYER_KEY = "model.language_model.embed_tokens_per_layer.weight"

# Substring prefixes (anchored to a leading "model.") used to drop the
# audio path from the saved checkpoint. Empty/missing modules are a
# no-op; HF Gemma4 forward without these still produces correct outputs
# for the text + vision paths (the iOS app never instantiates the audio
# branch — confirmed via hikeCompanion/scripts/strip-gemma-audio.py
# which uses the same name set).
AUDIO_STRIP_PREFIXES: tuple[str, ...] = (
    "model.audio_tower.",
    "model.embed_audio.",
)


# Per-embedding suffixes we emit under the packed scheme. The original
# `.weight` is dropped; HF `from_pretrained(strict=False)` will warn
# about the missing key, and our `load_hybrid_embeddings` post-load
# patches it. See B1-torchao-vs-gptqmodel.md §6 ("Stage 3 - serialize").
PACKED_SUFFIXES: tuple[str, ...] = (
    ".qweight_packed",  # uint8 [vocab, dim/2]
    ".scales",          # bf16  [vocab, dim/group_size]
    ".zero_point",      # int8  [vocab, dim/group_size]  (asymmetric only)
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig:
    """Knobs for the hybrid post-processing step.

    Defaults target the ~2.77 GB iOS-eligible structural size; lower
    bit widths are not implemented today (only 4-bit pack/unpack).
    """

    strip_audio: bool = True

    # embed_tokens_per_layer is the elephant in the room — 4.70 GB
    # bf16. ``None`` disables embed quant on this specific table
    # (intermediate variant for ablation).
    embed_per_layer_bits: int | None = 4
    embed_per_layer_group_size: int = 128
    embed_per_layer_mapping: str = "asymmetric"  # 'symmetric' | 'asymmetric'

    # embed_tokens is 0.81 GB bf16. Quantizing it shaves another
    # ~0.60 GB but is more accuracy-sensitive than the per-layer
    # table (every token decode touches it). ``None`` keeps it bf16.
    embed_tokens_bits: int | None = 4
    embed_tokens_group_size: int = 128
    embed_tokens_mapping: str = "asymmetric"

    # CUDA recommended. Pack/unpack works on CPU but torchao's
    # IntxWeightOnlyConfig is much slower without a GPU.
    device: str = "cuda"

    # Optional list of additional substring prefixes to drop. Mostly
    # for ablation experiments (e.g. drop one of the vision_tower
    # encoder layer ranges); audio is already covered by
    # ``strip_audio``. Plain substring match against the full
    # safetensors key.
    extra_drop_prefixes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Bit-packing primitives (int4 only today)
# ---------------------------------------------------------------------------


def pack_int4_to_uint8(qdata: torch.Tensor) -> torch.Tensor:
    """Pack signed-int4 values stored in an int8 tensor into uint8 with
    two nibbles per byte along the last dimension.

    ``torchao.IntxWeightOnlyConfig(weight_dtype=torch.int4)`` produces
    ``qdata`` of dtype ``int8`` with values in ``[-8, 7]`` (verified
    empirically; both ``SYMMETRIC`` and ``ASYMMETRIC`` mapping types
    fall in this range — torchao always uses signed int representation
    on CUDA). We shift to ``[0, 15]`` (unsigned) and pack pairs.

    Args:
        qdata: ``torch.int8`` tensor; last dim must be even; all values
            must fit in ``[-8, 7]``.

    Returns:
        ``torch.uint8`` tensor with last dim halved. Low nibble (bits
        0-3) holds ``qdata[..., 0::2]``, high nibble (bits 4-7) holds
        ``qdata[..., 1::2]``.
    """
    if qdata.dtype != torch.int8:
        raise TypeError(f"expected int8 qdata, got {qdata.dtype}")
    if qdata.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even for int4 packing, got shape {tuple(qdata.shape)}"
        )
    # Bounds check guards against out-of-range bugs upstream (mapping
    # misconfiguration, dtype confusion). Cheap on a 2.35 B-element
    # tensor (~0.3 s on a 4090).
    qmin = int(qdata.min().item())
    qmax = int(qdata.max().item())
    if qmin < -8 or qmax > 7:
        raise ValueError(
            f"qdata out of int4 range; got [{qmin}, {qmax}], expected [-8, 7]. "
            "This usually means the upstream quantizer wasn't actually int4."
        )
    # Shift signed → unsigned, then pack pairs.
    unsigned = (qdata + 8).to(torch.uint8)        # [..., D] in [0, 15]
    low = unsigned[..., 0::2]
    high = unsigned[..., 1::2]
    return low | (high << 4)                       # [..., D/2] uint8


def unpack_uint8_to_int4(
    packed: torch.Tensor, output_dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """Inverse of :func:`pack_int4_to_uint8`. Last dim doubles.

    Args:
        packed: ``torch.uint8`` tensor.
        output_dtype: result dtype (typically ``int8`` so downstream
            dequant can subtract zero_point in signed arithmetic).

    Returns:
        Tensor of shape ``packed.shape[:-1] + (packed.shape[-1] * 2,)``,
        values in ``[-8, 7]``.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected uint8 packed, got {packed.dtype}")
    low = (packed & 0x0F).to(output_dtype) - 8
    high = ((packed >> 4) & 0x0F).to(output_dtype) - 8
    # Interleave back along the last dim. ``torch.stack(..., dim=-1).flatten``
    # is the canonical pattern; flatten(-2) collapses just the last two
    # dims to undo the interleave.
    out = torch.stack([low, high], dim=-1).flatten(-2)
    return out


# ---------------------------------------------------------------------------
# Runtime Embedding module — owns the gather + unpack + dequant path
# ---------------------------------------------------------------------------


class _PackedWeightProxy:
    """Stand-in for ``nn.Embedding.weight`` that supports row-indexing.

    Why: HF Gemma 4's forward at
    ``transformers/models/gemma4/modeling_gemma4.py:2300`` reads the
    pad-token row directly via ``self.embed_tokens.weight[pad_id, :]``,
    bypassing the ``forward`` method. Without this proxy, the model
    crashes with ``AttributeError: 'PackedQuantizedEmbedding' object
    has no attribute 'weight'`` on the very first forward.

    Supported indexing:
        ``weight[int]``           → dequantized row, shape [embedding_dim]
        ``weight[int, :]``        → same as above (the ``:`` slice is the
                                    second axis selector — we serve the
                                    full row regardless)
        ``weight[slice|tensor]``  → dequantized rows, shape
                                    [N, embedding_dim]

    NOT supported:
        ``weight[None, None, :, :]`` → broadcasts the full table; only
        hit in the ``inputs_embeds is not None and input_ids is None``
        branch (``modeling_gemma4.py:1734``), which is dead code in
        normal eval. Raises ``NotImplementedError`` with a helpful
        message.

    Memory: each indexed lookup dequantizes only the requested rows,
    so the 4.7 GB embed_tokens_per_layer table stays packed.
    """

    def __init__(self, owner: "PackedQuantizedEmbedding"):
        self._owner = owner
        # HF training-time hooks query .requires_grad on the weight to
        # gate gradient propagation. We're inference-only, so report
        # False without instantiating an actual tensor.
        self.requires_grad = False

    @property
    def shape(self):
        return torch.Size([self._owner.num_embeddings, self._owner.embedding_dim])

    @property
    def dtype(self):
        return self._owner.compute_dtype

    @property
    def device(self):
        return self._owner.qweight_packed.device

    def __getitem__(self, idx):
        # Single int (e.g. ``pad_token_id``).
        if isinstance(idx, int):
            row = self._owner(torch.tensor([idx], dtype=torch.long, device=self.device))
            return row.squeeze(0)
        # tuple ``(row_idx, col_slice)`` — only support ``(int, :)`` and
        # ``(int, slice)`` where slice is a full slice. Other column
        # subsetting requires partial dequant, not implemented.
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            if isinstance(row_idx, int) and isinstance(col_idx, slice) and (
                col_idx.start in (None, 0)
                and col_idx.stop in (None, self._owner.embedding_dim)
                and col_idx.step in (None, 1)
            ):
                row = self._owner(torch.tensor([row_idx], dtype=torch.long, device=self.device))
                return row.squeeze(0)
            if isinstance(row_idx, type(None)) and isinstance(col_idx, type(None)):
                # weight[None, None, ...] — the dead-code path
                raise NotImplementedError(
                    "PackedQuantizedEmbedding._PackedWeightProxy does not support "
                    "full-table broadcast indexing (weight[None, None, :, :]). "
                    "This path is only used by HF Gemma4's inputs_embeds → input_ids "
                    "reverse-lookup branch (modeling_gemma4.py:1734), which is dead "
                    "code in normal eval (input_ids is always provided)."
                )
        # 1-D tensor / list / slice of indices.
        if torch.is_tensor(idx) or isinstance(idx, (list, slice)):
            if isinstance(idx, list):
                idx = torch.tensor(idx, dtype=torch.long, device=self.device)
            elif isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop if idx.stop is not None else self._owner.num_embeddings
                step = idx.step or 1
                idx = torch.arange(start, stop, step, dtype=torch.long, device=self.device)
            return self._owner(idx.to(self.device))
        raise NotImplementedError(
            f"PackedQuantizedEmbedding.weight indexing with {type(idx).__name__}={idx!r} "
            "is not supported. If this is hit on a real forward pass, add the case here."
        )


class PackedQuantizedEmbedding(nn.Module):
    """Drop-in replacement for :class:`nn.Embedding` (and the Gemma 4
    ``Gemma4TextScaledWordEmbedding`` subclass) that stores its weight
    as 4-bit-packed uint8.

    Forward = gather the row(s) requested by the indices, unpack just
    those rows from uint8 → int8, then dequantize with the per-row
    per-group scale (and zero_point for asymmetric mapping), and
    finally multiply by ``embed_scale`` (default 1.0, but Gemma 4 uses
    ``sqrt(hidden_size)`` for ``embed_tokens`` and
    ``sqrt(hidden_size_per_layer_input)`` for ``embed_tokens_per_layer``).
    **Critically does not dequant the full table** — only the rows
    actually indexed by the current batch.

    The ``embed_scale`` field is essential to keep this drop-in for
    HF Gemma 4. The original module's forward is::

        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

    Dropping the scale silently catastrophically degrades accuracy
    (e.g. n=300 PlantNet collapses from ~84% to ~4% — see
    B1-torchao-vs-gptqmodel.md §7).

    Storage (registered buffers):
        qweight_packed: ``uint8 [num_embeddings, embedding_dim // 2]``
        scales:         compute_dtype ``[num_embeddings, embedding_dim // group_size]``
        zero_point:     ``int8 [num_embeddings, num_groups]`` if asymmetric, else 0-length

    Persistent attributes (non-buffer):
        embed_scale:    float scalar applied to forward output. Must
                        be set to the value of the original
                        ``Gemma4TextScaledWordEmbedding.embed_scale``
                        when swapping; default 1.0 acts as a no-op
                        for vanilla ``nn.Embedding`` replacement.

    Dequant math (per row ``r``, per group ``g`` of values
    ``r[g*G : (g+1)*G]``):
        ``(q_int4 - zero_point[r, g]) * scales[r, g]``   if asymmetric
        ``q_int4 * scales[r, g]``                        if symmetric
    Followed by ``output *= embed_scale``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int,
        group_size: int,
        mapping: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        embed_scale: float = 1.0,
    ):
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only bits=4 is implemented today.")
        if embedding_dim % group_size != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be a multiple of group_size={group_size}"
            )
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim={embedding_dim} must be even for int4 packing")
        if mapping not in ("symmetric", "asymmetric"):
            raise ValueError(f"mapping must be 'symmetric' or 'asymmetric', got {mapping!r}")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        self.group_size = group_size
        self.mapping = mapping
        self.compute_dtype = dtype
        num_groups = embedding_dim // group_size
        self.num_groups = num_groups
        # Match Gemma4TextScaledWordEmbedding's storage: keep as a
        # non-persistent buffer (saved to state_dict only via meta —
        # we want the scale to come from the live config / orig
        # module, not from a stale .safetensors entry).
        self.embed_scale = float(embed_scale)

        self.register_buffer(
            "qweight_packed",
            torch.empty(num_embeddings, embedding_dim // 2, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(num_embeddings, num_groups, dtype=dtype, device=device),
        )
        if mapping == "asymmetric":
            self.register_buffer(
                "zero_point",
                torch.empty(num_embeddings, num_groups, dtype=torch.int8, device=device),
            )
        else:
            # Register as None-equivalent so state_dict load doesn't trip.
            # We use a 0-element tensor rather than ``None`` to keep
            # ``register_buffer`` semantics simple.
            self.register_buffer(
                "zero_point",
                torch.empty(0, dtype=torch.int8, device=device),
            )

        # Lazy-init proxy; created once so its identity stays stable
        # across .weight accesses. HF code at
        # modeling_gemma4.py:2300 directly reads
        # ``self.embed_tokens.weight[pad_token_id, :]``, so we expose a
        # tensor-like row-indexable object. See _PackedWeightProxy
        # docstring for what's supported.
        self._weight_proxy = _PackedWeightProxy(self)

    @property
    def weight(self):
        """Tensor-like proxy supporting ``[idx]`` and ``[idx, :]``.

        Returns :class:`_PackedWeightProxy`, NOT a real tensor. Calling
        ``.weight`` does NOT materialize the dequantized table; only
        the actually-indexed rows are dequantized on access.
        """
        return self._weight_proxy

    @classmethod
    def from_components(
        cls,
        qweight_packed: torch.Tensor,
        scales: torch.Tensor,
        zero_point: torch.Tensor | None,
        bits: int,
        group_size: int,
        mapping: str,
        embedding_dim: int,
        embed_scale: float = 1.0,
    ) -> "PackedQuantizedEmbedding":
        """Construct from already-packed tensors (e.g. read from disk).

        Shapes are inferred from the inputs; ``embedding_dim`` is taken
        as ground truth (not ``qweight_packed.shape[-1] * 2``) so a
        stray padding/trailing-dim bug surfaces as a shape mismatch
        assertion rather than silently wrong inference.

        ``embed_scale`` must be provided when swapping in for a
        ``Gemma4TextScaledWordEmbedding`` (or any other scaled
        embedding subclass). Defaults to 1.0 for vanilla
        ``nn.Embedding`` replacement.
        """
        num_emb = qweight_packed.shape[0]
        if qweight_packed.shape[-1] * 2 != embedding_dim:
            raise ValueError(
                f"packed qweight last dim {qweight_packed.shape[-1]} × 2 "
                f"≠ embedding_dim {embedding_dim}"
            )
        if scales.shape[0] != num_emb:
            raise ValueError(
                f"scales rows {scales.shape[0]} ≠ qweight rows {num_emb}"
            )
        m = cls(
            num_embeddings=num_emb,
            embedding_dim=embedding_dim,
            bits=bits,
            group_size=group_size,
            mapping=mapping,
            dtype=scales.dtype,
            device=qweight_packed.device,
            embed_scale=embed_scale,
        )
        m.qweight_packed.copy_(qweight_packed)
        m.scales.copy_(scales)
        if mapping == "asymmetric":
            if zero_point is None:
                raise ValueError("asymmetric mapping requires zero_point")
            m.zero_point.copy_(zero_point.to(torch.int8))
        return m

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: any shape; flatten for gather then reshape after.
        flat = x.reshape(-1)                           # [N]
        rows_packed = self.qweight_packed[flat]        # [N, D/2] uint8
        rows_scales = self.scales[flat]                # [N, num_groups] compute_dtype

        rows_int4 = unpack_uint8_to_int4(rows_packed)  # [N, D] int8 in [-8, 7]
        rows_float = rows_int4.to(self.compute_dtype)

        n = flat.numel()
        # Reshape to [N, num_groups, group_size] for per-group scale.
        rows_float = rows_float.view(n, self.num_groups, self.group_size)
        scales_b = rows_scales.unsqueeze(-1)           # [N, num_groups, 1]

        if self.mapping == "asymmetric":
            zp = self.zero_point[flat].to(self.compute_dtype).unsqueeze(-1)
            rows_dequant = (rows_float - zp) * scales_b
        else:
            rows_dequant = rows_float * scales_b

        rows_dequant = rows_dequant.view(n, self.embedding_dim)
        # Apply the scale that the wrapped Gemma4TextScaledWordEmbedding
        # would have applied. Default 1.0 makes this a no-op for plain
        # nn.Embedding swaps. See class docstring.
        if self.embed_scale != 1.0:
            rows_dequant = rows_dequant * self.embed_scale
        return rows_dequant.reshape(*x.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"mapping={self.mapping!r}"
        )


# ---------------------------------------------------------------------------
# Quantization helper (torchao does the math, we own the storage)
# ---------------------------------------------------------------------------


def quantize_embedding_components(
    bf16_weight: torch.Tensor,
    bits: int,
    group_size: int,
    mapping: str,
    device: str,
) -> dict[str, torch.Tensor | int | str]:
    """Quantize a bf16 ``[vocab, dim]`` weight via torchao and return
    the packed components.

    Returns a dict with at minimum::

        qweight_packed: uint8 [vocab, dim//2]
        scales:         bf16  [vocab, dim//group_size]
        zero_point:     int8  [vocab, dim//group_size]   # only if asymmetric
        bits:           int
        group_size:     int
        mapping:        str
        embedding_dim:  int   # original
        num_embeddings: int
    """
    if bits != 4:
        raise NotImplementedError(f"only bits=4 implemented; got {bits}")
    if bf16_weight.dtype != torch.bfloat16:
        raise ValueError(f"expected bf16 input, got {bf16_weight.dtype}")
    if bf16_weight.dim() != 2:
        raise ValueError(f"expected 2-D embedding weight, got shape {tuple(bf16_weight.shape)}")

    # Late import — heavy and torchao prints noisy warnings about CPP
    # extensions when imported, defer until we actually need it.
    from torchao.quantization import quantize_, MappingType
    from torchao.quantization.quant_api import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup

    map_type = {
        "symmetric": MappingType.SYMMETRIC,
        "asymmetric": MappingType.ASYMMETRIC,
    }[mapping]

    num_emb, dim = bf16_weight.shape
    if dim % group_size != 0:
        raise ValueError(
            f"embedding_dim={dim} must be a multiple of group_size={group_size}"
        )

    # Build a temp Embedding, copy in the weight, quantize in place.
    # We don't reuse the caller's nn.Embedding (if any) because we
    # need fresh CUDA storage; the caller passes a CPU bf16 tensor.
    emb = nn.Embedding(num_emb, dim, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        emb.weight.copy_(bf16_weight.to(device))

    quantize_(
        emb,
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(group_size),
            mapping_type=map_type,
            scale_dtype=torch.bfloat16,
        ),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
        device=device,
    )

    # torchao stashes the IntxUnpackedToInt8Tensor on emb.weight; pull
    # the data tensors out and move to CPU before we drop the GPU
    # working copy. Subsequent steps (pack, save) are CPU-friendly.
    qdata_cpu = emb.weight.qdata.detach().cpu().contiguous()
    scales_cpu = emb.weight.scale.detach().cpu().contiguous()
    zp_cpu = None
    if mapping == "asymmetric":
        zp_cpu = emb.weight.zero_point.detach().cpu().contiguous().to(torch.int8)
    # Free GPU memory ASAP — embed_tokens_per_layer at bf16 is 4.7 GB,
    # and torchao's intermediate state pushes a few times that
    # transiently.
    del emb
    if device == "cuda":
        torch.cuda.empty_cache()

    packed = pack_int4_to_uint8(qdata_cpu)

    out: dict[str, torch.Tensor | int | str] = {
        "qweight_packed": packed,
        "scales": scales_cpu,
        "bits": bits,
        "group_size": group_size,
        "mapping": mapping,
        "embedding_dim": dim,
        "num_embeddings": num_emb,
    }
    if zp_cpu is not None:
        out["zero_point"] = zp_cpu
    return out


# ---------------------------------------------------------------------------
# Production: post-process a GPTQ output dir into the hybrid artifact
# ---------------------------------------------------------------------------


def _should_drop(key: str, drop_prefixes: Iterable[str]) -> bool:
    """Substring-anchored drop check. Mirrors the same pattern used by
    ``bnb_nf4.skip_modules``: each prefix must occur as a substring
    of the dotted key. Empty list disables drops."""
    return any(prefix in key for prefix in drop_prefixes)


def quantize_hybrid(
    input_dir: Path | str,
    output_dir: Path | str,
    config: HybridConfig | None = None,
) -> Path:
    """Post-process an existing GPTQModel output directory into the
    hybrid artifact (GPTQ Linears + packed-int4 embeddings + optional
    audio strip).

    The input dir must contain HF-format safetensors (one or more
    shards) + ``config.json`` + the standard processor side-cars. The
    output dir is written fresh.

    Args:
        input_dir: GPTQ output dir (e.g. ``results/gptq_w4g128_da1/``).
        output_dir: Target dir for the hybrid artifact. Created if it
            doesn't exist; **not** cleared if it does (caller's
            responsibility to start from empty).
        config: Knobs. See :class:`HybridConfig`.

    Returns:
        The output dir path.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = config or HybridConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect drop substrings.
    drop_prefixes: list[str] = []
    if config.strip_audio:
        drop_prefixes.extend(AUDIO_STRIP_PREFIXES)
    drop_prefixes.extend(config.extra_drop_prefixes)

    # Map of original key → action.
    # Action codes: "drop", "passthrough", "embed_per_layer", "embed_tokens"
    embed_targets: dict[str, tuple[int, int, str]] = {}
    if config.embed_per_layer_bits is not None:
        embed_targets[EMBED_TOKENS_PER_LAYER_KEY] = (
            config.embed_per_layer_bits,
            config.embed_per_layer_group_size,
            config.embed_per_layer_mapping,
        )
    if config.embed_tokens_bits is not None:
        embed_targets[EMBED_TOKENS_KEY] = (
            config.embed_tokens_bits,
            config.embed_tokens_group_size,
            config.embed_tokens_mapping,
        )

    # Step 1: enumerate input safetensors shards.
    shard_paths = sorted(input_dir.glob("model*.safetensors"))
    if not shard_paths:
        # Fall back to a single ``model.safetensors`` (no shards).
        single = input_dir / "model.safetensors"
        if single.is_file():
            shard_paths = [single]
        else:
            raise FileNotFoundError(
                f"No safetensors shards found in {input_dir}"
            )
    log.info("Found %d input safetensors shard(s)", len(shard_paths))

    # Step 2: enumerate all keys + remember which shard holds each.
    all_keys: dict[str, Path] = {}   # tensor name → shard path
    for sp in shard_paths:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                all_keys[k] = sp
    log.info("Discovered %d tensors total across shards", len(all_keys))

    # Step 3: decide per-key action and build the output state dict.
    new_state: dict[str, torch.Tensor] = {}
    embed_meta: dict[str, dict] = {}    # original embed key → component metadata
    n_dropped = 0
    n_passthrough = 0
    for key, shard in all_keys.items():
        if _should_drop(key, drop_prefixes):
            n_dropped += 1
            continue
        if key in embed_targets:
            # Defer; handled in the embed pass below.
            continue
        with safe_open(shard, framework="pt", device="cpu") as f:
            new_state[key] = f.get_tensor(key)
        n_passthrough += 1
    log.info("Pass-through %d tensors, dropped %d (audio + extra)",
             n_passthrough, n_dropped)

    # Read config.json early so we can compute embed_scale for each
    # Gemma 4 scaled embedding target. Gemma4TextScaledWordEmbedding
    # at modeling_gemma4.py:1579,1595 sets:
    #   embed_tokens.embed_scale           = hidden_size ** 0.5
    #   embed_tokens_per_layer.embed_scale = hidden_size_per_layer_input ** 0.5
    # We embed these scalars into hybrid_quant.embeddings[...] so load
    # time doesn't have to re-derive them.
    src_cfg_for_scales = input_dir / "config.json"
    cfg_for_scales: dict = {}
    if src_cfg_for_scales.is_file():
        cfg_for_scales = json.loads(src_cfg_for_scales.read_text())
    text_cfg = cfg_for_scales.get("text_config", cfg_for_scales) or {}
    GEMMA4_SCALE_FORMULA: dict[str, str] = {
        EMBED_TOKENS_KEY: "hidden_size",
        EMBED_TOKENS_PER_LAYER_KEY: "hidden_size_per_layer_input",
    }

    # Step 4: quantize + pack each target embedding.
    for orig_key, (bits, gs, mapping) in embed_targets.items():
        if orig_key not in all_keys:
            log.warning(
                "Embed quant target %s not in input safetensors — skipping.",
                orig_key,
            )
            continue
        shard = all_keys[orig_key]
        with safe_open(shard, framework="pt", device="cpu") as f:
            bf16_weight = f.get_tensor(orig_key)
        log.info(
            "Quantizing %s: shape=%s bits=%d group_size=%d mapping=%s ...",
            orig_key, tuple(bf16_weight.shape), bits, gs, mapping,
        )
        comp = quantize_embedding_components(
            bf16_weight=bf16_weight,
            bits=bits,
            group_size=gs,
            mapping=mapping,
            device=config.device,
        )
        # Drop the original `.weight` key. Emit the packed components
        # under the same module FQN.
        module_fqn = orig_key.rsplit(".weight", 1)[0]
        new_state[f"{module_fqn}.qweight_packed"] = comp["qweight_packed"]
        new_state[f"{module_fqn}.scales"] = comp["scales"]
        if "zero_point" in comp:
            new_state[f"{module_fqn}.zero_point"] = comp["zero_point"]

        # Compute embed_scale for known Gemma 4 scaled embeddings. If
        # this is an unknown key (e.g. a future model with different
        # scaled embeddings), we leave embed_scale unset; load-time
        # falls back to probing the live module.
        embed_scale_val: float | None = None
        scale_src = GEMMA4_SCALE_FORMULA.get(orig_key)
        if scale_src is not None:
            dim = text_cfg.get(scale_src)
            if dim is not None:
                embed_scale_val = float(dim) ** 0.5
            else:
                log.warning(
                    "Couldn't read text_config.%s for %s; embed_scale will be "
                    "probed at load time from the live module.",
                    scale_src, orig_key,
                )

        embed_meta[orig_key] = {
            "module_fqn": module_fqn,
            "bits": bits,
            "group_size": gs,
            "mapping": mapping,
            "embedding_dim": comp["embedding_dim"],
            "num_embeddings": comp["num_embeddings"],
        }
        if embed_scale_val is not None:
            embed_meta[orig_key]["embed_scale"] = embed_scale_val
            log.info(
                "  embed_scale=%g (from text_config.%s=%s)",
                embed_scale_val, scale_src, text_cfg.get(scale_src),
            )
        log.info(
            "  → packed qweight %s bytes, scales %s bytes%s",
            new_state[f"{module_fqn}.qweight_packed"].element_size()
            * new_state[f"{module_fqn}.qweight_packed"].numel(),
            new_state[f"{module_fqn}.scales"].element_size()
            * new_state[f"{module_fqn}.scales"].numel(),
            (
                ", zp " + str(
                    new_state[f"{module_fqn}.zero_point"].element_size()
                    * new_state[f"{module_fqn}.zero_point"].numel()
                )
                if mapping == "asymmetric"
                else ""
            ),
        )

    # Step 5: write the unified safetensors shard. At ~2.77 GB the
    # output fits in a single safetensors file comfortably (the
    # safetensors format is happy up to 100 GB per file; HF's tooling
    # only shards for chunked download). We emit a single
    # ``model.safetensors`` plus an index for HF compatibility (the
    # GPTQ pipeline downstream tools sometimes look for the index).
    out_shard = output_dir / "model.safetensors"
    log.info("Writing %d tensors to %s ...", len(new_state), out_shard)
    save_file(new_state, str(out_shard))
    total_bytes = out_shard.stat().st_size
    log.info("Wrote %.2f GB", total_bytes / 1e9)

    # Step 6: write the safetensors index (single-shard variant) so
    # HF ``from_pretrained`` doesn't fall back to a slow scan.
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": {k: "model.safetensors" for k in new_state.keys()},
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2)
    )

    # Step 7: copy + patch config.json with our hybrid_quant block.
    src_cfg_path = input_dir / "config.json"
    dst_cfg_path = output_dir / "config.json"
    if src_cfg_path.is_file():
        cfg_obj = json.loads(src_cfg_path.read_text())
    else:
        cfg_obj = {}
    cfg_obj["hybrid_quant"] = {
        "embed_quant_method": "torchao_intx_packed_uint8",
        "audio_stripped": config.strip_audio,
        "extra_drop_prefixes": list(config.extra_drop_prefixes),
        "embeddings": {
            orig_key: meta for orig_key, meta in embed_meta.items()
        },
    }
    dst_cfg_path.write_text(json.dumps(cfg_obj, indent=2))
    log.info("Wrote patched config.json with hybrid_quant block")

    # Step 8: bring forward the rest of the side-cars
    # (tokenizer.json, processor_config.json, chat_template.jinja,
    # quantize_config.json, generation_config.json). copy_processor_assets
    # handles the canonical set; we additionally mirror
    # ``quantize_config.json`` since downstream GPTQ load paths need it.
    copy_processor_assets(input_dir, output_dir)
    for extra in ("quantize_config.json", "quant_log.csv"):
        src = input_dir / extra
        if src.is_file():
            shutil.copy2(src, output_dir / extra)

    log.info("Hybrid artifact ready at %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Load-side: patch a freshly-loaded HF model with our packed embeddings
# ---------------------------------------------------------------------------


def _extract_embed_scale(module: nn.Module) -> float | None:
    """Read ``embed_scale`` from a Gemma 4 scaled embedding module.

    Gemma4TextScaledWordEmbedding stores ``embed_scale`` two ways
    (modeling_gemma4.py:1448-1449):

    - ``self.scalar_embed_scale`` — Python float, original value.
    - ``self.embed_scale`` — non-persistent buffer (torch.tensor scalar).

    We prefer the Python float (no dtype quirks) and fall back to the
    tensor buffer. Returns ``None`` if neither attribute is present
    (vanilla nn.Embedding, no scale).
    """
    scalar = getattr(module, "scalar_embed_scale", None)
    if scalar is not None:
        return float(scalar)
    buf = getattr(module, "embed_scale", None)
    if torch.is_tensor(buf):
        return float(buf.detach().cpu().item())
    if isinstance(buf, (int, float)):
        return float(buf)
    return None


def load_hybrid_embeddings(
    model: nn.Module,
    model_dir: Path | str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """After ``AutoModelForImageTextToText.from_pretrained(model_dir, ...)``,
    call this to swap the (still-bf16, possibly random-initialized)
    embed modules with their packed-quantized replacements.

    ``from_pretrained(strict=False)`` will have warned about missing
    ``embed_tokens(.weight)`` and ``embed_tokens_per_layer(.weight)``
    keys and about unexpected ``qweight_packed``/``scales``/``zero_point``
    keys. This function bridges the gap by reading the packed components
    directly from the safetensors and installing
    :class:`PackedQuantizedEmbedding` modules.

    Args:
        model: A loaded HF Gemma 4 multimodal model (or any model whose
            embed modules sit at the FQNs in ``config.hybrid_quant``).
        model_dir: Path to the hybrid artifact dir (must contain
            ``config.json`` with ``hybrid_quant`` block and one or
            more ``model*.safetensors``).
        device: Target device for the packed embedding's buffers.

    Returns:
        The same ``model`` with patched embedding modules.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    hybrid = cfg.get("hybrid_quant")
    if not hybrid:
        raise RuntimeError(
            f"config.json at {model_dir} has no 'hybrid_quant' block — "
            "not a hybrid artifact."
        )

    shard_paths = sorted(model_dir.glob("model*.safetensors"))
    if not shard_paths:
        single = model_dir / "model.safetensors"
        if single.is_file():
            shard_paths = [single]
        else:
            raise FileNotFoundError(f"No safetensors in {model_dir}")

    # Build a key→shard map.
    key_to_shard: dict[str, Path] = {}
    for sp in shard_paths:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                key_to_shard[k] = sp

    for orig_key, meta in hybrid["embeddings"].items():
        module_fqn = meta["module_fqn"]
        pk = f"{module_fqn}.qweight_packed"
        sk = f"{module_fqn}.scales"
        zk = f"{module_fqn}.zero_point"

        if pk not in key_to_shard or sk not in key_to_shard:
            raise RuntimeError(
                f"Hybrid artifact references {pk}/{sk} but they are not in "
                f"any shard. Check that {model_dir} was produced by "
                "quantize_hybrid() against a matching config.json."
            )

        with safe_open(key_to_shard[pk], framework="pt", device="cpu") as f:
            qweight_packed = f.get_tensor(pk)
        with safe_open(key_to_shard[sk], framework="pt", device="cpu") as f:
            scales = f.get_tensor(sk)
        zero_point = None
        if meta["mapping"] == "asymmetric":
            if zk not in key_to_shard:
                raise RuntimeError(
                    f"asymmetric mapping declared for {orig_key} but {zk} is missing"
                )
            with safe_open(key_to_shard[zk], framework="pt", device="cpu") as f:
                zero_point = f.get_tensor(zk)

        # Walk the dotted FQN to find the parent module + capture the
        # original module's embed_scale (if any) BEFORE we swap. This
        # is the critical fix for the missing-scale bug — Gemma 4's
        # Gemma4TextScaledWordEmbedding multiplies its output by
        # embed_scale (sqrt(hidden_size) for embed_tokens,
        # sqrt(hidden_size_per_layer_input) for embed_tokens_per_layer).
        # A vanilla nn.Module replacement silently drops this, which
        # collapsed PlantNet eval from ~84% to ~4%. See
        # B1-torchao-vs-gptqmodel.md §7 + commit log.
        #
        # Precedence: meta["embed_scale"] (persisted in config at
        # quantize time) wins if present; otherwise probe the live
        # module attribute. Fall back to 1.0 (vanilla nn.Embedding)
        # with a debug log so we'd notice if a scaled wrapper was
        # silently treated as plain.
        parent = model
        parts = module_fqn.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        live_scale = _extract_embed_scale(old)
        meta_scale = meta.get("embed_scale")
        if meta_scale is not None:
            embed_scale = float(meta_scale)
            if live_scale is not None and abs(live_scale - embed_scale) > 1e-4:
                log.warning(
                    "embed_scale mismatch at %s: config=%g, live module=%g — using config",
                    module_fqn, embed_scale, live_scale,
                )
        elif live_scale is not None:
            embed_scale = live_scale
        else:
            embed_scale = 1.0
            log.info(
                "No embed_scale found at %s (neither in config nor on live module). "
                "Defaulting to 1.0; if this is a Gemma 4 scaled embedding the swap "
                "is silently wrong — check the source model class.",
                module_fqn,
            )

        packed_emb = PackedQuantizedEmbedding.from_components(
            qweight_packed=qweight_packed.to(device),
            scales=scales.to(device),
            zero_point=zero_point.to(device) if zero_point is not None else None,
            bits=meta["bits"],
            group_size=meta["group_size"],
            mapping=meta["mapping"],
            embedding_dim=meta["embedding_dim"],
            embed_scale=embed_scale,
        )

        if not isinstance(old, (nn.Embedding, PackedQuantizedEmbedding)):
            log.warning(
                "Replacing %s of unexpected type %s with PackedQuantizedEmbedding",
                module_fqn, type(old).__name__,
            )
        setattr(parent, parts[-1], packed_emb)
        log.info(
            "Installed PackedQuantizedEmbedding at %s "
            "(num_embeddings=%d, embedding_dim=%d, bits=%d, group_size=%d, "
            "mapping=%s, embed_scale=%g)",
            module_fqn, packed_emb.num_embeddings, packed_emb.embedding_dim,
            packed_emb.bits, packed_emb.group_size, packed_emb.mapping,
            packed_emb.embed_scale,
        )

    return model


# ---------------------------------------------------------------------------
# Smoke check (no GPU work; just verify the deps import)
# ---------------------------------------------------------------------------


def smoke_check() -> dict:
    """Verify torchao + safetensors + transformers are importable."""
    notes: list[str] = []
    result = {"deps_available": True, "notes": ""}
    try:
        import torchao  # noqa: F401
        notes.append(f"torchao: {torchao.__version__}")
    except Exception as e:  # noqa: BLE001
        result["deps_available"] = False
        notes.append(f"torchao import failed: {e}")
    try:
        from torchao.quantization.quant_api import IntxWeightOnlyConfig  # noqa: F401
        notes.append("torchao.quantization.IntxWeightOnlyConfig OK")
    except Exception as e:  # noqa: BLE001
        result["deps_available"] = False
        notes.append(f"IntxWeightOnlyConfig import failed: {e}")
    try:
        import safetensors  # noqa: F401
        notes.append(f"safetensors: {safetensors.__version__}")
    except Exception as e:  # noqa: BLE001
        result["deps_available"] = False
        notes.append(f"safetensors import failed: {e}")
    result["notes"] = " | ".join(notes)
    return result
