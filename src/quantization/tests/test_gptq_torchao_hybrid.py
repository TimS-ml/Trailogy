"""Unit tests for the GPTQ + torchao hybrid post-processor.

Three contracts that the hybrid runner relies on:

1. **Bit-packing is a pure round-trip.** ``unpack(pack(x)) == x`` for any
   ``int8`` tensor in the int4 value range.
2. **The packed-quantized embedding's forward matches torchao dequant
   numerically** (cosine_sim > 0.999 vs torchao's own
   ``IntxUnpackedToInt8Tensor`` dequantized output).
3. **The safetensors save → load → forward path preserves identity**:
   producing a hybrid artifact, then reloading and running the
   ``PackedQuantizedEmbedding`` forward yields the same values as the
   in-memory quantize path.

These tests use tiny toy embeddings (no real Gemma 4 weights) and run
in seconds on either CPU or CUDA. The full integration test
(production of an actual hybrid artifact from R2's GPTQ output) is a
separate manual smoke run, not pytest-driven, because it needs ~7 GB
of input on disk + ~10 GB of GPU memory for the
``embed_tokens_per_layer`` pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Pick CUDA only if available; otherwise skip the parts of the suite
# that need torchao on GPU (torchao prints noisy warnings on CPU but
# works). The pack/unpack tests are CPU-only.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def test_pack_unpack_round_trip_full_range():
    """Every int4 value in [-8, 7] must survive a pack/unpack cycle."""
    from src.methods.gptq_torchao_hybrid import (
        pack_int4_to_uint8, unpack_uint8_to_int4,
    )
    # Last dim 16 so all 16 unique values appear at least once after tile.
    qdata = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 1)  # shape [4, 16]
    packed = pack_int4_to_uint8(qdata)
    assert packed.dtype == torch.uint8
    assert packed.shape == (4, 8)  # last dim halved
    recovered = unpack_uint8_to_int4(packed)
    assert recovered.dtype == torch.int8
    assert recovered.shape == qdata.shape
    assert torch.equal(recovered, qdata)


def test_pack_rejects_out_of_range():
    """Values outside [-8, 7] must raise — silent truncation would
    corrupt the round-trip and we'd debug it the hard way at eval time."""
    from src.methods.gptq_torchao_hybrid import pack_int4_to_uint8

    # int8 with values outside int4 range
    qdata = torch.tensor([[-9, 0, 0, 0]], dtype=torch.int8)
    with pytest.raises(ValueError, match="out of int4 range"):
        pack_int4_to_uint8(qdata)

    qdata = torch.tensor([[0, 0, 0, 8]], dtype=torch.int8)
    with pytest.raises(ValueError, match="out of int4 range"):
        pack_int4_to_uint8(qdata)


def test_pack_rejects_odd_last_dim():
    """Last dim must be even — we pack two nibbles into one byte."""
    from src.methods.gptq_torchao_hybrid import pack_int4_to_uint8

    qdata = torch.zeros(4, 7, dtype=torch.int8)
    with pytest.raises(ValueError, match="last dim must be even"):
        pack_int4_to_uint8(qdata)


def test_pack_rejects_wrong_dtype():
    """Refuse anything that's not int8."""
    from src.methods.gptq_torchao_hybrid import pack_int4_to_uint8

    qdata = torch.zeros(4, 16, dtype=torch.int16)
    with pytest.raises(TypeError):
        pack_int4_to_uint8(qdata)


# ---------------------------------------------------------------------------
# Forward numerical match vs torchao dequantize
# ---------------------------------------------------------------------------


@pytest.mark.skipif(DEVICE != "cuda", reason="torchao IntxWeightOnlyConfig flow needs CUDA in our env")
@pytest.mark.parametrize("mapping", ["symmetric", "asymmetric"])
def test_packed_embedding_forward_matches_torchao_dequant(mapping):
    """End-to-end: torchao quantizes a bf16 Embedding, we pack, build
    a PackedQuantizedEmbedding from the packed components, then verify
    forward(x) ≈ original-torchao-dequantized-Embedding(x).

    Tolerance: cosine similarity > 0.999 per row. Strict equality is
    not expected because (a) torchao does the cast int8 → bf16 in a
    different order than our unpack-then-cast, and (b) bf16
    accumulation order can differ marginally. The cos-sim threshold
    catches any systematic sign / scale / offset bug.
    """
    from src.methods.gptq_torchao_hybrid import (
        quantize_embedding_components, PackedQuantizedEmbedding,
    )

    torch.manual_seed(0)
    num_emb, dim, group_size = 512, 1024, 128
    # Realistic init scale for a Gemma-style embedding (small std).
    w = torch.randn(num_emb, dim, dtype=torch.bfloat16) * 0.1

    # Run the production quant path.
    comp = quantize_embedding_components(
        bf16_weight=w, bits=4, group_size=group_size,
        mapping=mapping, device=DEVICE,
    )
    # Build our runtime module from the packed components.
    packed = PackedQuantizedEmbedding.from_components(
        qweight_packed=comp["qweight_packed"],
        scales=comp["scales"],
        zero_point=comp.get("zero_point"),
        bits=comp["bits"],
        group_size=comp["group_size"],
        mapping=comp["mapping"],
        embedding_dim=comp["embedding_dim"],
    ).to(DEVICE)

    # Reference: re-quant via torchao on the same weight, ask torchao
    # to dequantize, treat that as ground truth. We compare PACKED
    # forward against TORCHAO dequantize (NOT against the original
    # bf16 — both quantization paths agree on the quantized values;
    # any error vs bf16 is the same on both sides).
    from torchao.quantization import quantize_, MappingType
    from torchao.quantization.quant_api import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup
    map_type = {
        "symmetric": MappingType.SYMMETRIC,
        "asymmetric": MappingType.ASYMMETRIC,
    }[mapping]
    ref_emb = nn.Embedding(num_emb, dim, dtype=torch.bfloat16, device=DEVICE)
    with torch.no_grad():
        ref_emb.weight.copy_(w.to(DEVICE))
    quantize_(
        ref_emb,
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(group_size),
            mapping_type=map_type,
            scale_dtype=torch.bfloat16,
        ),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
        device=DEVICE,
    )

    # Probe with a deterministic spread of indices.
    idx = torch.tensor([0, 1, 5, 17, 42, 99, 256, 511],
                       dtype=torch.long, device=DEVICE)
    our_out = packed(idx)               # [N, dim]
    ref_out = ref_emb(idx)               # [N, dim], bf16, dequantized

    # Cosine similarity per row.
    cos = torch.nn.functional.cosine_similarity(
        our_out.float(), ref_out.float(), dim=-1,
    )
    assert cos.min().item() > 0.999, (
        f"mapping={mapping}: per-row cos-sim min={cos.min().item():.6f}, "
        f"mean={cos.mean().item():.6f}. Expected > 0.999. "
        "Probable sign/scale/zero_point math mismatch in PackedQuantizedEmbedding.forward."
    )

    # Also sanity-check absolute numerical agreement.
    max_err = (our_out.float() - ref_out.float()).abs().max().item()
    ref_max = ref_out.float().abs().max().item()
    assert max_err < 0.05 * max(ref_max, 1e-6), (
        f"abs error {max_err:.4f} exceeds 5% of ref max {ref_max:.4f}"
    )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(DEVICE != "cuda", reason="needs CUDA for torchao")
def test_save_load_round_trip_preserves_forward(tmp_path):
    """Produce a hybrid artifact from a tiny synthetic GPTQ-shaped
    safetensors dir, reload it, and check the embedding forward
    matches the in-memory quantize-then-forward bytewise.

    We synthesize a minimal GPTQ-shaped input: one safetensors file
    containing just the two embedding weights (no Linears needed for
    this round-trip test; the audio strip is exercised by a separate
    parametrize).
    """
    from safetensors.torch import save_file, safe_open

    from src.methods.gptq_torchao_hybrid import (
        EMBED_TOKENS_KEY,
        EMBED_TOKENS_PER_LAYER_KEY,
        HybridConfig,
        PackedQuantizedEmbedding,
        quantize_hybrid,
    )

    torch.manual_seed(123)

    # Build a synthetic input dir. Small dims so it runs in <1 s.
    num_emb, dim, dim_per_layer = 256, 512, 1024
    inp = tmp_path / "src"
    inp.mkdir()
    state = {
        EMBED_TOKENS_KEY: (torch.randn(num_emb, dim) * 0.1).to(torch.bfloat16),
        EMBED_TOKENS_PER_LAYER_KEY: (torch.randn(num_emb, dim_per_layer) * 0.1).to(torch.bfloat16),
        # An incidental non-target tensor that should pass through.
        "model.language_model.norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        # An audio tensor to verify strip works.
        "model.audio_tower.something.weight": torch.zeros(8, dtype=torch.bfloat16),
        "model.embed_audio.embedding_projection.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
    }
    save_file(state, str(inp / "model.safetensors"))
    (inp / "config.json").write_text(json.dumps({"model_type": "fake_gemma4"}))

    # Run the hybrid pass.
    out = tmp_path / "dst"
    quantize_hybrid(
        inp, out,
        HybridConfig(
            strip_audio=True,
            embed_per_layer_bits=4, embed_per_layer_group_size=128,
            embed_per_layer_mapping="asymmetric",
            embed_tokens_bits=4, embed_tokens_group_size=128,
            embed_tokens_mapping="asymmetric",
            device=DEVICE,
        ),
    )
    assert (out / "model.safetensors").is_file()
    assert (out / "config.json").is_file()
    assert (out / "model.safetensors.index.json").is_file()

    # Audio strip happened?
    with safe_open(out / "model.safetensors", framework="pt", device="cpu") as f:
        keys = set(f.keys())
    assert "model.audio_tower.something.weight" not in keys
    assert "model.embed_audio.embedding_projection.weight" not in keys
    # Pass-through preserved?
    assert "model.language_model.norm.weight" in keys
    # Embed weights replaced by packed components?
    assert EMBED_TOKENS_KEY not in keys
    assert "model.language_model.embed_tokens.qweight_packed" in keys
    assert "model.language_model.embed_tokens.scales" in keys
    assert "model.language_model.embed_tokens.zero_point" in keys
    assert EMBED_TOKENS_PER_LAYER_KEY not in keys
    assert "model.language_model.embed_tokens_per_layer.qweight_packed" in keys

    # Config has the hybrid_quant block?
    cfg = json.loads((out / "config.json").read_text())
    assert "hybrid_quant" in cfg
    hq = cfg["hybrid_quant"]
    assert hq["audio_stripped"] is True
    assert EMBED_TOKENS_KEY in hq["embeddings"]
    assert hq["embeddings"][EMBED_TOKENS_KEY]["bits"] == 4
    assert hq["embeddings"][EMBED_TOKENS_KEY]["embedding_dim"] == dim

    # Rebuild a PackedQuantizedEmbedding from disk and verify forward
    # matches the in-memory packed embedding produced directly from the
    # same weight. This is the cross-process equivalent of the
    # round-trip invariant: identical bytes → identical forward.
    with safe_open(out / "model.safetensors", framework="pt", device="cpu") as f:
        loaded_pk = f.get_tensor("model.language_model.embed_tokens.qweight_packed")
        loaded_sc = f.get_tensor("model.language_model.embed_tokens.scales")
        loaded_zp = f.get_tensor("model.language_model.embed_tokens.zero_point")

    loaded_emb = PackedQuantizedEmbedding.from_components(
        qweight_packed=loaded_pk.to(DEVICE),
        scales=loaded_sc.to(DEVICE),
        zero_point=loaded_zp.to(DEVICE),
        bits=4, group_size=128, mapping="asymmetric",
        embedding_dim=dim,
    )

    # Reproduce the in-memory packed embedding from the same source
    # weight (deterministic — torchao quant is data-only deterministic).
    from src.methods.gptq_torchao_hybrid import quantize_embedding_components
    comp = quantize_embedding_components(
        bf16_weight=state[EMBED_TOKENS_KEY], bits=4,
        group_size=128, mapping="asymmetric", device=DEVICE,
    )
    inmem_emb = PackedQuantizedEmbedding.from_components(
        qweight_packed=comp["qweight_packed"].to(DEVICE),
        scales=comp["scales"].to(DEVICE),
        zero_point=comp["zero_point"].to(DEVICE),
        bits=comp["bits"], group_size=comp["group_size"],
        mapping=comp["mapping"], embedding_dim=comp["embedding_dim"],
    )

    idx = torch.tensor([0, 1, 17, 250], dtype=torch.long, device=DEVICE)
    out_loaded = loaded_emb(idx)
    out_inmem = inmem_emb(idx)
    assert torch.equal(out_loaded, out_inmem), (
        "Forward output differs between loaded-from-disk and in-memory packed "
        "embeddings. This means save/load corrupted the packed state."
    )


# ---------------------------------------------------------------------------
# Regression: Gemma 4 scaled-embedding `embed_scale` must propagate
# ---------------------------------------------------------------------------


@pytest.mark.skipif(DEVICE != "cuda", reason="needs CUDA for torchao")
def test_embed_scale_matches_gemma4_scaled_embedding():
    """The drop-in must multiply by ``embed_scale`` to stay equivalent
    to ``Gemma4TextScaledWordEmbedding`` (modeling_gemma4.py:1441-1452).

    Forgetting the scale collapses HF Gemma 4 PlantNet accuracy from
    ~84% to ~4% because every PLE residual goes in at 1/16 amplitude.
    Asserted ratio here catches a regression long before that point.
    """
    from src.methods.gptq_torchao_hybrid import (
        quantize_embedding_components, PackedQuantizedEmbedding,
    )

    torch.manual_seed(0)
    num_emb, dim, group_size = 128, 256, 64
    scale = 16.0  # mirrors Gemma 4 embed_tokens_per_layer scale exactly
    w = (torch.randn(num_emb, dim) * 0.05).to(torch.bfloat16)

    comp = quantize_embedding_components(
        bf16_weight=w, bits=4, group_size=group_size,
        mapping="asymmetric", device=DEVICE,
    )
    unscaled = PackedQuantizedEmbedding.from_components(
        qweight_packed=comp["qweight_packed"], scales=comp["scales"],
        zero_point=comp.get("zero_point"), bits=comp["bits"],
        group_size=comp["group_size"], mapping=comp["mapping"],
        embedding_dim=comp["embedding_dim"], embed_scale=1.0,
    ).to(DEVICE)
    scaled = PackedQuantizedEmbedding.from_components(
        qweight_packed=comp["qweight_packed"], scales=comp["scales"],
        zero_point=comp.get("zero_point"), bits=comp["bits"],
        group_size=comp["group_size"], mapping=comp["mapping"],
        embedding_dim=comp["embedding_dim"], embed_scale=scale,
    ).to(DEVICE)

    idx = torch.tensor([0, 1, 17, 99], dtype=torch.long, device=DEVICE)
    out_unscaled = unscaled(idx).float()
    out_scaled = scaled(idx).float()

    # Direct elementwise check: scaled output must equal unscaled * scale,
    # within bf16 rounding tolerance. This is the strongest possible
    # assertion and sidesteps divide-by-zero on rows where q_int == zp.
    diff = (out_scaled - out_unscaled * scale).abs()
    tol = 0.01 * (out_unscaled.abs() * scale).max().item()
    assert diff.max().item() <= tol, (
        f"embed_scale not propagating: max |scaled - unscaled*{scale}| = "
        f"{diff.max().item():.6f}, tolerance {tol:.6f}. "
        "PackedQuantizedEmbedding.forward must multiply by embed_scale."
    )

    # Belt: nonzero unscaled entries must show ratio ≈ scale.
    nz = out_unscaled.abs() > 1e-3
    assert nz.any(), "Test setup error: synthetic embedding produced all-zero output"
    ratio = (out_scaled[nz] / out_unscaled[nz]).abs()
    assert (ratio - scale).abs().max().item() < 0.01 * scale, (
        f"embed_scale ratio range [{ratio.min().item():.4f}, "
        f"{ratio.max().item():.4f}] does not match expected {scale}."
    )


def test_embed_scale_defaults_to_no_op():
    """Default ``embed_scale=1.0`` must keep the module a drop-in for
    vanilla ``nn.Embedding``: forward output is unscaled."""
    from src.methods.gptq_torchao_hybrid import (
        PackedQuantizedEmbedding, pack_int4_to_uint8,
    )

    torch.manual_seed(1)
    num_emb, dim, group_size = 32, 64, 32
    # Build a deterministic small packed state directly (no torchao
    # needed for this test).
    qdata = torch.randint(-8, 8, (num_emb, dim), dtype=torch.int8)
    packed = pack_int4_to_uint8(qdata)
    scales = torch.ones(num_emb, dim // group_size, dtype=torch.bfloat16) * 0.1
    zp = torch.zeros(num_emb, dim // group_size, dtype=torch.int8)

    emb = PackedQuantizedEmbedding.from_components(
        qweight_packed=packed, scales=scales, zero_point=zp,
        bits=4, group_size=group_size, mapping="asymmetric",
        embedding_dim=dim,
    )
    assert emb.embed_scale == 1.0
    # Equivalent run with explicit scale=1.0 must match.
    emb2 = PackedQuantizedEmbedding.from_components(
        qweight_packed=packed, scales=scales, zero_point=zp,
        bits=4, group_size=group_size, mapping="asymmetric",
        embedding_dim=dim, embed_scale=1.0,
    )
    idx = torch.tensor([0, 5, 10, 31], dtype=torch.long)
    assert torch.equal(emb(idx), emb2(idx))


# ---------------------------------------------------------------------------
# Regression: quantize_hybrid persists embed_scale into config
# ---------------------------------------------------------------------------


@pytest.mark.skipif(DEVICE != "cuda", reason="needs CUDA for torchao")
def test_quantize_hybrid_persists_embed_scale(tmp_path):
    """When the source config.json carries Gemma 4's hidden sizes,
    quantize_hybrid must write the derived embed_scale into
    hybrid_quant.embeddings[...].embed_scale.

    This insulates the load path from depending on the live module
    attribute — it's a belt for the live-module-probe braces.
    """
    from safetensors.torch import save_file
    from src.methods.gptq_torchao_hybrid import (
        EMBED_TOKENS_KEY, EMBED_TOKENS_PER_LAYER_KEY,
        HybridConfig, quantize_hybrid,
    )

    num_emb, dim, dim_pl = 64, 128, 256
    inp = tmp_path / "src"
    inp.mkdir()
    state = {
        EMBED_TOKENS_KEY: (torch.randn(num_emb, dim) * 0.1).to(torch.bfloat16),
        EMBED_TOKENS_PER_LAYER_KEY: (torch.randn(num_emb, dim_pl) * 0.1).to(torch.bfloat16),
    }
    save_file(state, str(inp / "model.safetensors"))
    # Mimic the Gemma 4 config shape: hidden sizes live under text_config.
    (inp / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": dim,
            "hidden_size_per_layer_input": dim_pl // 4,  # arbitrary != dim
            "num_hidden_layers": 4,
        },
    }))

    out = tmp_path / "dst"
    quantize_hybrid(
        inp, out,
        HybridConfig(
            strip_audio=False,
            embed_per_layer_bits=4, embed_per_layer_group_size=64,
            embed_per_layer_mapping="asymmetric",
            embed_tokens_bits=4, embed_tokens_group_size=64,
            embed_tokens_mapping="asymmetric",
            device=DEVICE,
        ),
    )
    cfg = json.loads((out / "config.json").read_text())
    embs = cfg["hybrid_quant"]["embeddings"]
    import math
    assert embs[EMBED_TOKENS_KEY]["embed_scale"] == pytest.approx(math.sqrt(dim))
    assert embs[EMBED_TOKENS_PER_LAYER_KEY]["embed_scale"] == pytest.approx(math.sqrt(dim_pl // 4))
