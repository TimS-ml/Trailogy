#!/usr/bin/env python3
"""Hybrid-flow mlx-native quantization re-test (gptq / awq / dwq /
dynamic_quant), all at g=128 on the SFT'd merged bf16.

Hybrid flow:

    from mlx_vlm import load
    model, processor = load(bf16_dir)          # correct Gemma 4 tree
    from mlx_lm.quant.<method> import <quantize_fn>
    <quantize_fn>(model.language_model, calib, ...)
    from mlx_vlm.utils import save_weights, save_config
    save_weights(out, model); save_config(config, out / "config.json")

This is the corrected pipeline for mlx-native quantization on Gemma 4:
quant operates on the mlx_vlm-constructed Gemma 4 tree (with the right
RMSNormZeroShift / ScaledLinear classes and full KV layout), so the
output is directly loadable via ``mlx_vlm.load`` — no splice step
needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

# Ensure quantization/src/ is importable regardless of how this script
# is invoked (-m package.module, direct path, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("run_mlx_hybrid_quant")


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def load_via_mlx_vlm(bf16_dir: Path):
    """Load Gemma 4 multimodal via mlx_vlm (correct model class)."""
    from mlx_vlm import load as vlm_load

    log.info("Loading via mlx_vlm: %s", bf16_dir)
    model, processor = vlm_load(str(bf16_dir))
    return model, processor


def save_via_mlx_vlm(
    model,
    processor,
    src_dir: Path,
    out_dir: Path,
    quantization: dict,
) -> None:
    """Save the (now-quantized) model in mlx_vlm format.

    Mirrors what ``mlx_vlm.convert.convert`` does after quant: writes
    ``model.safetensors`` shards via ``save_weights``, copies over
    every ``*.json`` / ``*.py`` (incl. config.json, processor configs,
    chat_template etc.), and patches the ``quantization`` block into
    the saved config.json so ``mlx_vlm.load`` recognizes the variant.
    """
    from mlx_vlm.utils import save_weights, save_config

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving via mlx_vlm: %s", out_dir)
    save_weights(out_dir, model, donate_weights=False)

    # Copy side-cars from src (config, processor, tokenizer, chat_template).
    for pattern in ("*.py", "*.json"):
        for f in src_dir.glob(pattern):
            if f.name == "model.safetensors.index.json":
                continue
            shutil.copy(f, out_dir / f.name)
    for f in src_dir.iterdir():
        if f.is_dir():
            dst = out_dir / f.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(f, dst)
    # Save chat_template explicitly if present (some processors expect it)
    for extra in ("chat_template.jinja",):
        srcf = src_dir / extra
        if srcf.exists():
            shutil.copy(srcf, out_dir / extra)
    if processor is not None:
        try:
            processor.save_pretrained(str(out_dir))
        except Exception as e:
            log.warning("processor.save_pretrained failed (non-fatal): %s", e)

    # Patch quantization block into config.json
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization"] = quantization
    cfg["quantization_config"] = quantization
    save_config(cfg, str(cfg_path))


class _LogitsUnwrapper:
    """Adapter making mlx_vlm's ``LanguageModel`` look like mlx_lm's.

    mlx_vlm wraps logits in a ``LanguageModelOutput`` dataclass; mlx_lm
    quantization paths (dwq, dynamic_quant, awq) call ``model(batch)``
    and expect a raw ``mx.array`` so they can ``.size`` / ``.shape`` /
    take ``KL`` divergence on it. Wrapping passes attribute access and
    state ops through so ``nn.quantize`` and weight save still work.
    """

    def __init__(self, lm):
        # Bypass __setattr__ via object.__setattr__ for the wrapped ref.
        object.__setattr__(self, "_lm", lm)

    def __call__(self, *args, **kwargs):
        out = object.__getattribute__(self, "_lm")(*args, **kwargs)
        if hasattr(out, "logits"):
            return out.logits
        return out

    # Pass-through every attribute access to the underlying language_model.
    # Use object.__getattribute__ to avoid recursive __getattr__ on _lm.
    def __getattr__(self, name):
        lm = object.__getattribute__(self, "_lm")
        return getattr(lm, name)

    def __setattr__(self, name, value):
        if name == "_lm":
            object.__setattr__(self, name, value)
        else:
            lm = object.__getattribute__(self, "_lm")
            setattr(lm, name, value)


def text_calibration(processor, num_samples: int = 256, seq_len: int = 2048):
    """Text calibration from HF ``wikitext-2-raw-v1`` train split.

    The shipped ``mlx_lm.quant.gptq.load_data`` reads a small fixed
    text file (calibration_v5_rc.txt, ~110 K tokens). At seq_len=2048
    that yields only ~55 chunks — too few for GPTQ's Hessian estimate
    on Gemma 4 E2B (we saw NaN/Inf scales in 117 layers' output after
    the run). WikiText-2 train concatenated gives ~2.7 M tokens →
    ~1300 chunks at seq_len=2048 → plenty of calibration headroom.

    Returns an ``mx.array`` of shape ``(num_samples, seq_len)``.
    """
    from datasets import load_dataset
    import mlx.core as mx

    tokenizer = processor.tokenizer
    log.info(
        "Loading WikiText-2 train via HF datasets for calibration "
        "(target %d samples × %d tokens)...",
        num_samples, seq_len,
    )
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    log.info("WikiText concat: %.1f MB of text.", len(text) / 1e6)

    tokens = tokenizer.encode(text, return_tensors="mlx")[0]
    n_chunks = tokens.size // seq_len
    tokens = tokens[: n_chunks * seq_len].reshape(-1, seq_len)
    log.info("Tokenized into %d chunks of %d tokens each.", n_chunks, seq_len)

    perm = mx.random.permutation(tokens.shape[0])
    if num_samples > 0 and num_samples < tokens.shape[0]:
        perm = perm[:num_samples]
    data = tokens[perm]
    log.info("Calibration shape: %s", data.shape)
    return data


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------


def run_gptq(model, processor, bits: int, group_size: int) -> dict:
    from src.methods.gptq_stable import gptq_quantize_model

    calib = text_calibration(processor, num_samples=256, seq_len=2048)
    fallback_bits, fallback_group_size = 6, group_size

    log.info(
        "GPTQ (stable): bits=%d gs=%d fallback_bits=%d fallback_gs=%d",
        bits, group_size, fallback_bits, fallback_group_size,
    )
    t0 = time.perf_counter()
    _, lm_qconfig = gptq_quantize_model(
        model.language_model,
        calib,
        bits=bits,
        group_size=group_size,
        fallback_bits=fallback_bits,
        fallback_group_size=fallback_group_size,
    )
    elapsed = time.perf_counter() - t0

    from mlx_lm.quant.gptq import compute_bits_per_weight
    bpw = compute_bits_per_weight(model.language_model)
    log.info("GPTQ stable done in %.1fs. language_model bpw = %.3f", elapsed, bpw)

    qconfig: dict = {
        "group_size": group_size,
        "bits": bits,
        "mode": "affine",
    }
    for k, v in lm_qconfig.items():
        if isinstance(v, dict):
            full_key = f"language_model.{k}" if not k.startswith("language_model") else k
            qconfig[full_key] = v
    return qconfig


# ---------------------------------------------------------------------------
# AWQ
# ---------------------------------------------------------------------------


def _register_gemma4_text_awq_config():
    """Patch AWQ_MODEL_CONFIGS with a 'gemma4_text' entry that re-uses
    the gemma3 layer-block topology (Gemma 4's per-layer structure
    matches: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
    input_layernorm, pre_feedforward_layernorm)."""
    from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS

    if "gemma4_text" in AWQ_MODEL_CONFIGS:
        return
    base = AWQ_MODEL_CONFIGS["gemma3"]
    # Same topology — shallow copy is fine.
    AWQ_MODEL_CONFIGS["gemma4_text"] = base
    log.info("Registered AWQ_MODEL_CONFIGS['gemma4_text'] = (reuse of gemma3 entry).")


def run_awq(model, processor, bits: int, group_size: int) -> dict:
    from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS, awq_quantize, update_config

    _register_gemma4_text_awq_config()
    awq_cfg = AWQ_MODEL_CONFIGS["gemma4_text"]
    # AWQ's ``search_best_scale`` runs an n_grid sweep over the
    # calibration batch, materializing (n_grid, B, T, H) tensors.
    # Default upstream uses 128 × 512 = 64 K tokens at n_grid=20 ->
    # ~7 GB. We saw OOM at 256 × 2048 with n_grid=20: on a Mac Pro-
    # class unified-memory device Metal's per-buffer ceiling sits at
    # ~14 GB and AWQ tried to allocate ~17 GB. Drop to (64, 512) and
    # n_grid=10 to fit.
    inputs = text_calibration(processor, num_samples=64, seq_len=512)
    n_grid = 10

    log.info(
        "AWQ: bits=%d gs=%d calib=%s n_grid=%d (config reused from gemma3)",
        bits, group_size, tuple(inputs.shape), n_grid,
    )
    t0 = time.perf_counter()
    # AWQ extracts model[lm_key] internally. lm_key = 'language_model',
    # so we pass the full multimodal model.
    awq_quantize(
        model,
        inputs,
        awq_cfg,
        group_size=group_size,
        bits=bits,
        embed_group_size=32,
        embed_bits=4,
        n_grid=n_grid,
    )
    elapsed = time.perf_counter() - t0
    log.info("AWQ done in %.1fs.", elapsed)
    # update_config walks every leaf, recording each module's actual
    # bits/group_size or False for unquantized. This becomes the
    # ``config["quantization"]`` dict that mlx_vlm.load consults.
    cfg: dict = {}
    update_config(model, cfg)
    cfg["quantization"]["mode"] = "affine"
    return cfg["quantization"]


# ---------------------------------------------------------------------------
# DWQ
# ---------------------------------------------------------------------------


def run_dwq(model, processor, bits: int, group_size: int) -> dict:
    """Distillation-based weight quantization. Needs a target model
    (the bf16 teacher) and validation data. Returns the in-place
    quantized model on success."""
    from copy import deepcopy
    from mlx_lm.quant.dwq import dwq_quantize, compute_dwq_targets

    # We'll use the bf16 LM itself as teacher BEFORE quant (target_fn
    # is computed up front and frozen).
    log.info("DWQ: preparing teacher targets from bf16 language_model...")
    train_data = text_calibration(processor, num_samples=128, seq_len=2048)
    valid_data = text_calibration(processor, num_samples=32, seq_len=2048)

    # Build a target callable: capture bf16 logits over train + valid
    # batches before quantization mutates the model.
    teacher = model.language_model
    batch_size = 1
    max_seq_length = 2048

    # `compute_dwq_targets` returns a (target_fn, train_iter, valid_iter)
    # tuple suitable for dwq_quantize.
    try:
        target_fn, train_iter, valid_iter = compute_dwq_targets(
            teacher,
            None,
            train_data,
            valid_data,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            seed=0,
        )
    except Exception as e:
        log.error("compute_dwq_targets failed: %s", e)
        raise

    # Quantize language_model in place via DWQ training loop.
    # `nn.quantize` the model first with the target bit/gs, then DWQ
    # trains the dequant residuals to minimize KL vs teacher.
    # Apply the static quant first:
    import mlx.nn as nn
    from mlx_lm.quant.dwq import quantize_model as _qm
    _qm(model.language_model, model.language_model.config.__dict__, group_size, bits)

    opt = optim.AdamW(learning_rate=1e-5)
    log.info("DWQ: starting training loop bits=%d gs=%d", bits, group_size)
    t0 = time.perf_counter()
    dwq_quantize(
        model.language_model,
        target_fn,
        opt,
        train_iter,
        valid_iter,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        seed=0,
    )
    log.info("DWQ done in %.1fs.", time.perf_counter() - t0)
    return {"group_size": group_size, "bits": bits, "mode": "affine"}


# ---------------------------------------------------------------------------
# dynamic_quant
# ---------------------------------------------------------------------------


def run_dynamic_quant(model, processor, bits: int, group_size: int) -> dict:
    """Dynamic_quant: estimates per-layer sensitivity, then assigns
    low_bits (3) to sensitive layers and high_bits (4) to robust ones,
    targeting an average bpw of ``bits``. Mirrors ``dynamic_quant.main``."""
    from mlx_lm.quant.dynamic_quant import (
        estimate_sensitivities,
        estimate_threshold,
        compute_bits_per_weight,
        quantize_model as _qm,
    )
    from mlx_lm.quant.awq import update_config

    target_bpw = float(bits)
    low_bits, high_bits = 3, 4
    low_gs, high_gs = group_size, group_size

    # Calibration budget. Previous run (128 × 512, no gradient_checkpoint)
    # timed out at 2 h on M5 Pro 32 GB — iter rate degraded under memory
    # pressure (~70 MB free, 600 MB compressed at iter 14/32).
    # Reduce to 64 × 512 + gradient_checkpoint=True to trade compute
    # for memory headroom and avoid page compression stalls.
    calib = text_calibration(processor, num_samples=64, seq_len=512)

    log.info(
        "dynamic_quant: target_bpw=%.2f low=(%d,%d) high=(%d,%d)",
        target_bpw, low_bits, low_gs, high_bits, high_gs,
    )
    t0 = time.perf_counter()
    # mlx_lm.quant.dynamic_quant expects raw logits from the model
    # call; mlx_vlm wraps them in LanguageModelOutput. Adapter unwraps.
    lm_unwrapped = _LogitsUnwrapper(model.language_model)
    sensitivities = estimate_sensitivities(
        lm_unwrapped,
        calib,
        low_bits=low_bits,
        low_group_size=low_gs,
        high_bits=high_bits,
        high_group_size=high_gs,
        gradient_checkpoint=True,
    )
    log.info(
        "Sensitivities done in %.1fs. %d entries.",
        time.perf_counter() - t0, len(sensitivities),
    )
    sensitivities = dict(sensitivities)
    threshold = estimate_threshold(
        lm_unwrapped,
        sensitivities,
        target_bpw=target_bpw,
        low_bits=low_bits,
        low_group_size=low_gs,
        high_bits=high_bits,
        high_group_size=high_gs,
    )
    log.info("Threshold = %.6e", threshold)

    def predicate(p, m):
        if not hasattr(m, "to_quantized"):
            return False
        if p in sensitivities and sensitivities[p] > threshold:
            return {"bits": high_bits, "group_size": high_gs}
        return True

    _qm(
        model.language_model,
        {},
        group_size=low_gs,
        bits=low_bits,
        quant_predicate=predicate,
    )

    bpw = compute_bits_per_weight(model.language_model)
    log.info(
        "dynamic_quant done in %.1fs. Final bpw = %.3f (target %.2f).",
        time.perf_counter() - t0, bpw, target_bpw,
    )
    cfg: dict = {}
    update_config(model, cfg)
    cfg["quantization"]["mode"] = "affine"
    return cfg["quantization"]


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


METHODS = {
    "gptq": run_gptq,
    "awq": run_awq,
    "dwq": run_dwq,
    "dynamic_quant": run_dynamic_quant,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--hf-path", required=True, type=Path)
    parser.add_argument("--mlx-path", required=True, type=Path)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    src = args.hf_path
    dst = args.mlx_path
    if not src.is_dir():
        log.error("HF source dir not found: %s", src)
        return 2

    model, processor = load_via_mlx_vlm(src)
    runner = METHODS[args.method]
    quantization = runner(model, processor, bits=args.bits, group_size=args.group_size)

    save_via_mlx_vlm(model, processor, src, dst, quantization)
    log.info("Done. Output: %s", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
