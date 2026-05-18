#!/usr/bin/env python3
"""Tensor-level LoRA merge that preserves the base safetensors key set.

Motivation
----------
The standard ``PEFT merge_and_unload`` → ``save_pretrained`` recipe goes
through HF transformers' Gemma 4 model class, which (correctly, per the
v5.8 architecture) does NOT allocate k_proj / v_proj / k_norm / v_norm
on KV-shared layers (E2B: layers 15-34). ``save_pretrained`` only
writes registered ``nn.Parameter``s, so those tensors are dropped from
the output safetensors.

This is invisible at training time (the model class never references
the dropped weights) but kills downstream conversion:
``mlx_vlm.convert`` 0.4.3 instantiates the dead modules unconditionally
and strict-loads, failing with ``Missing N parameters``.

This module exists to keep that root cause documented in-repo and to
make the safe merge path executable/testable.

What this tool does
-------------------
1. Open the base model's safetensors directly (no HF model object).
2. For every LoRA ``(lora_A, lora_B)`` pair in the adapter, compute
   ``(alpha / r) * lora_B @ lora_A`` and add it in place to the matching
   base tensor.
3. For every ``modules_to_save`` tensor in the adapter (PEFT writes
   these as direct weights under the original module path, no
   ``modules_to_save.default`` infix), REPLACE the matching base
   tensor with the trained version.
4. Write the result as a single safetensors with the SAME key set as
   the base — including the dead K/V keys mlx-vlm wants.
5. Copy ``config.json`` / ``processor_config.json`` / tokenizer side-
   cars so the output is a self-contained HF-loadable directory.

This bypasses HF's transformers Gemma 4 class entirely. Forward-
compatible: if HF ever changes how ``_keys_to_ignore_on_load_unexpected``
works, this tool keeps doing the right thing because it never reads
the live model class.

Naming convention
-----------------
The PEFT adapter saves tensors under
``base_model.model.<base-name-without-leading-model.>``. For Gemma 4
(``Gemma4ForConditionalGeneration``) this resolves to
``base_model.model.model.<actual_name>``. We strip ``base_model.model.``
to recover the base safetensors key.

LoRA tensor naming convention (peft 0.19.x, no named-adapter suffix):
    base_model.model.<module_path>.lora_A.weight
    base_model.model.<module_path>.lora_B.weight

``modules_to_save`` saved tensors (peft 0.19.x):
    base_model.model.<module_path>.weight   (the trained weight directly)

We detect the LoRA target by the ``.lora_A.weight`` /
``.lora_B.weight`` suffix. Anything else (including the projector full-
param tensors) we treat as a direct replacement.

Limitations
-----------
- Supports LoRA-on-Linear only. Conv1d/Conv2d LoRA targets would
  require additional reshape handling.
- Supports ``modules_to_save`` for tensors that resolve 1:1 against a
  base tensor name (the common case for projector / vision-layer
  full-param tuning).
- Trusts the adapter's ``adapter_config.json`` for ``r`` and
  ``lora_alpha``. Scale = ``lora_alpha / r``.

Usage
-----
    python -m scripts.repair.merge_safetensors \\
        --base unsloth/gemma-4-E2B-it \\
        --adapter outputs/plantnet-50k-baseline2-qlora-.../final-adapter \\
        --output quantization/results/baseline2_qlora_safemerged_bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Canonical sidecar list lives in common/model_io.py so all PTQ paths
# (gptqmodel, bnb_nf4, mlx_vlm, this merge tool, splice_lm_into_multimodal)
# share the same set. ``config.json`` is added here because this tool
# also copies the model architecture config alongside the sidecars
# (gptqmodel-style PTQ wouldn't need it because save_quantized handles
# config.json itself).
from src.common.model_io import PROCESSOR_SIDECAR_FILES

log = logging.getLogger("merge_safetensors")

_PROCESSOR_FILES: tuple[str, ...] = ("config.json", *PROCESSOR_SIDECAR_FILES)

_PEFT_PREFIX = "base_model.model."  # PEFT wraps base into this dotted prefix


def _resolve_base_dir(base: str) -> Path:
    """Accept either a local dir path or an HF Hub repo id; return a
    local directory containing the safetensors + processor side-cars.
    """
    p = Path(base)
    if p.is_dir():
        return p
    # Fall back to HF Hub download — populate the cache, then return
    # the snapshot directory.
    from huggingface_hub import snapshot_download

    log.info("Resolving %s via huggingface_hub.snapshot_download", base)
    local = snapshot_download(
        repo_id=base,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            *_PROCESSOR_FILES,
        ],
    )
    return Path(local)


def _enumerate_base_safetensors(base_dir: Path) -> list[Path]:
    idx_path = base_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        shards = sorted({base_dir / shard for shard in idx["weight_map"].values()})
        log.info(
            "Base is sharded: %d shards under %s", len(shards), base_dir
        )
        return shards
    mono = base_dir / "model.safetensors"
    if mono.exists():
        log.info("Base is monolithic: %s", mono)
        return [mono]
    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json under {base_dir}"
    )


def _load_base_tensors(base_dir: Path) -> Dict[str, torch.Tensor]:
    """Load every base tensor into RAM as bf16/torch.Tensor."""
    shards = _enumerate_base_safetensors(base_dir)
    tensors: Dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    log.info("Loaded %d base tensors from %s", len(tensors), base_dir)
    return tensors


def _load_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not in {adapter_dir}")
    return json.loads(cfg_path.read_text())


def _strip_peft_prefix(adapter_key: str) -> str:
    """Strip ``base_model.model.`` from a PEFT-stored key to recover the
    name used in the base safetensors.
    """
    if not adapter_key.startswith(_PEFT_PREFIX):
        raise ValueError(
            f"Adapter key {adapter_key!r} does not start with the "
            f"expected PEFT prefix {_PEFT_PREFIX!r}. This tool assumes "
            "peft 0.19.x layout."
        )
    return adapter_key[len(_PEFT_PREFIX):]


def _is_lora_a(k: str) -> bool:
    return k.endswith(".lora_A.weight")


def _is_lora_b(k: str) -> bool:
    return k.endswith(".lora_B.weight")


def _lora_target_base_key(adapter_a_key: str) -> str:
    """``...module_path.lora_A.weight`` → ``...module_path.weight`` (base-side)."""
    if not _is_lora_a(adapter_a_key):
        raise ValueError(f"Not a lora_A key: {adapter_a_key}")
    base_unprefixed = _strip_peft_prefix(adapter_a_key)
    # Drop ".lora_A.weight" → append ".weight"
    return base_unprefixed[: -len(".lora_A.weight")] + ".weight"


def _modules_to_save_target_base_key(adapter_key: str) -> str:
    """Non-LoRA adapter tensor → base key (just strip the PEFT prefix)."""
    return _strip_peft_prefix(adapter_key)


def merge(
    base_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> None:
    """Run the full safetensors-level merge."""
    cfg = _load_adapter_config(adapter_dir)
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    scale = alpha / r
    log.info(
        "Adapter config: r=%d, lora_alpha=%d, scale=%.4f; "
        "modules_to_save=%s",
        r, alpha, scale, cfg.get("modules_to_save"),
    )

    base = _load_base_tensors(base_dir)
    log.info("Base key set: %d tensors", len(base))

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not in {adapter_dir}")

    # Pass 1: collect LoRA pairs and direct replacements.
    lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    direct_replacements: Dict[str, torch.Tensor] = {}

    # First pass: enumerate keys so we can pair A with B without
    # ambiguity on iteration order.
    lora_a_keys: list[str] = []
    other_keys: list[str] = []
    with safe_open(str(adapter_file), framework="pt") as f:
        for k in f.keys():
            if _is_lora_a(k):
                lora_a_keys.append(k)
            elif _is_lora_b(k):
                # Will be paired with its A counterpart below.
                pass
            else:
                other_keys.append(k)
        # Now load the actual tensors.
        for a_key in lora_a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.")
            target = _lora_target_base_key(a_key)
            a_t = f.get_tensor(a_key)
            b_t = f.get_tensor(b_key)
            lora_pairs[target] = (a_t, b_t)
        for k in other_keys:
            target = _modules_to_save_target_base_key(k)
            direct_replacements[target] = f.get_tensor(k)

    log.info(
        "Adapter: %d LoRA pairs, %d direct replacements",
        len(lora_pairs), len(direct_replacements),
    )

    # Sanity: every LoRA / direct target must exist in the base.
    missing = [k for k in lora_pairs if k not in base]
    if missing:
        # Show first 5 so we can spot the pattern.
        raise KeyError(
            f"{len(missing)} LoRA targets are not in the base. "
            f"Sample: {missing[:5]}. This means either the wrong "
            "--base was supplied, or the adapter targets a module "
            "path that does not exist in this checkpoint. Aborting."
        )
    missing_direct = [k for k in direct_replacements if k not in base]
    if missing_direct:
        raise KeyError(
            f"{len(missing_direct)} direct-replacement targets are not "
            f"in the base. Sample: {missing_direct[:5]}. Aborting."
        )

    # Pass 2: apply LoRA deltas in fp32, cast back to base dtype.
    applied_lora = 0
    for target, (a, b) in lora_pairs.items():
        base_t = base[target]
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
        merged = base_t.to(torch.float32) + delta
        base[target] = merged.to(base_t.dtype)
        applied_lora += 1
    log.info("Applied %d LoRA deltas in fp32, cast back to base dtype", applied_lora)

    # Pass 3: replace modules_to_save tensors.
    applied_direct = 0
    for target, t in direct_replacements.items():
        base_t = base[target]
        if t.shape != base_t.shape:
            raise ValueError(
                f"Shape mismatch on direct replacement {target}: "
                f"adapter has {t.shape}, base has {base_t.shape}."
            )
        base[target] = t.to(base_t.dtype)
        applied_direct += 1
    log.info("Applied %d direct replacements (modules_to_save)", applied_direct)

    # Save: same key set as base, monolithic file (Gemma 4 E2B fits).
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "model.safetensors"
    log.info(
        "Writing merged safetensors → %s (%d tensors)",
        out_path, len(base),
    )
    save_file(base, str(out_path))

    # Copy side-cars from the BASE (config.json reflects the base
    # architecture, which is what mlx-vlm convert reads). Tokenizer
    # comes from the adapter if present (post-train state); fall back
    # to the base.
    _copy_sidecars(base_dir, adapter_dir, output_dir)


def _copy_sidecars(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    """Copy processor / tokenizer / config side-cars.

    Order of preference for each file:
      1. adapter_dir (post-train state, e.g. updated chat template)
      2. base_dir
    """
    import shutil

    copied = []
    for name in _PROCESSOR_FILES:
        for src_dir in (adapter_dir, base_dir):
            src = src_dir / name
            if src.exists():
                dst = output_dir / name
                shutil.copy2(src, dst)
                copied.append((name, src_dir.name))
                break
    log.info(
        "Copied side-cars: %s",
        ", ".join(f"{n} (<-{src})" for n, src in copied),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--base",
        required=True,
        help="HF repo id or local directory of the bf16 base model. "
             "MUST be the bf16 source even if training used a quantized "
             "variant — LoRA deltas merge into the dequantized weights.",
    )
    p.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="PEFT adapter directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the merged safetensors. Must not "
             "exist or must be empty.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.output.exists() and any(args.output.iterdir()):
        log.error(
            "Output directory %s exists and is non-empty. Refusing to "
            "overwrite — pass a fresh path or rm -rf first.",
            args.output,
        )
        return 2

    base_dir = _resolve_base_dir(args.base)
    merge(base_dir, args.adapter, args.output)

    # Quick post-write sanity report.
    out_st = args.output / "model.safetensors"
    out_sz = out_st.stat().st_size if out_st.exists() else 0
    log.info(
        "Merged dir: %s (model.safetensors %.2f GB)",
        args.output, out_sz / 1024 / 1024 / 1024,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
