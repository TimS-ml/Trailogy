#!/usr/bin/env python3
"""Inject K/V/k_norm tensors for KV-shared layers (15-34 in Gemma 4
E2B) by copying from the v5.5-era ``unsloth/gemma-4-E2B-it`` base
checkpoint. Without this, ``mlx_vlm.load`` fails with
``Missing 60 parameters`` and ``mlx_lm.quant.*`` produces NaN logits
because dead modules calibrate against garbage weights.

Background
----------
``transformers ≥ 5.8`` 's Gemma 4 attention class allocates
``k_proj`` / ``v_proj`` / ``k_norm`` ONLY for non-KV-shared layers
(0-14 on E2B). For layers 15-34 those modules don't exist as
``nn.Parameter``\\s, so ``save_pretrained`` after ``merge_and_unload``
emits a safetensors with **no K/V for layers 15-34**.

``mlx_vlm 0.4.3`` allocates K/V for ALL 35 layers and calls
``load_weights`` with default ``strict=True``, so the convert fails.

The cleanest band-aid: copy the **real v5.5-era K/V/k_norm bytes**
from the unsloth base model. These are the same bytes
``mlx-community/gemma-4-e2b-it-4bit`` carries today. The KV-shared
layers' forward path never reads them at inference
(``language.py:204-218`` short-circuits to ``cache.state``), but they
make GPTQ / AWQ / DWQ / dynamic_quant Hessians sane during
calibration (calibration runs without a cache, so the forward DOES
call ``self.k_proj`` on these layers — random / zero weights produce
NaN downstream).
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


log = logging.getLogger("prep_inject_kv_shared")


SIDECAR_NAME = "model-kv-shared-pad.safetensors"


def _safe_open(path: Path):
    """Read tensors by name from a safetensors file, working around the
    fact that ``safetensors.safe_open`` with ``framework='numpy'``
    refuses BF16 (`TypeError: data type 'bfloat16' not understood`).

    Strategy: parse the header to find each tensor's byte slice, then
    memory-map the data and view as uint16 (BF16 is 16-bit). We hand
    the result to MLX via a numpy view + ``mx.array(...).view(bfloat16)``.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    # Data section starts at 8 + n.
    data_offset = 8 + n
    return path, header, data_offset


def _read_bf16_tensor(path: Path, header: dict, data_offset: int, key: str) -> mx.array:
    entry = header[key]
    if entry["dtype"] != "BF16":
        raise ValueError(f"{key} has dtype {entry['dtype']!r}, expected BF16")
    shape = tuple(entry["shape"])
    start, end = entry["data_offsets"]
    nbytes = end - start
    buf = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        offset=data_offset + start,
        shape=(nbytes // 2,),
    )
    arr_u16 = mx.array(np.ascontiguousarray(buf).copy())  # detach from mmap
    return arr_u16.view(mx.bfloat16).reshape(shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--base-safetensors",
        required=True,
        type=Path,
        help=(
            "Path to a v5.5-era Gemma 4 base checkpoint's model.safetensors "
            "(carries the real K/V/k_norm bytes for KV-shared layers). "
            "e.g. ``$HF_HOME/hub/models--unsloth--gemma-4-E2B-it/snapshots/<HASH>/model.safetensors``."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_dir: Path = args.model_dir
    base_st: Path = args.base_safetensors
    if not base_st.exists():
        log.error("base safetensors not found: %s", base_st)
        return 2

    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        log.error("config.json not found in %s", model_dir)
        return 2
    config = json.loads(cfg_path.read_text())
    tc = config.get("text_config", {})
    num_hidden_layers = tc["num_hidden_layers"]
    num_kv_shared = tc.get("num_kv_shared_layers", 0)
    if num_kv_shared <= 0:
        log.info("num_kv_shared_layers = %d; nothing to inject.", num_kv_shared)
        return 0
    first_shared = num_hidden_layers - num_kv_shared

    base_path, base_header, base_data_offset = _safe_open(base_st)
    log.info(
        "Source: %s (%d tensors). Injecting K/V/k_norm for layers [%d, %d).",
        base_path, len(base_header) - (1 if "__metadata__" in base_header else 0),
        first_shared, num_hidden_layers,
    )

    tensors: dict[str, mx.array] = {}
    for layer_idx in range(first_shared, num_hidden_layers):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        for sub in ("k_proj", "v_proj", "k_norm"):
            key = f"{prefix}.{sub}.weight"
            if key not in base_header:
                log.error("Base missing %s — cannot copy.", key)
                return 3
            tensors[key] = _read_bf16_tensor(base_path, base_header, base_data_offset, key)

    sidecar_path = model_dir / SIDECAR_NAME
    mx.save_safetensors(str(sidecar_path), tensors, metadata={"format": "pt"})
    log.info(
        "Wrote %d tensors (real v5.5-era K/V/k_norm) to %s.",
        len(tensors), sidecar_path,
    )
    log.info(
        "These bytes are the same ones mlx-community/gemma-4-e2b-it-4bit "
        "ships at int4 cost. Forward path bypasses them (KV-shared layers "
        "read cache.state at language.py:204-218); GPTQ / AWQ / DWQ / "
        "dynamic_quant calibration sees real bf16 activations through them, "
        "avoiding NaN propagation that random-init triggered."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
