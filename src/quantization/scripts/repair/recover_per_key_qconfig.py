#!/usr/bin/env python3
"""Reconstruct per-key ``config["quantization"]`` entries from a saved
GPTQ output where the runner forgot to capture the returned qconfig.

mlx_vlm.load reads ``config["quantization"]`` as either:

  - top-level ``{group_size, bits, mode}`` applied to all linears, OR
  - ``{path: {group_size, bits}, ..., group_size: ..., bits: ..., mode: ...}``
    where per-path entries override the default for that module.

For GPTQ runs that used ``fallback_bits != bits``, some modules (often
``embed_tokens`` / ``lm_head`` / layers whose width isn't divisible by
group_size) get the fallback bits. Without per-path config the saved
quantized weight shape doesn't match what ``nn.quantize`` reproduces
at load → ``ValueError: Expected shape … but received …``.

This script scans ``model.safetensors``, infers bits per quantized
linear from shape, and patches ``config.json`` with per-key entries
where bits differ from the default.

Shape arithmetic
----------------
For ``nn.QuantizedLinear`` at (bits, group_size):

    packed_uint32_per_row = (in_features / group_size) * (bits * group_size / 32)
                          = in_features * bits / 32

So ``bits = (saved_shape[1] * 32) / in_features`` where in_features is
the *unquantized* input dim. We recover in_features from the matching
``.scales`` tensor whose shape is
``(out_features, in_features / group_size)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from collections import defaultdict
from pathlib import Path


log = logging.getLogger("recover_per_key_qconfig")


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def collect_safetensors(model_dir: Path) -> dict:
    """Return merged header from all *.safetensors shards."""
    merged: dict = {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    for shard in shards:
        meta = read_safetensors_header(shard)
        for k, v in meta.items():
            if k == "__metadata__":
                continue
            merged[k] = v
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--default-bits", type=int, default=4)
    parser.add_argument("--default-group-size", type=int, default=128)
    parser.add_argument("--mode", type=str, default="affine")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    md = args.model_dir
    headers = collect_safetensors(md)

    # Find every (key.weight, key.scales) pair under the same prefix.
    weight_keys = {k[: -len(".weight")]: v for k, v in headers.items() if k.endswith(".weight")}
    scales_keys = {k[: -len(".scales")]: v for k, v in headers.items() if k.endswith(".scales")}

    quantized_prefixes = sorted(set(weight_keys) & set(scales_keys))
    log.info("Found %d quantized linears.", len(quantized_prefixes))

    per_key: dict[str, dict] = {}
    bits_hist: dict[int, int] = defaultdict(int)

    for prefix in quantized_prefixes:
        w_shape = weight_keys[prefix]["shape"]
        s_shape = scales_keys[prefix]["shape"]
        if len(w_shape) != 2 or len(s_shape) != 2:
            log.warning("Skipping %s: unexpected shapes w=%s s=%s.", prefix, w_shape, s_shape)
            continue
        out_f = w_shape[0]
        groups_per_row = s_shape[1]
        # in_features = groups_per_row * group_size; we test default first.
        gs_candidates = [args.default_group_size, 64, 32]
        best = None
        for gs in gs_candidates:
            in_f = groups_per_row * gs
            bits_f = (w_shape[1] * 32) / in_f
            if abs(bits_f - round(bits_f)) < 1e-6 and 2 <= round(bits_f) <= 8:
                best = (int(round(bits_f)), gs, in_f)
                break
        if best is None:
            log.warning("Could not infer bits for %s w=%s s=%s.", prefix, w_shape, s_shape)
            continue
        bits, gs, in_f = best
        bits_hist[bits] += 1
        if (bits, gs) != (args.default_bits, args.default_group_size):
            per_key[prefix] = {"group_size": gs, "bits": bits}

    log.info("Bits histogram: %s", dict(bits_hist))
    log.info("Per-key (non-default) entries: %d", len(per_key))

    cfg_path = md / "config.json"
    cfg = json.loads(cfg_path.read_text())
    new_q: dict = {
        "group_size": args.default_group_size,
        "bits": args.default_bits,
        "mode": args.mode,
    }
    new_q.update(per_key)
    cfg["quantization"] = new_q
    cfg["quantization_config"] = new_q
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log.info("Wrote patched %s with %d per-key entries.", cfg_path, len(per_key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
