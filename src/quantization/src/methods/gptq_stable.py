"""Numerically stable GPTQ for MLX — fixes NaN on Gemma 4.

Ported from parameter-golf PR#1855's ``gptq_quantize_weight``.  Three
fixes over ``mlx_lm.quant.gptq.gptq_quantize``:

1. **desc_act** — process columns in descending Hessian-diagonal order
   so the most informative dimensions are quantized first, reducing
   error amplification into later columns.
2. **Dead-column handling** — zero Hessian diagonal ⇒ zero the weight
   column and set H[i,i]=1 before Cholesky.  Prevents NaN from
   singular matrices on Gemma 4's KV-shared layers.
3. **Symmetric per-row clipping** — adaptive ``clip_sigmas * row_std``
   quantization range instead of mlx_lm's per-group min/max asymmetric
   grid.  Avoids Inf scales when GPTQ error propagation shifts weights
   outside the original group range.

The output is packed into mlx ``QuantizedLinear`` format so it drops
into ``mlx_vlm.load`` without any downstream changes.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hessian collection (reuses the Catcher hook pattern from mlx_lm)
# ---------------------------------------------------------------------------


class Catcher(nn.Module):
    """Forward-hook wrapper that accumulates X^T X for GPTQ Hessians."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.H = mx.zeros((1,), dtype=mx.float32)  # will be resized on first call
        self.n_samples = 0

    def __call__(self, x, *args, **kwargs):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        xtx = xf.T @ xf
        if self.H.size == 1:
            self.H = xtx
        else:
            self.H = self.H + xtx
        self.n_samples += xf.shape[0]
        return self.module(x, *args, **kwargs)


# ---------------------------------------------------------------------------
# Core GPTQ quantize (per weight matrix)
# ---------------------------------------------------------------------------


def _pack_to_mlx_format(Q_int: mx.array, bits: int) -> mx.array:
    """Pack signed int quantized weights into mlx uint32 format.

    mlx QuantizedLinear expects unsigned [0, 2^bits-1] packed into
    uint32.  We shift from signed [-range, range] to unsigned first.
    """
    n_bins = 2**bits - 1
    el_per_int = 32 // bits
    # Shift from signed to unsigned: [-range, range] -> [0, 2*range]
    half_range = 2 ** (bits - 1)
    Q_unsigned = (Q_int + half_range).astype(mx.uint32)
    # Pack el_per_int values into each uint32
    rows, cols = Q_unsigned.shape
    # Pad cols to multiple of el_per_int
    pad = (el_per_int - cols % el_per_int) % el_per_int
    if pad > 0:
        Q_unsigned = mx.pad(Q_unsigned, [(0, 0), (0, pad)])
    Q_unsigned = Q_unsigned.reshape(rows, -1, el_per_int)
    shifts = mx.power(mx.array(2, dtype=mx.uint32), mx.arange(0, 32, bits, dtype=mx.uint32))
    packed = mx.sum(Q_unsigned * shifts[None, None, :], axis=-1)
    return packed


def gptq_quantize_weight(
    W: mx.array,
    H: mx.array,
    bits: int = 4,
    group_size: int = 128,
    clip_sigmas: float = 3.0,
    block_size: int = 128,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """GPTQ with desc_act + dead-col handling + symmetric per-row clip.

    Args:
        W: (out_features, in_features) weight matrix, float32.
        H: (in_features, in_features) Hessian X^T X, float32.
        bits: target bit width (default 4).
        group_size: quantization group size (default 128).
        clip_sigmas: symmetric clipping range in row-std units.
        block_size: GPTQ block size for batched error propagation.

    Returns:
        (W_packed, scales, biases, perm):
        - W_packed: uint32 packed quantized weights in mlx format
        - scales: per-group scales for dequantization
        - biases: per-group biases for dequantization
        - perm: column permutation used (desc_act order)
    """
    rows, cols = W.shape
    W = W.astype(mx.float32)
    H = H.astype(mx.float32)

    clip_range = 2 ** (bits - 1) - 1  # e.g. 7 for 4-bit symmetric
    n_bins = 2**bits - 1              # e.g. 15 for 4-bit

    # --- Dead column handling ---
    diag = mx.diag(H)
    dead = diag == 0
    n_dead = int(mx.sum(dead).item())
    if n_dead > 0:
        log.debug("  %d dead columns (zero Hessian diagonal)", n_dead)
        # Set dead diagonal to 1 to prevent singular Cholesky
        diag_fix = mx.where(dead, mx.ones_like(diag), mx.zeros_like(diag))
        H = H + mx.diag(diag_fix)
        # Zero out dead weight columns — they contribute nothing
        dead_mask = mx.broadcast_to(dead[None, :], W.shape)
        W = mx.where(dead_mask, mx.zeros_like(W), W)

    # --- Damping ---
    damp = 1e-2 * mx.mean(mx.diag(H))
    H = H + mx.diag(mx.full((cols,), damp, dtype=mx.float32))

    # --- desc_act: column reordering ---
    perm = mx.argsort(-mx.diag(H))  # descending by Hessian diagonal
    invperm = mx.argsort(perm)
    W_perm = W[:, perm]
    H_perm = H[perm][:, perm]

    # --- Cholesky inverse (must run on CPU in current MLX) ---
    # H_perm = L L^T  →  H_perm^{-1} = L^{-T} L^{-1}
    # Then Cholesky of H^{-1} = U (upper triangular)
    # This is the "Hinv" used in GPTQ's error formula: err = (w - q) / d
    # where d = Hinv[j, j]
    with mx.stream(mx.cpu):
        L = mx.linalg.cholesky(H_perm)
        Hinv_full = mx.linalg.cholesky_inv(L)
        Hinv = mx.linalg.cholesky(Hinv_full, upper=True)
    mx.eval(Hinv)

    # --- Per-row symmetric scales ---
    # Compute on ORIGINAL W (before permutation) — matches reference.
    # Use true std (with mean subtraction), not RMS.
    row_mean = mx.mean(W, axis=1, keepdims=True)
    row_std = mx.sqrt(mx.mean((W - row_mean) ** 2, axis=1))
    scale_per_row = (clip_sigmas * row_std / clip_range)
    scale_per_row = mx.maximum(scale_per_row, mx.array(1e-10, dtype=mx.float32))
    # Round-trip through float16 so quantization grid aligns with stored
    # precision — matches reference which stores scales as fp16.
    scale_per_row = scale_per_row.astype(mx.float16).astype(mx.float32)

    # --- GPTQ quantize loop ---
    # MLX arrays support __setitem__ for slice assignment (like numpy).
    Q = mx.zeros((rows, cols), dtype=mx.float32)
    W_work = mx.array(W_perm)  # working copy

    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = mx.array(W_work[:, i1:i2])  # copy for local mutation
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = mx.zeros((rows, i2 - i1), dtype=mx.float32)

        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]

            # Symmetric quantize: round(w / scale), clamp to [-range, range]
            q_col = mx.clip(
                mx.round(w_col / scale_per_row),
                -clip_range,
                clip_range,
            )
            Q[:, i1 + j] = q_col

            # GPTQ error propagation
            err = (w_col - q_col * scale_per_row) / d
            Err[:, j] = err

            # Update remaining columns in this block
            if j + 1 < i2 - i1:
                W_block[:, j + 1:] -= err[:, None] * Hinv_block[j, j + 1:][None, :]

            mx.eval(Q, Err, W_block)

        # Propagate error to columns beyond this block
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
            mx.eval(W_work)

    # --- Un-permute ---
    Q_unperm = Q[:, invperm]
    scale_per_row_expanded = scale_per_row  # (rows,)

    # --- Convert to mlx QuantizedLinear format ---
    # mlx uses per-group asymmetric: w_deq = scales * w_q + biases
    # We have symmetric: w_deq = scale_per_row * q
    # Convert: for each group, scales = scale_per_row, biases = -scale_per_row * clip_range
    # so that: scales * (q + clip_range) + biases = scale_per_row * q
    #
    # Actually mlx expects: w_q in [0, n_bins], w_deq = scales * w_q + biases
    # With symmetric q in [-clip_range, clip_range]:
    #   w_q_unsigned = q + clip_range  (in [0, 2*clip_range] = [0, n_bins])
    #   w_deq = scales * w_q_unsigned + biases
    #         = scale_per_row * w_q_unsigned + biases
    #         = scale_per_row * (q + clip_range) + biases
    #         = scale_per_row * q  (if biases = -scale_per_row * clip_range)
    n_groups = (cols + group_size - 1) // group_size
    scales_out = mx.broadcast_to(
        scale_per_row[:, None], (rows, n_groups)
    ).astype(mx.bfloat16)
    biases_out = mx.broadcast_to(
        (-scale_per_row * clip_range)[:, None], (rows, n_groups)
    ).astype(mx.bfloat16)

    # Pack quantized values
    Q_unsigned = (Q_unperm + clip_range).astype(mx.uint32)
    el_per_int = 32 // bits
    # Ensure cols is multiple of el_per_int
    pad_cols = (el_per_int - cols % el_per_int) % el_per_int
    if pad_cols > 0:
        Q_unsigned = mx.pad(Q_unsigned, [(0, 0), (0, pad_cols)])
    Q_unsigned = Q_unsigned.reshape(rows, -1, el_per_int)
    shifts = mx.power(
        mx.array(2, dtype=mx.uint32),
        mx.arange(0, 32, bits, dtype=mx.uint32),
    )
    W_packed = mx.sum(Q_unsigned * shifts[None, None, :], axis=-1)

    mx.eval(W_packed, scales_out, biases_out)
    return W_packed, scales_out, biases_out, perm


# ---------------------------------------------------------------------------
# Full model quantize (drop-in for mlx_lm.quant.gptq.gptq_quantize)
# ---------------------------------------------------------------------------


def gptq_quantize_model(
    model: nn.Module,
    data: mx.array,
    bits: int = 4,
    group_size: int = 128,
    fallback_bits: int = 6,
    fallback_group_size: int = 128,
    batch_size: int = 8,
    clip_sigmas: float = 3.0,
    block_size: int = 128,
) -> tuple[nn.Module, dict]:
    """GPTQ quantize a model with numerical stability fixes.

    Drop-in replacement for ``mlx_lm.quant.gptq.gptq_quantize``.

    Returns:
        (model, config) where config is the per-key quantization dict
        that mlx_vlm.load needs.
    """
    # Step 1: Install catchers on all quantizable Linears
    layers = []
    for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(l, nn.Linear):
            layers.append((k, Catcher(l)))
    model.update_modules(tree_unflatten(layers))

    # Step 2: Collect Hessians
    log.info("Collecting Hessians (%d batches of %d)...", len(data) // batch_size, batch_size)
    for s in tqdm(
        range(0, len(data), batch_size),
        total=len(data) // batch_size,
        desc="Computing Hessians",
    ):
        batch = data[s : s + batch_size]
        model(batch)
        mx.eval(layers)

    # Step 3: Quantize each Linear
    quantized_layers = []
    for lid, (key, catcher) in tqdm(
        list(enumerate(layers)),
        desc="Quantizing (stable)",
    ):
        H = catcher.H
        if catcher.n_samples > 0:
            H = H / catcher.n_samples
        del catcher.H

        orig_module = catcher.module
        W = orig_module.weight.astype(mx.float32)
        orig_dtype = orig_module.weight.dtype

        # Check if this layer is too small for GPTQ (fallback to simple quant)
        if W.size <= 65536:
            log.debug("  %s: small (%d params), fallback quantize", key, W.size)
            fb_layer = orig_module.to_quantized(
                bits=fallback_bits, group_size=fallback_group_size
            )
            quantized_layers.append((key, fb_layer))
            continue

        log.debug("  %s: GPTQ stable, shape=%s", key, W.shape)
        W_packed, scales, biases, perm = gptq_quantize_weight(
            W, H,
            bits=bits,
            group_size=group_size,
            clip_sigmas=clip_sigmas,
            block_size=block_size,
        )

        # Check for NaN — the whole point of this fix
        has_nan = bool(mx.any(mx.isnan(scales)).item()) or bool(mx.any(mx.isinf(biases)).item())
        if has_nan:
            log.warning("  %s: NaN/Inf DETECTED after stable GPTQ! Falling back.", key)
            fb_layer = orig_module.to_quantized(
                bits=fallback_bits, group_size=fallback_group_size
            )
            quantized_layers.append((key, fb_layer))
            continue

        # Build a QuantizedLinear with our weights
        q_layer = orig_module.to_quantized(bits=bits, group_size=group_size)
        q_layer.weight = W_packed
        q_layer.scales = scales
        q_layer.biases = biases
        q_layer.set_dtype(orig_dtype)
        mx.eval(q_layer)
        quantized_layers.append((key, q_layer))

    model.update_modules(tree_unflatten(quantized_layers))

    # Step 4: Fallback-quantize remaining quantizable layers (embed_tokens etc.)
    remaining = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    config: dict = {"bits": bits, "group_size": group_size}
    fallback_config = {"bits": fallback_bits, "group_size": fallback_group_size}
    fb_layers = []
    for k, l in remaining:
        if hasattr(l, "to_quantized") and not isinstance(l, nn.QuantizedLinear):
            config[k] = fallback_config
            fb_layers.append((k, l.to_quantized(**fallback_config)))
    if fb_layers:
        model.update_modules(tree_unflatten(fb_layers))

    return model, config
