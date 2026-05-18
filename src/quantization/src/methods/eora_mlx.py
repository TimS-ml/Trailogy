"""EoRA: Training-free post-quantization quality recovery via
eigenspace low-rank approximation.

Reference: arXiv:2410.21271 (NVIDIA, NeurIPS 2024)
MLX port for B.2 (Mac/MLX-native) quantization pipeline.

Algorithm:
  1. Collect calibration X^T X per quantized linear (same pattern as
     GPTQ Hessian collection, but we only need the covariance).
  2. For each quantized linear:
     a. delta = W_bf16 - dequant(W_q)
     b. Eigendecompose X^T X to get the input eigenspace
     c. Scale delta by the eigenspace so SVD captures directions that
        matter for the actual input distribution
     d. Truncated SVD (via eigendecomposition on the smaller dim) to
        get (B, A) where B @ A ~= delta
  3. At inference: y = W_q(x) + x @ A^T @ B^T  (LoRA-shaped correction)

Output format is LoRA-compatible:
  lora_a = A^T  (in_dim, rank)
  lora_b = B^T  (rank, out_dim)
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration: collect X^T X per quantized linear
# ---------------------------------------------------------------------------


class XTXCatcher(nn.Module):
    """Forward-hook wrapper that accumulates X^T X for EoRA."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.xtx = None
        self.n_tokens = 0

    def __call__(self, x, *args, **kwargs):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        batch_xtx = xf.T @ xf
        if self.xtx is None:
            self.xtx = batch_xtx
        else:
            self.xtx = self.xtx + batch_xtx
        self.n_tokens += xf.shape[0]
        return self.module(x, *args, **kwargs)


def collect_xtx(
    model: nn.Module,
    data: mx.array,
    batch_size: int = 4,
) -> dict[str, tuple[mx.array, int]]:
    """Hook all QuantizedLinear, run calibration, return {key: (XTX, n_tokens)}.

    After collection the original modules are restored (catchers removed).
    """
    catchers: list[tuple[str, XTXCatcher]] = []
    for k, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(m, nn.QuantizedLinear):
            catchers.append((k, XTXCatcher(m)))
    model.update_modules(tree_unflatten(catchers))

    n_batches = (len(data) + batch_size - 1) // batch_size
    log.info(
        "EoRA calibration: %d samples, batch_size=%d, %d batches, "
        "%d QuantizedLinear hooks",
        len(data), batch_size, n_batches, len(catchers),
    )

    t0 = time.perf_counter()
    for s in tqdm(
        range(0, len(data), batch_size),
        total=n_batches,
        desc="EoRA calibration",
    ):
        batch = data[s : s + batch_size]
        model(batch)
        # Force eval to free activation graph while keeping XTX accumulators
        mx.eval([c for _, c in catchers])

    elapsed = time.perf_counter() - t0
    log.info("Calibration done in %.1f s.", elapsed)

    # Extract XTX and restore original modules
    result: dict[str, tuple[mx.array, int]] = {}
    restore: list[tuple[str, nn.Module]] = []
    for k, catcher in catchers:
        if catcher.xtx is not None and catcher.n_tokens > 0:
            # Normalize: mean X^T X (divide by token count)
            result[k] = (catcher.xtx / catcher.n_tokens, catcher.n_tokens)
        restore.append((k, catcher.module))
    model.update_modules(tree_unflatten(restore))

    log.info("Collected XTX for %d / %d layers.", len(result), len(catchers))
    return result


# ---------------------------------------------------------------------------
# Core EoRA decomposition
# ---------------------------------------------------------------------------


def eora_decompose(
    delta: mx.array,
    xtx: mx.array,
    max_rank: int = 128,
) -> tuple[mx.array, mx.array]:
    """Eigenspace-weighted low-rank decomposition of quantization error.

    Args:
        delta: (out_dim, in_dim) float32 -- W_bf16 - dequant(W_q)
        xtx:   (in_dim, in_dim) float32 -- normalized calibration X^T X
        max_rank: maximum adapter rank

    Returns:
        (A, B) where B @ A ~= delta (weighted by input distribution).
        A: (rank, in_dim) bfloat16     -- becomes lora_a.T
        B: (out_dim, rank) bfloat16    -- becomes lora_b.T
    """
    out_dim, in_dim = delta.shape
    rank = min(max_rank, in_dim, out_dim)
    EPS = 1e-10

    # --- Step 1: Eigendecompose calibration covariance ---
    with mx.stream(mx.cpu):
        eigenvalues, Q = mx.linalg.eigh(xtx.astype(mx.float32))
    mx.eval(eigenvalues, Q)

    # Clamp eigenvalues: at least 1% of max to prevent 1/sqrt(λ) blowup.
    # Rank-deficient XTX (n_tokens < in_dim) or fp32 noise can produce
    # zero/negative eigenvalues whose inverse would dominate A.
    max_eig = float(mx.max(eigenvalues).item())
    clamp = max(max_eig * 0.01, 1e-8)
    n_clamped = int(mx.sum(eigenvalues < clamp).item())
    if n_clamped > 0:
        log.debug("  Clamped %d/%d eigenvalues below %.2e", n_clamped, in_dim, clamp)
    eigenvalues = mx.maximum(eigenvalues, mx.array(clamp, dtype=mx.float32))

    sqrt_eig = mx.sqrt(eigenvalues)           # (in_dim,)
    inv_sqrt_eig = 1.0 / mx.maximum(sqrt_eig, mx.array(EPS))

    # --- Step 2: Scale delta by eigenspace ---
    # delta_scaled = delta @ Q @ diag(sqrt_eig)
    #             = (delta @ Q) * sqrt_eig        (broadcast)
    delta_f32 = delta.astype(mx.float32)
    dQ = delta_f32 @ Q                           # (out_dim, in_dim)
    delta_scaled = dQ * sqrt_eig[None, :]        # (out_dim, in_dim)
    del dQ
    mx.eval(delta_scaled)

    # --- Step 3: Truncated SVD via eigendecomposition on smaller dim ---
    # This avoids the O(max(m,n)^2 * min(m,n)) full SVD memory cost.
    # We always work with the min(out_dim, in_dim) side.

    if out_dim <= in_dim:
        # M = delta_scaled @ delta_scaled^T  (out_dim x out_dim)
        M = delta_scaled @ delta_scaled.T
        with mx.stream(mx.cpu):
            eig_sq, U_full = mx.linalg.eigh(M)
        mx.eval(eig_sq, U_full)
        del M

        # Top-r (eigh ascending → last r are largest)
        sigma_sq = eig_sq[-rank:]
        U_r = U_full[:, -rank:]                   # (out_dim, rank)
        del U_full, eig_sq
        sigma_r = mx.sqrt(mx.maximum(sigma_sq, mx.array(0.0)))
        sigma_safe = mx.maximum(sigma_r, mx.array(EPS))
        # V_r^T = diag(1/sigma) @ U_r^T @ delta_scaled
        Vt_r = (U_r.T @ delta_scaled) / sigma_safe[:, None]   # (rank, in_dim)
    else:
        # M = delta_scaled^T @ delta_scaled  (in_dim x in_dim)
        M = delta_scaled.T @ delta_scaled
        with mx.stream(mx.cpu):
            eig_sq, V_full = mx.linalg.eigh(M)
        mx.eval(eig_sq, V_full)
        del M

        sigma_sq = eig_sq[-rank:]
        V_r = V_full[:, -rank:]                   # (in_dim, rank)
        del V_full, eig_sq
        sigma_r = mx.sqrt(mx.maximum(sigma_sq, mx.array(0.0)))
        sigma_safe = mx.maximum(sigma_r, mx.array(EPS))
        # U_r = delta_scaled @ V_r @ diag(1/sigma)
        U_r = (delta_scaled @ V_r) / sigma_safe[None, :]     # (out_dim, rank)
        Vt_r = V_r.T                              # (rank, in_dim)
        del V_r

    del delta_scaled
    mx.eval(U_r, Vt_r, sigma_r)

    # --- Step 4: Construct B and A ---
    # B @ A ~= delta  (in original, un-scaled space)
    # B = U_r @ diag(sqrt(sigma_r))
    # A = diag(sqrt(sigma_r)) @ Vt_r @ S^{-1}
    # where S^{-1} = diag(1/sqrt_eig) @ Q^T

    sqrt_sigma = mx.sqrt(mx.maximum(sigma_r, mx.array(0.0)))
    B = U_r * sqrt_sigma[None, :]                 # (out_dim, rank)

    # A = diag(sqrt_sigma) @ Vt_r @ diag(inv_sqrt_eig) @ Q^T
    # Step by step to avoid large intermediates:
    Vt_r_inv = Vt_r * inv_sqrt_eig[None, :]       # (rank, in_dim) -- undo eigenscaling
    A = (sqrt_sigma[:, None] * Vt_r_inv) @ Q.T    # (rank, in_dim)

    del Q, Vt_r, Vt_r_inv, U_r
    mx.eval(A, B)

    return A.astype(mx.bfloat16), B.astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Inference wrapper: QuantizedLinear + EoRA correction
# ---------------------------------------------------------------------------


class EoRALinear(nn.Module):
    """QuantizedLinear with additive low-rank EoRA correction.

    Forward:  y = qlinear(x) + x @ lora_a @ lora_b

    Where lora_a = A^T (in_dim, rank), lora_b = B^T (rank, out_dim).
    """

    def __init__(self, qlinear: nn.QuantizedLinear, lora_a: mx.array, lora_b: mx.array):
        super().__init__()
        self.qlinear = qlinear
        self.lora_a = lora_a    # (in_dim, rank)
        self.lora_b = lora_b    # (rank, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.qlinear(x)
        z = (x.astype(self.lora_a.dtype) @ self.lora_a) @ self.lora_b
        return y + z

    # Pass through attributes that downstream code may check
    @property
    def weight(self):
        return self.qlinear.weight

    @property
    def scales(self):
        return self.qlinear.scales

    @property
    def biases(self):
        return self.qlinear.biases

    @property
    def bits(self):
        return self.qlinear.bits

    @property
    def group_size(self):
        return self.qlinear.group_size

    @property
    def shape(self):
        return self.qlinear.shape


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def compute_adapters(
    quant_model: nn.Module,
    bf16_weights: dict[str, mx.array],
    xtx_dict: dict[str, tuple[mx.array, int]],
    max_rank: int = 128,
    bf16_key_prefix: str = "model.language_model.",
    quant_key_prefix: str = "model.",
) -> dict[str, tuple[mx.array, mx.array]]:
    """Compute EoRA (A, B) adapters for all quantized linears.

    Args:
        quant_model: language_model from the quantized mlx_vlm model
        bf16_weights: dict from mx.load on the bf16 safetensors
        xtx_dict: output of collect_xtx
        max_rank: maximum adapter rank (compute once, truncate later)
        bf16_key_prefix: prefix in bf16 safetensors keys
        quant_key_prefix: prefix in quantized model tree keys

    Returns:
        {key: (A, B)} where A (rank, in_dim), B (out_dim, rank), both bf16.
    """
    leaves = tree_flatten(
        quant_model.leaf_modules(), is_leaf=nn.Module.is_module
    )
    qlinears = [(k, m) for k, m in leaves if isinstance(m, nn.QuantizedLinear)]

    adapters: dict[str, tuple[mx.array, mx.array]] = {}
    t0 = time.perf_counter()
    skipped = 0

    for k, ql in tqdm(qlinears, desc="Computing EoRA adapters"):
        # Skip layers without calibration data
        if k not in xtx_dict:
            skipped += 1
            continue

        xtx, n_tok = xtx_dict[k]

        # Map key to bf16 weight key
        # quant key: "model.layers.0.self_attn.q_proj"
        # bf16 key:  "model.language_model.layers.0.self_attn.q_proj.weight"
        suffix = k[len(quant_key_prefix):]  # "layers.0.self_attn.q_proj"
        bf16_key = f"{bf16_key_prefix}{suffix}.weight"

        if bf16_key not in bf16_weights:
            log.warning("  %s: bf16 key '%s' not found, skipping", k, bf16_key)
            skipped += 1
            continue

        # Get dequantized weight
        W_q = mx.dequantize(
            ql.weight, ql.scales, ql.biases,
            group_size=ql.group_size, bits=ql.bits,
        ).astype(mx.float32)

        # Get bf16 weight
        W_bf16 = bf16_weights[bf16_key].astype(mx.float32)

        if W_q.shape != W_bf16.shape:
            log.warning(
                "  %s: shape mismatch q=%s bf16=%s, skipping",
                k, W_q.shape, W_bf16.shape,
            )
            skipped += 1
            continue

        # Delta
        delta = W_bf16 - W_q
        del W_q, W_bf16

        # Decompose
        A, B = eora_decompose(delta, xtx, max_rank=max_rank)
        del delta, xtx  # free calibration data for this layer

        adapters[k] = (A, B)

        # Periodically clear MLX cache
        if len(adapters) % 50 == 0:
            mx.metal.clear_cache()

    elapsed = time.perf_counter() - t0
    log.info(
        "EoRA adapters computed: %d layers, %d skipped, %.1f s total.",
        len(adapters), skipped, elapsed,
    )
    return adapters


def apply_adapters(
    model: nn.Module,
    adapters: dict[str, tuple[mx.array, mx.array]],
    rank: int,
) -> int:
    """Patch quantized model with EoRA corrections at the given rank.

    Truncates pre-computed adapters (which may be at max_rank > rank) to
    the requested rank, then wraps each QuantizedLinear with EoRALinear.

    Returns the number of layers patched.
    """
    patches: list[tuple[str, EoRALinear]] = []
    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    ql_dict = {k: m for k, m in leaves if isinstance(m, nn.QuantizedLinear)}

    for key, (A, B) in adapters.items():
        if key not in ql_dict:
            log.warning("apply_adapters: key '%s' not in model tree, skipping", key)
            continue

        actual_rank = min(rank, A.shape[0])
        A_trunc = A[-actual_rank:]           # top-r rows (largest σ, stored last)
        B_trunc = B[:, -actual_rank:]        # corresponding columns

        lora_a = A_trunc.T                   # (in_dim, rank)
        lora_b = B_trunc.T                   # (rank, out_dim)

        patches.append((key, EoRALinear(ql_dict[key], lora_a, lora_b)))

    if patches:
        model.update_modules(tree_unflatten(patches))
    log.info("Applied EoRA adapters: %d layers, rank=%d.", len(patches), rank)
    return len(patches)


def apply_adapters_from_file(
    model: nn.Module,
    saved_path: str,
    rank: int | None = None,
) -> int:
    """Load adapters from a ``save_adapters`` safetensors file and patch
    quantized linears in-place.

    Mirror of ``apply_adapters`` for the disk path. The saved tensors are
    already in EoRALinear-final shape — ``lora_a`` is ``(in_dim, R_saved)``
    and ``lora_b`` is ``(R_saved, out_dim)`` — so they are plugged into
    ``EoRALinear`` directly without re-transposing.

    Args:
        model: the language_model (or any subtree containing the
            ``QuantizedLinear``\\s referenced by the saved keys).
        saved_path: path to the safetensors written by ``save_adapters``.
        rank: if not None and < saved rank, truncate to top-``rank``
            (largest-sigma columns / rows are stored last per
            ``save_adapters`` convention). If None, use the saved rank
            as-is.

    Returns:
        Number of layers patched.

    Notes:
        - ``apply_adapters`` consumes the raw ``(A, B)`` tuples returned
          by ``compute_adapters`` (shapes ``(rank, in_dim)`` and
          ``(out_dim, rank)``). Routing a *saved* dict back through it
          would double-transpose and crash inference. Use this loader
          whenever the adapters live on disk.
    """
    raw = mx.load(saved_path)

    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    ql_dict = {k: m for k, m in leaves if isinstance(m, nn.QuantizedLinear)}

    base_keys = sorted({
        k[: -len(".lora_a")] for k in raw if k.endswith(".lora_a")
    })

    patches: list[tuple[str, EoRALinear]] = []
    missing_in_model = 0
    missing_pair = 0
    for bk in base_keys:
        b_key = f"{bk}.lora_b"
        if b_key not in raw:
            log.warning(
                "apply_adapters_from_file: '%s' has no matching lora_b, skipping",
                bk,
            )
            missing_pair += 1
            continue
        if bk not in ql_dict:
            log.warning(
                "apply_adapters_from_file: '%s' not in model tree, skipping", bk,
            )
            missing_in_model += 1
            continue

        lora_a = raw[f"{bk}.lora_a"]    # (in_dim, R_saved)
        lora_b = raw[b_key]             # (R_saved, out_dim)
        r_saved = lora_a.shape[1]
        if rank is not None and rank < r_saved:
            # save_adapters writes top-r columns of lora_a / rows of lora_b
            # (largest sigma stored last). Truncate symmetrically.
            lora_a = lora_a[:, -rank:]
            lora_b = lora_b[-rank:, :]
        elif rank is not None and rank > r_saved:
            log.warning(
                "apply_adapters_from_file: requested rank=%d > saved rank=%d for '%s'; using %d",
                rank, r_saved, bk, r_saved,
            )

        patches.append((bk, EoRALinear(ql_dict[bk], lora_a, lora_b)))

    if patches:
        model.update_modules(tree_unflatten(patches))
    log.info(
        "Applied EoRA adapters from %s: %d patched, %d unmatched-key, "
        "%d unpaired (rank=%s).",
        saved_path, len(patches), missing_in_model, missing_pair,
        rank if rank is not None else "saved",
    )
    return len(patches)


def save_adapters(
    adapters: dict[str, tuple[mx.array, mx.array]],
    output_path: str,
    rank: int | None = None,
) -> None:
    """Save adapters as safetensors in LoRA-compatible format.

    Keys: <module_path>.lora_a  (in_dim, rank) bf16
          <module_path>.lora_b  (rank, out_dim) bf16
    """
    tensors: dict[str, mx.array] = {}
    for key, (A, B) in adapters.items():
        r = A.shape[0] if rank is None else min(rank, A.shape[0])
        A_r = A[-r:]       # (rank, in_dim)
        B_r = B[:, -r:]    # (out_dim, rank)
        tensors[f"{key}.lora_a"] = A_r.T.astype(mx.bfloat16)   # (in_dim, rank)
        tensors[f"{key}.lora_b"] = B_r.T.astype(mx.bfloat16)   # (rank, out_dim)

    mx.save_safetensors(output_path, tensors)
    total_bytes = sum(t.nbytes for t in tensors.values())
    log.info(
        "Saved %d adapter pairs (rank=%s) to %s (%.1f MB)",
        len(adapters), rank or "max", output_path, total_bytes / 1e6,
    )
