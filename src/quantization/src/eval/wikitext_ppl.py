"""WikiText-103 perplexity on a small fixed-segment subset.

The "catastrophic-language-damage" guard. INT4 quantization sometimes
trashes the language head (especially `lm_head` and final norm). A bf16
→ INT4 PPL increase of >2× is the tripwire — fix by promoting those
layers to fp16 (Unsloth-UD-style).

Default subset size = 200 segments * ~512 tokens = ~100K tokens. Small
enough to run in ~2 minutes on bf16 / 4090; ~1 minute on INT4 / MLX.

This benchmark uses the language-only forward pass — image inputs not
required. Works fine on top of ``Gemma4ForConditionalGeneration``
because the language sub-module accepts text-only inputs natively.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_loaders import ModelHandle

log = logging.getLogger(__name__)


@dataclass
class WikiTextPPLConfig:
    n_segments: int = 200
    segment_tokens: int = 512
    stride: int = 256  # overlap-stride for sliding-window PPL
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    split: str = "test"
    seed: int = 0
    # If the loader's backend is "mlx_vlm" we can't easily call .forward()
    # for log-probs. In that case the runner returns ppl=None and
    # marks the benchmark "not supported on backend".
    skip_on_mlx: bool = True


@dataclass
class WikiTextPPLResult:
    n_segments: int
    n_tokens: int
    perplexity: float | None
    nll_per_token: float | None
    backend_supported: bool
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def run(handle: ModelHandle, config: WikiTextPPLConfig) -> WikiTextPPLResult:
    """Compute perplexity over a deterministic WikiText-103 subset."""
    notes: list[str] = []
    if config.skip_on_mlx and handle.backend == "mlx_vlm":
        notes.append(
            "PPL skipped on MLX backend (forward-pass logprobs not "
            "exposed by mlx_vlm.generate). Run PPL on bf16 reference "
            "and rely on the size/match metrics for INT4 variants, "
            "or implement an mlx_vlm forward-logprob hook."
        )
        return WikiTextPPLResult(
            n_segments=0, n_tokens=0, perplexity=None, nll_per_token=None,
            backend_supported=False, notes=notes,
        )

    # Deferred imports
    import torch  # noqa: F401  (handle.model is a torch module on bf16 backend)
    from datasets import load_dataset

    log.info(
        "Loading %s / %s / %s for PPL eval",
        config.dataset_name, config.dataset_config, config.split,
    )
    ds = load_dataset(
        config.dataset_name, config.dataset_config, split=config.split
    )

    # Concatenate non-empty rows; tokenize once; chunk into fixed segments.
    text_iter = (row["text"] for row in ds if row.get("text", "").strip())
    big_text = "\n\n".join(text_iter)
    tokenizer = _get_tokenizer(handle)
    enc = tokenizer(big_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    log.info("Tokenized corpus: %d tokens total", input_ids.numel())

    # Gemma 4 (and most modern Gemma / Llama variants) was trained with
    # a BOS prefix on every sequence. Scoring a segment without BOS puts
    # the model strictly OOD and collapses NLL to near-uniform (~ln(vocab)).
    # Empirically on bf16 Gemma 4 E2B: PPL ≈ 40,000 without BOS, ≈ 18
    # with BOS on the same clean English. Prepend BOS to every segment
    # so the PPL we report reflects model quality, not a missing prefix.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        log.warning(
            "tokenizer has no bos_token_id — PPL may be inflated for "
            "models trained with mandatory BOS (Gemma, Llama)."
        )

    seg_len = config.segment_tokens
    stride = config.stride
    n_seg = config.n_segments
    # Deterministic non-overlapping segment selection from front
    # (stride lets us cover a wider sweep without re-running).
    starts = list(range(0, max(1, input_ids.numel() - seg_len), stride))
    if len(starts) > n_seg:
        # subsample deterministically
        step = len(starts) // n_seg
        starts = starts[::step][:n_seg]

    log.info("Eval segments: %d × %d tokens", len(starts), seg_len)

    model = handle.model
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in starts:
            chunk = input_ids[s : s + seg_len]
            if bos_id is not None:
                bos_t = torch.tensor([bos_id], dtype=chunk.dtype)
                ids = torch.cat([bos_t, chunk], dim=0).unsqueeze(0).to(device)
            else:
                ids = chunk.unsqueeze(0).to(device)
            # Cross-entropy w/ shifted labels via the model's own loss.
            # AutoModelForImageTextToText forwards text inputs through
            # the language sub-module when no image is provided.
            out = model(input_ids=ids, labels=ids)
            # ``out.loss`` is the mean CE over the (T-1) shift positions.
            # When we prepended BOS, the (T-1)-shifted positions cover
            # exactly the ``seg_len`` original tokens (BOS predicts t0,
            # t0 predicts t1, ..., t_{T-2} predicts t_{T-1}).
            n_tokens_in_seg = ids.shape[1] - 1
            total_nll += float(out.loss.item()) * n_tokens_in_seg
            total_tokens += n_tokens_in_seg
    elapsed = time.perf_counter() - t0
    nll_per_tok = total_nll / max(1, total_tokens)
    ppl = math.exp(nll_per_tok)
    log.info(
        "PPL eval done: %d tokens, NLL/tok=%.4f, PPL=%.3f, %.1fs",
        total_tokens, nll_per_tok, ppl, elapsed,
    )
    return WikiTextPPLResult(
        n_segments=len(starts),
        n_tokens=total_tokens,
        perplexity=ppl,
        nll_per_token=nll_per_tok,
        backend_supported=True,
        elapsed_s=elapsed,
        notes=notes,
    )


def _get_tokenizer(handle: ModelHandle):
    """Pull a tokenizer out of the handle. HF processors expose a
    ``.tokenizer`` attribute; some custom processors expose ``.tokenize``.
    """
    proc = handle.processor
    tok = getattr(proc, "tokenizer", None)
    if tok is None:
        # Some HF processors ARE the tokenizer.
        tok = proc
    return tok
