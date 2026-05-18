#!/usr/bin/env python3
"""GPU smoke test for the silent-PEFT-orphan-tensor bug class.

The bug
-------
The adapter was trained on an older ``transformers`` that exposed
``k_proj`` / ``v_proj`` as ``nn.Linear`` in all language-tower layers.
On reload under a different version, PEFT silently dropped the
"orphan" tensors (no warning, no error) and the resulting adapter was
effectively empty during evaluation. This was the catastrophic blocker
called out in the project write-up ("Silent PEFT loading failure") and
the fix was an upstream ``transformers`` upgrade.

This script exercises the full save -> fresh-process reload roundtrip
against the actual Gemma 4 E2B model and the same FastModel + LoRA +
projector wrapping the finetune pipeline uses, so any reintroduction of
the bug (HF / PEFT / unsloth version drift, modeling restructure)
surfaces here before it eats a multi-hour real finetune.

What it checks (all four mismatch criteria from the design):

  1. Trainable LoRA tensor key set: every key saved to disk reloads to
     a parameter on the fresh model — no orphans, no extras.
  2. Byte equality of every LoRA tensor.
  3. Byte equality of every modules_to_save tensor (projector + tuned
     vision layers when present).
  4. Forward-pass logits equality on a fixed text-only input, with a
     tight bf16 tolerance.

Run
---
    export LD_LIBRARY_PATH=/path/to/torch-env/lib:$LD_LIBRARY_PATH
    python -m scripts.inspect.save_reload --config configs/<smoke>.yaml

Exits 0 if every check passes, non-zero with an actionable error on any
mismatch. Total runtime on a 24 GB consumer GPU: ~30-60s (two cold
Gemma loads).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

# Make `src.<module>` importable when running as `python scripts/inspect/save_reload.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import FinetuneConfig, load_config, validate_config  # noqa: E402
from src.save_reload_check import (  # noqa: E402
    StateDiff,
    assert_no_diff,
    diff_state,
    extract_savable_state,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("smoke_save_reload")


# ---------------------------------------------------------------------------
# Determinism + dtype helpers
# ---------------------------------------------------------------------------


# A fixed, short text-only probe that's safe regardless of vision tower
# handling. The bug class is in language-tower LoRA so a text-only
# forward is sufficient and avoids dragging in image preprocessing.
PROBE_PROMPT = "Identify this plant in three words."


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _randomize_trainable_params(model: Any, seed: int) -> int:
    """Make every trainable param non-zero so byte equality is meaningful.

    LoRA tensors are init-zero on lora_B by design; without this the
    save -> reload roundtrip would pass trivially on the all-zeros
    payload (and miss any save-side bug that only manifests on
    non-trivial values).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Generate on CPU then copy_ to keep this deterministic and
            # cheap on bf16/fp16 trainable params.
            target = torch.empty(p.shape, dtype=torch.float32).normal_(generator=gen) * 0.01
            p.copy_(target.to(p.dtype).to(p.device))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Model construction (mirrors finetune.py's real_train head)
# ---------------------------------------------------------------------------


def _build_peft_model(cfg: FinetuneConfig) -> Tuple[Any, Any]:
    """Load FastModel + apply LoRA + (optionally) projector wrap.

    Returns (peft_model, processor). This is the same construction the
    real finetune pipeline performs, minus the dataset / trainer setup.
    """
    from unsloth import FastModel

    log.info("Loading base model: %s", cfg.model.base_model)
    model, processor = FastModel.from_pretrained(
        model_name=cfg.model.base_model,
        dtype=_resolve_dtype(cfg.model.dtype),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        full_finetuning=cfg.model.full_finetuning,
    )

    modules_to_save = None
    if cfg.lora.tune_projector:
        try:
            from src.projector import find_projector_module_names
        except ImportError:
            from projector import find_projector_module_names  # type: ignore
        proj_modules = find_projector_module_names(model)
        if not proj_modules:
            raise RuntimeError(
                "tune_projector=True but no projector modules found. "
                "Check src.projector.PROJECTOR_CANDIDATE_TOKENS against the "
                "current base model."
            )
        modules_to_save = list(proj_modules)
        log.info("Projector modules under modules_to_save: %s", modules_to_save)

    peft_model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_audio_layers=cfg.lora.finetune_audio_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        random_state=cfg.lora.random_state,
        modules_to_save=modules_to_save,
    )
    return peft_model, processor


def _reload_peft_model(cfg: FinetuneConfig, adapter_dir: Path) -> Tuple[Any, Any]:
    """Fresh FastModel + PeftModel.from_pretrained on adapter_dir."""
    from peft import PeftModel
    from unsloth import FastModel

    log.info("Re-loading base model (fresh): %s", cfg.model.base_model)
    base, processor = FastModel.from_pretrained(
        model_name=cfg.model.base_model,
        dtype=_resolve_dtype(cfg.model.dtype),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        full_finetuning=cfg.model.full_finetuning,
    )
    log.info("Loading adapter from %s", adapter_dir)
    reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
    return reloaded, processor


# ---------------------------------------------------------------------------
# Forward-pass probe
# ---------------------------------------------------------------------------


def _capture_logits(model: Any, processor: Any) -> torch.Tensor:
    """Run a deterministic text-only forward pass and return last-token logits.

    Returns shape [vocab]; deviates immediately if any LoRA tensor in
    the language tower changed between in-memory and reloaded model.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    inputs = tokenizer(PROBE_PROMPT, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    # last-position logits; cast to fp32 on CPU for cross-process compare.
    logits = out.logits[0, -1, :].detach().to(torch.float32).cpu()
    return logits


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------


def phase_1_snapshot_and_save(
    cfg: FinetuneConfig,
    adapter_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Build the PEFT model, perturb trainable params, snapshot, save, return baseline."""
    peft_model, processor = _build_peft_model(cfg)
    n_perturbed = _randomize_trainable_params(peft_model, seed=cfg.training.seed)
    log.info("Randomized %d trainable params (non-zero LoRA + modules_to_save)", n_perturbed)

    in_memory_state = extract_savable_state(peft_model)
    log.info("Snapshotted %d savable tensors from in-memory PEFT state.", len(in_memory_state))

    logits = _capture_logits(peft_model, processor)
    log.info(
        "Captured in-memory logits: shape=%s dtype=%s norm=%.4f",
        tuple(logits.shape), logits.dtype, float(logits.norm()),
    )

    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    log.info("Saved adapter to %s", adapter_dir)

    # Best-effort free of phase-1 VRAM before phase 2 reloads the base.
    del peft_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return in_memory_state, logits


def phase_2_reload_and_snapshot(
    cfg: FinetuneConfig,
    adapter_dir: Path,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Reload the adapter onto a fresh base and return the same snapshot tuple."""
    reloaded, processor = _reload_peft_model(cfg, adapter_dir)
    state = extract_savable_state(reloaded)
    log.info("Snapshotted %d savable tensors from RELOADED PEFT state.", len(state))
    logits = _capture_logits(reloaded, processor)
    log.info(
        "Captured reloaded logits: shape=%s dtype=%s norm=%.4f",
        tuple(logits.shape), logits.dtype, float(logits.norm()),
    )
    return state, logits


# ---------------------------------------------------------------------------
# Diff bucketing — split the state diff by tensor class so the operator
# can immediately see which subsystem regressed.
# ---------------------------------------------------------------------------


def _is_modules_to_save_key(key: str) -> bool:
    return ".modules_to_save." in key or ".original_module." in key


def _split_diff_by_class(diff: StateDiff) -> Dict[str, StateDiff]:
    """Bucket diff entries into 'lora' and 'modules_to_save' for reporting."""

    def _split(keys: set[str]) -> Tuple[set[str], set[str]]:
        lora = {k for k in keys if not _is_modules_to_save_key(k)}
        mts = {k for k in keys if _is_modules_to_save_key(k)}
        return lora, mts

    lora_a, mts_a = _split(diff.only_in_a)
    lora_b, mts_b = _split(diff.only_in_b)
    lora_mm = [(k, r) for k, r in diff.value_mismatched if not _is_modules_to_save_key(k)]
    mts_mm = [(k, r) for k, r in diff.value_mismatched if _is_modules_to_save_key(k)]
    return {
        "lora": StateDiff(only_in_a=lora_a, only_in_b=lora_b, value_mismatched=lora_mm),
        "modules_to_save": StateDiff(only_in_a=mts_a, only_in_b=mts_b, value_mismatched=mts_mm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Tight bf16 tolerance: same weights + same hardware should give bitwise
# equal logits on a sequential forward pass; the small floor accommodates
# nondeterministic CUDA reductions in some kernels.
LOGITS_ATOL = 1e-4
LOGITS_RTOL = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/plantnet-50k-baseline-v2.yaml",
        help="Path to the smoke config (default: configs/plantnet-50k-baseline-v2.yaml).",
    )
    parser.add_argument(
        "--keep-adapter",
        action="store_true",
        help="Keep the temporary adapter directory after the run (default: delete).",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 2
    cfg = load_config(cfg_path)
    errs = validate_config(cfg)
    if errs:
        for e in errs:
            log.error("Config error: %s", e)
        return 2

    tmp_root = Path(tempfile.mkdtemp(prefix="smoke_save_reload_"))
    adapter_dir = tmp_root / "adapter"
    log.info("Temp adapter directory: %s", adapter_dir)

    failed = False
    try:
        in_memory_state, logits_pre = phase_1_snapshot_and_save(cfg, adapter_dir)
        reloaded_state, logits_post = phase_2_reload_and_snapshot(cfg, adapter_dir)

        # 1+2+3: tensor-state diff (key sets + byte equality), bucketed.
        diff = diff_state(in_memory_state, reloaded_state)
        buckets = _split_diff_by_class(diff)

        log.info(
            "Save->reload tensor diff summary: %d only_in_memory, %d only_in_reloaded, "
            "%d value_mismatched (total tensors compared: %d).",
            len(diff.only_in_a), len(diff.only_in_b), len(diff.value_mismatched),
            len(set(in_memory_state) | set(reloaded_state)),
        )

        for label, b in buckets.items():
            if b.is_empty():
                log.info("  %-18s OK", label)
            else:
                log.error(
                    "  %-18s FAIL — only_in_memory=%d only_in_reloaded=%d value_mismatched=%d",
                    label, len(b.only_in_a), len(b.only_in_b), len(b.value_mismatched),
                )

        if not diff.is_empty():
            try:
                assert_no_diff(diff, label="save->reload roundtrip (full state)")
            except RuntimeError as exc:
                log.error("%s", exc)
                failed = True

        # 4: forward-pass logits equality on the fixed probe.
        if logits_pre.shape != logits_post.shape:
            log.error(
                "Logits shape mismatch: %s vs %s — model architecture differs "
                "across save->reload, definite regression.",
                tuple(logits_pre.shape), tuple(logits_post.shape),
            )
            failed = True
        else:
            allclose = torch.allclose(
                logits_pre, logits_post, atol=LOGITS_ATOL, rtol=LOGITS_RTOL
            )
            max_abs = (logits_pre - logits_post).abs().max().item()
            if allclose:
                log.info(
                    "Logits OK — max |Δ| = %.2e (atol=%.0e rtol=%.0e).",
                    max_abs, LOGITS_ATOL, LOGITS_RTOL,
                )
            else:
                log.error(
                    "Logits MISMATCH — max |Δ| = %.2e exceeds atol=%.0e rtol=%.0e. "
                    "In-memory vs reloaded model produced different outputs on "
                    "the fixed probe: %r. This is the AGENTS.md bug surface.",
                    max_abs, LOGITS_ATOL, LOGITS_RTOL, PROBE_PROMPT,
                )
                failed = True

        if failed:
            log.error(
                "Smoke FAILED. AGENTS.md orphan-tensor bug class may have "
                "regressed under the current transformers + peft + unsloth "
                "versions. Adapter kept at %s for inspection.", adapter_dir,
            )
            args.keep_adapter = True
            return 1
        log.info("Smoke PASSED on all 4 criteria.")
        return 0
    finally:
        if args.keep_adapter:
            log.info("Adapter directory retained: %s", adapter_dir)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
