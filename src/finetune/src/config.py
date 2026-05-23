"""Typed configuration for the unsloth finetune pipeline.

Loaded from a YAML file plus optional CLI overrides. Pure-Python; no torch
import, so it can be exercised by pytest on a CPU-only Mac.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Base model. The iOS app ships INT4 (mlx-community/gemma-4-e2b-it-4bit)
    # but training pulls the unquantized unsloth mirror (no HF gate, easier
    # CI). Project policy is no QLoRA — we train in full bf16; the INT4
    # quantization happens later via mlx_vlm in scripts/run/export.sh.
    base_model: str = "unsloth/gemma-4-E2B-it"
    max_seq_length: int = 1024
    # Project policy: NO QLoRA by default. Set load_in_4bit=true ONLY
    # for the explicit QLoRA-baseline comparison runs (see
    # configs/plantnet-50k-baseline2-qlora-...yaml). All other configs
    # leave this at False and train in full bf16 precision.
    #
    # When True, the recommended ``base_model`` is
    # ``unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`` (or any base whose
    # ``quantization_config.llm_int8_skip_modules`` keeps
    # ``embed_vision`` / ``vision_tower`` / ``audio_tower`` /
    # ``embed_audio`` in bf16). The 4-bit-in-trainable tripwire in
    # finetune.py guarantees a loud failure if the projector or any
    # tuned vision layer ends up as ``bnb.Params4bit`` — i.e. ANY base
    # that does not skip-quantize those modules will fail-fast before
    # training, so the user finds out at the smoke-test stage rather
    # than burning hours on a silently-broken optimizer group.
    load_in_4bit: bool = False
    full_finetuning: bool = False  # unsloth's full-finetune flag
    # bf16 is the right default for Gemma 4 on Ampere+ (A100/H100): native
    # hardware support, no fp16 softmax-overflow issues in attention.
    # Unsloth would auto-detect bf16 on Ampere+ anyway, but locking it in
    # explicitly avoids silently falling back to fp16 on older GPUs
    # (V100/T4) where Gemma 4 attention is numerically unstable in fp16.
    # Set to "float16" / "float32" / None (auto-detect) to override.
    dtype: Optional[str] = "bfloat16"


@dataclass
class LoraConfig:
    # Mirrors `FastModel.get_peft_model` arguments.
    finetune_vision_layers: bool = False    # MUST be False — frozen
    finetune_audio_layers: bool = False     # MUST be False — frozen (manual)
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    # Vision-language projector tuning (feature/lora-plus-projector).
    # When True, embed_vision.* (Gemma4MultimodalEmbedder) is unfrozen as
    # FULL parameters via PEFT's modules_to_save, co-trained with the
    # language LoRA. The vision encoder (vision_tower.encoder, etc.)
    # stays frozen. See docs/superpowers/specs/2026-05-10-lora-plus-projector-design.md.
    tune_projector: bool = False
    # When tune_projector is True and this is None, the trainer auto-sets
    # to training.learning_rate / 10 (LLaVA-style: lower LR for the
    # full-param projector than for the low-rank LoRA).
    projector_learning_rate: Optional[float] = None
    # Last-N vision encoder layer tuning
    # (feature/lora-plus-projector-plus-vision-tower).
    # 0 = off (default, bit-identical to feature/lora-plus-projector).
    # N > 0 unfreezes vision_tower.encoder.layers[(total-N) .. total-1]
    # as FULL parameters via PEFT's modules_to_save, co-trained with the
    # language LoRA AND the projector. Requires tune_projector=True
    # (moving visual features without letting the projector track them
    # creates a feature-space misalignment that the language LoRA cannot
    # fix downstream). See docs/superpowers/specs/
    # 2026-05-10-lora-plus-projector-plus-vision-tower-design.md.
    tune_last_n_vision_layers: int = 0
    # When tune_last_n_vision_layers > 0 and this is None, the trainer
    # auto-sets to training.learning_rate / 20. Lower than the projector
    # LR because pretrained SigLIP weights are more fragile than the
    # freshly-needed projector adaptation.
    vision_layers_learning_rate: Optional[float] = None
    # DoRA (Weight-Decomposed Low-Rank Adaptation). When True, PEFT applies
    # DoRA instead of standard LoRA: each adapted weight is decomposed into
    # a magnitude vector and a direction matrix, with LoRA applied only to
    # the direction component. This generally improves training stability
    # and final accuracy at the cost of ~10-15% slower step time (extra
    # magnitude normalization per forward pass). Default False for
    # backward compatibility with existing sweep configs.
    use_dora: bool = False


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/hike-gemma4-lora"
    run_name: Optional[str] = None  # auto-generated: config_stem + timestamp
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = None  # set to int to override epochs
    learning_rate: float = 2.0e-4
    logging_steps: int = 1
    # Default: fused PyTorch AdamW. On Ampere/Ada (4090) this is ~5-15%
    # faster step time than "adamw_torch" with identical memory and
    # numerics. Project policy forbids 8-bit / 4-bit optimizer states
    # (validate_config rejects "adamw_8bit" and bnb paged_*_8bit names).
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    # warmup_ratio: fraction of total optimizer steps used for warmup.
    # When > 0, HF's TrainingArguments treats this as taking PRECEDENCE
    # over warmup_steps. Prefer this over warmup_steps for configs whose
    # total step count varies (different epochs / batch sizes) because
    # the ratio scales automatically. None = leave HF default (0), in
    # which case warmup_steps wins.
    warmup_ratio: Optional[float] = None
    seed: int = 3407
    save_steps: int = 200
    # Cap how many checkpoint dirs the trainer keeps under output_dir.
    # Each LoRA-r256 checkpoint is hundreds of MB; a long run with
    # save_steps=200 will fill disk without this. None = unlimited (HF
    # default).
    save_total_limit: Optional[int] = None
    report_to: str = "none"  # "none", "wandb", "tensorboard"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    # group_by_length: bucket records by approximate sequence length so
    # each batch has similar lengths => less pad waste. In this project
    # every image contributes a constant ~280 pooled vision tokens, so
    # length variance is dominated by the text part of `messages`.
    # When True, finetune.py auto-populates a `length` column from
    # message text char counts and passes
    # `train_sampling_strategy="group_by_length"` to SFTConfig. Without
    # the auto-population step, the HF LengthGroupedSampler crashes on
    # our raw vision dataset (no `length` / `input_ids` column).
    group_by_length: bool = False
    # tf32: enable TensorFloat-32 matmul on Ampere+ GPUs (A100 / 4090).
    # Only affects FP32 ops (optimizer math, any residual FP32 path) —
    # bf16 matmuls already use the bf16 tensor cores regardless. Free
    # ~1-3% wall-time perf on this project. None = HF/PyTorch default
    # (False since PyTorch 1.12). Recommend True on 4090.
    tf32: Optional[bool] = None

    # ----- v2: ModalityAware sampler + mid-training multi-eval -----
    #
    # modality_aware_sampler: when True, the trainer is wrapped in
    # ModalityAwareSFTTrainer which uses ModalityAwareBatchSampler so
    # every batch is homogeneous in image-presence. This unlocks the
    # text-only batches' vision-skip optimization (saves ~30-40% compute
    # on smoltalk records). Required for v2 mix-100k / mix-50k configs.
    # MUST be paired with group_by_length=False (the sampler does its
    # own within-modality length-sort; double-sorting is a config error).
    modality_aware_sampler: bool = False

    # eval_strategy: HF SFTConfig field. "no" disables mid-training
    # eval (v1 default — only final eval via src/evaluate.py). "steps"
    # enables eval every eval_steps steps; "epoch" enables eval at
    # end-of-epoch. Use "steps" + eval_steps=500 for v2 to track
    # per-bucket val loss during the long 100K run.
    eval_strategy: str = "no"
    eval_steps: Optional[int] = None
    # Mirrors per_device_train_batch_size but applied during eval. When
    # None, the trainer falls back to per_device_train_batch_size. With
    # a r=256 LoRA + 2 vision layers + projector tuned, eval batch
    # size = 4 typically fits on a 4090 alongside the train step's
    # gradient state.
    per_device_eval_batch_size: Optional[int] = None


@dataclass
class DataConfig:
    train_file: str = "data/train.jsonl"
    val_file: Optional[str] = "data/val.jsonl"
    # v2: multi-eval dataset. When set, the trainer ignores val_file and
    # passes a dict to SFTTrainer.eval_dataset so each key gets its own
    # eval_<key>_loss in the log (e.g. plant / nonplant / negative for
    # the mid-training catastrophic-forget watch).
    val_files: Optional[Dict[str, str]] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    # Online data augmentation.  When True, random image transforms
    # (horizontal flip, rotation, colour jitter, perspective) are applied
    # at collation time so each epoch sees different augmented versions of
    # the same training images.  No extra disk space needed.  Default OFF
    # so existing configs are bit-identical.
    augmentation: bool = False
    # v4: conditional-FT input gate keyed on **image presence**. Dict
    # with up to two keys: ``camera_on`` (record carries an image) and
    # ``camera_off`` (record has no image / text-only). The matching
    # value is a literal string prepended to the first user turn's
    # text. Either key may be omitted (= no prefix for that branch);
    # the dict as a whole may be ``None`` (= no prefix anywhere,
    # bit-identical to v2 behaviour).
    #
    # Typical training config::
    #
    #     prompt_prefixes:
    #       camera_on:  "[camera=on] "
    #       camera_off: "[camera=off] "
    #
    # The on-device app (``GemmaService.streamResponse``) prepends the
    # matching marker at inference time, so the model sees the same
    # two-state contract it was trained with regardless of question
    # topic. See ``build_vision_messages`` for the dispatch.
    prompt_prefixes: Optional[Dict[str, str]] = None


@dataclass
class EvalConfig:
    """Post-training evaluation behaviour.

    When `enabled` is True (default), `scripts/run/train.sh` automatically
    invokes `python -m src.evaluate --config <yaml>` after the training
    process exits successfully, teeing output to
    `<output_dir>/eval.log`. The eval JSON summary still goes to
    `results/<run_name>_eval.json` (unchanged path).
    """

    enabled: bool = True
    # null = evaluate the full val set; positive int = subsample for speed.
    max_eval_samples: Optional[int] = 300
    max_new_tokens: int = 256
    # Required for vision-tower models. False = AutoModelForImageTextToText,
    # which on Gemma 4 silently drops vision_tower.* — never want this for
    # multimodal eval in this project.
    use_unsloth: bool = True
    batch_size: int = 1
    # Inference-time 4-bit quantization of the merged model. Default
    # OFF — matches the training-side no-QLoRA policy for numerics
    # consistency: eval runs in the same bf16 dtype as training, so
    # eval scores reflect the actual deployed-quality of the merged
    # weights rather than a quantization-distorted approximation.
    # This is NOT QLoRA (no gradients here, just post-training
    # quantization for memory), so the validator does NOT hard-reject
    # True — users with memory-constrained eval boxes can opt in via
    # CLI (`--quantize`) or YAML (`eval.load_in_4bit: true`).
    load_in_4bit: bool = False


@dataclass
class RegularizationConfig:
    """Two regularizers, both off by default (bit-identical to v2).

    * **KL penalty** against the base model's outputs on the current
      training batch — implementation in ``src.regularization.KLPenalty``.
      Mirrors what RLHF / DPO use to keep the policy close to a
      reference. Teacher = the same trained model under PEFT's
      ``disable_adapter()`` context (zero extra GPU memory).
      Cost: ~+1 forward pass per training step.

    * **L2 weight anchor** toward the parameters' value at trainer init
      — i.e. pretrained for ``modules_to_save`` full-rank params
      (projector, last-N vision layers), approximately-zero for LoRA
      delta params. This is the "L2 toward original weights" /
      elastic-weight-consolidation idea from the continual-learning
      literature (full Fisher-weighted EWC is out of scope for v3;
      this is the cheaper diagonal-ones approximation).

    Both regularizers are wired by ``src.regularization.build_regularizers``
    and consumed inside the trainer's ``compute_loss`` override. When
    everything is disabled the trainer takes a fast-path that has no
    extra cost.
    """

    # KL penalty
    kl_enabled: bool = False
    kl_weight: float = 0.05
    # Softmax temperature applied identically to student + teacher logits
    # before the KL is taken. Hinton-style T^2 scaling is applied internally
    # so the loss magnitude stays comparable across temperatures.
    kl_temperature: float = 1.0
    # KL observe-only (diagnostic logging without loss contribution).
    # When True, KLPenalty is constructed even if kl_enabled=False;
    # every Nth optimizer step (N = kl_log_every_n_steps) the teacher
    # forward + KL math runs, the value is logged to wandb under
    # train/reg_kl, but kl_weight is forced to 0 so the result does
    # NOT contribute to the optimizer's loss. Use this on memory-tight
    # GPUs (e.g. 4090 24 GB) where running the teacher forward EVERY
    # step OOMs but you still want a periodic KL trace for diagnostics.
    # The teacher forward is no-grad and re-uses the same model via
    # disable_adapter(), so cost == 1 extra forward pass once per
    # kl_log_every_n_steps. Default off (bit-identical to v3 baseline).
    kl_log_only: bool = False
    # When kl_log_only=True, run the diagnostic KL every N optimizer
    # steps. N must be >= 1. At 1 the cost is the same as fully-enabled
    # KL (just without the loss contribution). Pick a value that
    # amortizes the OOM risk — 50 or 100 is usually safe on a 4090.
    kl_log_every_n_steps: int = 100

    # Optional chunk size for the KL fp32 math along the N (supervised
    # positions) axis.  When None (default), KL is computed in one shot
    # over all N positions — fast but peak fp32 buffer is ~5 × N × V × 4
    # bytes.  When set to a positive int, the fp32 log_softmax / kl_div
    # is done in chunks of this many rows, bounding peak to
    # ~5 × chunk_size × V × 4 bytes at the cost of extra kernel launches.
    # Only relevant when kl_enabled or kl_log_only is True.
    kl_chunk_size: Optional[int] = None

    # L2 weight anchor
    l2_enabled: bool = False
    # Coefficient is intentionally conservative — at 1e-4 with bf16 LoRA
    # the L2 term is generally an order of magnitude below the CE term
    # at convergence. Tune up if you see catastrophic forgetting; tune
    # down if training stalls early.
    l2_weight: float = 1.0e-4


@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _merge_into_dataclass(dc: Any, src: Dict[str, Any]) -> None:
    """Deep-merge `src` dict into `dc` dataclass instance.

    Unknown keys are warned and skipped. Nested dataclasses recurse.
    """
    if not is_dataclass(dc):
        raise TypeError(f"Expected dataclass instance, got {type(dc).__name__}")
    valid_names = {f.name for f in fields(dc)}
    for key, value in src.items():
        if key not in valid_names:
            log.warning("Unknown config key ignored: %s", key)
            continue
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(dc, key, value)


# Map of CLI flag → dotted dataclass path.
CLI_OVERRIDE_MAP: Dict[str, tuple[str, ...]] = {
    "base_model": ("model", "base_model"),
    "max_seq_length": ("model", "max_seq_length"),
    "load_in_4bit": ("model", "load_in_4bit"),
    "learning_rate": ("training", "learning_rate"),
    "num_train_epochs": ("training", "num_train_epochs"),
    "max_steps": ("training", "max_steps"),
    "per_device_train_batch_size": ("training", "per_device_train_batch_size"),
    "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
    "output_dir": ("training", "output_dir"),
    "run_name": ("training", "run_name"),
    "report_to": ("training", "report_to"),
    "train_file": ("data", "train_file"),
    "val_file": ("data", "val_file"),
    "max_train_samples": ("data", "max_train_samples"),
    "max_val_samples": ("data", "max_val_samples"),
    "augmentation": ("data", "augmentation"),
    "lora_r": ("lora", "r"),
    "lora_alpha": ("lora", "lora_alpha"),
    "lora_dropout": ("lora", "lora_dropout"),
    "tune_projector": ("lora", "tune_projector"),
    "projector_learning_rate": ("lora", "projector_learning_rate"),
    "tune_last_n_vision_layers": ("lora", "tune_last_n_vision_layers"),
    "vision_layers_learning_rate": ("lora", "vision_layers_learning_rate"),
    "use_dora": ("lora", "use_dora"),
    # EvalConfig (auto-eval after training).
    "eval_enabled": ("eval", "enabled"),
    # CLI uses `max_eval_samples_eval` to avoid collision with
    # `data.max_val_samples` (already in this map under different key).
    "max_eval_samples_eval": ("eval", "max_eval_samples"),
    "eval_max_new_tokens": ("eval", "max_new_tokens"),
    "eval_use_unsloth": ("eval", "use_unsloth"),
    "eval_batch_size": ("eval", "batch_size"),
}


def apply_cli_overrides(cfg: FinetuneConfig, overrides: Dict[str, Any]) -> None:
    """Apply CLI flag overrides (dict of flag→value, None values ignored)."""
    for flag, value in overrides.items():
        if value is None:
            continue
        path = CLI_OVERRIDE_MAP.get(flag)
        if path is None:
            log.warning("CLI override '%s' not mapped; ignored", flag)
            continue
        target: Any = cfg
        for attr in path[:-1]:
            target = getattr(target, attr)
        setattr(target, path[-1], value)


def load_config(
    yaml_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> FinetuneConfig:
    """Build a FinetuneConfig: defaults → YAML overlay → CLI overrides."""
    cfg = FinetuneConfig()
    if yaml_path is not None:
        import yaml  # lazy: pyyaml is in requirements but we keep import local

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Top-level YAML in {yaml_path} must be a mapping")
        _merge_into_dataclass(cfg, raw)
    if cli_overrides:
        apply_cli_overrides(cfg, cli_overrides)
    return cfg


def validate_config(cfg: FinetuneConfig) -> List[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    errors: List[str] = []
    if cfg.model.max_seq_length <= 0:
        errors.append("model.max_seq_length must be positive")
    if cfg.training.per_device_train_batch_size <= 0:
        errors.append("training.per_device_train_batch_size must be positive")
    if cfg.training.gradient_accumulation_steps <= 0:
        errors.append("training.gradient_accumulation_steps must be positive")
    if cfg.training.learning_rate <= 0:
        errors.append("training.learning_rate must be positive")
    # Project policy: NO 8-bit / 4-bit anywhere — including optimizer
    # states. bnb's *_8bit optimizers store moments in INT8 which
    # silently degrades numerics for full-param modules_to_save (the
    # projector / vision-layer groups) and contradicts the rest of the
    # no-QLoRA stance. Reject up front so config typos don't slip past.
    optim_name = cfg.training.optim.lower()
    if (
        "8bit" in optim_name
        or "4bit" in optim_name
        or "_bnb" in optim_name
        or optim_name.startswith("paged_")
    ):
        errors.append(
            f"training.optim={cfg.training.optim!r} is rejected by project "
            "policy (no 8-bit / 4-bit anywhere; AGENTS.md). Use "
            "'adamw_torch_fused' (recommended on Ampere+) or 'adamw_torch'."
        )
    # warmup_ratio in [0, 1] when explicitly set. Sanity-check obvious
    # typos like 50 (meaning "50%" but interpreted as 5000% by HF).
    if cfg.training.warmup_ratio is not None and not (
        0.0 <= cfg.training.warmup_ratio <= 1.0
    ):
        errors.append(
            f"training.warmup_ratio must be in [0, 1] (got "
            f"{cfg.training.warmup_ratio}). Use a decimal fraction "
            "(e.g. 0.03 for 3% warmup), not a percentage."
        )
    if (
        cfg.training.save_total_limit is not None
        and cfg.training.save_total_limit <= 0
    ):
        errors.append(
            "training.save_total_limit must be null (= unlimited) or a "
            f"positive int (got {cfg.training.save_total_limit})"
        )
    if (
        cfg.training.num_train_epochs is None
        and cfg.training.max_steps is None
    ):
        errors.append(
            "Either training.num_train_epochs or training.max_steps must be set"
        )
    if cfg.lora.r <= 0:
        errors.append("lora.r must be positive")
    if cfg.lora.lora_alpha <= 0:
        errors.append("lora.lora_alpha must be positive")
    # Vision/audio MUST stay frozen for this project — guard against
    # accidental config drift.
    if cfg.lora.finetune_vision_layers:
        errors.append(
            "lora.finetune_vision_layers must be False — vision tower is frozen"
        )
    if cfg.lora.finetune_audio_layers:
        errors.append(
            "lora.finetune_audio_layers must be False — audio tower is frozen"
        )
    # Projector tuning (feature/lora-plus-projector). When enabled, the
    # vision encoder must STILL be frozen — we are unfreezing only the
    # vision-language projector (embed_vision.*), not the SigLIP encoder.
    if cfg.lora.tune_projector and cfg.lora.finetune_vision_layers:
        errors.append(
            "lora.tune_projector=True is incompatible with "
            "lora.finetune_vision_layers=True — projector tuning unfreezes "
            "ONLY the vision-language projector (embed_vision.*), not the "
            "SigLIP vision encoder. Set finetune_vision_layers=False."
        )
    # Project policy: NO QLoRA BY DEFAULT.  All training runs in full
    # bf16 precision UNLESS the config explicitly opts in to QLoRA by
    # setting model.load_in_4bit=true.  Opting in is reserved for the
    # explicit QLoRA-baseline comparison runs (see configs/
    # plantnet-50k-baseline2-qlora-...yaml) — every other config must
    # keep load_in_4bit=false.
    #
    # When load_in_4bit=true the projector / vision-layer tuning paths
    # still need to *work* (not silent-no-op).  The hard guarantee comes
    # from `_assert_no_4bit_in_trainable_full_param_modules` in
    # finetune.py, which walks trainable params after PEFT wrapping and
    # fails loudly if any trainable projector / vision-layer param ended
    # up as ``bnb.nn.Params4bit``.  The recommended base for QLoRA +
    # projector tuning is
    # ``unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`` whose
    # ``quantization_config.llm_int8_skip_modules`` already keeps
    # ``embed_vision``, ``vision_tower``, ``audio_tower``, and
    # ``embed_audio`` in bf16 — exactly what ``modules_to_save`` needs
    # to be differentiable.
    #
    # We only emit a config-level note here; no hard reject.  Callers
    # that want a hard reject for "no QLoRA in this experiment series"
    # should add their own preflight check.
    if cfg.model.load_in_4bit:
        log.warning(
            "model.load_in_4bit=True — running QLoRA. Project default is "
            "bf16 LoRA; QLoRA is reserved for the explicit baseline-2 "
            "comparison configs. Verify this is intentional. The "
            "4-bit-in-trainable tripwire in finetune.py will still "
            "fail loudly if any modules_to_save copy ends up as "
            "bnb.Params4bit (e.g. base_model doesn't skip-quantize the "
            "projector)."
        )
    # DoRA + QLoRA: PEFT's DoRA layer requires computing a weight norm
    # on the full-precision weight, but under 4-bit quantization the
    # original weight is not available — this causes NaN or crashes.
    # Hard-reject the combination.
    if cfg.lora.use_dora and cfg.model.load_in_4bit:
        errors.append(
            "lora.use_dora=True is incompatible with model.load_in_4bit=True "
            "(QLoRA). DoRA requires the full-precision base weight for "
            "magnitude normalization, which is unavailable under 4-bit "
            "quantization. Use standard LoRA for QLoRA runs, or disable "
            "load_in_4bit to use DoRA."
        )
    if (
        cfg.lora.projector_learning_rate is not None
        and cfg.lora.projector_learning_rate <= 0
    ):
        errors.append(
            "lora.projector_learning_rate must be positive when set "
            f"(got {cfg.lora.projector_learning_rate})"
        )
    # Last-N vision encoder layer tuning
    # (feature/lora-plus-projector-plus-vision-tower).
    if cfg.lora.tune_last_n_vision_layers < 0:
        errors.append(
            "lora.tune_last_n_vision_layers must be >= 0 "
            f"(got {cfg.lora.tune_last_n_vision_layers})"
        )
    if cfg.lora.tune_last_n_vision_layers > 0 and not cfg.lora.tune_projector:
        errors.append(
            "lora.tune_last_n_vision_layers > 0 requires lora.tune_projector=true. "
            "Moving visual features (encoder layer updates) without letting "
            "the projector track them creates a feature-space misalignment "
            "that the language LoRA cannot fix downstream."
        )
    if (
        cfg.lora.vision_layers_learning_rate is not None
        and cfg.lora.vision_layers_learning_rate <= 0
    ):
        errors.append(
            "lora.vision_layers_learning_rate must be positive when set "
            f"(got {cfg.lora.vision_layers_learning_rate})"
        )
    # v2: ModalityAware sampler must be paired with group_by_length=False.
    if cfg.training.modality_aware_sampler and cfg.training.group_by_length:
        errors.append(
            "training.modality_aware_sampler=True conflicts with "
            "training.group_by_length=True. ModalityAwareBatchSampler does "
            "its own within-modality length sort; double-sorting via HF's "
            "group_by_length is a config error. Set group_by_length=False."
        )

    # v2: eval_strategy must be one of HF's supported values.
    valid_eval_strategies = {"no", "steps", "epoch"}
    if cfg.training.eval_strategy not in valid_eval_strategies:
        errors.append(
            f"training.eval_strategy={cfg.training.eval_strategy!r} not "
            f"in {sorted(valid_eval_strategies)}"
        )

    # v2: eval_steps only meaningful when eval_strategy='steps'.
    if cfg.training.eval_strategy == "steps":
        if cfg.training.eval_steps is None or cfg.training.eval_steps <= 0:
            errors.append(
                "training.eval_strategy='steps' requires "
                "training.eval_steps to be a positive int "
                f"(got {cfg.training.eval_steps})"
            )
    elif cfg.training.eval_steps is not None:
        errors.append(
            "training.eval_steps is set but training.eval_strategy is not "
            f"'steps' (got {cfg.training.eval_strategy!r}). Remove eval_steps "
            "or set eval_strategy='steps'."
        )

    # v2: data.val_files (dict) and data.val_file (string) are mutually
    # exclusive. Allowing both invites a config typo where the user
    # thinks one is active but the other wins.
    if cfg.data.val_files is not None and cfg.data.val_file is not None:
        # Special case: default cfg has val_file='data/val.jsonl' which a
        # v2 yaml might not bother to null out. Only flag when val_files
        # is set AND val_file was deliberately overridden away from the
        # default. We approximate "deliberate" with "non-empty string".
        if cfg.data.val_file:
            errors.append(
                "data.val_files (v2 dict) and data.val_file (v1 string) "
                "cannot both be set. Use one or the other. To use v2 "
                "multi-val, explicitly null out val_file: 'data.val_file: null'."
            )

    # Post-training auto-eval (EvalConfig).
    if cfg.eval.max_new_tokens <= 0:
        errors.append(
            "eval.max_new_tokens must be positive "
            f"(got {cfg.eval.max_new_tokens})"
        )
    if cfg.eval.batch_size <= 0:
        errors.append(
            "eval.batch_size must be positive "
            f"(got {cfg.eval.batch_size})"
        )
    if (
        cfg.eval.max_eval_samples is not None
        and cfg.eval.max_eval_samples <= 0
    ):
        errors.append(
            "eval.max_eval_samples must be null (= all) or a positive int "
            f"(got {cfg.eval.max_eval_samples})"
        )

    # v3: regularization block. When disabled, the numeric values are
    # dead code — don't reject them (lets users leave 0.0 stubs around).
    # When enabled, all coefficients must be strictly positive.
    rcfg = cfg.regularization
    if rcfg.kl_enabled:
        if rcfg.kl_weight <= 0.0:
            errors.append(
                "regularization.kl_weight must be positive when "
                f"kl_enabled=True (got {rcfg.kl_weight})"
            )
        if rcfg.kl_temperature <= 0.0:
            errors.append(
                "regularization.kl_temperature must be positive when "
                f"kl_enabled=True (got {rcfg.kl_temperature})"
            )
    # kl_log_only validations: independent of kl_enabled (they can
    # coexist — when both are True the teacher forward runs every step,
    # period is ignored, and kl_log_only is effectively redundant).
    if rcfg.kl_log_only:
        if rcfg.kl_log_every_n_steps < 1:
            errors.append(
                "regularization.kl_log_every_n_steps must be >= 1 when "
                f"kl_log_only=True (got {rcfg.kl_log_every_n_steps})"
            )
        if rcfg.kl_temperature <= 0.0:
            errors.append(
                "regularization.kl_temperature must be positive when "
                f"kl_log_only=True (got {rcfg.kl_temperature})"
            )
    # QLoRA caveat: with load_in_4bit=true the base weights are
    # quantized, so the teacher distribution under disable_adapter()
    # is a 4-bit-quantized base, not the true pretrained base. Same
    # caveat applies whether KL is contributing to loss (kl_enabled)
    # or just being logged for diagnostics (kl_log_only). Warn but
    # don't hard-reject so users can opt in if they really want
    # KL-to-quantized-base.
    if (rcfg.kl_enabled or rcfg.kl_log_only) and cfg.model.load_in_4bit:
        log.warning(
            "regularization.kl_enabled or kl_log_only = True together "
            "with model.load_in_4bit=True (QLoRA): the KL teacher is "
            "the 4-bit-quantized base, not the bf16 pretrained base. "
            "The KL signal is therefore distorted by quantization "
            "noise. Set load_in_4bit=False (project default) for a "
            "clean KL-to-pretrained signal."
        )
    if rcfg.l2_enabled:
        if rcfg.l2_weight <= 0.0:
            errors.append(
                "regularization.l2_weight must be positive when "
                f"l2_enabled=True (got {rcfg.l2_weight})"
            )
    # v3 wiring constraint: the KL + L2 hooks live on
    # ModalityAwareSFTTrainer. Plain SFTTrainer would silently ignore
    # them, so reject the combo at config time rather than fail at
    # trainer construction.
    if (rcfg.kl_enabled or rcfg.l2_enabled or rcfg.kl_log_only) and not cfg.training.modality_aware_sampler:
        errors.append(
            "regularization.kl_enabled, l2_enabled, or kl_log_only "
            "requires training.modality_aware_sampler=true (the "
            "regularization compute_loss hook lives on "
            "ModalityAwareSFTTrainer). Either enable the modality-aware "
            "sampler, or disable all regularizers."
        )

    return errors
