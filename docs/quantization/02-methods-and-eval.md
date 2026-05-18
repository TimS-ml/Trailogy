# Quantization methods & eval

What we test and how we measure it. Companion to:

- `B1-sft-results.md` — per-variant HF/CUDA results
- `B2-sft-results.md` — per-variant MLX results (iOS-deployable)
- `B1-bnb-nf4-vision-collapse.md` — why bnb NF4 kills vision-tower-quantized variants
- `05-mlx-vlm-design.md` — the active design for the MLX-VLM path

## Methods — what we test

Ranked by combination of expected effort, expected size, and risk to
the deadline.

| Method | Stack | Where it runs | Reach goal |
|---|---|---|---|
| `mlx_vlm.convert -q` (4-bit affine, vary group_size) | mlx-vlm | Mac (Apple Silicon) | Reproduce 3.58 GB public baseline on SFT'd merge; sweep g32/g64/g128 |
| `mlx_vlm.convert -q --quant-predicate mixed_X_Y` | mlx-vlm | Mac | Mixed-precision recipes (Apple's analogue of Unsloth UD) — same packer, per-tensor bit map |
| MLX hybrid: mlx-vlm `load` + mlx-lm `*_quantize()` core | mlx-vlm tree + mlx-lm quant algo | Mac | Use mlx-lm's research-grade calibration-driven PTQ (gptq/awq/dwq/dynamic_quant) without losing iOS load compatibility — see `05-mlx-vlm-design.md` |
| Unsloth UD MLX 4-bit (`9ee11f5` recipe) | mlx-vlm packer + Unsloth recipe | Mac | Replicate 3.55 GB recipe; understand bit-promotion policy |
| GPTQ (HF / gptqmodel) | HF + gptqmodel | 4090 (CUDA) | Cross-framework reference. ~7 GB ceiling (LM-only scope). Already covered in 01b. |
| AWQ (HF) | HF | 4090 | Backup PTQ if GPTQ tooling fails on Gemma 4 |
| bitsandbytes NF4 | HF | 4090 | Reference-only — quantizes vision tower → 0.1 % match. See `B1-bnb-nf4-vision-collapse.md`. |
| QAT (fake-quant fwd, bf16 grads) | finetune + QAT recipe | 4090 | Stretch goal; **policy ruling needed** before any code |

### Method notes

#### 1. `mlx_vlm.convert` — the deployable default

What it does:

- Loads bf16 safetensors into mlx-vlm's Gemma 4 model class
- Calls `mlx_lm.utils.quantize_model` internally
  (`mlx_vlm/convert.py:158-170`) — the bit-packing kernel is shared
  with mlx-lm
- Stores quantized weights as packed uint32 + bf16 scales/biases
- Saves MLX-format safetensors + processor configs
- **Skips** `vision_tower.*` / `audio_tower.*` via
  `skip_multimodal_module` (`mlx_vlm/utils.py:82-103`).
  **Does NOT** skip `embed_vision.*` / `embed_audio.*` — those
  projectors get quantized by default. Override with a custom
  `quant_predicate` callable if you need them at bf16 too (see
  `01c-*.md` M1 for the working recipe on the data-aug-enwiki SFT).

Why this is the baseline:

- It's the only format `mlx-swift-lm` consumes on iOS today.
- The mlx-community 3.58 GB public reference was produced by exactly
  this command (with g64). See `01-baselines.md` for the bit-by-bit
  scope.
- Zero new code; the work is in the eval harness.

CLI surface (`mlx_vlm/convert.py:104-291`):

```bash
mlx_vlm.convert \
    --hf-path  <bf16-dir> \
    --mlx-path <out> \
    -q \
    --q-bits         {2, 3, 4, 6, 8}                       # flat-precision
    --q-group-size   {32, 64, 128}
    --q-mode         {affine, mxfp4, nvfp4, mxfp8}
    --quant-predicate {mixed_2_6, mixed_3_4, mixed_3_5,
                       mixed_3_6, mixed_3_8, mixed_4_6,
                       mixed_4_8}                          # mixed-precision recipes
```

The mixed-precision recipes encode a fixed per-tensor policy:
`mixed_3_4` → most tensors at 3-bit, "use-more-bits" tensors at 4-bit
(specifically `v_proj` + `down_proj` in the first 1/8 + last 1/8 +
every-3rd middle layer, plus `lm_head` and `embed_tokens`). This is
functionally equivalent to what Unsloth's "UD" recipe does.

#### 2. MLX hybrid — research quant in the right model tree

Use mlx-lm's calibration-driven PTQ implementations
(`mlx_lm/quant/{gptq, awq, dwq, dynamic_quant}.py`) but build the
forward-pass tree via `mlx_vlm.load` so the result is iOS-loadable.
Active design in `05-mlx-vlm-design.md`. Caveats: AWQ needs a
5-line `AWQ_MODEL_CONFIGS["gemma4"]` patch; DWQ has an upstream
broadcast bug in the validation step; only `dynamic_quant` was proven
end-to-end in the 2026-05-13 round (with caveats about which model
tree it was run through).

**Note:** of the `mlx_lm` package, we ONLY trust the
`mlx_lm/quant/*` quant cores (the actual bit-packing + sensitivity
loops). Everything else in `mlx_lm` (model classes, sanitize, load)
is buggy for our use case — see the 2026-05-13 post-mortem in
[`../general/15-postmortems.md`](../general/15-postmortems.md) §2.

#### 3. Unsloth UD (Unsloth Dynamic) MLX 4-bit

What it does: mixed-precision quantization with a per-tensor bit-width
schedule, packed by MLX's standard `QuantizedLinear`. Verified
2026-05-13: HEAD averages 6.14 bits/param (NOT 4-bit) across the LM
body (see `01-baselines.md` for the bit-width map). The `9ee11f5`
recipe is text-only (strips vision + audio towers entirely).

What we need to do: find the script that produced `9ee11f5` (if
public), or reproduce the bit-map via `mlx_vlm.convert
--quant-predicate mixed_4_6` and diff per-tensor dtypes.

Risk: Unsloth's pipeline may depend on Unsloth-specific training-loop
hooks that don't apply to a vanilla bf16 HF checkpoint.

#### 4. GPTQ (HF backend)

Already covered in 01b (R1/R2/R3/R4). Useful as a cross-framework
accuracy reference (68-69 % at n=2,870 on the data-aug-enwiki SFT
under `desc_act=True` + 256-sample PlantNet calibration). The HF
output is **not** MLX-loadable on iOS; treat these rows as quality
reference, not ship candidates.

Calibration data choice (when re-running):

- PlantNet **train.jsonl** (never `val.jsonl` — eval-leak guarded)
- WikiText for general-language preservation
- Mix: 256 plant + 256 WikiText
- Never `overfit100.jsonl` — that set has `train == eval` by
  construction.

#### 5. AWQ — defer

Similar tooling maturity issues to GPTQ. Run only if GPTQ shows
something interesting on this SFT.

#### 6. bitsandbytes NF4 — reference only, do not ship

Quantizes ALL `nn.Linear` including the SigLIP vision tower. The
2026-05-13 ablation (skip_ev / skip_vt / skip_both) confirmed the
vision tower is the sole non-negotiable: NF4'ing the projector while
keeping the tower at bf16 is essentially harmless, but NF4'ing the
tower drops PlantNet match from ~70 % → 0.1 %. Details in
`B1-bnb-nf4-vision-collapse.md`. Useful as the "floor" data point;
NOT iOS-deployable (no MLX NF4 kernel).

#### 7. QAT — open policy question

Modifies the SFT training loop: forward applies fake-quant (rounds
weights to their 4-bit representation, computes loss against
quantized outputs); backward updates the underlying bf16 master
weights. Output is a bf16 adapter + a QAT recipe file;
`quantization/src/methods/_stubs/qat_export.py` materializes the 4-bit MLX output
at export time.

Project policy says "8-bit / 4-bit are not allowed for training" — but
QAT uses **bf16 master weights and bf16 gradients**; the only thing
that's 4-bit is the forward-pass *simulation* of quantization.
Strictly speaking this is not "4-bit training". **Ruling needed from
the project owner before any QAT code is written.** Even with a green light, QAT is
days of GPU time on 50k samples — stretch goal.

### What we explicitly do NOT test

- **HQQ**, **SpQR**, **OmniQuant**, **SmoothQuant** — too many options,
  not enough days. GPTQ is our PTQ representative.
- **8-bit MLX** — outside the 4 GB ceiling for Gemma 4 E2B
  (8-bit would be ~5.1 GB, base size halved).
- **Mixed-precision per-tensor manual recipes** — only viable if the
  Unsloth UD path teaches us a generalizable rule, or if the
  `mlx_vlm.convert --quant-predicate mixed_*` recipes hit the target
  size with acceptable accuracy.

## Eval

### Two test sets

- **quick** = **PlantNet val n=300, seed=0** (random sample via
  `random.Random(0).sample(records, 300)` in
  `src/quantization/src/eval/plantnet.py:54-59`). Fast feedback
  for sweep work; ~2-5 min wall on M5 Pro for an MLX 4-bit variant.
- **full** = **PlantNet300K full eval** (3,090 val samples cap n=2,870
  for cross-doc comparison with 01b's earlier rows). ~30 min on M5 Pro
  for an MLX 4-bit variant; ~3-5 h on 4090 for an HF GPTQ variant.

Workflow:

1. New method or recipe → run **quick** first. If it passes the
   tripwires (see below), graduate to **full**.
2. Final deliverable candidate → ALWAYS report **full** number.

Quick is for sweep economics; full is for the deliverable.

### Metrics

- **species_match** — case-insensitive, whitespace-normalized exact
  match between the model's extracted species name and the reference.
  Primary metric. `extract_species` regex handles both Title Case
  English common names and Latin binomials.
- **ROUGE-L** — mean and median across samples. Secondary signal
  (catches fluent-but-wrong answers).
- **avg response length** — sanity check (sudden drop = pad-spam
  failure).

### Deferred evals

The following were considered and are **deferred** for the current
deliverable cycle:

- WikiText-103 PPL (catastrophic-language-damage tripwire) — useful
  cross-method signal but not MLX-loadable on the iOS path; kept on
  the HF rows in 01b for reference.
- VQAv2 dev-test / MMMU val / OK-VQA / TextVQA — domain-irrelevant
  since the SFT specialized to plant-ID-only (VQAv2 = 0 % by design).
- MMLU few-shot / HellaSwag — same domain mismatch.
- Hiking-QA holdout — no clean train/val split exists yet.

Single-metric focus (PlantNet species_match) is intentional for this
deliverable cycle.

### Methodology — what we must NOT do

- **Don't cherry-pick subsets per method.** Lock the eval splits
  before the first quant run, write a deterministic data manifest,
  never change it during the sweep. Both `quick` and `full` are
  defined here.
- **Don't change the generation hyperparameters between bf16 and INT4.**
  Greedy decode, `max_new_tokens=128`, identical chat template.
- **Don't run eval only at the end.** Eval the bf16 baseline FIRST on
  the same split; all quant-cost numbers are relative to it.
- **Don't leak eval data into GPTQ (or other PTQ) calibration.**
  Calibration MUST come from `train.jsonl`, never `val.jsonl`.
  Enforced by `src.methods.gptq._reject_calibration_leak`,
  which hard-fails on `val.jsonl` paths and on any path containing
  `overfit100`.
- **Don't use overfit100 data anywhere in the deliverable pipeline.**
  Overfit100 sets have `train == eval` by construction
  (memorization-ceiling test). Calibrating or evaluating on them is
  meaningless. The bash run scripts call `_reject_overfit100` on every
  input path; the Python guard catches direct API misuse.

### Output format

Each eval run writes a single JSON file at
`src/quantization/results/<variant>/eval.json`:

```json
{
  "variant": "mlx_vlm_g128_sft_aug_enwiki",
  "model_path": "results/mlx_vlm_g128_sft_aug_enwiki/",
  "model_size_gb": 3.2,
  "base_model_size_gb_bf16": 9.51,
  "backend": "mlx_vlm",
  "benchmarks": {
    "plantnet_val": {
      "n": 300,
      "species_match": 0.497,
      "rouge_l_mean": 0.573,
      "rouge_l_median": 0.561,
      "species_matches": 149,
      "avg_response_len": 175.0,
      "elapsed_s": 145.7
    }
  },
  "eval_seed": 0,
  "generation_kwargs": {"max_new_tokens": 128, "do_sample": false}
}
```

Plus an `eval_per_sample.json` sidecar holding per-sample
`{image, ref_species, pred_species, rouge_l, species_match,
response_len}`.

`compare_runs.py` aggregates these into the per-method matrices in
01b / 01c.

### Tripwires (quick test, n=300)

- **PlantNet drop > 10 pts vs bf16 reference on the SAME split** —
  stop and investigate. Likely a quant scope error (vision tower
  silently quantized, or projector regression). Run
  `inspect_vision_dtype` first.
- **species_match < 5 %** — pad-spam / NaN / load failure; check
  `avg_response_len` and the per-sample `pred_species` field.
- **Output size > 4.0 GB** — exceeds iOS jetsam ceiling; fails the
  ship gate regardless of quality.

For full-eval runs the same tripwires apply at the n=2,870 cap.

### Per-variant wall-time budget

For the quick test (n=300) on M5 Pro 32 GB:

| Backend | Wall time |
|---|---|
| `mlx_vlm` bf16 | ~5 min (10 GB load + bf16 decode) |
| `mlx_vlm` 4-bit | ~2-3 min |
| `hf_bf16` (4090) | ~5-10 min |
| `hf_gptq` (4090) | ~5 min |

For the full test (n=2,870):

| Backend | Wall time |
|---|---|
| `mlx_vlm` bf16 | ~45 min |
| `mlx_vlm` 4-bit | ~25 min |
| `hf_bf16` (4090) | ~2 h |
| `hf_gptq` (4090) | ~3-5 h depending on kernel |

Budget per variant (quant + smoke + quick eval): ~10 min wall on Mac
for an MLX baseline; ~30-60 min for a research method.
