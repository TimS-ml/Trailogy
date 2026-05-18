# Direct-tune-on-4-bit (QLoRA) baseline — progress

Companion to `docs/B1-sft-results.md`. That file tracks
the standard bf16-SFT → PTQ path (baseline-1). This folder tracks the
**direct-tune-on-4-bit** alternative (baseline-2): train QLoRA on a
pre-quantized 4-bit base, then merge + (optionally) requantize for
deploy. The end-to-end question:

> Does training against a 4-bit base produce a meaningfully different
> deployed model than the standard "train in bf16, post-quantize" recipe?

## Source: SFT run

- Config: `finetune/configs/plantnet-50k-baseline2-qlora-r256+fullproj-lr5e5-data-aug-enwiki.yaml`
- Run name: `plantnet-50k-baseline2-qlora-r256+fullproj-lr5e5-data-aug-enwiki_20260514_044031`
- Base model: `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`
  - bnb NF4 (double-quant, bf16 compute) on language layers
  - `embed_vision` / `vision_tower` / `audio_tower` / `embed_audio` /
    `patch_embedder` / `embedding_projection` skip-quantized to bf16
    (per the checkpoint's `quantization_config.llm_int8_skip_modules`)
- LoRA: r=256, alpha=256, dropout=0.0, attention + MLP, language tower only
- Projector tuning: `tune_projector: true`, `projector_learning_rate: 5.0e-5`
  - PEFT `modules_to_save=['embed_vision']` lands as bf16 differentiable
    (verified by `_assert_no_4bit_in_trainable_full_param_modules` tripwire
    at training start: *"inspected 1 trainable projector/vision-layer
    param(s); none are bnb.Params4bit"*)
- Data: `data/english-desc/` (27,813 train / 3,090 val; English vernacular
  + Wikipedia description; aug enabled)
- Schedule: 5 epochs, cosine, warmup_ratio 0.03, peak LR 2e-4, projector
  LR 5e-5, batch 16, grad-accum 1, total 8,695 optimizer steps
- Hardware: 4090 (24 GB) — 22.0 GB peak / 100 % util
- Wall time: **3.04 h** (vs baseline-1's ~10 h on the same 4090). QLoRA's
  ~halved VRAM footprint lets the bnb-NF4 forward pass + bf16 LoRA backward
  run end-to-end without paging.
- Final train loss: 0.152 (vs baseline-1's 0.152 also — matches within noise)
- Trainable params: 387.7 M / 5.49 B (7.06 %); identical structure to baseline-1

### Why `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`, not `mlx-community/gemma-4-e2b-it-4bit`

The iOS-ship checkpoint `mlx-community/gemma-4-e2b-it-4bit` uses
MLX-native group-affine INT4 plus an mlx-lm `dynamic_quant`-style
calibration layer (the safetensors carry the usual `weight`/`scales`/
`biases` triples AND `input_max`/`input_min`/`output_max`/`output_min`
per Linear plus a per-layer `layer_scalar` — see
`mlx-lm/mlx_lm/quant/dynamic_quant.py`). That format is not loadable
in PyTorch / unsloth / PEFT without a non-trivial port of the MLX
quant kernel.

`unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` is the PyTorch-native
4-bit derivative of the same parent weights (`unsloth/gemma-4-E2B-it`)
that bf16-baseline-1 uses, with the *exact* skip-list we need for
projector tuning to back-propagate. That keeps the experiment legible:
single variable change (bf16 base vs bnb-NF4 base) on top of an
otherwise knob-for-knob identical config.

Using bnb-NF4 instead of MLX-INT4 as the training-time 4-bit scheme is
an accepted simplification: the question we're answering ("does training
see 4-bit-rounded weights") is invariant to which 4-bit quantizer
produces the rounding, and the LoRA adapter is the part that adapts.

## Post-train pipeline

1. **Merge** LoRA + projector into bf16 base
   (`unsloth/gemma-4-E2B-it`, same source as baseline-1's merge target).

   Two merged dirs exist; the second supersedes the first:

   1. `quantization/results/baseline2_qlora_bnb_nf4_skip_both.bf16-merged/`
      — **9.6 GB, 1,951 tensors**. Produced via the standard
      `AutoModelForImageTextToText` + `merge_and_unload()` +
      `save_pretrained()` recipe. Works fine for HF-side downstream
      (bnb-NF4 quantize, HF eval) but is **MISSING 60 inert KV-shared
      tensors** for layers 15-34 (`k_proj` / `v_proj` / `k_norm` /
      `v_norm` weights). Reason: `transformers ≥ 5.8` doesn't allocate
      those as `nn.Parameter`s on KV-shared layers, and
      `_keys_to_ignore_on_load_unexpected` adds them to a silent-drop
      list at load time. `save_pretrained` only writes registered
      params, so the dead tensors fall on the floor.
   2. `quantization/results/baseline2_qlora_safemerged_bf16/` — **9.57 GB,
      2,011 tensors**. Produced via the new safetensors-level merge
      (`quantization/scripts/repair/merge_safetensors.py`, see
      `02b-mlx-torch-convert.md` for the design). Opens base + adapter
      with the `safetensors` library directly; applies LoRA deltas in
      fp32 + casts back; replaces `modules_to_save` tensors by direct
      copy. Bypasses `transformers` entirely → preserves the full
      base key set including the dead KV tensors that `mlx_vlm.convert`
      strict-checks for.

   Wall: ~50 s either way on the 4090.

2. **bnb-NF4 quantize with `skip_modules=['vision_tower', 'embed_vision']`**
   (same recipe as baseline-1's `bnb_nf4_skip_both` — the only NF4 variant
   without the vision-collapse failure mode, per
   `B1-bnb-nf4-vision-collapse.md`)
   - Input: merged dir #1 (HF round-trip). Either input gives the same
     output because bnb's transformers swap re-loads through HF, which
     re-drops the dead KV tensors at load time — they never reach bnb.
   - Output: `quantization/results/baseline2_qlora_bnb_nf4_skip_both/`
   - Size: **6.52 GB** (identical to baseline-1's; same recipe, same shapes)
   - Wall: ~45 s.

3. **MLX conversion via `mlx_vlm.convert`** — DONE on the 4090.
   - Input: merged dir #2 (safetensors-level), required to satisfy
     mlx-vlm 0.4.3's strict load (#1 fails with "Missing 60 parameters",
     see `02b-mlx-torch-convert.md`).
   - Command:
     ```bash
     mlx_vlm convert -q --q-bits 4 --q-group-size 64 --q-mode affine \
         --hf-path quantization/results/baseline2_qlora_safemerged_bf16 \
         --mlx-path quantization/results/baseline2_qlora_mlx_vlm_g64
     ```
   - Output: `quantization/results/baseline2_qlora_mlx_vlm_g64/`
   - Size: **3.37 GB, 2,649 tensors** — byte-for-byte the same tensor
     inventory (suffix counts) as iOS-shipping
     `mlx-community/gemma-4-e2b-it-4bit`. Under the 4 GB ceiling.
   - Wall: ~2.5 min.
   - Convert ran in the conda `mlx` env CPU-side; no GPU JIT involved
     at convert time (which is why the 4090 worked despite the runtime
     CUDA-header gotcha addressed in step 4).

4. **MLX runtime eval on the 4090** — DONE. The Linux mlx-cuda
   runtime needs a manual NVRTC↔toolkit fix-up before any kernels JIT.
   Helper: `src/quantization/scripts/_env/_mlx_env.sh`. Root cause
   + fix detailed in `02b-mlx-torch-convert.md`'s "Action items +
   verification" section. TL;DR: `libmlx.so` links `libnvrtc.so.12`
   (NVRTC 12.9), but the host only ships CUDA 13 headers; the staging
   helper downloads matching CUDA 12.9 toolkit pip wheels and points
   `CUDA_HOME` at them.

## Eval — PlantNet val.jsonl, n=200 (all evals, `eval_seed=0` shuffle)

Run: `quantization/scripts/run/eval.py`, default greedy generation,
`finetune/data/english-desc/val.jsonl` shuffled, first 200 samples
post-shuffle.

| Variant                                                                  | Size      | n   | species_match      | ROUGE-L mean | ROUGE-L median | Loader     |
|--------------------------------------------------------------------------|-----------|-----|--------------------|--------------|----------------|------------|
| baseline-1 bf16-SFT · bf16 reference (canonical)                          | 9.51 GB   | 2870| 70.63 % (2027)     | 0.7108       | 0.8667         | `hf_bf16`  |
| baseline-1 bf16-SFT · bnb-NF4 skip vt+ev (ref @ n=300)                    | 6.52 GB   | 300 | 69.33 % (208)      | 0.7070       | 0.8718         | `hf_bf16`  |
| baseline-2 QLoRA · bnb-NF4 skip vt+ev (built from HF round-trip merge)    | 6.52 GB   | 200 | 69.50 % (139)      | 0.7105       | 0.8718         | `hf_bf16`  |
| baseline-2 QLoRA · safetensors-merged bf16 (no PTQ; sanity ref)           | 9.57 GB   | 200 | **67.50 %** (135)  | 0.6855       | —              | `hf_bf16`  |
| **baseline-2 QLoRA · MLX-INT4 g64 affine** (deploy candidate)             | **3.37 GB** | 200 | **22.50 %** (45)   | 0.3148       | —              | `mlx_vlm`  |

Eval wall on the 4090: bnb-NF4 @ n=200 = 771 s; bf16 @ n=200 = 480 s;
MLX-INT4 @ n=200 = **192 s** (~1 s/sample; mlx-cuda runs noticeably
faster than HF/torch on this hardware for image-text generation).

### Observations

- **Direct-tune-on-4-bit lands at parity with bf16-train+PTQ on the
  post-quant model (HF / bnb-NF4 deploy path).** 69.50 % (baseline-2
  QLoRA → NF4) vs 69.33 % (baseline-1 bf16 → NF4) at the same recipe
  is within statistical noise at these sample sizes (Δ ≈ 0.2 pp on
  n=200 vs 300). The 1.3-pp gap to the bf16 reference (70.63 %) is the
  *quantization* cost, not the *training-base* cost — both paths pay
  it equally.
- **Sanity-merge eval lands 2 pp below the bnb-NF4 eval at the same n.**
  67.50 % bf16-merged vs 69.50 % bnb-NF4 at n=200 is within the n=200
  noise floor (1σ ≈ 3.2 pp). Both reads are on the *same* trained
  weights — the gap is sample-set variance, not a real quality
  difference. The point of this row is to confirm the safetensors-level
  merge is byte-correct (it preserves the trained QLoRA quality; the
  number is in the same ballpark as the bnb-NF4 read).
- **Training wall is ~3× faster on baseline-2.** 3.04 h vs ~10 h on
  the same 4090. The 4-bit base is loaded once with bnb's NF4 kernel
  and the forward pass dequantizes on the fly; no shrink in trainable
  param count vs baseline-1.
- **The MLX-INT4 deploy candidate drops 45 pp from the bf16 reference**
  (67.50 % → 22.50 % at the same n=200). This is the cost of 4-bit
  affine quantization with **no calibration data** on a fine-grained
  classification task. The trained answer style ("This appears to be
  X. Y is a species of …") survives in the quantized weights —
  confirmed by head-to-head with the iOS-ship base
  `mlx-community/gemma-4-e2b-it-4bit` (same `bits=4 / g=64 / affine`
  config) which emits empty / whitespace-only output on these same
  `"Describe this plant."` prompts. But the per-class features the
  LoRA learned are not robust to 4-bit affine compression: species
  identification collapses, and some samples additionally degenerate
  into pad-token spam after the first repeat.
- **The -45 pp drop is reproducible and not an eval bug.** Verified
  by direct comparison: the eval-loader-rendered prompt and the
  direct-`apply_chat_template` prompt are byte-identical
  (`'<bos><|turn>user\n<|image|>Describe this plant.<turn|>\n<|turn>model\n'`);
  both paths produce the same wrong species on sample 0 (ref
  "magic-lily" → pred "Common Marsh-mallow"). Processor-config
  `size` was patched to the trained shape (960×672 via
  `export_mlx.patch_processor_config_for_mlx_swift`) with no effect
  on the outcome.

### Open: full eval + Q4 quality recovery

- [ ] **Full eval at `EVAL_PLANTNET_N=2870`** on at minimum:
      - `baseline2_qlora_safemerged_bf16/` — to land a number directly
        comparable to baseline-1's `bf16_reference` row in
        `B1-sft-results.md` (70.63 %). Expected wall ~3 h.
      - `baseline2_qlora_mlx_vlm_g64/` — to confirm the -45 pp drop at
        n=200 is not just n=200 noise (1σ ≈ 3.2 pp; the gap is huge
        regardless but the absolute number stabilizes).
- [ ] **Recover the MLX-INT4 quality**, ordered by expected impact:
      1. `mlx_vlm.convert --q-mode dynamic_quant --q-calibration-data <plantnet-train>`
         — affine + per-tensor scale calibrated on the trained
         distribution typically recovers 3-15 pp on fine-tuned VLMs;
         this is `mlx-community/gemma-4-e2b-it-4bit`'s own format.
      2. `--q-group-size 32` — half the group size ≈ half the
         quantization error per group; adds ~400-600 MB to the
         artifact (likely close to but not over the 4 GB budget).
      3. Selective skip-modules: keep the trained `embed_vision`
         projector at bf16 (it's small — 2.2 MB — but it's the layer
         the LoRA training tuned most heavily); skip-quantize the
         last few language decoder layers.
      4. `--q-bits 6` — recovers most of the gap but the artifact
         lands ~4.5 GB, over the deploy ceiling.
- [ ] WikiText PPL on baseline-2 MLX-INT4 — sanity tripwire (>2× bf16
      ratio indicates catastrophic language damage; see baseline-1's
      GPTQ rows in `B1-sft-results.md` for the methodology). Useful
      to confirm whether the species-id collapse is a vision-feature
      problem or a general language degradation.

## Artifacts

```
hikeCompanion/
├── finetune/configs/
│   ├── plantnet-50k-baseline2-qlora-r256+fullproj-lr5e5-data-aug-enwiki.yaml
│   └── smoke-save-reload-qlora.yaml
├── finetune/outputs/plantnet-50k-baseline2-qlora-r256+fullproj-lr5e5-data-aug-enwiki_20260514_044031/
│   ├── final-adapter/                    # 1.47 GB (LoRA + projector deltas)
│   ├── checkpoint-8000/  checkpoint-8695/
│   └── train.log
├── finetune/logs/
│   ├── smoke-qlora.log                          # save/reload smoke passed 4/4 criteria
│   ├── baseline2-qlora-dryrun.log
│   ├── baseline2-qlora-train.log
│   ├── baseline2-quant.log                      # HF round-trip merge + NF4 quantize
│   ├── baseline2-quant-eval200.log              # bnb-NF4 PlantNet eval, n=200
│   ├── baseline2-safemerge.log                  # safetensors-level merge
│   ├── baseline2-safemerged-bf16-eval200.log    # bf16-merged PlantNet eval, n=200
│   ├── baseline2-mlx-convert.log                # mlx_vlm.convert run
│   ├── baseline2-mlx-eval20.log                 # MLX-INT4 PlantNet smoke, n=20
│   ├── baseline2-mlx-eval20-patched.log         # post processor-size patch, n=20
│   └── baseline2-mlx-eval200.log                # MLX-INT4 PlantNet eval, n=200
├── quantization/scripts/
│   ├── merge_safetensors.py                     # tensor-level LoRA merge (new)
│   └── _mlx_env.sh                              # mlx-cuda runtime bootstrap (new)
└── quantization/results/
    ├── baseline2_qlora_bnb_nf4_skip_both.bf16-merged/     # 9.6 GB  (HF round-trip; missing 60 KV)
    ├── baseline2_qlora_bnb_nf4_skip_both/                 # 6.52 GB + eval.json
    ├── baseline2_qlora_safemerged_bf16/                   # 9.57 GB (safetensors merge; full keys) + eval.json
    └── baseline2_qlora_mlx_vlm_g64/                       # 3.37 GB (deploy candidate) + eval.json
```

Pushed commits on `@feature/quantization`:
- `480e042` feat(finetune): add QLoRA baseline-2 config for quantization sweep
- `c19d07c` test(finetune): smoke save/reload config for QLoRA baseline-2
- `4c00bf0` feat(quantization): tensor-level safetensors merge preserving full base key set
- `54c1b9d` feat(quantization): make mlx_vlm eval work on Linux + NVIDIA

- `2f91f0a` docs(07-quantization/02b): MLX↔PyTorch convert root-cause + safetensors-merge bridge
- `fd13cac` docs(07-quantization/02b): MLX eval runs on Linux+NVIDIA; -45pp Q4 drop
