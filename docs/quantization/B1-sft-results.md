# SFT'd Gemma 4 E2B — HF/CUDA quantization baselines

## TL;DR

- This doc reports the HF/CUDA quantization results for the SFT'd Gemma 4 E2B model.
- The bf16 reference reaches 86.7 % PlantNet on n=300, while pure GPTQ variants land around 80-84 % but stay near 7 GB.
- Pure GPTQ misses the iOS size budget because the large `embed_tokens_per_layer` table remains bf16.
- The R6 hybrid, combining GPTQ Linears, torchao int4 embeddings, and audio stripping, is the first sub-4 GB HF artifact at 3.41 GB and 83.7 %.
- Readers should use R6 as a CUDA/HF reference point, not as the final iOS packaging route.

Per-variant detail for every HF/CUDA quantization we've run on our
**SFT'd** Gemma 4 E2B. MLX-side variants (the actual deployable
candidates) live in `B2-sft-results.md`. Public 4-bit MLX
references on the un-SFT'd base live in `01-baselines.md`.

## At-a-glance

Legend: ✅ done · 🟡 partial · ⏳ queued · ⛔ blocked

| # | Variant | Machine | Eval | Size (GB) | PlantNet match (n=300) | ROUGE-L mean | Notes |
|---|---|---|---|---|---|---|---|
| R0 | `bf16_reference` (baseline LoRA r=256 + fullproj SFT) | 4090 desktop (24G) | ✅ | 9.54 | **86.7 %** (260/300) | 0.821 | reference ceiling, HF CUDA bf16 |
| R1 | `gptq_w4g128_da0` | 4090 desktop | ✅ | 6.97 | **82.3 %** (247/300) | 0.802 | −4.4 pts vs bf16 ✅; Marlin kernel |
| R2 | `gptq_w4g128_da1` | 4090 desktop | ✅ | 6.97 | **80.3 %** (241/300) | 0.784 | −6.4 pts; ExllamaV2 kernel (`desc_act=True`) |
| R3 | `gptq_w4g64_da0` | 4090 laptop  | ✅ | 7.01 | **83.7 %** (251/300) | 0.804 | −3.0 pts; finer group beats g128 on this sample |
| R4 | `gptq_w4g128_lmhead` | 4090 laptop  | ✅ | 6.97 | **81.0 %** (243/300) | 0.788 | −5.7 pts; `lm_head=True` auto-downgraded to `False` |
| R6 | `gptq_w4g64_da0_hybrid_pl_g128` (GPTQ Linears + torchao int4 `embed_tokens_per_layer` + audio strip) | 4090 desktop | ✅ | **3.41** | **83.7 %** (251/300) | 0.804 | **−3.0 pts vs R0; 0.0 pts vs R3 source** ✅ first sub-4 GB HF/CUDA artifact. Loader `hf_gptq_hybrid` (with `embed_scale` carried through). |
| R6.32 | `..._hybrid_pl_g32`  (same, finer group) | 4090 desktop | ✅ | 3.57 | **83.0 %** (249/300) | 0.804 | −3.7 pts vs R0; −0.7 vs R3 — group_size barely moves quality once runtime is correct |
| R6.16 | `..._hybrid_pl_g16`  (same, finest group)| 4090 desktop | ✅ | 3.79 | **82.7 %** (248/300) | 0.797 | −4.0 pts vs R0; −1.0 vs R3 |
| R6.dq | `..._dequantpl_g128` (quant-then-dequant control: noise-only, stock runtime) | 4090 desktop | ✅ | 7.01 | **82.7 %** (248/300) | 0.809 | runtime-isolation control row — see "What R6 proves" below |

### Cross-SFT references (different SFT recipes, same bf16 ceiling test)

These are not quantizations — they are different SFT runs of the same
base, included here to put R0-R4 in context with the `baseline2` /
`baseline3` SFT recipes from the project timeline.

| Variant | Eval | Size (GB) | PlantNet match (n=300) | ROUGE-L mean | Notes |
|---|---|---|---|---|---|
| `baseline2_qlora_safemerged_bf16` | ✅ | 9.57 | **85.7 %** (257/300) | 0.820 | QLoRA on bnb-nf4 base + fullproj, merged back to bf16; −1.0 pt vs R0 |
| `baseline3_bnb_qlora_safemerged_bf16` | ✅ | 9.57 | **80.3 %** (241/300) | 0.788 | QLoRA + bnb base + fullproj (route A in the project timeline); −6.4 pts vs R0 |

The 1-pt gap between R0 (baseline LoRA, 86.7 %) and `baseline2` (QLoRA,
85.7 %) confirms QLoRA is essentially as good as full-precision LoRA on
this task. `baseline3` drops 5 pts further; the additional bnb-base
penalty during training stacks on top of QLoRA.

## Run config

Source model: SFT run
`plantnet-50k-baseline-lora-r256+fullproj-lr5e5-data-aug-enwiki_20260513_031214`,
LoRA r=256 + full projector tuning, 5 epochs, final train loss 0.1515.
Merged bf16 lives at `src/quantization/results/_merged_bf16/`.

Eval split: `finetune/data/english-desc/test.jsonl` (29,880 samples,
782 species; paper-grade from PlantNet `test/`). Per-variant cap
**n=300** at `eval_seed=0` via `random.Random(0).sample(records, 300)`.
Output dirs end in `_test300` so they don't clobber the old n=2870
artifacts.

Eval driver: `quantization/scripts/run/eval.py` (sys.path fix in
`b7d88a1`).

Toolchain: `gptqmodel==7.0.0`, `bitsandbytes==0.49.2`,
`transformers==5.8.0`, `peft==0.19.1`, `torch==2.10.0+cu130`,
NVIDIA RTX 4090 24 GB.

## Tripwires

A variant trips if any of:

- `SIZE>4.0GB` — exceeds the iOS jetsam ceiling. **All rows above
  trip** (this is the iOS-deployability cost of HF backends — they
  leave the towers bf16 or quantize them with no MLX kernel). The MLX
  rows in `B2-sft-results.md` clear this.
- `PlantNet drop >10 pts` — none trip. R2's −6.4 pts is the widest gap.
- `PPL >2× bf16` — not re-run on n=300 (PPL not part of the quick-test
  bench); old n=2870 PPLs are in the historical table at the bottom of
  this doc for reference.

## Per-variant detail

### R0 — `bf16_reference` ✅ (HF CUDA bf16)

- size = **9.54 GB**; species_match = **86.7 %** (260/300); ROUGE-L
  mean = 0.821, median = 0.882.
- Wall: **683 s** ≈ 11.4 min on 4090.
- Source: `quantization/results/bf16_r0_sft_aug_enwiki_test300/eval.json`.

This row is the reference ceiling. All HF tripwires compare against it.
Cross-framework reference for comparison:

| Framework | PlantNet match (n=300) | ROUGE-L mean | Notes |
|---|---|---|---|
| HF CUDA bf16 (4090, this row) | **86.7 %** | 0.821 | R0 canonical reference |
| HF MPS bf16 (macbook) | 81.0 % | 0.787 | M0_hf in B2, on macbook's pre-fix sample — see "sample alignment" caveat below |
| mlx_vlm bf16 (macbook) | 85.7 % | 0.831 | M0 in B2, also on macbook's pre-fix sample |

The MPS vs CUDA framework gap (−5.7 pts on bf16) is meaningful; HF
MPS on Apple Silicon has known numerical drift on bf16 attention.
mlx_vlm comes closer to CUDA bf16, consistent with MLX using Apple's
own bf16 lowering rather than torch's MPS shim.

**Sample alignment caveat**: the macbook M0/M0_hf/M1 results in
`B2-sft-results.md` were measured on the macbook's pre-fix
`test.jsonl` (different per-species image hashes due to APFS-native
order). After macbook pulls the `discover_images` patch and re-preps,
those numbers will be re-run on the canonical sample. See
[`../general/15-postmortems.md`](../general/15-postmortems.md) §4.

### R1 — `gptq_w4g128_da0` ✅

- size = **6.97 GB**; species_match = **82.3 %** (247/300); ROUGE-L
  mean = 0.802, median = 0.880.
- PlantNet drop vs bf16: **−4.4 pts** ✅ (well within the 10 pt tripwire).
- Wall: **853 s** ≈ 14.2 min. Marlin kernel (compatible with
  sym + g128 + `desc_act=False`).
- Calibration: 256 PlantNet train (text-only) + 256 WikiText. Quant
  wall ~18 min (unchanged from the original run; this row reuses the
  existing checkpoint).
- Source: `quantization/results/gptq_w4g128_da0_test300/eval.json`.
- On-disk: language_model 6.06 GB (−2.57 GB / −30 % vs bf16); vision
  tower / audio tower stay bf16.

### R2 — `gptq_w4g128_da1` ✅ (GPTQ winner by size, third by accuracy on n=300)

- size = **6.97 GB**; species_match = **80.3 %** (241/300); ROUGE-L
  mean = 0.784, median = 0.866.
- PlantNet drop vs bf16: **−6.4 pts** ✅.
- Wall: **885 s** ≈ 14.75 min. ExllamaV2 kernel (Marlin classic does
  NOT support `desc_act=True`).
- Quant config: `bits=4, group_size=128, desc_act=True`. Quant wall
  ~18 min, originally ranked above da=0 (R1) on the old n=2870 split
  (68.8 % vs 68.4 %) but slightly below on this n=300 sample. The
  swap is within sampling variance for a 300-row run — see "Sampling
  variance" note below.
- Source: `quantization/results/gptq_w4g128_da1_test300/eval.json`.

### R3 — `gptq_w4g64_da0` ✅

- size = **7.01 GB**; species_match = **83.7 %** (251/300); ROUGE-L
  mean = 0.804, median = 0.875.
- PlantNet drop vs bf16: **−3.0 pts** ✅ (best of the four GPTQ rows
  on this n=300 sample).
- Wall: **832 s** ≈ 13.9 min. Marlin kernel.
- **Old result was 47.7 %** on n=2870 (B1 historical table). The
  large delta is partly checkpoint difference (this row is the new
  laptop re-quant after the calibration-size diagnosis) and partly
  sampling variance from the 300-row cap. The B1 historical doc
  pre-noted "Pending: re-quant laptop variant with 256 PlantNet
  samples" — that re-quant landed and this is the result.
- Finer group (g64) gives slightly better numerics than g128 on this
  task, at +0.04 GB on-disk.
- Source: `quantization/results/gptq_w4g64_da0_test300/eval.json`.

### R4 — `gptq_w4g128_lmhead` ✅ (effectively R1)

- size = **6.97 GB**; species_match = **81.0 %** (243/300); ROUGE-L
  mean = 0.788, median = 0.857.
- PlantNet drop vs bf16: **−5.7 pts** ✅.
- Wall: **827 s** ≈ 13.8 min. Marlin kernel.
- `lm_head=True` requested but auto-downgraded to `False` (Gemma 4
  has `tie_word_embeddings=True`; see `_resolve_lm_head` in the
  GPTQ method module). Variant is therefore equivalent in scope to
  R1 — but it was originally quantized on the laptop with only 54
  PlantNet calib samples, so the 1.3 pt gap from R1 (82.3 %) is the
  calibration-size effect surviving on this n=300 sample.
- Source: `quantization/results/gptq_w4g128_lmhead_test300/eval.json`.
- Note: this checkpoint dir lacks `processor_config.json` (artifact
  of the original quant run); the eval falls back to
  `--base_model_for_processor unsloth/gemma-4-E2B-it` over HTTP and
  proceeds normally. Production runs should re-quant with the
  side-car-fixed flow.

### R6 — `gptq_w4g64_da0_hybrid_pl_g128` ✅ (first sub-4 GB HF/CUDA, zero accuracy loss vs source)

- size = **3.41 GB** on disk; species_match = **83.7 %** (251/300);
  ROUGE-L mean = 0.804, median = 0.875. **0.0 pp vs the R3 GPTQ
  source** that fed it (R3 = 83.7 %, bf16 embeds); −3.0 pp vs R0.
- Wall: 875 s ≈ 14.6 min on the 4090 desktop. Marlin kernel via the
  `hf_gptq_hybrid` loader (`quantization/src/eval/model_loaders.py:
  load_hf_gptq_hybrid`).
- Pipeline:
  1. Start from R3 (`gptq_w4g64_da0`, 7.01 GB GPTQ Linears + bf16
     embeds + bf16 vision/audio).
  2. `quantize_hybrid()` (`src/methods/gptq_torchao_hybrid.py`,
     CLI `scripts/run/gptq_torchao_hybrid.py`) does: strip
     `audio_tower` + `embed_audio` (free, never used on iOS); torchao
     `IntxWeightOnlyConfig(weight_dtype=int4, granularity=PerGroup(128),
     mapping=ASYMMETRIC)` on `embed_tokens_per_layer` (4.70 GB →
     1.23 GB); custom uint8 packing into `qweight_packed` (50%
     storage save vs torchao's int8 default on CUDA); GPTQ Linears
     pass through unchanged.
  3. Load via `load_hybrid_embeddings` which installs
     `PackedQuantizedEmbedding` (carries `embed_scale=16` probed
     from the live `Gemma4TextScaledWordEmbedding`) and reads packed
     components from the safetensors.
- Source: `quantization/results/gptq_w4g64_da0_hybrid_pl_only_FIXED_test300/eval.json`.
- On-disk breakdown: 0.94 GB GPTQ Linears (unchanged from R3), 1.23 GB
  packed `embed_tokens_per_layer`, 0.81 GB bf16 `embed_tokens`, 0.34 GB
  bf16 `vision_tower`, ~0.05 GB norms / `embed_vision` / projectors.
  Audio (0.61 GB) stripped.

### R6.32 / R6.16 — `_hybrid_pl_g32`, `_hybrid_pl_g16` ✅ (group_size sweep)

- Same pipeline as R6 with `--embed_per_layer_group_size 32` and `16`
  respectively.
- 83.0 % (g32, 3.57 GB) and 82.7 % (g16, 3.79 GB). Per-row cos_sim vs
  the original bf16 improves from 0.994 (g128) → 0.997 (g32) → 0.998
  (g16), but **eval accuracy does not** — the noise level is already
  at or below the bf16 rounding floor on this distribution
  (`embed_tokens_per_layer` std ≈ 0.064). Group_size is no longer a
  quality lever once the runtime is correct.
- **R6 g128 dominates this trio** for deployment: smallest variant,
  highest accuracy, equal to the bf16-embed source.

### R6.dq — `_dequantpl_g128` (control row, isolates noise from runtime)

- size = **7.01 GB** (same as R3 — only the embedding tensor was
  quantize-then-dequantized in place; no packing, no module swap);
  species_match = **82.7 %** (248/300).
- Built by running torchao asymmetric int4 g128 on
  `embed_tokens_per_layer.weight`, then dequantizing back to bf16 and
  saving the result into a copy of R3's shard 2. Stock `hf_gptq`
  loader, stock `Gemma4TextScaledWordEmbedding`, stock `nn.Embedding`
  forward — only difference vs R3 is the embedding tensor's bytes.
- **Purpose**: A/B isolate "quantization noise" from "runtime swap".
  If the noise is the cause of an accuracy collapse, R6.dq should
  collapse identically. R6.dq holds at 82.7 % (within 1 pp of R3),
  proving the noise itself is harmless on this model. This row drove
  the diagnosis that surfaced the `embed_scale` bug. Full narrative
  in `B1-torchao-vs-gptqmodel.md` §7.1.
- Not a deployment candidate (still 7.01 GB); kept on disk for
  reproducibility of the diagnostic and as documentation of the
  quant-noise upper-bound.

### What R6 proves (and what it doesn't)

**Proves**:

- The hybrid pipeline produces a deployment-grade PyTorch/CUDA
  artifact at < 4 GB with **zero accuracy loss vs its bf16-embed
  source**. The b1 route is no longer ceiling-limited at 7 GB.
- The "negative result" written up earlier on 2026-05-15 was a runtime
  bug (`Gemma4TextScaledWordEmbedding.embed_scale` dropped on module
  swap), not a fundamental limitation of int4 affine embedding quant
  on this model.
- The b1 PyTorch reference and the b2 MLX deploy artifact (M2 at
  83.7 % / 3.4 GB, `B2-sft-results.md`) match on both axes within the
  eval seed noise band — the two independent quantization routes
  cross-validate.

**Does not prove (yet)**:

- That `embed_tokens` (the main token-id → 1536-dim table, 0.81 GB
  bf16) can be quantized without further work. The earlier
  "tied-weights gotcha" was almost certainly the same `embed_scale`
  bug (scale on `embed_tokens` ≈ 39.19 → catastrophic on first
  lookup), but full-embed-quant has not yet been measured. Expected
  size if it works: ~2.77 GB. Tracked in `B1-torchao-vs-gptqmodel.md`
  §8 follow-up #1.
- That MLX-side quantization on the same SFT carries the same
  number when re-run on the canonical sorted `test.jsonl` n=300
  split (M2 already does at 83.7 %, but only on the macbook's
  pre-fix sample — see "Sample alignment caveat" in R0).

### Cross-SFT — `baseline2_qlora_safemerged_bf16` ✅

- size = **9.57 GB**; species_match = **85.7 %** (257/300); ROUGE-L
  mean = 0.820, median = 0.871.
- Drop vs R0: **−1.0 pt** — QLoRA + fullproj is essentially equivalent
  to the baseline full-precision LoRA on this task.
- Wall: 705 s ≈ 11.75 min. Backend: HF CUDA bf16.
- Source:
  `quantization/results/baseline2_qlora_safemerged_bf16_test300/eval.json`.
- Recipe:
  `finetune/configs/plantnet-50k-baseline2-qlora-r256+fullproj-lr5e5-data-aug-enwiki.yaml`.

### Cross-SFT — `baseline3_bnb_qlora_safemerged_bf16` ✅

- size = **9.57 GB**; species_match = **80.3 %** (241/300); ROUGE-L
  mean = 0.788, median = 0.863.
- Drop vs R0: **−6.4 pts** — the bnb-quantized base during training
  costs ~5 pts on top of QLoRA's already-1pt cost vs baseline LoRA.
- Wall: 705 s ≈ 11.75 min. Backend: HF CUDA bf16.
- Source:
  `quantization/results/baseline3_bnb_qlora_safemerged_bf16_test300/eval.json`.
- Recipe:
  `finetune/configs/plantnet-50k-baseline3-qlora-r256+bnb+fullproj-lr5e5-data-aug-enwiki.yaml`.
- Note: this matches the route A spec in the project timeline —
  "Gemma 4 4-bit direct bnb-QLoRA + projection → 4-bit output." The
  measured accuracy (80.3 %) is in the same range as the route-A
  prediction (~69.5 % on the older val); the jump is the n=300 sample
  shift rather than a recipe change.

## Sampling variance — context for the absolute numbers

The new n=300 absolute numbers are higher than the old n=2870 numbers
(e.g. R0: 70.6 % → 86.7 %; R1: 68.4 % → 82.3 %). This is **not**
a model improvement — none of the checkpoints changed. Three reasons:

1. **Different eval set.** Old runs used `val.jsonl` (3,090 rows,
   932 species, filtered with a looser English-name filter). New
   runs use `test.jsonl` (29,880 rows, 782 species, canonical
   filter). Paper-grade PlantNet `test/` images skew toward more
   visually distinctive species; the in-distribution holdout had a
   long-tail of rare species the SFT didn't fit as cleanly.
2. **Smaller sample.** n=300 has a 95 % CI of roughly ±5 pp around a
   point estimate of 85 % (binomial). Differences inside ~3 pp
   between R1/R2/R3/R4 are within noise.
3. **Sample is fixed across rows.** Same n=300 records for every
   variant, so relative ordering between rows is far more reliable
   than the absolute numbers. The −1 to −6 pt drops below R0 are
   robust signals; absolute placements of R3 above R1 above R2 are
   noisier.

For absolute deployment numbers, use the full n=29,880 run; for the
quick-comparison sweep at hand, the n=300 numbers are sufficient to
rank methods and confirm "GPTQ is iOS-jetsam-ineligible-but-accurate".

## Per-variant detail — historical n=2870 table (kept for reference)

These are the original B1 results on `val.jsonl` (3,090 rows / 932
species) before the canonical eval-split fix. Superseded by the
n=300 table above; kept here so anyone navigating from older notes
can find the matching row.

| # | Variant | Size (GB) | PlantNet match (n=2870) | ROUGE-L mean | PPL |
|---|---|---|---|---|---|
| R0 | `bf16_reference` | 9.51 | 70.6 % (2027/2870) | 0.711 | 2,873 |
| R1 | `gptq_w4g128_da0` | 7.0 | 68.4 % (1962/2870) | 0.694 | 3,149 |
| R2 | `gptq_w4g128_da1` | 6.97 | 68.8 % (1975/2870) | 0.699 | 2,932 |
| R3 | `gptq_w4g64_da0` | 7.0 | 47.7 % (1370/2870) | 0.568 | 2,920 |
| R4 | `gptq_w4g128_lmhead` | 7.0 | 46.9 % (1347/2870) | 0.560 | 3,011 |
| R5 | `bnb_nf4` (removed) | 6.31 | 0.1 % (2/2870) | 0.238 | 4,086 |

## Caveats / known issues (carried over from the original B1)

1. **GPTQModel's stand-alone 4-bit ceiling is ~7 GB on this model.**
   With everything in the language sub-module quantized to INT4,
   `gptqmodel` alone drops 9.5 GB (bf16) → ~7.0 GB. Sources of the
   remaining 7 GB:
   - Embedding tables, especially `embed_tokens_per_layer` (~4.7 GB
     bf16, `[262144, 8960]`) and `embed_tokens` (~0.81 GB bf16,
     `[262144, 1536]`), are `nn.Embedding`, not `nn.Linear` — GPTQ
     doesn't touch them.
   - Tied `lm_head` shares `embed_tokens` weights; `lm_head=True`
     auto-downgrades to `False` via `_resolve_lm_head`.
   - Vision tower (0.34 GB) and audio tower (0.61 GB) stay bf16
     under `Gemma4ForConditionalGenerationGPTQ`'s scope.
   - RMSNorm / per-layer scaling weights / position caches stay bf16.
   - Per-group `(scale, zero_point)` metadata adds ~4 % overhead on
     top of theoretical 4 bits/weight.

   **This is no longer the final ceiling.** R6 (`gptq_w4g64_da0_hybrid_pl_g128`)
   adds torchao int4 packing on `embed_tokens_per_layer` + audio
   strip on top of GPTQModel's output and lands at **3.41 GB** with
   **zero accuracy loss** vs the GPTQ source. The combined hybrid
   path is documented in `B1-torchao-vs-gptqmodel.md`. With both
   embeddings quantized (deferred follow-up) the budget falls to
   ~2.77 GB.

2. **GPTQ calibration size matters more than group_size for accuracy.**
   The old R3 result (47.7 % on n=2870) was caused by only 54
   PlantNet calib samples; the new R3 result (83.7 % on n=300) reflects
   a re-quant with 256 calib samples. Calibration size is the lever;
   group_size is a secondary tweak.

3. **WikiText PPL on a domain-SFT model is high by design.** PPL in
   the 2,900–4,100 range is the new normal after SFT, not a regression.
   The model's distribution shifted toward botanical descriptions; the
   laptop's pre-SFT smoke showed PPL=274 on the same checkpoint.
   These numbers are useful for **relative** comparison between quant
   methods, not as absolute quality indicators.

4. **VQAv2 = 0 % on every variant is by design.** After SFT on
   plant-ID-only data the model answers every VQA question with a
   plant identification ("What color is the bus?" → "You've spotted
   Breadseed Poppy."). Benchmark stays wired purely as a regression
   tripwire; result excluded from the row format.

5. **Two merge bugs fixed.** (a) `save_bf16_merged` wasn't saving the
   tokenizer/processor side-cars, breaking GPTQ load; (b)
   `gptq.quantize()` wasn't copying side-cars into the output dir,
   breaking `AutoProcessor.from_pretrained()` at eval time. Both
   fixed in `src/quantization/src/common/model_io.py`
   (`copy_processor_assets`).

## How to refresh a row

```bash
python3 -c "
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
p = d['benchmarks']['plantnet_val']
ppl = d['benchmarks'].get('wikitext_ppl', {})
print(f'size = {d[\"model_size_gb\"]:.2f} GB; '
      f'plantnet species_match = {p[\"species_match\"]*100:.1f}% '
      f'({p[\"species_matches\"]}/{p[\"n\"]}); '
      f'ROUGE-L mean = {p[\"rouge_l_mean\"]:.3f}, '
      f'median = {p[\"rouge_l_median\"]:.3f}; '
      f'wall = {p[\"elapsed_s\"]:.0f}s')
if ppl: print(f'  WikiText PPL: {ppl[\"perplexity\"]:.1f}')
" quantization/results/<variant>_test300/eval.json
```
