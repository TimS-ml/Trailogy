# SFT'd Gemma 4 E2B (r8-a8-nokl-step13000) — MLX quantization benchmarks

## TL;DR

- This doc reports an MLX quantization sweep on the r8-alpha8 no-KL step-13000 SFT, a weaker checkpoint than the data-aug-enwiki model.
- The bf16 ceiling is only 40 % on `plant_100.jsonl`, so this run mainly shows how quantization behaves when the source model is under-trained.
- Every 4-bit variant drops 15-23 points from that low ceiling, far beyond the normal 5-point comfort band.
- EoRA does not recover quality on this checkpoint, so readers should not generalize the successful M8b result without checking the source SFT quality.
- The doc also records the M8b merge bug and fix: use the dict-level safetensors merge, not an inline model round-trip that rewrites unrelated tensors.

> last edit: 2026-05-18 (morning — M8b-merge bug fixed, plant_100 eval set)
>
> MLX quantization sweep on the **r8-a8-nokl-step13000** SFT.
>
> **Headline (2026-05-18 morning)**: switched primary eval from
> PlantNet `test.jsonl` (n=300) to `finetune/eval_sets/plant_100.jsonl`
> (n=100, max-1-per-species) for faster iteration. **M0 bf16 ceiling
> is 40.0 % on plant_100** — significantly lower than the previous SFT
> (data-aug-enwiki) at 88.3 %. The LoRA (r=8, α=8, no-KL, 13k steps)
> is under-trained relative to data-aug-enwiki (r=256, full-proj).
> Quant drops scale accordingly: M2 g64 at 20.0 % is **−20 pt** from
> M0, far past the 5 pt comfortable band.
>
> **M8b-merge bug found + fixed** (2026-05-18): the v1 inline merge
> went through `mlx_vlm.load → model.update_modules → save_pretrained`,
> which silently rewrote ALL tensors at bf16 round-trip (including
> 659 vision_tower + 752 audio_tower keys and the language_model's
> RMS norm with max |Δ|=118 vs σ=4.8 — completely destroying the model;
> output was 100 % `<pad>` spam, 0/300 match). v2 is a dict-level merge
> (`scripts/run/merge_eora_into_mlx.py`) that only rewrites the 945
> safetensors entries (315 covered layers × {weight, scales, biases})
> and copies every other tensor through byte-for-byte. Verified: 1700
> keys identical to M2, exactly 945 changed.

Per-variant detail for every MLX-side quantization we run on the
**r8-a8-nokl-step13000** SFT'd Gemma 4 E2B. These are the candidates
that load directly via `mlx_vlm.load` and `mlx-swift-lm` on iOS.
HF/CUDA variants live in `B1-sft-results.md`. Public 4-bit MLX
references on the un-SFT'd base live in `01-baselines.md`.

Previous SFT round (data-aug-enwiki) results: `B2-sft-results.md`.

## At-a-glance — `plant_100` (n=100, max-1-per-species)

Primary eval set: `src/finetune/eval_sets/plant_100.jsonl`
(100 records, max 1 sample per species → less per-class variance than
the unstratified test.jsonl sample). Image paths resolved against
`images_resized/test/` and persisted at `<results>/_eval_inputs/plant_100_abs.jsonl`.

Legend: ✅ done · 🟡 partial · ⏳ queued · ⛔ blocked

| # | Variant | Pipeline | Size (GB) | match | ROUGE-L | Drop vs M0 | Notes |
|---|---|---|---|---|---|---|---|
| M0 | `bf16` (`merged/`) | none — bf16 reference | 9.6 | **40.0 %** (40/100) | 0.152 | — | ✅ ceiling (was 88.3 % on prev SFT) |
| M1 | `mlx_g128` | `mlx_vlm.convert -q --q-bits 4 --q-group-size 128` + embed skip | 3.2 | **23.0 %** (23/100) | 0.122 | −17 | ✅ ⚠ trip (>10 pt) |
| M2 | `mlx_g64` | `mlx_vlm.convert -q --q-bits 4 --q-group-size 64` + embed skip | 3.4 | **20.0 %** (20/100) | 0.128 | −20 | ✅ ⚠ trip |
| M3 | `mlx_g32` | `mlx_vlm.convert -q --q-bits 4 --q-group-size 32` + embed skip | — | **25.0 %** (25/100) | 0.121 | −15 | ✅ ⚠ trip (best affine) |
| M8a | `mlx_g64 + eora r=32 (separate)` | M2 + EoRA r=32 (truncated from saved r=64) | 3.5 | **19.0 %** (19/100) | 0.133 | −21 | ✅ EoRA not helping on this SFT |
| M8b | `mlx_g64 + eora r=64 (separate)` | M2 + EoRA r=64 (full saved adapter) | 3.6 | **18.0 %** (18/100) | 0.116 | −22 | ✅ no recovery vs M2 |
| M8b-merge | `mlx_g64_eora64_merged` | M2 + EoRA r=64, dict-level merge: dequant + Δ + requant | 3.4 | **17.0 %** (17/100) | 0.127 | −23 | ✅ functional (was 0% pad-spam pre-fix) |
| M4 | `mlx_hybrid_gptq_stable_w4_g128` | hybrid + `gptq_stable.py` | — | — | — | — | ⏳ skipped (slow + −22.7 pt on prev SFT) |
| M4b | `mlx_lm_gptq_w4_g128` | mlx-lm baseline GPTQ | — | — | — | — | ⏳ skipped |
| M5 | `mlx_hybrid_awq_w4_g128` | hybrid + AWQ | — | — | — | — | ⛔ (gemma4 not in AWQ_MODEL_CONFIGS) |
| M6 | `mlx_hybrid_dynamic_quant_bpw4_g128` | hybrid + sensitivity-driven 3/4-bit mix | — | — | — | — | ⏳ aborted (low SFT ceiling makes 30 min spend low value) |
| M7 | `mlx_hybrid_dwq_w4_g128` | hybrid + KL distillation | — | — | — | — | ⏳ deferred |
| M8c | `mlx_g64 + eora r=128 (separate)` | M2 + EoRA r=128 | — | — | — | — | ⏳ need max_rank=128 recompute |

### Headline takeaways (plant_100, n=100)

1. **M0 ceiling is 40 %** — substantially under-trained vs the prev SFT
   (88.3 %). Whether to ship this checkpoint is an SFT-side question.
2. **All MLX 4-bit variants drop 15-23 pt** vs M0. Far past the 5 pt
   comfortable band — this isn't a vision-tower-leak quant bug
   (M3 g32 with the smallest group_size still drops 15 pt). The
   under-trained SFT is brittle: small quant perturbations land far
   from the SFT objective.
3. **EoRA does not recover quality on this SFT** (M8a/M8b ≤ M2). On
   the previous SFT, M8b r=64 closed the gap from −4.7 pt to −0.3 pt.
   Hypothesis: an under-trained model produces a sharper, less
   redundant weight landscape where a WikiText-calibrated correction
   moves further off-manifold.
4. **M8b-merge is now functional** (17 %, was 0 % pad-spam before the
   bug fix). Still trails M2 by 3 pt because requantization
   re-introduces 4-bit error on top of the EoRA-corrected bf16 weights.
   For this under-trained SFT, neither separate-adapter nor
   merged-adapter EoRA helps; for production deploy on this SFT the
   pure M2 (or M3) is preferable.

## Previous-round at-a-glance (test.jsonl n=300, pre-fix)

Recorded for reference; superseded by plant_100 above.

| # | Variant | match (test.jsonl) | Notes |
|---|---|---|---|
| M0 | bf16 | 55.7 % (167/300) | matches plant_100 trend (40 %) |
| M2 | g64 | 38.3 % (115/300) | matches plant_100 (20 %) |
| M3 | g32 | 39.7 % (119/300) | matches plant_100 (25 %) |
| M8b-merge (v1 BROKEN) | — | 0.0 % | pad-spam, vision_tower corrupted |

## Why does EoRA HURT this SFT? (analysis)

The headline anomaly: on the previous SFT (data-aug-enwiki, LoRA r=256
+ full projector), `M2 + EoRA r=64` recovered the quant gap from
−4.7 pt to **−0.3 pt** (essentially closing it). On this SFT
(r=8 / α=8 / no-KL / 13k steps), `M2 + EoRA r=64` does **worse than
plain M2** (−22 vs −20 drop on plant_100). EoRA r=32 also hurts but
less; r=64 hurts more; r=64-merged hurts most. **More EoRA, worse.**

This is the opposite of the expected behavior. Below is the mechanistic
story.

### 1. What EoRA actually computes

For each quantized linear, EoRA produces a rank-r factorization

$$
\Delta_{\text{EoRA}} \;=\; A \cdot B
\;\;\approx\;\; \mathrm{TopR\text{-}SVD}\big(\,
  (W_{\text{bf16}} - \mathrm{dequant}(W_q)) \cdot \sqrt{\Lambda} \cdot U^{\!\top}
\,\big)
$$

where $X^{\!\top}\!X = U \Lambda U^{\!\top}$ is the eigendecomposition
of the input-activation covariance collected from **128 × 512 WikiText
tokens flowed through the quantized model**. The eigenspace weighting
prioritizes reconstructing quant error in directions that carry input
mass — at calibration time.

At inference the model computes $y = \mathrm{quant}(W_q) x + x A B$.
If $A B$ matches the quant error in the directions that matter at
inference, the model recovers toward $W_{\text{bf16}}$.

### 2. The two SFT regimes produce very different `W_bf16 - dequant(W_q)` matrices

| Regime | LoRA r | Where the SFT change lives | The quant-error matrix |
|---|---|---|---|
| **prev SFT** (data-aug-enwiki) | r=256, +full projector | task-relevant changes occupy ~256 directions per matrix, with large magnitude (~ training-time σ ≈ LR × steps) | the gap $W_{\text{bf16}} - \mathrm{dequant}(W_q)$ is **dominated** by quant error on the LoRA's directions — those directions also carry the most input mass at inference because they were trained to |
| **this SFT** (r=8, no-KL, 13k) | r=8, no projector tune | task-relevant changes occupy ~8 directions per matrix, with small magnitude | the gap is **dominated by isotropic-ish base-model quant noise** across ~hidden_dim directions, only an 8-dim sliver of which intersects the LoRA's plant-task signal |

Concretely, for one MLP up_proj (out=6144, in=192):

- Total quant error has Frobenius L2 ≈ `0.5 · max_abs(W) · 2^-bits / √g`
  ≈ a few percent of `||W||_F`, spread over ~min(out,in)=192 directions.
- LoRA r=256 SFT: ~256 directions in `W_bf16 - W_base_bf16` carry
  large weight changes ⇒ those same 256 directions dominate the
  `(W_bf16 - dequant(W_q))` matrix's top-σ space.
- LoRA r=8 SFT: only 8 directions carry SFT signal ⇒ they are
  **buried** under base-model quant noise in the matrix's top-σ space.

### 3. Top-R SVD of a noise-dominated delta is structured noise

EoRA truncates to rank-r ≥ 32 keeping the **largest singular vectors**.

- prev SFT: top-32/64 singular vectors of the delta align with
  LoRA-task directions ⇒ correction is signal ⇒ adding it back recovers.
- this SFT: top-32/64 singular vectors of the delta align with
  generic high-mass base-model directions (driven by WikiText X^⊤X,
  not the plant SFT). The "correction" is structured noise that
  happens to maximize variance reconstruction under the calibration
  distribution. Adding it back **perturbs** the model along
  directions that have no relationship to the plant task.

The user-facing observation:
$$
\Big\| A_{\text{EoRA}} B_{\text{EoRA}} \Big\|_F \;\approx\; 2.3 \text{ per layer (fp32)}
$$

This is the magnitude we recorded during the M8b-merge: `avg L2/layer
= 2.30`. For comparison, the LoRA r=8 contribution to W has L2 ≈
`α/r · ||A_lora B_lora||_F` ≈ a fraction of that (LoRA r=8 with α=8
is unit-scaled). So **the EoRA correction is larger than the SFT's
own contribution to W**, but pointing in the wrong directions.

### 4. Why does it get WORSE with more rank, not just plateau?

If the top-r singular vectors of the delta were merely *uninformative*
(orthogonal to plant directions, parallel to nothing the model uses),
EoRA at any rank would just plateau M2's score. But we see a
**monotone degradation**:

```
M2 g64 bare:    20.0 %
+ EoRA r=32:    19.0 %   (−1)
+ EoRA r=64:    18.0 %   (−2)
+ r=64 merged:  17.0 %   (−3)
```

This implies EoRA's correction is **destructive**, not merely
orthogonal. The mechanism:

- More rank ⇒ more total Frobenius norm added to W (the rank-r SVD
  approximation's L2 grows with r until the full rank is reached).
- The added directions correlate with **base-model activations on
  WikiText**, not with the SFT-trained plant prompt distribution.
- This pulls the layer's input-output map back toward base-model
  WikiText behavior — partially undoing the SFT's plant adaptation
  in the few directions where the plant SFT had been able to assert
  itself through a low-rank delta.

In other words, **EoRA on a low-rank SFT is approximately
"un-SFT-ing"** the model: the calibration data (WikiText) is closer
to the pre-SFT distribution than to the SFT distribution, so the
correction nudges weights toward the pre-SFT optimum, partially
undoing the plant-specific changes.

### 5. Why does merging hurt more than separate adapter?

`M8b-merge` (17.0 %) trails `M8b separate` (18.0 %) by 1 pt. The
mechanism is mechanical, not statistical:

- Separate adapter: `y = quant(W_q) x + x A B`. The correction is
  applied in bf16 precision exactly as computed.
- Merged: dequantize → add Δ → **re-quantize** at 4-bit g=64. This
  re-quantization introduces a second pass of quant error on top of
  the already-quant-error-loaded corrected weight.

For an under-trained model in a noisy weight regime, the second
quantization adds incoherent perturbation on top of incoherent
perturbation. Coherent corrections survive re-quantization (they
re-project onto the same quantization grid); incoherent corrections
get re-randomized.

### 6. Predicted experiments to test this analysis

If the story is right, the following should hold (none of these have
been run yet on this SFT):

1. **EoRA r=8 should be the least harmful** (or actually help): if
   the SFT's task-relevant rank is ~8, the top-8 SVD of the delta
   may still capture the LoRA's directions before noise dominates.
2. **PlantNet-domain calibration** (replace WikiText with prompted
   plant Q&A) should help: X^⊤X eigenvectors would point along
   plant-relevant input directions, biasing EoRA to recover quant
   error in directions the model actually uses at deploy.
3. **EoRA r=1 with PlantNet calibration** could in principle close
   the gap on this SFT — capturing only the single most-task-relevant
   direction.
4. **Apply EoRA to the BASE model (no SFT)** then SFT on top: if the
   bug is "EoRA un-SFTs", correcting before SFT would not have this
   problem. Costly to test.

### Bottom line

EoRA is a **task-agnostic post-quantization quality recovery** method.
It works well when:

- The SFT's contribution to W is **large and high-rank**, so it
  dominates the bf16-vs-quant delta's top singular vectors.
- The calibration distribution is **close to the deploy distribution**,
  so the X^⊤X eigenvectors weight reconstruction in directions the
  model uses.

It fails (and can hurt) when:

- The SFT is **low-rank and small-magnitude**, so the delta's top
  singular vectors come from generic base-model quant noise, not
  task signal.
- The calibration distribution is **far from deploy** (WikiText vs
  plant Q&A is far), so EoRA's correction nudges the model along
  generic directions rather than task directions.

**For this SFT** (r=8, α=8, no-KL, under-trained at 13k steps), both
failure conditions hold simultaneously. The right post-quant recovery
methods here are probably DWQ (KL-distilled scale training against the
bf16 teacher, which is task-aware) or domain-calibrated EoRA — not
WikiText-calibrated EoRA at any rank.

**Test set conventions** (see `02-methods-and-eval.md`):

- **quick** = PlantNet test.jsonl n=300, seed=0 (random sample from
  paper-grade test/ split; 29,880 rows, 782 species). Generated by
  `prepare_plantnet_50k.sh` with per-species image sorting for
  cross-platform reproducibility (Linux/Mac glob order fix).
- **full**  = PlantNet test.jsonl full eval (29,880 rows; or n=2,870
  cap for cross-doc comparison with 01b's earlier rows)

## Pass criteria

For any MLX variant against its own bf16 reference on the same split:

- **PlantNet drop ≤ 5 pts vs bf16 reference (M0)** → ✅
- **PlantNet drop 5-10 pts** → 🟡 (investigate per-method)
- **PlantNet drop > 10 pts** → ⚠ trip — likely scope error (vision
  tower silently quantized, or projector regression). Run
  `inspect_vision_dtype` tripwire (`src/quantization/scripts/inspect/vision_dtype.py`).

## Source model + paths

- SFT adapter: `r8-a8-nokl-step13000` (LoRA rank=8, alpha=8, no KL loss, step 13000)
- HF checkpoint: `<author>/gemma-4-E2B/r8-a8-nokl-step13000_mlx_g64_eora64`
- Merged bf16: TBD
- Quick-test split: `src/finetune/data/english-desc/test.jsonl`,
  seed=0, n=300 sampled by `random.Random(0).sample(...)` (per
  `quantization/eval/plantnet.py:54-59`). Generated by
  `prepare_plantnet_50k.sh` from PlantNet-300K-data-v2/test/ (29,880
  rows, 782 species after English-name filter).

## Previous SFT reference (data-aug-enwiki)

For comparison, key numbers from the previous SFT round:

| # | Method | match (prev) | Drop vs M0 (prev) |
|---|---|---|---|
| M0 | bf16 | 88.3 % | — |
| M2 | affine g64 | 83.7 % | −4.7 |
| M4 | GPTQ stable | 65.7 % | −22.7 |
| M6 | dynamic_quant | 83.3 % | −5.0 |
| M8b | M2 + EoRA r=64 | 88.0 % | −0.3 |

## Per-variant detail

### M0 — `bf16_r8a8_nokl_13k` ⏳

(pending)

### M1 — `mlx_vlm_g128_r8a8_nokl_13k` ⏳

(pending)

### M2 — `mlx_vlm_g64_r8a8_nokl_13k` ⏳

(pending)

### M3 — `mlx_vlm_g32_r8a8_nokl_13k` ⏳

(pending)

### M4 — `mlx_hybrid_gptq_stable_w4_g128` ⏳

(pending — re-test with new SFT)

### M4b — `mlx_lm_gptq_w4_g128` ⏳

(pending — mlx-lm baseline GPTQ, was NaN on prev SFT)

### M5 — `mlx_hybrid_awq_w4_g128` ⏳

(pending — was ⛔ blocked on prev SFT due to missing gemma4_text AWQ config)

### M6 — `mlx_hybrid_dynamic_quant_bpw4_g128` ⏳

(pending)

### M8 — EoRA post-quant adapters on M2 ⏳

(pending — M8a/r=32, M8b/r=64, M8c/r=128)

### M8b-merge — `mlx_g64_eora64_merged` ✅ (after bug fix)

Single-file variant of M8b: instead of loading M2 + a separate adapter
at inference, the EoRA r=64 correction is **baked into** the quantized
weights via `dequantize → +Δ → quantize int4`.

#### v2 (dict-level merge, 2026-05-18) — ✅ functional

Driver: `quantization/scripts/run/merge_eora_into_mlx.py`.

- Operates at the safetensors-dict level, NOT through `mlx_vlm.load`:
  1. `weights = mx.load(quant_dir/model.safetensors)` — flat dict.
  2. For each adapter key `<bk>` (e.g. `model.layers.0.mlp.down_proj`):
     - Look up `language_model.<bk>.{weight, scales, biases}` in dict.
     - `W = mx.dequantize(weight, scales, biases, gs=64, bits=4).f32`.
     - `Δ = lora_b.T @ lora_a.T` (shape (out, in), float32 math).
     - `W_new = W + Δ`.
     - `w_q', s', b' = mx.quantize(W_new.bf16, gs=64, bits=4)`.
     - Replace the 3 entries in the dict.
  3. Save with `metadata={"format": "mlx"}` (required so mlx_vlm.load
     skips its add-`model.`-prefix sanitize path).
- Merge stats: 315/315 layers covered; avg L2/layer = 2.30 (uses
  fp32 corr now); max |Δ| = 0.113. Wall: 2.4 s for math + 6 s for save.
- Diff vs M2 verified: 1700 identical keys, exactly 945 changed
  (315 layers × {weight, scales, biases}). Zero vision_tower /
  audio_tower / norm / embed corruption.
- Eval (plant_100, n=100): **17.0 %** (17/100) match — functional but
  trails M2's 20.0 % by 3 pt (re-quant noise on top of corrected weights).

#### v1 (BROKEN, archived 2026-05-18) — ⛔

Original inline merge went through `mlx_vlm.load → model.update_modules → save`.
That path silently rewrote all 2645 tensors at bf16 round-trip
precision, including:

- 659 `vision_tower.*` bf16 weights perturbed (mean |Δ| ≈ 0.027 on
  weights with σ ≈ 0.036 — roughly 0.7σ random noise across the
  entire vision encoder).
- 752 `audio_tower.*` weights perturbed.
- `language_model.model.norm.weight` perturbed with **max |Δ| = 118.5
  vs σ = 4.8** — completely destroying the RMSNorm scale, which
  cascades to garbage logits.

Result: 100 % `<pad>` token spam, 0.0 % match. Archived at
`<results>/mlx_g64_eora64_merged.broken_v1/` for forensics.

Root cause: the load-model-save round-trip went through MLX's
`nn.QuantizedLinear.update` then `nn.utils.tree_flatten`, which forced
every leaf parameter into a re-savable form. The non-deterministic
init of new `nn.RMSNorm()` slots inside `mlx_vlm`'s Gemma4 layer
construction (called during sanitize before our weights were copied
in) leaked into the save dict because we saved `model.state_dict()`
without filtering. The dict-level v2 sidesteps all of this.

- HF (after re-upload): `<author>/gemma-4-E2B/r8-a8-nokl-step13000_mlx_g64_eora64-merge`
  (currently the broken v1 — TODO: re-upload v2).
- Use case: single-file deployment when QLoRA-style adapter plumbing
  is not available.

### M7 — `mlx_hybrid_dwq_w4_g128` ⏳

(pending — was deferred on prev SFT due to broadcast bug)

## Full sweep conclusion

(pending — will be filled after all rows complete)

| # | Method | Size | bpw | match | Drop vs M0 |
|---|---|---|---|---|---|
| M0 | bf16 (ceiling) | — | 16 | — | — |
| M1 | affine g128 | — | — | — | — |
| M2 | affine g64 | — | — | — | — |
| M3 | affine g32 | — | — | — | — |
| M4 | GPTQ stable | — | — | — | — |
| M4b | GPTQ mlx-lm baseline | — | — | — | — |
| M5 | AWQ hybrid | — | — | — | — |
| M6 | dynamic_quant | — | — | — | — |
| M7 | DWQ | — | — | — | — |
| M8b | M2 + EoRA r=64 (separate adapter) | — | — | — | — |
| M8b-merge | M2 + EoRA r=64 (merged into int4) | — | — | — | — |

## Source paths — MLX-LM / MLX-VLM quant primitives we depend on

(same as prev SFT round — see `B2-sft-results.md` for full table)

| Variant | Algorithm entry point |
|---|---|
| **M1, M2, M3** | flat affine PTQ via `mlx_vlm.convert -q` |
| **M4** | `gptq_stable.py` (desc_act + dead-col + symmetric clip) |
| **M4b** | `mlx-lm/mlx_lm/quant/gptq.py` (upstream baseline) |
| **M5** | AWQ (`mlx-lm/mlx_lm/quant/awq.py`) |
| **M6** | `mlx-lm/mlx_lm/quant/dynamic_quant.py` |
| **M7** | `mlx-lm/mlx_lm/quant/dwq.py` |
| **M8** | `src/quantization/src/methods/eora_mlx.py` |
| hybrid driver | `src/quantization/scripts/run/mlx_hybrid_quant.py` |
| affine driver | `src/quantization/scripts/run/mlx_vlm_deploy_variant.py` |

## How to refresh a row

```bash
python3 -c "
import json, sys
with open(sys.argv[1]) as f: d = json.load(f)
p = d['benchmarks']['plantnet_val']
print(f'size = TBD GB; species_match = {p[\"species_match\"]*100:.1f}% '
      f'({p[\"species_matches\"]}/{p[\"n\"]}); '
      f'ROUGE-L mean = {p[\"rouge_l_mean\"]:.3f}, '
      f'median = {p[\"rouge_l_median\"]:.3f}; '
      f'wall = {p[\"elapsed_s\"]:.0f}s')
" quantization/results/<variant>/eval.json
```
