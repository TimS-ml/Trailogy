# bnb 4-bit vs torchao QAT in the SFT stage — what they actually solve

## TL;DR

- This doc separates two often-confused ideas: low-memory 4-bit training and quantization-aware training for better deployed accuracy.
- Bitsandbytes QLoRA saves training memory by keeping the base frozen in 4-bit form, while torchao QAT keeps bf16 weights but trains with fake-quant noise.
- The recorded results suggest QLoRA did not help deployment: it tied bf16 under one NF4 path but collapsed under the MLX INT4 path.
- The practical conclusion is to keep the vision tower out of bnb NF4, avoid treating QLoRA as a deployment-accuracy fix, and reserve torchao QAT for a focused follow-up test.

## Conceptual Distinction

- **bnb 4-bit is a train-time VRAM tool.** It stores the frozen base
  weights as NF4/FP4 so a LoRA adapter can train on top in bf16. The
  end product is "4-bit base + bf16 LoRA". It does **not** by itself
  improve deployed 4-bit accuracy.
- **torchao QAT is a deploy-time accuracy tool.** It keeps weights in
  bf16 during training but injects fake-quant noise on the forward
  pass so the loss landscape becomes "int4-rounding-aware". The end
  product is a real int4 model. It does **not** save train-time VRAM.
- They are not interchangeable. Picking one because "I want 4-bit"
  conflates two different problems: *fitting training* vs *protecting
  deploy accuracy*.
- Empirical signal we already have:
  - At **bnb-NF4 deploy**, bf16-SFT and QLoRA-SFT tie within noise
    (69.3 % vs 69.5 %, see §3). QLoRA didn't help.
  - At **MLX-INT4 g64 deploy**, the same QLoRA checkpoint **collapses
    to 22.5 %** vs the bf16-SFT checkpoint's 78.0 % through the same
    PTQ recipe. QLoRA-base hurt.
  - bnb NF4 on the vision tower drops PlantNet match to 0.1 % even
    on a bf16-SFT checkpoint (`B1-bnb-nf4-vision-collapse.md`). The
    SigLIP encoder is non-negotiably bf16 in every deploy variant we
    ship.
- torchao QAT remains **untested on our pipeline**. Plausible upside
  is modest (+0.3 to +1.0 pp on deployed int4 vs GPTQ + desc_act),
  but a half-day cooldown experiment (§7) can settle it.

## 1. Mechanism — the only diagram that matters

|  | bnb 4-bit (QLoRA) | torchao QAT | bf16 LoRA (our baseline-1) |
|---|---|---|---|
| Base weight on disk | NF4 / FP4 packed | bf16 | bf16 |
| Base weight in VRAM during training | NF4 / FP4 packed | bf16 | bf16 |
| Base weight gradient flow | **frozen** | bf16 grads with STE through fake-quant | n/a (frozen) |
| Forward path on a `Linear` | dequant → bf16 matmul | bf16 weight → fake-quant(round→dequant) → bf16 matmul | bf16 matmul |
| Trainable params | LoRA + (optional) extras (projector, lm_head, …) | LoRA (typically) and/or full-param subset | LoRA + extras |
| Optimizer state precision | bf16 (LoRA only) | bf16 (LoRA / chosen params) | bf16 (LoRA + extras) |
| Training VRAM | **lowest** (~½ of bf16) | ≈ bf16 baseline | bf16 baseline |
| Output after `convert` / merge | "4-bit base + bf16 LoRA" or merged bf16 | real int4 weights + scales | merged bf16 |
| What it optimizes | training fits the box | deployed int4 stays accurate | upstream accuracy |
| Native deploy backend | bnb runtime (not iOS) | torchao + HF transformers; needs bridge for MLX | n/a (PTQ chooses later) |

The two columns do **opposite** things to the training graph. bnb
hardcodes a rounded base and adapts LoRA around it. torchao adapts
the base itself so that rounding it later hurts less.

## 2. Why "4-bit training" is two different sentences

> "bnb 4-bit lets some layers participate in fine-tuning in 4-bit form,
> and torchao QAT does the same thing in a cleaner way"

is the wrong unification. HF's own bnb docs are explicit: bnb's 4/8-bit
training only supports **training extra parameters** — LoRA / adapter
/ head — not the 4-bit base itself
(`huggingface.co/docs/bitsandbytes/reference/nn/linear4bit`). The
quantized base is frozen storage.

torchao QAT replaces target `nn.Linear`s with `FakeQuantizedLinear`,
which keeps the underlying `weight` as a normal bf16 `Parameter` and
applies `(fake_quant ∘ dequant)` in the forward pass. Backward flows
through the STE and updates the bf16 weight. After training,
`quantize_(model, QATConfig(step="convert"))` materializes the int4
weight + scales
(`docs.pytorch.org/ao/stable/workflows/qat.html`).

The qwen3 Unsloth notebook (`qwen3_(4b)_instruct_qat.py`) mixes both
worlds in a way that's easy to misread: it loads at bf16
(`load_in_4bit=False, load_in_8bit=False`), then sets
`qat_scheme="int4"` inside the LoRA config — that's torchao QAT, not
bnb. The post-train `quantize_(model, QATConfig(step="convert"))` is
torchao again. There is no bnb path in that notebook.

## 3. Empirical signal from our own data — three reads that matter

### 3.1 bnb-NF4 deploy: bf16-SFT and QLoRA-SFT tie within noise

From `../../quantization/A-baseline2-qlora-progress.md` and
`../../quantization/B1-sft-results.md`:

| Train | Deploy (PTQ) | n | PlantNet match |
|---|---|---|---|
| baseline-1 (bf16 SFT) | bnb-NF4 skip vt+ev | 300 | 69.33 % (208) |
| baseline-2 (bnb-NF4 QLoRA SFT) | bnb-NF4 skip vt+ev | 200 | 69.50 % (139) |
| baseline-1 (bf16 SFT) | bf16 reference | 2,870 | 70.63 % (2027) |

Δ between the two SFT recipes at the same NF4 PTQ recipe ≈ 0.2 pp,
inside the n≈200-300 noise floor. **Training against a 4-bit base did
not reduce the bnb-NF4 PTQ loss.** Both SFT recipes pay the same
~1.3 pp quantization cost vs bf16.

### 3.2 MLX-INT4 deploy: QLoRA-SFT degrades severely

| Train | Deploy (PTQ) | n | PlantNet match |
|---|---|---|---|
| baseline-1 (bf16 SFT) | `mlx_vlm.convert -q g64 affine` | 300 | **78.0 %** (M2) |
| baseline-1 (bf16 SFT) | `mlx_vlm.convert -q g128 affine` | 300 | **78.3 %** (M1) |
| baseline-2 (bnb-NF4 QLoRA SFT) | `mlx_vlm.convert -q g64 affine` | 200 | **22.5 %** |

Source: `../../quantization/B2-sft-results.md` rows M1/M2 vs
the baseline-2 MLX-INT4 row in
`../../quantization/A-baseline2-qlora-progress.md`.

Δ ≈ 55 pp at the same MLX-INT4 affine recipe. The split / n differ
(paper-grade test/ n=300 vs val/ n=200) so the absolute numbers are
not perfectly comparable, but the magnitude is far beyond any
sample-set variance (1σ at n=200 is ~3 pp).

**Read: training against a frozen NF4 base produced an adapter whose
visual features survive bnb-NF4 redeploy but do NOT survive MLX-INT4
affine redeploy.** The most natural mechanistic story is that the
LoRA adapted around NF4-rounded base features that have no equivalent
in the MLX-INT4 grid; an adapter trained against bf16 features
generalizes across PTQ schemes, an adapter trained against
NF4-rounded features generalizes only back to the same NF4 grid.

This is the user-observed effect: **bnb-trained models lose more
under (a different) quantization than bf16-trained models do.** The
direction is the opposite of what one might hope from "training-time
quantization awareness".

### 3.3 The vision tower is non-negotiable in every deploy

From `../../quantization/B1-bnb-nf4-vision-collapse.md`,
n=300 ablation on the bf16-SFT checkpoint:

| Variant | skip list | PlantNet n=300 |
|---|---|---|
| `bnb_nf4_skip_ev` (projector skip only) | embed_vision | 0.00 % |
| `bnb_nf4_skip_vt` (vision tower skip only) | vision_tower | 67.67 % |
| `bnb_nf4_skip_both` | vt + ev | 69.33 % |

The SigLIP vision tower **must** stay bf16. Skipping the projector
without the tower does nothing. Skipping the tower recovers to
GPTQ-level accuracy regardless of whether the projector is quantized.

This sets a hard module-level constraint that any "4-bit training"
variant has to respect.

## 4. Module-by-module decision matrix for Gemma 4 VLM

Combining §3.3, baseline-2's projector-tuning verification
(`../../quantization/A-baseline2-qlora-progress.md` § Source),
and the `tune_last_n_vision_layers` constraint from
`03-vision-mode.md`:

| Module | bnb 4-bit (frozen base storage) | torchao QAT (fake-quant forward) | bf16 (recommended default) |
|---|---|---|---|
| `vision_tower` (SigLIP encoder) | ❌ collapse to 0.1 % at deploy. Skip in any bnb config. | ❌ fragile; fake-quant noise compounds the undocumented-decoder problem from `03-vision-mode.md`. **Do not target.** | ✅ Mandatory. |
| `vision_tower.encoder.layers[-2:]` (last-2, when `tune_last_n_vision_layers=2`) | n/a (full-param, not LoRA) | ❌ same fragility argument. **Do not target.** | ✅ bf16 full-param, capped at N=2. |
| `embed_vision` (projector, 1.18 M params) | 🟡 OK to NF4 if vision tower is bf16 (no visible deploy hit). But projector is what `tune_projector: true` actively trains — usually loaded as bf16 differentiable via PEFT `modules_to_save`. | 🟡 Plausible target but small. The 1.18 M params are heavily tuned; QAT cost-benefit is dominated by language linears. **Optional include.** | ✅ Default. |
| `language_model.layers.*.{q,k,v,o,gate,up,down}_proj` | ✅ bnb's primary target. baseline-2 confirms LoRA-on-frozen-NF4 trains cleanly on these. | ✅ **Primary QAT target.** This is where group-affine int4 PTQ pays its rounding cost; QAT can recover some of it. | n/a (LoRA on top). |
| `lm_head` | ❌ Conventionally skip-quantize. Last-layer rounding tends to destabilize generation. | ❌ Skip. Same reason. | ✅ bf16. |
| `embed_tokens` | ❌ Skip in bnb (HF default skips embeddings). | ❌ Skip. Embeddings rarely benefit from fake-quant. | ✅ bf16. |

**Net rule.** Both 4-bit-training paths have an identical "do not
touch the vision side" boundary. torchao QAT and bnb QLoRA differ in
what they do to the language linears, not in what they're allowed to
do to the vision side.

## 5. Where each tool sits

| Tool | Route | Status |
|---|---|---|
| bnb-NF4 QLoRA SFT | A.1 — 4-bit SFT | done (`../../quantization/A-baseline2-qlora-progress.md`) |
| torchao QAT (cooldown or from-scratch) | A.2 — proposed | not started; needs §7 experiment first |
| bf16 LoRA SFT → GPTQModel PTQ | B.1 — PTQ via GPTQModel | done (R1/R2 in `../../quantization/B1-sft-results.md`) |
| bf16 LoRA SFT → mlx_vlm/mlx_lm PTQ | B.2 — PTQ via MLX | partial (`../../quantization/B2-sft-results.md`) |

Route A.1 (bnb QLoRA) is **kept as a training-cost optimization tool**,
not as a deploy-accuracy tool. The 3× train wall speedup is real and
useful when iterating on data / hparams; the deployed int4 accuracy
ranking is unchanged from bf16-SFT and possibly worse under PTQ
schemes other than bnb-NF4 itself.

Route A.2 (torchao QAT), if it pans out, is **the only training-side
intervention that targets deploy accuracy**. Every other training-side
tuning lever (`02`, `03`, prefix gate, KL/L2, data mix) only moves the
bf16 ceiling.

## 6. Honest skepticism on torchao QAT's magnitude

The mechanism is sound — adapting bf16 weights toward an
int4-rounding-robust region of weight space does reduce post-quant
loss. The empirical question is **how much** on our specific setup:

1. **PTQ already does most of the work at 4-bit weight-only.** GPTQ
   with `desc_act=True` recovers 0.4 pp over `desc_act=False` at zero
   inference cost (R1 vs R2 in `B1-sft-results.md`). The remaining
   1.8 pp gap to bf16 is what QAT can attack — that's the ceiling,
   not the floor.
2. **Public 4-bit weight-only QAT vs strong-PTQ deltas are typically
   sub-1 pp**, not 2-3 pp. LLM-QAT-style multi-pp wins are mostly
   at w4a8 (activation quantization), which we are not doing.
3. **An unknown fraction of our 1.8 pp deploy gap may come from the
   projector / vision side**, where QAT-on-language can't help (see
   §4 — vision side stays bf16 in every variant).
4. **The small-model term cuts both ways.** Gemma 4 E2B is small
   enough that QAT helps proportionally more than on 70 B-class
   models, but the SFT loss is already very low (0.152) so the
   model is already operating near its capacity — QAT noise during
   training may have less to fix because there isn't much slack.

Realistic prior on QAT cooldown vs R2 (68.8 %): **+0.3 to +1.0 pp**,
with ~20-30 % probability of being inside noise (no detectable
benefit) and small probability of mild regression on small-data
runs.

This is not a "transformative" outcome. It is potentially a
"close the last pp to bf16" outcome, which has value for the
deliverable but does not justify a large engineering investment up
front. See §7 for the right-sized first step.

## 7. Minimum decision experiment

The decision "do we invest in a Route A.2 spec + a torchao→MLX
bridge" should be gated on a half-day experiment, not on
literature alone.

### Setup

- **Base**: `_merged_bf16/` from baseline-1 (the SFT winner we already
  trust at 70.63 %).
- **Tool**: torchao directly, not via Unsloth's `qat_scheme`. Use
  `torchao.quantization.qat.QATConfig` and pass a `filter_fn` that
  matches **only** language-side `nn.Linear` modules:
  `q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj`
  under `language_model.layers.*`. Explicitly exclude
  `vision_tower.*`, `embed_vision`, `audio_tower.*`, `embed_audio`,
  `lm_head`, `embed_tokens`.
- **Optimizer**: `adamw_torch` or `adamw_torch_fused`. **Forbidden**:
  `adamw_8bit`, `paged_adamw_8bit`, `bnb_*` — see project policy.
- **Training**: 500-1,000 steps QAT-aware cooldown on PlantNet-50k
  subset, LR 5e-5, batch 16, grad-accum 1, single epoch fraction.
- **Convert**: `quantize_(model, QATConfig(step="convert"))` →
  torchao Int4WeightOnlyConfig output dir.
- **Bridge (sanity, lossy)**: dequant the torchao int4 back to bf16
  → feed through `mlx_vlm.convert -q --q-bits 4 --q-group-size 64
  --q-mode affine`. This is **lossy and represents the lower bound**
  of QAT's potential
  on the MLX deploy path; if QAT helps, this experiment will
  understate the help.
- **Eval**: PlantNet n=2,870 paper-grade, plus quick eval on the
  bnb-NF4 deploy path for cross-route comparison.

### Decision matrix

Compare R_QAT against R2 (GPTQModel w4g128 desc_act=True, 68.8 %):

| Δ (R_QAT − R2) | Verdict |
|---|---|
| ≥ +1.0 pp | QAT hypothesis confirmed. Invest in Route A.2 spec + Path 1 (real torchao→MLX bridge). |
| +0.3 to +1.0 pp | Marginal but cheap. Keep QAT cooldown as a Stage 3 deploy candidate. Skip from-scratch QAT (cost too high for the delta). Skip Path 1 bridge until B.1 bridge lands first. |
| −0.3 to +0.3 pp | Inside noise. Conclusion: GPTQModel + desc_act is near-optimal PTQ for our model scale. Drop torchao QAT from the roadmap. |
| < −0.3 pp | QAT regressed. Drop and document the empirical refutation. |

Important: the lossy Path-2 bridge means an inside-noise result is
**ambiguous** — could be "QAT didn't help" or "QAT helped but the
re-quantization through mlx_vlm.convert erased it". A Δ ≥ +1.0 pp
result, by contrast, is **unambiguous** in the positive direction.
This asymmetry is acceptable for a gating experiment because we
care about whether to invest more, not about the exact magnitude.

### Cost

- Cooldown training: ~1-2 h on 4090
- Convert + lossy bridge: ~10 min
- Eval (n=2,870 on 4090): ~1.5 h
- Total: **< 4 h GPU time**, no new bridge code

## 8. Open questions

These are intentionally not answered in this doc; they are the
follow-up work if §7 returns a positive signal.

### Q1 — module-level mix for Gemma 4 VLM

§4 is a priors-based recommendation, not an empirical one. The
specific question we'd want to settle if §7 succeeds: at QAT
convert time, is there value in keeping the **last few language
decoder layers** at bf16 (analogous to `tune_last_n_vision_layers`
but on the deploy side)? Apple's mlx-lm `dynamic_quant` does a
sensitivity-weighted version of this; torchao supports it via
per-module `filter_fn` exclusions.

### Q2 — bnb QLoRA adapter → torchao / GPTQModel / MLX export

If we keep using baseline-2-style QLoRA for iteration speed but
want to deploy via a non-bnb backend (MLX, GPTQModel, GGUF), the
adapter has to round-trip through bf16 merge first, and §3.2 says
the merged bf16 carries QLoRA-base-induced fragility into the
non-bnb PTQ. This may be why M2 = 22.5 %. Worth a controlled
ablation: same QLoRA adapter, three deploys (bnb-NF4 / GPTQModel /
MLX-INT4), and compare.

### Q3 — attribution of the deploy gap

The 1.8 pp bf16 → GPTQ-int4 gap currently has unknown
decomposition between:

- language-side rounding (QAT can attack this)
- projector rounding (QAT optional, projector is small)
- vision-side compounding via the language adapter's cross-modal
  reliance on the projector output distribution (QAT can't attack
  this directly)

A clean ablation would PTQ the language linears only / projector
only / both, then read PlantNet match. If language-only PTQ already
recovers most of the gap, QAT-on-language is the right intervention.
If projector PTQ dominates, QAT priorities shift.

## File pointers

| Concern | Path |
|---|---|
| Route A.1 status (bnb QLoRA baseline-2) | `../../quantization/A-baseline2-qlora-progress.md` |
| bnb-NF4 vision collapse evidence | `../../quantization/B1-bnb-nf4-vision-collapse.md` |
| bf16-SFT → MLX-INT4 deploy numbers (M0-M3, M6) | `../../quantization/B2-sft-results.md` |
| bf16-SFT → GPTQModel deploy numbers (R0-R2) | `../../quantization/B1-sft-results.md` |
| Project policy: no 8/4-bit mixing unless explicitly QLoRA-style | Public repo policy |
| qwen3 QAT notebook (reference, not for direct adoption) | `../../../qwen3_(4b)_instruct_qat.py` |

## External references

- HF bitsandbytes 4-bit Linear: <https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit>
- HF transformers bitsandbytes quantization: <https://huggingface.co/docs/transformers/quantization/bitsandbytes>
- HF transformers torchao quantization: <https://huggingface.co/docs/transformers/en/quantization/torchao>
- torchao QAT (prepare / convert workflow): <https://docs.pytorch.org/ao/stable/workflows/qat.html>
