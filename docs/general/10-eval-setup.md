# Evaluation Pipeline — `finetune/src/evaluate.py` + `quantization/src/eval/`

What the eval driver actually does, what metrics it computes, what the
default PlantNet subset looks like, and — importantly — why numbers
from different phases of the project are not directly comparable.

## ⚠️ Benchmark drift across phases (read this first)

The eval surface evolved during the 10-day sprint. **Numbers in
different docs were produced under different settings.** Some
combinations that look like apples-to-apples comparisons are not:

| Axis | What changed |
|---|---|
| Sample count `n` | 200 → 300 → 2870 across phases. The first quant sweep used n=200 for speed; the bf16 SFT baseline reports n=2870 (full curated val). |
| Val pool | Multiple iterations of the PlantNet 50k subset (v1 random, v2 LLaVA mix, v3 + offline_qa, v4 mix-50k variants). Per-bucket val splits land in `val/<source>.jsonl` after Track A v2; older single `val.jsonl` is not the same pool. |
| Per-species split | The pre-`6e49ec2` random tail in `prepare_plantnet.py` dropped 23 % of species (177/782) from val. Post-fix uses stratified split. Numbers reported before this fix overstate accuracy on the held-out species. |
| Prefix dispatch | v3 was source-keyed (`[task=*]`); v4 is modality-keyed (`[camera=on/off]`). A v4-trained checkpoint evaluated under v3 (or no) prompt shape falls back toward base behaviour. Eval-side wiring fixes landed in `300d52e` (PlantNet) and `e98ec18` (text-only generality). Numbers reported pre-fix for v4 checkpoints may not be directly comparable to post-fix numbers. |
| Loader backend | `hf_bf16` (MPS or CUDA), `hf_gptq`, `hf_gptq_hybrid`, `hf_bnb_nf4`, `mlx_vlm`, `mlx_lm`. Cross-backend numbers are not bit-exact; see [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md). |
| Metric methods | `species_match` (exact regex), `rouge_l`, WikiText PPL, and **LLM-as-judge** (Qwen2.5-VL-72B, structured JSON, scores accuracy/richness/hallucination) were introduced in different phases. Older runs do not have judge scores. |
| Subset identifiers (M0, M1, M2) | Track C η docs reference "M0 subset", "M1 subset", "M2 subset" for the per-quant-variant cross-checks. These are **specific 200-image slices** chosen to share calibration with the corresponding quant config; they are not interchangeable with the canonical n=300 / n=2870 PlantNet val. Doc-local context matters. |
| Camera-state in generality eval | `e98ec18` wired the v4 `[camera=off]` prefix into the text-only generality domains (plant_100 + mmlu_50 + aime_20). Pre-fix v4 numbers on those domains evaluated under a v3 prompt shape. |
| Refusal markers | `359069d` extended `REFUSAL_MARKERS` to catch the v4 model's actual idiom. Pre-fix refusal scores understate the model's correct refusals. |

**How to read a number in this repo**:

1. Look at the doc footer for `n`, the subset name (`val.jsonl` /
   `val/<source>.jsonl` / `M0` etc.), and the loader (`hf_bf16` /
   `mlx_vlm` etc.).
2. Look at the phase. Track C η+ numbers post-`f94697a` are the first
   ones where YAML config was actually loaded for GPTQ; earlier
   `desc_act` ablations are confounded with calibration-set drift.
3. Never aggregate numbers across phases without re-running under a
   single eval invocation.

This doc describes the **current** (May 18, 2026) eval surface.

## TL;DR

We have two entry points, both ending in the **same metric code** so
numbers from one invocation are internally comparable:

| Entry point | Use case | Loader options |
|---|---|---|
| `finetune/src/evaluate.py` | Eval a HF/PEFT adapter or merged HF model during training iteration | bf16 HF, BnB-NF4, unsloth FastModel |
| `quantization/scripts/run/eval.py` | Cross-backend eval of a finalized quantized model | `hf_bf16`, `hf_gptq` (triton / marlin / exllama_v2), `hf_gptq_hybrid`, `mlx_vlm`, `mlx_lm` |

The quant runner imports `rouge_l`, `extract_species`, and
`load_test_data` directly from `finetune/src/evaluate.py`, so
species-match scoring is **bit-identical** across both entry points.

Three metric families on the PlantNet split:

- **`species_match`** — exact string equality of
  `extract_species(pred)` and `extract_species(ref)`, both non-empty.
  The headline number; binary; strict.
- **`rouge_l`** — LCS-based F1 over whitespace-lowercased tokens.
  Soft similarity; less binary than `species_match`.
- **LLM-as-judge** (Qwen2.5-VL-72B, structured JSON) — accuracy,
  richness, hallucination per sample. Lets us numerically separate
  variants whose `species_match` is in the noise floor; non-trivial
  inference cost.

The default 300-sample subset (`--plantnet_n 300 --eval_seed 0`) is
deterministically chosen via `random.Random(0).sample(records, 300)`,
covers **218 of the 605** PlantNet species in the val pool, and uses
**12 paraphrased identification prompts** so the metric reflects
species recognition rather than memorizing one phrasing.

## File layout

```
finetune/src/
  evaluate.py             # 899 lines: HF+PEFT eval CLI + shared metric kernels
  data.py                 # build_vision_messages() — used by both entry points
  judge.py                # Qwen2.5-VL-72B LLM-as-judge

quantization/
  scripts/run/eval.py     # CLI wrapper: parses args, calls runner
  src/eval/
    runner.py             # benchmark registry, RunnerConfig, run_all()
    plantnet.py           # PlantNet benchmark — reuses finetune/src metrics
    wikitext_ppl.py       # text-only PPL benchmark
    vqav2.py              # VQAv2 dev-test benchmark
    model_loaders.py      # ModelHandle (uniform infer_text() interface)
                          # + concrete loaders: hf_bf16, hf_gptq, hf_gptq_hybrid,
                          #   hf_bnb_nf4, mlx_vlm, mlx_lm
```

## Metric kernels

Listing by function in `finetune/src/evaluate.py`:

| Function | Purpose |
|---|---|
| `_lcs_length(a, b)` | Space-optimized LCS DP over two token lists. |
| `rouge_l(prediction, reference)` | ROUGE-L F1: tokenize both via `.lower().split()`, compute LCS, return `2·P·R/(P+R)`. Whitespace tokenization, no nltk dependency. |
| `extract_species(text)` | Three-pattern regex cascade, returns lowercased best-match string or `""`. See "Species extraction" below. |

### Species extraction — the 3-pattern cascade

The exact-match metric depends entirely on how cleanly we can pull a
species name out of free-form model output. `extract_species` tries
patterns in this order and stops at the first hit:

1. **`_SPECIES_PHRASE_RE`** — trigger phrase + name + sentence terminator.
   - Trigger phrases match the templates used by `prepare_plantnet.py`
     and `prepare_plantnet_enriched.py`:
     `This is`, `That's`, `You're looking at`, `That looks like`,
     `Looks like`, `This appears to be`, `appears to be`,
     `looking at is`, `identified as`, `plant is`, `species of`,
     `specimen of`, `type of`, `You've spotted`,
     `Good eye[non-alpha]*this is`.
   - Capture group: non-greedy run of non-punctuation chars,
     optionally surrounded by `**...**` markdown bold.
   - Terminator lookahead: `.!?,\n` OR the literal `" to me"` OR end-of-string.
   - Case-insensitive trigger, case-sensitive capture (so Title-Case
     English vs Latin binomials can be distinguished downstream).

2. **`_ITALIC_BINOMIAL_RE`** — `*Genus species*` (single-star italics, optional author after).

3. **`_BOLD_SPECIES_RE`** — `**Title Case Name**`.

4. **Fallback** — first sentence (everything before the first `.`), lowercased.

Trade-off: the fallback **always returns something non-empty**, but
`species_match` additionally requires
`pred_species == ref_species AND pred_species != ""`, which keeps the
fallback from inflating the score when the model emits garbage like
pure `<pad>` spam.

## What the quantization runner adds

`quantization/src/eval/plantnet.py` is a thin re-implementation of
`evaluate_batch` that imports the same metric kernels but routes
generation through a `ModelHandle` interface so each loader plugs in
uniformly:

```python
from finetune.src.evaluate import (
    extract_species, load_test_data, rouge_l,
)
```

```python
records = load_test_data(str(config.val_jsonl), require_image=True)
if config.n_samples is not None and config.n_samples < len(records):
    rng = random.Random(config.seed)              # seed=0 by default
    records = rng.sample(records, config.n_samples)
```

The two important behaviors that differ from `evaluate.py`'s batch loop:

1. **Subsetting is seeded**. The full 5000-row jsonl is reduced to
   `--plantnet_n` rows via `random.Random(seed).sample(...)`, so
   `--eval_seed 0 --plantnet_n 300` always picks the same 300 rows in
   the same order on any platform / mlx version / backend.
2. **`ModelHandle.infer_text`** is the abstraction layer. Each loader
   in `quantization/src/eval/model_loaders.py` returns a `ModelHandle`
   with the same `infer_text(messages, image_path, max_new_tokens)`
   signature, so swapping `--loader mlx_vlm` ↔ `--loader hf_bf16`
   ↔ `--loader hf_gptq` requires no eval-side code changes.

Metric computation is byte-for-byte the same as `evaluate.py`.

## The 300-sample PlantNet subset (default eval slice)

This is what `--plantnet_n 300 --eval_seed 0` evaluates on, against
the curated `val_mac.jsonl` (5000 rows).

### Selection

```python
rng = random.Random(0)                     # Mersenne Twister, fixed seed
selected = rng.sample(all_records, 300)    # bit-exact across platforms
```

Empirically verified: sample 0 in every `eval.json` produced (Mac,
Linux broken, Linux fixed) is the same image, ref species "bush
lupine".

### Species coverage

| Dimension | 300-sample subset | Full 5000-row val pool |
|---|---|---|
| Unique PlantNet sids (species class folders) | **218** | 605 |
| Unique `ref_species` extracted from assistant turns | 216 | — |
| Samples per sid: min / median / max | 1 / 1 / 5 | 1 / 7 / 28 |

Per-sample-count breakdown in the 300:

| Samples per species (sid) | # of sids |
|---|---|
| 1 | **158 (72.5 %)** |
| 2 | 43 |
| 3 | 13 |
| 4 | 3 |
| 5 | 1 |

So 72 % of the 218 species in the subset are represented by **a single
image**. The 300-sample slice is **much more even than the underlying
5000-row pool** (which has median 7 samples per sid and a max of 28),
because random sampling at low density disperses across species.

Implications for `species_match`:

- The aggregate is closer to a **macro / per-species accuracy** than
  a per-image accuracy.
- Eval results are directly comparable to PlantNet-paper-style
  macro metrics, not to the head-heavy raw test-set distribution.
- One bonus side-effect of "1 sample per species" structure: when
  CUDA jitter flips a `species_match` hit/miss on Linux, the
  aggregate moves by 1/300 ≈ 0.33 pp — exactly the run-to-run
  variance documented in [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md).

### Prompt diversity

The `user` turn is one of **12 paraphrased identification prompts**,
roughly uniformly distributed:

| # | Prompt | Count |
|---|---|---|
| 1 | "What plant is this?" | 36 |
| 2 | "I saw this growing near the trail. Any idea what it is?" | 29 |
| 3 | "Describe this plant." | 29 |
| 4 | "What's the name of this plant?" | 29 |
| 5 | "Can you identify this species?" | 28 |
| 6 | "Help me identify this — is it a common species?" | 25 |
| 7 | "I found this on the trail — what is it?" | 25 |
| 8 | "Can you tell me about this plant I just spotted?" | 24 |
| 9 | "What species is this plant?" | 23 |
| 10 | "What am I looking at?" | 20 |
| 11 | "I'm curious about this plant. What can you tell me?" | 17 |
| 12 | "Do you know what kind of plant this is?" | 15 |

This is intentional — it tests whether the model's species-ID skill
generalizes across question phrasings.

### Reference answer shape

The assistant reference follows the `prepare_plantnet.py`
ANSWER_TEMPLATES + enrichment:

- **First sentence** uses one of the trigger phrases that
  `_SPECIES_PHRASE_RE` matches.
- **Following sentences** add Wikipedia-style enrichment (Latin
  binomial, family, range, common synonyms).

Reference-answer length distribution (chars):

| min | median | mean | max |
|---|---|---|---|
| 48 | 162 | 170 | 341 |

`species_match` only looks at the first sentence; `rouge_l` covers
the full enrichment text.

## How the metrics behave in practice

### species_match — binary, strict

`pred_species == ref_species AND pred_species != ""` after both go
through `extract_species`. Common failure modes:

| Failure | Cause | Counts as |
|---|---|---|
| Wrong species name | Genuine model error | miss |
| Pad-spam corrupting the trigger sentence | mlx-cuda QMM bug (see [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)) | miss |
| Model uses scientific name instead of common (or vice versa) | Template mismatch | miss |
| Truncated common name (`"common st"` instead of `"common stonecrop"`) | max_tokens hit mid-name | miss |
| Different valid common name for same species (e.g. "large-leaved lupine" vs "bush lupine") | string equality only — no synonym handling | miss |

The last row is the biggest honestly-good-but-marked-miss class —
this is why `species_match` is typically ~40-50 % while `rouge_l` is
~0.40-0.60 for the same model. The LLM-as-judge pipeline addresses
this by scoring **accuracy** separately from string match.

### rouge_l — soft, gradient-friendly

LCS-based F1 over whitespace-lowercased tokens. Tracks "how much of
the reference vocabulary the prediction recovered, in order." Useful
as a continuous secondary signal; sensitive to padding tokens,
word-order changes, and length mismatch.

### LLM-as-judge — Qwen2.5-VL-72B

Default judge in `finetune/src/judge.py`. Device_map=auto across
available GPUs. Per-sample structured JSON output:

```json
{"accuracy": 0.85, "richness": 0.70, "hallucination": 0.10}
```

Includes a retry path and a fallback regex extractor for malformed
JSON. Becomes essential as remaining quality deltas shrink to the
noise floor of exact-string comparison.

## v4 camera-state prefix integration (May 17-18)

Track B v4 replaced the v3 source-keyed task-tag prefix dispatch
(`[task=plantnet]` / `[task=refuse]` keyed on `record.source`) with a
modality-state prefix gate (`[camera=on]` / `[camera=off]` keyed on
`record["image"]` truthiness). The dispatch is a property of the
request, not its topic — an image record gets `[camera=on]` regardless
of whether the user is asking about the plant in the photo or about
the weather.

**Eval-side wiring** (two follow-up commits):

| Commit | Fix |
|---|---|
| `300d52e` | `quantization/src/eval/plantnet.py` — wire v4 camera-state prefix into PlantNet eval (image records get `[camera=on]`, text-only get `[camera=off]`) |
| `e98ec18` | `finetune/src/evaluate.py` — wire v4 `[camera=off]` gate into text-only generality domains (mmlu_50, aime_20) |

**Why this matters for numerical comparison**: a v4-trained model
evaluated under v3 prompt shape (or no prompt shape) sees a different
input distribution from training and falls back toward base behavior.
Numbers reported pre-`300d52e` for v4 checkpoints on PlantNet (and
pre-`e98ec18` for v4 on generality eval) **are not apples-to-apples**
with the same checkpoints evaluated post-fix.

The on-device side (`GemmaService.swift`, commit `f468523`) emits the
same prefix at iOS inference time — see [`05-rag-runtime.md`](05-rag-runtime.md).

### Per-record `prefix_key` override

`26790ca` added per-record override: a JSONL record may carry a
`prefix_key` field that overrides the default image-presence dispatch.
Future-proofs the format for multi-axis tags
(e.g. `camera_on_plant_true`) without extending the dispatcher. Empty
/ missing field → keeps v4 image-presence semantics —
backward-compatible.

## EoRA eval helpers

EoRA (`2c1ce0b`) produces a `.lora_a/.lora_b` safetensors adapter
that, when applied, recovers +4.3 pp on M2 g64-affine (83.7 % → 88.0 %).
Three eval
helpers wire this into the standard eval surface:

| Script | Purpose |
|---|---|
| `quantization/scripts/run/eora_post_quant.py` | One-shot: calibrate EoRA on a quantized model (128×512 WikiText, ~28 min M5 Pro) → save adapter |
| `quantization/scripts/run/eval_eora_only.py` | Eval a quantized base model + EoRA adapter, applied via `EoRALinear` wrapper. Skip the merge step for fast A/B. |
| `quantization/scripts/run/merge_eora_into_mlx.py` | Fold EoRA into the deploy checkpoint — output is directly `mlx_vlm.load`-able. Used when shipping to iOS bundle. |

`58ef09b` fixed an off-by-one in the EoRA merge step;
`dd7ae11` added `apply_adapters_from_file` for saved-adapter
re-evaluation.

## Hybrid loader (`hf_gptq_hybrid`)

`9ef88d4`'s GPTQ + torchao hybrid produces a CUDA artifact at ~2.77 GB
(under the iOS jetsam ceiling, but not directly MLX-loadable). The
`hf_gptq_hybrid` eval loader handles the packed embedding table via
`PackedWeightProxy` for `embed_tokens.weight` access. Used for
CUDA-side quality validation of the hybrid pack.

## Common eval commands

### Quantized variant on Mac (authoritative)

```bash
python -m scripts.run.eval \
  --variant mlx_vlm_g128 \
  --loader mlx_vlm \
  --model_dir results/mlx_vlm_g128 \
  --plantnet_val_jsonl finetune/data/english-desc-v2/val.jsonl \
  --plantnet_n 300 --eval_seed 0 \
  --benchmarks plantnet_val \
  --output_dir results/mlx_vlm_g128
```

### Same model on Linux — requires from-source mlx + env

```bash
source quantization/scripts/_env/_mlx_env.sh
CUDA_VISIBLE_DEVICES=0 $MLX_PYTHON -m scripts.run.eval \
  --variant mlx_vlm_g128_linux \
  --loader mlx_vlm \
  --model_dir results/mlx_vlm_g128 \
  --plantnet_val_jsonl finetune/data/english-desc-v2/val_mac.jsonl \
  --plantnet_n 300 --eval_seed 0 \
  --benchmarks plantnet_val \
  --output_dir results/mlx_vlm_g128_linux
```

(See [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)
for why Linux needs the from-source mlx build and the `_mlx_env.sh`
wrapper.)

### bf16 reference via HF transformers

```bash
python -m scripts.run.eval \
  --variant bf16_reference \
  --loader hf_bf16 \
  --model_dir exports/sft-50k-fullproj/merged-bf16 \
  --plantnet_val_jsonl finetune/data/english-desc-v2/val.jsonl \
  --plantnet_n 300 --eval_seed 0 \
  --benchmarks plantnet_val wikitext_ppl \
  --output_dir results/bf16_reference
```

### HF/PEFT during training iteration

```bash
python -m finetune.src.evaluate \
  --base_model google/gemma-4-e2b-it \
  --adapter_path outputs/hike-gemma4-lora \
  --test_file finetune/data/english-desc-v2/val.jsonl \
  --output_file results/iter_eval.json
```

## Cross-references

- Cross-platform numerical contract:
  [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)
- KV-shared parity audit (which adapters are safe to reuse across
  `transformers` versions):
  [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md)
- Tested package versions where the v5.8 fix lands:
  [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md)
- Where eval numbers feed into model decisions:
  [`../quantization/B2-sft-results.md`](../quantization/B2-sft-results.md),
  [`../quantization/B1-sft-results.md`](../quantization/B1-sft-results.md),
  [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
