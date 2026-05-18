# Postmortems

Concise rollup of dated debugging sessions from the May 2026 sprint.
Each entry: **what fired, root cause, fix, lesson**. Originals are
longer; this doc keeps only the results and the tripwires that catch
the same class of bug next time.

Reading order roughly matches the order in which we hit the bugs.

---

## 1. Silent PEFT loading on `transformers 5.5 → 5.8` (2026-05-12)

**Symptom.** A LoRA-finetune of Gemma 4 E2B trained to apparent
convergence — trainer-reported `loss ≈ 1e-4`, `mean_token_accuracy =
1.0` over 30 epochs on a 20-100 sample overfit set. PlantNet
post-train evaluation stayed at **0 %**. In-memory teacher-forced loss
on the freshly-trained PEFT model gave per-sample CE ≈ 1e-4 (correct).
After `save_pretrained` → `from_pretrained` round-trip, per-sample CE
jumped to ~6 (base model).

**Triage that mattered.** Boiling the task down to image → single
integer label (one-token target, identical prompt) and running it
through two completely independent training stacks (unsloth +
`SFTTrainer` AND pure HF + PEFT + TRL) reproduced the same failure
under both. **The bug was not unsloth-specific.**

A diagnostic ladder isolated the failure point: in-memory PEFT model
was correct in `train()` and `eval()` modes, after `save_pretrained`,
and after `merge_and_unload`. The reload from disk killed it. The 490
LoRA tensors written to disk were byte-equal to their in-memory
values. And yet the reloaded model behaved like the base.

**Root cause.** `transformers 5.8.0` restructured Gemma 4 E2B's hybrid
attention. Layers 0-14 use global attention with standalone `q/k/v/o_proj`
modules; layers 15-34 use sliding-window attention that **does not
expose standalone `k_proj` / `v_proj` modules**. The previous
`transformers` exposed `k_proj` / `v_proj` on every layer (dead, but
present).

The adapter was trained on the old layout → PEFT wrapped 245 modules
(490 LoRA tensors). On reload with the new layout, PEFT's
`from_pretrained` found only 205 matching modules and **silently
dropped the remaining 80**. No warning, no error. The model lost
16 % of its LoRA adaptations and reverted to base behavior.

**Fix.** Upgrade `transformers` to ≥ 5.8 and **retrain the adapter**.
Adapters trained under 5.5 are not compatible with the 5.8 model
layout. After retraining on 5.8: reload tensor inventory matches
saves, TF loss after reload matches in-memory, generation produces
the trained tokens.

**Tripwires added** (`4fab396`):

1. `src/save_reload_check.py` — comparison primitives that diff
   in-memory PEFT state against `adapter_model.safetensors` bytewise.
2. **In-pipeline tripwire** in `finetune.py` after every
   `save_pretrained` — runs the diff and fails the training run if
   any tensor is missing.
3. `scripts/smoke_save_reload.py` — heavyweight pre-finetune ritual.
   Verified on a 4090: **411 savable tensors, 0 orphans, 0 value
   mismatches, logits max delta 0.00e+00.**

**Co-discovered bugs the same day.** While the same author was on the
hot trail, three independent issues surfaced and got fixed in the
debug-marathon's tail:

| | Bug | Fix |
|---|---|---|
| 6ee0112 | `adamw_8bit` silently selected for projector/vision param groups despite YAML specifying `adamw_torch`. Hardcoded dispatcher in `finetune.py` bypassed config. | All configs → `adamw_torch_fused`. Config validator rejects every `*8bit*` / `*4bit*` / `paged_*` optim name string. |
| beb30cd | Training-arg modernization: `warmup_ratio`, `group_by_length` (new `train_sampling_strategy` API for transformers 5.x), `tf32`, `save_total_limit`. | Resolved warmup_ratio → `warmup_steps = ceil(ratio × total_steps)`; translated `group_by_length: true` → `train_sampling_strategy="group_by_length"` + auto-populated `length` column. |
| dc7b1ba | Eval/train parity bugs: (a) eval used different chat template, (b) eval used `AutoModelForCausalLM` (drops `vision_tower`) instead of `AutoModelForImageTextToText`, (c) `max_new_tokens` not threaded into generate, (d) eval included text-only records that training filtered out, (e) eval collapsed multi-turn to last pair while training preserved full conversation, (f) `warmup_ratio` used `round()` but transformers 5.x uses `ceil()`, (g) `tf32=True` forwarded on pre-Ampere (rejected by transformers 5.x). | All eight fixed; 381 tests pass. |

**Validation** (afternoon, post-merge): 5 overfit100 configs retrained
under the fixed pipeline:

| Config | LoRA r | Extra modules | Species Match | ROUGE-L |
|---|---|---|---|---|
| lora-r8-a8-lr2e4 | 8 | none | 0 % | 0.073 |
| lora-r256-a8-lr2e4 | 256 | none | 93 % | 0.898 |
| r256+fullproj-lr5e5 | 256 | projector | **100 %** | 0.9998 |
| r256+fullproj+vision2-lr1e5 | 256 | proj + vision last 2 | 96 % | 0.960 |
| classify-r8-a8-lr2e4 | 8 | none (classify) | 72 % | 0.720 |

A concurrent plantnet-50k run (r=8, bf16, full 50k dataset) reached
**46 % exact binomial match** on 2870 val samples, confirming the
pipeline works on production-scale data.

**Cross-platform check.** [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md)
later confirmed `mlx_vlm` and `mlx-swift-lm` both match `transformers
5.8` semantics. Old 5.5 adapters loaded on iPhone behave the same way
they behave on `transformers 5.8` — dead LoRA tensors land but never
fire, net 16 % capacity loss.

**Lesson.** In a fast-moving model ecosystem, a minor `transformers`
version bump can silently invalidate every adapter trained under the
prior version. The tripwire isn't optional. The Trailogy writeup's
Technical Challenge #1 is this bug.

---

## 2. MLX quantization debug — `mlx_lm` Gemma 4 forward pass is the wrong tree (2026-05-14)

**Symptom.** Sweeping the four `mlx_lm.quant.*` methods on Gemma 4
E2B:

| Variant | Outcome |
|---|---|
| `dynamic_quant_bpw4` | converts cleanly (~85 min), coherent text generation in `mlx_lm` |
| `gptq_w4g64` (multiple configs) | **NaN logits**, argmax → `<pad>` token |
| `awq_w4g64` | `AWQ_MODEL_CONFIGS` has no `gemma4` entry — fails before convert |
| `dwq_w4g64` | `mx.take_along_axis` broadcast error `(2,576) vs (2,288)` |

Initial conclusion: **3 of 4 methods broken on Gemma 4**, attributed to
upstream algorithmic bugs in mlx-lm.

**The deeper bug.** Trying to deploy the one working `dynamic_quant`
output to iOS via `mlx_vlm` exposed four architectural mismatches in
sequence:

1. `per_layer_model_projection` quantized by `dynamic_quant` (mlx_lm
   models it as `nn.Linear`); `mlx_vlm` models it as `ScaledLinear`
   which has no `to_quantized` method. Load fails with
   `ValueError: Unable to quantize model of type ... ScaledLinear`.
2. `mlx_lm`'s `gemma4_text.sanitize` drops `k_proj` / `v_proj` /
   `k_norm` for KV-shared layers (15-34); `mlx_vlm`'s `gemma4.language`
   allocates them unconditionally. Load fails with `Missing weights`.
3. `mlx_vlm.utils.load_model:230` runs
   `config.setdefault("audio_config", {})`. Even if the merged config
   has no audio block, an empty `{}` triggers audio-tower allocation.
4. After patching the above, model **loads** — but generation produces
   `<pad>` with `logprobs = -12.5` (uniform over the vocab). **Root
   cause**: `mlx_lm`'s `Gemma4TextModel` uses `nn.RMSNorm` (with `+1`
   weight offset); `mlx_vlm` uses `RMSNormZeroShift` (no offset). Same
   tensors, different math. The norm-layer mismatch corrupts the
   hidden state.

**Conclusion.** The bug is **not in the quant algorithms**. It is in
the **forward-pass tree** that `mlx_lm.quant.*.main()` walks — it
loads via `mlx_lm.load`, which builds the wrong Gemma 4 tree for our
deploy stack.

Stack-level mental model:

```
Library         Gemma 4 tree                    Quant primitive       Deployable?
--------------- ------------------------------- --------------------- -----------
mlx_lm          gemma4.py + gemma4_text.py      mlx_lm.quant.*        NO
                (nn.RMSNorm, nn.Linear,                                 (iOS rejects)
                 sanitize drops KV-shared K/V)

mlx_vlm         gemma4/{gemma4,language,...}    Internally calls      YES
                (RMSNormZeroShift, ScaledLinear, mlx_lm.utils.          (iOS accepts)
                 KV-shared layers allocated)     quantize_model
```

`mlx_vlm` doesn't ship its own quantization algorithm —
`mlx_vlm/convert.py` imports `mlx_lm.utils.quantize_model` directly.
The bit-packing kernel is identical. What differs is **which model
tree gets walked**.

**Fix.** The **hybrid flow**, landed as
`quantization/scripts/run/mlx_hybrid_quant.py` (`da6af9f`):

```python
from mlx_vlm import load
model, _ = load(bf16_dir)              # mlx_vlm's tree (correct)
from mlx_lm.quant.gptq import gptq_quantize
gptq_quantize(model.language_model, calib_data, bits=4, group_size=64, ...)
from mlx_vlm.utils import save_weights, save_config
save_weights(out, model); save_config(model.config, out / "config.json")
```

Plus two prep helpers:

- `prep_inject_kv_shared.py` — repairs `transformers 5.8` KV-shared
  drops; without it `mlx_vlm.load` fails with `Missing 60 parameters`.
- `recover_per_key_qconfig.py` — reconstructs missing config dict by
  reading safetensors shape arithmetic.

**Easier path discovered same day.** `mlx_vlm.convert` already exposes
`--quant-predicate {mixed_2_6, mixed_3_4, ..., mixed_4_8}`, which
implements the same mixed-precision policy `dynamic_quant` was
discovering — **inside the right model class**. For deployable
artifacts, prefer `mlx_vlm.convert -q --quant-predicate mixed_3_4`
over any hybrid flow.

**Implications for prior conclusions.** Every "method X failed on
Gemma 4" verdict from the 2026-05-13 round needs re-testing under the
hybrid flow before being treated as an algorithm fault. NaN-logits
GPTQ, DWQ broadcast crash, dynamic_quant deploy-collapse — all very
likely **forward-pass bugs of the wrong Gemma 4 implementation**, not
**quantizer bugs**.

**Lesson.** Apple's `mlx_lm` is **baseline-grade** for Gemma 4 — the
same way their model class is a baseline reference, not a production
implementation. Community / VLM forks carry the production-grade
behavior. When debugging a quantization method on Gemma 4, the very
first triage question is "which forward-pass tree did `quantize_model`
walk?".

---

## 3. mlx-cuda on Linux — pypi 0.31.1 wheel is broken for Gemma 4 INT4 (2026-05-15)

**Symptom.** Same Mac-quantized M1 INT4 model rsynced to a Linux +
4090 box, run via `mlx-cuda-12==0.31.1` (the latest pypi wheel): 9-13
correct tokens, then `<pad>` spam to `max_tokens`. `species_match`
collapses from Mac's 49.7 % (n=300) to **7.7 %**.

Triage on the 4090, same model bytes:

| Test | Output | Verdict |
|---|---|---|
| Quant M1 + image | 9-13 correct tokens → pad-spam | bug |
| Quant M1 + text-only short prompt | 135 chars coherent | bug masked |
| bf16 base + text-only | 532 chars coherent | bf16 OK |
| bf16 base + image | "please provide an image" (template issue) | bf16 generation OK |

bf16 generation works → Gemma 4 itself is fine on mlx-cuda. The bug
needs **(quantized weights) AND (long-enough K-dim that vision tokens
land past a tile boundary)**.

**Root cause.** Missing QMM (quantized matmul) kernel fixes in the
pypi `mlx-cuda-12==0.31.1` wheel. Apple shipped `mlx-metal` 0.31.2 on
2026-04-22 (with PRs #3255 / #3268 / #3321 / #3352 / #3417 landed) but
never published a matching CUDA wheel. Additional QMM CUDA fixes
landed on `main` after `v0.31.2`: most suggestively **#3509 "Guard
qmm_naive scale and bias loads at tile boundaries"** — which reads
exactly like our pad-spam symptom.

**Fix.** Build mlx `main` HEAD from source with CUDA 12.9 toolchain
from conda-forge. Full recipe + verification numbers in
[`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md).

**Result.** Same M1 model, same val_mac.jsonl, n=300:

| Build | species_match |
|---|---|
| Mac mlx-metal 0.31.2 | **49.7 %** |
| Linux 4090 mlx-cuda-12 0.31.1 (broken) | 7.7 % |
| Linux 4090 mlx 0.32.0.dev+main (fixed) | **40.6 %** (3-run mean, ±0.33 pp) |

**Caveat.** The source build only **partially** fixes pad-spam: ~48 %
of responses still pad-spam stochastically across runs. But aggregates
are stable, so Linux is usable for **within-machine sweep
comparisons**; Mac remains the authoritative reporting backend. See
[`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md) for
the full numerical contract (deterministic dataset selection,
non-deterministic CUDA generation, and how to read a Linux ↔ Mac gap).

**Lesson.** Where the pypi wheel doesn't track upstream, the
"convenient" version may be broken in user-invisible ways. The
deviation surfaces only when you take a known-good artifact from the
authoritative backend and run it under the lagging one. **Cross-backend
parity testing is not a hypothetical — it's how you find this class of
bug.**

---

## 4. Train/eval data mismatch — stale `val.jsonl` + non-portable image discovery (2026-05-15 → 2026-05-16)

This was three findings stacked, each invalidating earlier-reported
numbers in a different way.

### Finding A — 4090 had a stale `val.jsonl`

The Mac side ran the canonical `prepare_plantnet_50k.sh`; the 4090
side ran the inner `prepare_plantnet_enriched.py` by hand at some
point and never re-ran the wrapper. Result:

| Side | File | Rows | Species | Image paths |
|---|---|---|---|---|
| Mac (canonical) | `test.jsonl` | 29,880 | 782 | `images_resized/test/` (pre-stretched) |
| 4090 (stale) | `val.jsonl` | 3,090 | 932 | `PlantNet-300K-data-v2/test/` (raw) |

Two divergences: species filter (932 vs 782 — older looser filter) and
image paths (raw vs pre-resized). Since
`random.Random(0).sample(records, 300)` is deterministic in its input
list, the 300-sample subsets on Mac vs 4090 don't match even with the
same seed.

**Sub-cause** — `prepare_plantnet_enriched.py` writes to
`train.jsonl` + `val.jsonl` keyed on a loop label, NOT on `--split`.
So `--split test --val_ratio 0.1` produces `val.jsonl ≈ 10 % of
filtered test images`, not "all test images". The canonical wrapper
uses `--val_ratio 1.0` and then promotes the staged val → test.jsonl.

**Fix.** Always run the wrapper. Don't call the inner script by hand.

### Finding B — `discover_images` cross-platform non-reproducibility

After re-running the canonical script on the 4090, the resulting
`test.jsonl` **still** had a different `n=300 seed=0` sample than the
Mac's. Species IDs matched, but specific image hashes per species
differed.

**Root cause.** `prepare_plantnet.py:discover_images` sorted species
directories but not the images within each species. `Path.iterdir()`
returns entries in filesystem-native order — APFS (Mac) and ext4
(Linux) produce different orders. Same script, two filesystems → two
different JSONLs.

**Fix.** Single `sorted(...)` wrap on the inner iterator
(lexicographic ordering, FS-independent):

```python
for img in sorted(species_dir.iterdir()):  # ← lexicographic
    ...
```

### Finding C — `val.jsonl` ≠ `test.jsonl` at n=300 (structural)

Even after both files are canonical and byte-identical across
machines, `val.jsonl` and `test.jsonl` differ in species coverage and
per-species image distribution badly enough that the **same SFT
checkpoint scores ~16-29 pp differently** on each at n=300.

| File | Rows | Species | Per-species (min/median/mean/max) | n=300 top-50 head representation |
|---|---|---|---|---|
| `val.jsonl` | 5,000 | 605 | 1 / 7 / 8.3 / 28 | 20 % |
| `test.jsonl` | 29,880 | **782** | 1 / 5 / 38.2 / **901** | **66 %** |

So an n=300 quick-eval on `test.jsonl` is two-thirds dominated by the
50 head-of-tail species (which the SFT adapter fit most cleanly); on
`val.jsonl` it's a much more uniform draw across 218 species.

**Bonus structural bug**: val was missing **177 / 782 species** that
the training set saw. The pre-fix `samples[:val_count]` random tail of
the stratified pool systematically dropped sparser classes.

**Fix.** `class_stratified_split` in `prepare_plantnet.py` — for each
species, deterministically partition first
`max(1, floor(N · val_ratio))` to val, rest to train. Species set in
val is now **identical** to species set in train.

**Lesson.** *Same metric* + *same model* + *different eval file* =
wildly different numbers. Cross-file comparisons need a "which file?"
column. **Stage 1 / Stage 2 SFT sweeps use `val.jsonl`** (in-distribution,
broad coverage — sensitive to long-tail behavior). **B1 / B2
quantization sweeps use `test.jsonl`** (paper-grade, head-of-tail
biased — closer to the PlantNet paper's reported ceiling). Both are
valid; mixing them is the trap.

---

## 5. Eval-pipeline silent failures — adapter, sampling, paths (2026-05-17)

**Symptom.** The cloud_sweep generality eval
(`cloud_sweep_v*_sweep.sh`) had three independent silent failure
modes that compounded into a single observation: **every**
`generality_*.json` reported the **base** model's numbers
(`mmlu_acc ≈ 0.46`, `aime_acc ≈ 0.10`, `plant_match = 0.00`)
regardless of which adapter, sweep, or hyperparameter was nominally
being scored.

Surfaced via a code-review session noticing five completed sweeps had
triple-matched the base-model fingerprint documented in the wrapper's
own comments.

### Finding A — `--adapter_path` silently dropped on `hf_bf16` path

The registry loader for `hf_bf16` accepts only `model_dir /
base_model_for_processor`. It has no concept of a PEFT adapter. So
`args.adapter_path` was parsed by argparse and dropped on the floor.
PEFT-merging code did exist on the `_load_hf_direct` fallback branch
but cloud_sweep always specified `--loader hf_bf16`, so the working
branch was never exercised.

**No error**: the runner produced a valid `Gemma4ForConditionalGeneration`
from the base repo. It evaluated fine, just on a model that had not
seen any SFT updates. Plant species match degraded to 0 (base never
emits the trained lead phrase); MMLU and AIME matched the base
distribution exactly.

**Fix.** After the registry loader returns its handle, apply PEFT
explicitly + assert no `lora_*` params survive merge:

```python
handle = loader_fn(args.base_model)
if getattr(args, "adapter_path", None):
    from peft import PeftModel
    peft_wrapped = PeftModel.from_pretrained(handle.model, args.adapter_path)
    merged = peft_wrapped.merge_and_unload()
    merged.eval()
    leftover = [n for n, _ in merged.named_parameters() if "lora_" in n.lower()]
    assert not leftover, f"merge_and_unload left {len(leftover)} lora_* params behind"
    handle.model = merged
```

**Tripwire** in `sweep_eval_all.sh` — refuses wandb push if every
freshly-produced eval JSON matches the base fingerprint to 3 decimals.

### Finding B — non-deterministic generation

`finetune/src/evaluate.py:386` (pre-patch) called
`model.generate(**inputs, max_new_tokens=...)` with no `do_sample=False`.
Inherits Gemma 4's `generation_config.json` default which has
`do_sample=True, temperature=1.0, top_p=0.95`. Two consecutive evals
on the same checkpoint produced ~5 pp jitter on `plant_100`.

The MLX path already set `temperature=0.0`; the `_generate_hf` path
in `evaluate_generality.py` already passed `do_sample=False`. But
`hf_bf16` routed through `generate_response` which had neither.

**Fix.** Force greedy beam-1 with explicit overrides that defeat any
`generation_config.json` injection:

```python
output_ids = model.generate(
    **inputs, max_new_tokens=max_new_tokens,
    do_sample=False, num_beams=1,
    temperature=1.0, top_p=1.0, top_k=0,
)
```

Plus `_set_eval_determinism(seed)` at the top of `main()` that pins
`random` / `numpy` / `torch.manual_seed`, sets
`cudnn.deterministic=True`, and warns if `CUBLAS_WORKSPACE_CONFIG` is
unset before cuda init (bash wrappers now set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before invoking python).

### Finding C — `plant_100.jsonl` was non-portable by design

`image` fields shipped with absolute paths to one developer's home
directory. The wrapper auto-built a parallel `_eval_dir/` per run as a
workaround — fragile, easy to skip, and required machine-specific path
references in shared code.

**Fix.** One-time JSONL rewrite to relative `<species_id>/<hash>.jpg`,
plus `--plant_image_root` (env: `PLANT_IMAGE_ROOT`) resolver that
joins each relative path against the root and **raises** on first
missing file. Silent skip would shrink the eval set and bias the
metric — that was the original sin and we don't want to replicate it.

### Common pattern

All three bugs share the same shape:

1. Caller passes data X.
2. Receiver doesn't use X.
3. No error.
4. Output is structurally valid.
5. Output is semantically wrong.
6. Downstream system happily consumes (wandb push, summary tables,
   cross-config comparisons) and pollutes the comparison.

Defense in each case is a **tripwire that asserts an invariant the
receiver SHOULD satisfy**, placed at the boundary where data X
crosses the silent-default surface.

**Lesson.** Argparse args that flow through optional-load paths must
have a reception tripwire — `--adapter_path` exists in argparse,
therefore there must be a code path or assertion that proves the
adapter actually got applied before the first generation. Generation
in eval is deterministic by default; `generation_config.json` belongs
to *deployment*, not to *measurement*. Eval data files are content-
addressable and portable, never absolute paths.

Same lesson as Postmortem #4's `discover_images` non-determinism bug —
silent defaults compounding into metric differences. Both are now
defended by per-call tripwires.

---

## 6. HF Trainer LR-schedule on `resume_from_checkpoint` is NOT broken (2026-05-17)

A doc misdiagnosis worth recording so it doesn't repeat.

**The claim** (in `_HANDOFF_STATUS.md`, since deleted): *"Locked to
step 4000 by HF Trainer's resume behavior — `state.max_steps` loaded
from `trainer_state.json` overrides the new `--num_train_epochs 3` CLI
arg."*

**Reality.** `transformers 5.8`'s `Trainer.train(resume_from_checkpoint=...)`
correctly continues the LR schedule when the training plan is
lengthened or shortened. The LR scheduler is **not** locked.

**Source check** (`trainer.py:1538-1554`):

```python
self.state.compute_steps(self.args, max_steps)
self.state = TrainerState.load_from_json(...)   # checkpoint overrides state
self.state.init_training_references(self, max_steps, num_train_epochs, trial)  # KEY: args override checkpoint
```

`init_training_references` directly sets `self.max_steps = max_steps`,
so the args-derived value wins over the checkpoint-loaded one. **The
scheduler also continues correctly**: PyTorch's `LambdaLR.load_state_dict`
restores `last_epoch` / `_step_count` from the checkpoint but the
cosine `partial`'s closure (where `num_training_steps` lives) is
re-built from the new args. The `.__dict__.update(fn)` line at the
end of `load_state_dict` is a no-op for `partial` objects because
their closure args live in `.keywords`, not `.__dict__`.

**Empirical verification.** Resume run with new `num_training_steps =
9291`:

| step | actual lr | theory (new 9291 cosine) | theory (old 4000 cosine) |
|---|---|---|---|
| 2000 | 1.83e-4 | **1.79e-4** ✓ | 1.01e-4 ✗ |
| 4000 | 1.27e-4 | **1.22e-4** ✓ | 0.00 ✗ |
| 6480 | 4.43e-5 | **4.34e-5** ✓ | (wrap) ✗ |

New 9291 schedule matches within 4 % across the full range. Old 4000
schedule would have hit zero at step 4000; actual run was nowhere near
zero. **The schedule is correct.**

**What was actually broken in that run.** The cloud sweep's
wandb-during-eval path hung mid-step-6480 (a known wandb / network /
eval-generation stability issue) — completely unrelated to LR
resumption. The doc author conflated "training stuck" with "LR
schedule frozen".

**Lesson.** `trainer_state.json` records what the trainer state was
**at checkpoint write time** — it cannot be used to predict the
future training plan because args win at resume. Any
`--resume_from_checkpoint --num_train_epochs N --max_steps -1` (or
explicit `--max_steps M`) is safe; no monkey-patch is needed. Verify
by reading `metrics.jsonl`'s lr column against expected cosine, not by
reading `trainer_state.json`.

If `transformers 6.x` ever changes `init_training_references` such
that it skips on resume, this conclusion needs revisiting — verify
with a 10-step resume + cosine-curve comparison.

---

## Common defenses, summarized

| Class of bug | Defense |
|---|---|
| Version-bump silently invalidates saved adapter | In-pipeline `save_reload` tripwire + standalone smoke check |
| Wrong forward-pass tree under quantization | Cross-stack parity audit ([`12`](12-mlx-vlm-vs-hf-kv-sharing.md)) + always quant under `mlx_vlm.load` tree |
| Backend wheel ships missing kernel fixes | Cross-backend eval on a known-good artifact |
| Data file drift between machines | Single canonical prep script; per-file row-count + species-count tripwire |
| Same-script different-order cross-platform | `sorted(...)` on every filesystem-iteration step |
| Two eval files conflated in cross-stage comparison | Documentation discipline: every result row carries the file it came from |
| Adapter silently not applied | Tripwire on `merge_and_unload`: assert no `lora_*` survive |
| Sampling instead of greedy decode in eval | Explicit `do_sample=False, num_beams=1, temperature=1.0, top_p=1.0, top_k=0`; eval driver pins all RNGs at start |
| Absolute paths in shared eval data | Relative path + `--plant_image_root` env; resolver raises on miss |
| LR scheduler "locked" on resume | Read `metrics.jsonl` lr column, not `trainer_state.json` |

All defenses live in code, not in process. Each one was added because
a real run hit the underlying class of bug.
