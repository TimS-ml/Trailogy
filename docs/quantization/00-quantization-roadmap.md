# Quantization roadmap — three routes, current priority, and why

## TL;DR

- This roadmap chooses between three quantization routes: 4-bit-aware SFT, CUDA/HF post-training quantization, and MLX-native post-training quantization.
- The current iOS deliverable ships from the MLX-native route because M8b reaches 3.6 GB and 88.0 % PlantNet, within noise of the bf16 MLX reference.
- The CUDA/HF hybrid route still matters as an independent reference: it reaches 3.41 GB and 83.7 %, validating the same size-quality region with different tooling.
- Use this doc to decide which route to work on; use the route-specific result files for exact commands, variants, and caveats.

> last edit: 2026-05-18 (B.1 hybrid PyTorch route landed at 3.41 GB /
> 83.7 %; B.2 + EoRA closes gap to 88.0 % at 3.6 GB; MLX bridge no
> longer the only path to a sub-4 GB deliverable. iOS deploy artifact
> remains a B.2 MLX export — only format `mlx-swift-lm` consumes.)
> Strategic-decision doc. Read after `README.md` and before any
> route-specific spec.

## Route Summary

Three routes to a deployable 4-bit Gemma 4 E2B:

| Route | Pipeline | Status | Priority |
|---|---|---|---|
| **A** — 4-bit SFT | QAT or QLoRA during training; output is already 4-bit | `A-*.md` in this dir | parallel track |
| **B.1** — bf16 SFT → GPTQModel → MLX bridge | Train at bf16, calibration-driven PTQ via HF GPTQModel (mature CUDA tools), then bridge HF GPTQ format to MLX | hybrid PyTorch artifact **landed** at 3.41 GB / 83.7 % (R6, `B1-torchao-vs-gptqmodel.md` §7); MLX-format bridge still TBW | accuracy reference |
| **B.2** — bf16 SFT → MLX → mlx-lm quantize | Train at bf16, convert to MLX, then quantize via `mlx_lm.quant.*` (Apple's research methods) | M1-M3 affine PTQ done; M8b EoRA r=64 lands at **88.0 % / 3.6 GB** (−0.3 pp vs M0 bf16, within noise) | **PRIORITY** — iOS deliverable |

**Current decision: the iOS deliverable ships from B.2 (MLX-native).**
The B.2 affine path (M1-M3) lands at 83-84 % / 3.2-3.6 GB, and the
**M8b EoRA post-quant adapter** (`B2-sft-results.md`) recovers the
4.3 pp quant drop, landing at 88.0 % / 3.6 GB total — within noise
of the bf16 ceiling. No MLX-format bridge from GPTQModel is needed
to ship.

**B.1 stays as the accuracy reference.** The hybrid PyTorch artifact
(GPTQModel Linears + torchao int4 `embed_tokens_per_layer` packing +
audio strip; R6 in `B1-sft-results.md`) lands at **3.41 GB / 83.7 %**
on the 4090 desktop, matching B.2's M2 (83.7 % / 3.4 GB) within
seed noise. The two independent routes cross-validate the size-vs-
quality trade.

## Why B.1 was prioritized originally (and what changed)

Going into this round, Apple's `mlx-lm/quant/{gptq, awq, dwq, dynamic_quant}.py`
implementations did not look strong enough to be the foundation of a
shipped quant. That assessment is still true for the mlx-lm research
methods themselves — but it turned out we did not need them to ship.
Two findings flipped the priority back to B.2 for the iOS artifact:

1. **B.2 affine PTQ via `mlx_vlm.convert -q` already lands at
   83-84 % on the canonical sorted test n=300 split**
   (`B2-sft-results.md` M1-M3). Once vision/audio scope is set
   correctly (`embed_vision` / `embed_audio` added to the skip list),
   the data-free per-group asymmetric grid alone is competitive with
   GPTQModel + desc_act on this model.
2. **M8b EoRA post-quant adapter** (NVIDIA training-free recovery
   method, `B2-sft-results.md`) closes the residual 4.3 pp quant
   drop on M2 to 0.3 pp at 0.2 GB extra cost — total 3.6 GB / 88.0 %
   vs M0's 88.3 %. mlx-swift-lm already supports QLoRALinear so the
   adapter ships as a sidecar safetensors with zero iOS plumbing.

B.1's role is now the **accuracy reference and PyTorch testbed**, not
the deliverable path. The original reasons B.1 looked appealing
remain valid as caveats on mlx-lm's research methods:

- **GPTQ** (`mlx_lm/quant/gptq.py`): NaN logits on Gemma 4 in the
  2026-05-13 mac_mlx_lm round. The fix is non-trivial — needs the
  hybrid flow at minimum (see `05-mlx-vlm-design.md`), and even then
  the algorithm itself lacks the production-grade tricks present in
  `gptqmodel`: act-order (`desc_act`), dead-column handling,
  Hessian-weighted auto-clip search, LQER low-rank residual
  compensation. These are documented in `B2-research-spec.md`
  §8 as "out of scope for the spec"; they are in fact the **core of
  any credible MLX-native PTQ effort**. None of them are landed today.
- **AWQ** (`mlx_lm/quant/awq.py`): no `gemma4` entry in
  `AWQ_MODEL_CONFIGS`. Convert fails immediately.
  Patch is ~5 lines but mandatory.
- **DWQ** (`mlx_lm/quant/dwq.py`): broadcast bug at line 113 in the
  validation-loss path; reproduces on Gemma 4 regardless of which
  model tree builds the forward pass.
- **`dynamic_quant`**: the only method that ran end-to-end in
  mac_mlx_lm. Produced 2.99 GB on disk but is text-only without a
  vision splice; the splice fails because of the architectural
  mismatches in `05-mlx-vlm-design.md`. Even the size number is
  preliminary — `dynamic_quant` does *sensitivity-based* bit
  allocation, not the OBS/Hessian-driven PTQ that gptqmodel does, so
  it's a different family of recipe.

`gptqmodel` on CUDA, by contrast, is mature: `desc_act=True` recovered
0.5-1.0 pp on our SFT (R1 vs R2 on the old n=2870 val; the ordering
swaps on the new n=300 sorted-test split inside sampling variance,
see `B1-sft-results.md` "Sampling variance" note) at near-zero extra
inference cost. It runs at any PlantNet val cap. The original
diagnosis — "B.1 bottleneck is format compatibility, not quality" —
held empirically: the embedding-quant coverage was the structural
size driver, and torchao int4 on `embed_tokens_per_layer` closed
that gap inside PyTorch (`B1-torchao-vs-gptqmodel.md` §1, §7).

**Updated asymmetry as of 2026-05-18:**

- **B.1 deliverable shape**: hybrid PyTorch artifact at 3.41 GB /
  83.7 % is shippable on CUDA/server, NOT on iOS (no MLX kernel
  for the torchao packed-uint8 embeddings or for GPTQ I32-packed
  Linears on Metal). The HF→MLX format bridge is still TBW for
  any iOS use of this artifact and is not blocking the deliverable.
- **B.2 deliverable shape**: MLX-native artifact at 3.6 GB / 88.0 %
  (M2 + EoRA r=64) loads via `mlx_vlm.load` and `mlx-swift-lm`
  on iPhone today. No cross-machine dependency at inference time.

B.2 wins as the deliverable. B.1 stays for cross-validation, for
serving on non-Apple stacks, and as the source of truth for "what
PyTorch-grade calibration-driven PTQ gets you on this model" so we
can size the EoRA recovery against a known-good baseline.

## Route A — 4-bit at training time

Two distinct sub-routes that the "4-bit SFT" label has historically
conflated. Their decision framing lives in
`../../finetune/06-bnb-vs-torchao-sft.md`
(SFT-side companion); per-route status lives in the `A-*.md` files
in this directory.

| Sub-route | Mechanism | What it optimizes | Status |
|---|---|---|---|
| **A.1** — bnb-NF4 QLoRA SFT | base stored as NF4, LoRA in bf16 on top | **train-time VRAM** (3× wall speedup on 4090) | done — `A-baseline2-qlora-progress.md` |
| **A.2** — torchao QAT (cooldown or from-scratch) | bf16 base + fake-quant on forward; `convert` after | **deploy-time int4 accuracy** | not started; gated on the experiment in `../../finetune/06-bnb-vs-torchao-sft.md` §7 |

Key empirical readings that constrain A.2's expected upside:

- A.1's deployed bnb-NF4 model ties baseline-1's bf16-SFT-then-NF4
  model within noise (69.50 % vs 69.33 %). QLoRA did not narrow the
  PTQ gap.
- A.1's deployed MLX-INT4 model lands at 22.5 % vs baseline-1's
  78.0 % through the same `mlx_vlm.convert -q g64` recipe — a 55 pp
  collapse. QLoRA-base appears to produce adapters that survive
  same-grid redeploy (NF4 → NF4) but **not** cross-grid redeploy
  (NF4-trained → MLX-INT4-deployed). Detail and mechanism in
  `06-bnb-vs-torchao-sft.md` §3.2.
- The 1.8 pp gap between bf16 reference (70.6 %) and best PTQ
  (GPTQModel + desc_act, 68.8 %) is the entire ceiling A.2 can
  attack. Realistic prior on QAT cooldown vs GPTQ + desc_act:
  **+0.3 to +1.0 pp**.

Policy gate: fake-quant fwd + bf16 grads is technically not
"4-bit training" under AGENTS.md rule [0] because no 4-bit weights
or 4-bit optimizer state ever exist. A.2 still needs an explicit
ruling on this before any code lands. The qwen3 reference notebook
uses `adamw_8bit`, which **does** violate rule [0] — any A.2
implementation must lock the optimizer to `adamw_torch` /
`adamw_torch_fused`.

## Route B.1 — bf16 SFT → GPTQModel → (torchao hybrid | MLX bridge)

Three stages, all three with a landed PyTorch artifact; the MLX-format
bridge is still TBW but no longer on the critical path:

### B.1.1 bf16 SFT — done

`src/finetune/` produces the merged bf16 checkpoint
`src/quantization/results/_merged_bf16/`. The data-aug-enwiki
SFT run is the current baseline (LoRA r=256 + full projector tuning,
5 epochs, train loss 0.1515). Eval ceiling on PlantNet val n=2,870 is
70.6 % (`bf16_reference` row R0 in 01b).

### B.1.2 GPTQModel calibration-driven PTQ — done (on CUDA)

`src/quantization/src/methods/gptq.py` runs `gptqmodel`-side
PTQ. The 4090 desktop rows in 01b are at 68.4-68.8 % match (R1 = da=0,
R2 = da=1) with 256-sample PlantNet+WikiText calibration.
`desc_act=True` wins by ~0.5 pp at zero inference cost.

The calibration data design (text-only / mixed-text / multimodal,
eval-leak guards) is being separated into a route-agnostic doc:
`04-calibration-data-design.md`. The same calibration sources will
feed both B.1 and B.2.

### B.1.3a torchao hybrid (PyTorch-side sub-4 GB) — done

R6 in `B1-sft-results.md` + full design in
`B1-torchao-vs-gptqmodel.md` §6-§7. Pipeline: start from R3
(`gptq_w4g64_da0`, 7.01 GB), strip `audio_tower` + `embed_audio`
(iOS-unused), torchao `IntxWeightOnlyConfig(int4, PerGroup(128),
ASYMMETRIC)` on `embed_tokens_per_layer` (4.70 GB → 1.23 GB), pack
to uint8 to halve torchao's default int8 storage. Load via
`hf_gptq_hybrid` loader (`PackedQuantizedEmbedding` carries
`embed_scale=16` probed from the live `Gemma4TextScaledWordEmbedding`).

Result: **3.41 GB on disk / 83.7 % PlantNet n=300 / ROUGE-L 0.804**
— zero accuracy loss vs the GPTQ source. Cross-validates B.2's M2
(3.4 GB / 83.7 %) within seed noise.

Open follow-up: full `embed_tokens` quant (`B1-torchao-vs-gptqmodel.md`
§8 #1) could shave another ~0.6 GB to ~2.77 GB. Earlier "tied-weights
gotcha" was almost certainly the same `embed_scale` runtime bug,
not a fundamental limit.

### B.1.3b HF GPTQ → MLX bridge — to be specified (not blocking)

The original bridge work, deferred because B.2 + EoRA ships a
sub-4 GB iOS artifact without it. Kept in the roadmap because a
working bridge would let GPTQModel's mature CUDA calibration tools
seed MLX deploys (instead of relying on the data-free `mlx_vlm.convert`
affine + post-hoc EoRA recovery). What's needed:

| Concern | HF GPTQModel output | MLX `QuantizedLinear` expected |
|---|---|---|
| Weight storage | I32 packed (8 nibbles/int32) | U32 packed |
| Scales dtype | F16 | BF16 |
| Per-group bias | combined with scale (`qzeros`) | separate `biases` tensor |
| `desc_act=True` artifact | `g_idx` permutation tensor per layer | not natively supported; must be baked into weight order |
| Tensor key naming | `model.layers.N.mlp.gate_proj.qweight` etc. | mlx-vlm-style `language_model.model.layers.N.mlp.gate_proj.weight` etc. |
| `config.json` quant manifest | `quantization_config: {bits, group_size, desc_act, ...}` | `quantization: {bits, group_size, mode}` + per-tensor entries |

Two implementation paths to evaluate when the spec is written:

1. **Direct format translation**: write a function that reads HF GPTQ
   tensors and emits MLX-format tensors. Lossless (preserves the GPTQ
   quantization values exactly). Has to handle the `desc_act`
   permutation by reordering the weight columns before re-packing.
2. **Dequant → bf16 → mlx_vlm.convert**: dequantize HF GPTQ back to
   bf16 in HF land, then feed through `mlx_vlm.convert -q` with the
   correct predicate. **Lossy** — re-quantizes with mlx-vlm's
   data-free affine, throwing away GPTQ's Hessian-aware choices.
   This is the "naive bridge" — would be a useful sanity baseline
   but is not the actual deliverable goal.

Path 1 is the real bridge. Spec TBW; expected number is "match R2's
68.8 % match while shipping ~3.6 GB on disk via vision/audio bf16".

## Route B.2 — bf16 SFT → MLX → mlx-lm quantize (parallel research)

`B2-research-spec.md` is the active spec for this route.
It uses the **hybrid flow** (`mlx_vlm.load` first, then `mlx_lm.quant.*`
core on the resulting tree) from `05-mlx-vlm-design.md`.

Status this round:

- The 8-row ablation matrix in 08 §3 is **B.2 research backlog**, not
  the priority deliverable.
- 08 §8 ("out of scope") is actually the **core MLX-native quant
  work**: porting stable GPTQ algorithm tricks
  (act-order, dead-column, auto-clip, LQER) into `mlx_lm/quant/gptq.py`.
  Until those land, B.2's GPTQ is just a weaker version of B.1's
  GPTQModel.
- The simple `mlx_vlm.convert -q` baselines (no calibration, data-free
  affine) tracked in 01c are *not* B.2 PTQ — they're the no-research
  baseline that any B.2 effort has to beat.

## Current state summary

> All n=300 numbers below are on the canonical sorted `test.jsonl`
> seed=0 split (see "sorted-split" note in `B1-sft-results.md`).

| Component | Where | Status |
|---|---|---|
| bf16 SFT merged checkpoint | `_merged_bf16/` | ✅ |
| HF bf16 reference (R0) | `B1-sft-results.md` | ✅ 86.7 % n=300 (CUDA), 88.3 % n=300 (Mac mlx_vlm M0) |
| HF GPTQModel PTQ (R1-R4) | `B1-sft-results.md` | ✅ 80.3-83.7 % n=300 |
| **B.1 torchao hybrid (R6)** | `B1-sft-results.md`, `B1-torchao-vs-gptqmodel.md` | ✅ **3.41 GB / 83.7 % n=300** — sub-4 GB on PyTorch/CUDA, zero loss vs R3 source |
| MLX affine PTQ (M1-M3, vision-frozen) | `B2-sft-results.md` | ✅ 83-84 % n=300, 3.2-3.6 GB |
| **B.2 + EoRA (M8b, r=64)** | `B2-sft-results.md` | ✅ **3.6 GB / 88.0 % n=300** — iOS deliverable, −0.3 pp vs M0 ceiling |
| Cross-backend Linux mlx-cuda validation | `../general/11-cuda-vs-mlx-eval-parity.md` | ✅ 40-41 % n=300 on 4090 box after source-build of `mlx` main (CUDA QMM kernel bug fixed); 9 pp gap vs Mac is CUDA/Metal numerics |
| HF GPTQ → MLX format bridge | TBW (B.1.3b) | deferred — not blocking deliverable |
| Calibration data design | `04-calibration-data-design.md` | ✅ |
| B.2 algorithm-stability port | `B2-research-spec.md` §8 | research backlog |

## What this roadmap does NOT cover

- Route A details (live in `A-*.md` files in this dir).
- Specific GPTQModel calibration sweeps — see `04-calibration-data-design.md`
  once written.
- mlx_vlm vs mlx_lm mental model — see `05-mlx-vlm-design.md`.
- Per-variant eval numbers — see `01b-` and `01c-`.

## Next steps

1. ~~Write `04-calibration-data-design.md`~~ — done.
2. ~~Restructure `B2-research-spec.md`~~ — done.
3. Domain-calibrated EoRA — replace WikiText-2 `X^T X` with PlantNet
   image+text pairs; r=128 regression (87.0 % < r=64's 88.0 %)
   suggests text-only calibration loses direction at high rank.
4. EoRA on M1 (smaller base, 3.2 GB) — if r=32 adapter matches M8b's
   88.0 % at ~3.3 GB total, that becomes the new smallest viable
   iOS artifact.
5. (Deferred / not blocking) Spec the B.1 MLX bridge — would unlock
   GPTQModel's CUDA calibration as input to MLX deploys. Likely
   lives at a new `06-route-b1-gptqmodel-mlx-bridge.md`.
6. (Deferred / research) Full `embed_tokens` quant on B.1 hybrid —
   shave another ~0.6 GB toward ~2.77 GB. Tracked in
   `B1-torchao-vs-gptqmodel.md` §8 #1.
