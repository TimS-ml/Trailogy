# B1 quantization tooling — torchao vs GPTQModel

Scope: pick the right PyTorch-side quantization toolchain for the
**b1 route** (`bf16 SFT → quantize → optional MLX bridge`). Question
that triggered this doc: GPTQModel's b1 ceiling sits at ~7 GB on the
SFT'd Gemma 4 E2B (B1-sft-results.md R1/R2). MLX gets to 3.2 GB on
the same model. Is the gap a Linear-quant-quality gap, an embedding-
quant-coverage gap, or both?

Answer: **embedding-quant coverage**, not Linear quality. Section 1
shows the size budget, sections 2-5 justify the tooling pick (use
**both**, hybrid), section 6 lays out the resulting pipeline, and
§7 reports the end-to-end empirical result on this SFT after a
critical `embed_scale` bug fix: **3.41 GB at 83.7 % n=300** —
zero accuracy loss vs the bf16-embed R3 baseline.

## 1. The actual size budget (R2 inspected, 2026-05-15)

Reading `gptq_w4g128_da1/model*.safetensors` headers directly:

| Bucket | Size | Note |
|---|---:|---|
| `language_model.embed_tokens_per_layer` | **4.70 GB** | `[262144, 8960]` bf16. Gemma 4 per-layer-input lookup. **`nn.Embedding`, not quantized by GPTQ.** |
| `language_model.embed_tokens` | 0.81 GB | `[262144, 1536]` bf16, `nn.Embedding`. Untouched. |
| `audio_tower` + `embed_audio` | 0.61 GB | bf16. iOS-unused — strippable. |
| `vision_tower` | 0.34 GB | bf16. **Must stay bf16** (NF4 collapse proven). |
| Quantized Linears (35 layers × q/k/v/o + g/u/d + per_layer_*) | ~0.94 GB | INT32-packed at 4-bit g128. |
| Norms, scales, qzeros, g_idx, embed_vision, etc. | ~0.05 GB | |
| **TOTAL** | **~7.45 GB** | |

The dominant size driver is **`embed_tokens_per_layer`** — not the
decoder Linears. Gemma 4 inherits the per-layer-input architecture
from Gemma 3n: each decoder layer reads an extra additive embedding
sliced from a `[vocab, num_layers × per_layer_input_dim]` table. With
`vocab=262144`, `num_layers=35`, `per_layer_input_dim=256`, the table
is 2.35 B params at bf16 = 4.7 GB. **No GPTQ code path touches it.**

This is the entire structural reason b1 ≠ b2. mlx-lm's `nn.quantize()`
walks every module with a `to_quantized` method, including
`mlx.nn.QuantizedEmbedding`, so its 3.2 GB output is what you get
when you quantize this table to 4-bit g64 alongside everything else.

## 2. Overlap: nn.Linear quantization comparison

These are the modules **both** toolchains can quantize: the 35 ×
{q/k/v/o_proj, gate/up/down_proj, per_layer_input_gate,
per_layer_projection} Linears under
`model.language_model.layers.#.*`. The comparison is honest about
where each one is best.

### Algorithm + Gemma 4 readiness

| Axis | GPTQModel main (`3437e60`) | torchao 0.16 |
|---|---|---|
| GPTQ (Hessian-based, calibration-driven) | Production — `gptqmodel/looper/module_looper.py` | `torchao.prototype.gptq.GPTQConfig` — explicit prototype |
| AWQ (activation-aware scaling) | Production + per-arch shims (`awq_processor.py`); Gemma 4 entry not yet shipped | `torchao.prototype.awq.AWQConfig` — prototype |
| HQQ (proximal-optimized data-free) | Not directly | `Int4WeightOnlyConfig(int4_choose_qparams_algorithm=HQQ)` stable; kernel needs external `hqq` pip package |
| RTN (data-free affine) | Available via `RTN` looper | `Int4WeightOnlyConfig` (tinygemm/preshuffled/plain), `IntxWeightOnlyConfig` for 1–8 bit |
| SmoothQuant / AutoRound / SpinQuant | No | `torchao.prototype.{smoothquant, autoround, spinquant}` — all prototype |
| **Gemma 4 model shim** | `gptqmodel/models/definitions/gemma4.py:1-256` — `_capture_gemma4_positional_inputs`, `_patch_gemma4_per_layer_input_capture`, `_prepare_gemma4_replay_kwargs`. **Per-layer-input cache + decoder replay handled.** | **None.** Calibration-driven methods (GPTQ/AWQ) need ~200-400 LOC of Gemma 4-aware adapter code. |
| **Documented quality on this exact model** | R2 = 68.8 % (−1.8 pp vs bf16 70.6 %), B1-sft-results.md | **Zero.** New territory. |

### HF integration

| Axis | GPTQModel | torchao |
|---|---|---|
| Save | `model.save_quantized(dir)` → HF-format safetensors + `config.json` + `quantize_config.json`. Loadable via `AutoModelForCausalLM.from_pretrained` natively. | `model.save_pretrained(dir)` **fails** for tensor-subclass tensors (`RuntimeError: Attempted to access the data pointer on an invalid python storage`). Must use `torchao.prototype.safetensors.safetensors_support.flatten_tensor_state_dict` + manual `safetensors.save_file`. |
| Load | Native via `quantization_config=GPTQConfig(...)` | `quantization_config=TorchAoConfig(...)` (transformers ≥ 4.43). Tensor-subclass registration via `torchao.prototype.safetensors.safetensors_utils.ALLOWED_CLASSES`. |
| Side-car carry-over (processor_config / chat_template) | **Broken** — `save_quantized` drops them. We patch this in `src/quantization/src/common/model_io.py:copy_processor_assets`. | Same — `save_pretrained` carries tokenizer assets but not processor side-cars for multimodal. Same patch reused. |
| Per-FQN config | `QuantizeConfig.dynamic: Dict[str regex pattern, dict]` with `-:` / `+:` prefix semantics | `FqnToConfig({fqn or "re:pattern": AOBaseConfig, "_default": AOBaseConfig})` |

### Verdict on Linear

**Use GPTQModel.** Three reasons:

1. **Validated number on this model exists**: R2's 68.8 % was produced
   with the Gemma 4 shim doing the per-layer-input capture + replay
   correctly. Reproducing that quality in torchao means either:
   (a) writing the Gemma 4 calibration adapter — ~1-2 days,
       duplicates `gemma4.py` work, prototype API stability risk;
   (b) accepting HQQ data-free quality — usually 1-3 pp below GPTQ
       on transformer LMs, no Gemma 4 head-to-head exists.

2. **Production vs prototype**: torchao's calibration-driven methods
   are in `torchao.prototype.*`. APIs and tensor formats change
   between minor versions. GPTQModel's GPTQ path is the public
   surface.

3. **HF-native save/load**: GPTQ checkpoints round-trip through
   `from_pretrained` without custom code. torchao tensor subclasses
   need `flatten_tensor_state_dict` glue both on save and on load.

torchao would only win on Linear if the project chose to:
- Drop calibration entirely and go HQQ data-free (cleaner code,
  unknown quality on Gemma 4); OR
- Target sub-4-bit (Int3/Int2) — `IntxWeightOnlyConfig` has multiple
  packing formats while GPTQModel's 3-bit kernel is Torch-only and
  slow.

Neither matches our current constraints.

## 3. Non-overlap: `nn.Embedding` quantization

This is the **structural** difference, and the reason we need torchao
at all on the b1 route.

| Toolchain | Quantizes `nn.Embedding`? | Evidence |
|---|---|---|
| GPTQModel (main `3437e60`) | **No** | `gptqmodel/models/_const.py:37` defines `SUPPORTS_MODULE_TYPES = [nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D]`. No `nn.Embedding`. Confirmed by `rtn.py:77-82` assertion. No `dynamic` pattern overrides this. |
| AutoGPTQ | **No** | Same `nn.Linear`-only scope inherited by the fork chain. |
| Optimum-Quanto | Module-swap framework exists but no shipped `QEmbedding`; would need user code. |
| HF bitsandbytes | **Effectively no** | `BitsAndBytesConfig.llm_int8_skip_modules` controls Linear scope. NF4 quantizes `nn.Linear`; no embedding quant. `StableEmbedding` is for training, not inference storage compression. |
| AWQ (the standalone repo) | **No** | Linear-only. |
| HQQ (mobiusml/hqq) | Math primitive can pack any 2D tensor; **no shipped `HQQEmbedding` runtime**. Would need ~80 LOC. |
| **torchao 0.16** | **Yes — stable public API** | `IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(...))` applied via `quantize_(..., filter_fn=lambda m,fqn: isinstance(m, nn.Embedding))`. `quant_api.py:1781-1806` handles both `nn.Linear` and `nn.Embedding` swap, including the `aten.embedding.default` op override at `intx_unpacked_to_int8_tensor.py:371-380` and `intx_opaque_tensor.py:340-367`. |
| PyTorch native `torch.ao.nn.quantized.Embedding` | **Yes**, but CPU/FBGEMM-targeted, not bf16 CUDA inference. |
| FBGEMM_GPU / TorchRec | **Yes**, mature INT4/INT2 embedding-bag kernels, but recsys-shaped API (table-batched, not drop-in `nn.Embedding`). Not HF safetensors-compatible without a wrapper. |
| MLX (`mlx.nn.QuantizedEmbedding`) | **Yes**, drop-in. This is exactly why MLX gets to 3.2 GB. |

So on the PyTorch side, **torchao is the only mature, in-process
package that quantizes `nn.Embedding` end-to-end**.

## 4. Why not "just use torchao for everything"

Three reasons we're not pivoting the whole pipeline:

1. **Linear quality on Gemma 4 is unproven in torchao**. GPTQModel's
   R2 = 68.8 % is a published number we built on. Re-running with
   torchao HQQ/RTN means resetting the quality baseline.

2. **Calibration-driven torchao requires a Gemma 4 calibration
   adapter** that doesn't exist. GPTQModel's `gemma4.py` already
   solved this problem; re-solving it in `torchao.prototype.gptq`
   for prototype-stable code is unattractive.

3. **No structural advantage to unifying**. torchao writes
   tensor-subclass safetensors, which already requires custom
   `flatten/unflatten` glue. Adding GPTQ-packed tensors to the same
   file is straightforward — both end up as raw `torch.Tensor`s with
   metadata. The output checkpoint is "mixed" in any case.

## 5. Storage gotcha: torchao stores int4 as int8 on CUDA

A late finding: `torchao.IntxWeightOnlyConfig(weight_dtype=torch.int4, ...)`
applied to `nn.Embedding` produces an `IntxUnpackedToInt8Tensor`
where **`qdata.dtype == torch.int8`** regardless of `weight_dtype`.
Verified with `bf16[2048, 1024]` embedding, group_size=128, on CUDA:

| Variant | qdata storage | size vs bf16 |
|---|---|---:|
| `int4` SYMMETRIC | int8, values in `[-8, 7]`, 16 unique | 51% |
| `int3` SYMMETRIC | int8, values in `[-4, 3]`,  8 unique | 51% |
| `int2` SYMMETRIC | int8, values in `[-2, 1]`,  4 unique | 51% |

So torchao on CUDA does **not** deliver true 4-bit packing
out-of-the-box. `IntxPackingFormat.OPAQUE_TORCHAO_LOWBIT` (true
4-bit packed via `_pack_embedding_{n}bit` aten ops) exists in
`torchao.prototype.quantization.embedding`, but it's
`device == cpu` + `dtype == float32` only, and requires the
torchao experimental kernel library to be built from source
(`mslk >= 1.0.0` / `libtorchao_ops_mps_aten.dylib`). Neither
condition holds in our pytorch env on the 4090.

**Workaround**: torchao computes correct int4 scales/zero_points
on CUDA, but the values stored in `qdata` are constrained to the
int4 range. We pack them ourselves at save time:

```python
# qdata: int8 [V, D], values in [-8, 7]
unsigned = (qdata + 8).to(torch.uint8)      # shift to [0, 15]
packed   = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
# packed: uint8 [V, D/2]
```

Inverse at load time. This is mechanical bit-shuffling (~20 LOC).
The **quantization math** (selecting scales/zero_points, rounding)
stays in torchao — we only own the storage layout and the inference
forward pass. See `src/quantization/src/methods/gptq_torchao_hybrid.py:pack_int4_to_uint8`.

## 6. The resulting hybrid pipeline

Implemented in `src/quantization/src/methods/gptq_torchao_hybrid.py`,
dispatched via `scripts/run/quant.py --method gptq_torchao_hybrid`.

Stage diagram:

```
                bf16 merged Gemma 4 (~9.51 GB, hf_bf16)
                          │
                          ▼ stage 1: GPTQModel (existing)
                  bits=4, group_size=128, desc_act=True
                  calibration: 256 PlantNet + 256 WikiText
                  Gemma 4 multimodal shim (per-layer-input replay)
                          │
                          ▼  ~7.45 GB GPTQ checkpoint
                          │
                          ▼ stage 2: hybrid runner (new)
                  ┌───────┴────────────────────────┐
                  │                                │
                  ▼ 2a: strip audio                ▼ 2b: torchao embed quant
                  del model.model.audio_tower     IntxWeightOnlyConfig
                  del model.model.embed_audio     weight_dtype=torch.int4
                          │                       granularity=PerGroup(128)
                          │                       filter_fn = match
                          │                         language_model.embed_tokens_per_layer
                          │                         and (optional) embed_tokens
                          ▼                                │
                          └────────────────┬───────────────┘
                                           ▼
                          stage 3: serialize via
                          flatten_tensor_state_dict
                          + safetensors.save_file
                          + patch config.json
                                           │
                                           ▼
                          hybrid checkpoint (~3.4-3.6 GB)
                          loadable via hybrid_load()
```

Size accounting target (`embed_per_layer_bits=4, group_size=128`,
both embeddings packed, asymmetric mapping):

| Component | bf16 | Hybrid output | Saving |
|---|---:|---:|---:|
| GPTQ Linears | 0.94 GB | 0.94 GB | — |
| `embed_tokens_per_layer` qweight uint8 (`262144 × 4480`) | — | 1.17 GB | |
| `embed_tokens_per_layer` scales bf16 (`262144 × 70`) | — | 0.037 GB | |
| `embed_tokens_per_layer` zero_point int8 (`262144 × 70`) | — | 0.018 GB | |
| `embed_tokens_per_layer` subtotal | 4.70 GB | **~1.23 GB** | **−3.47 GB** |
| `embed_tokens` (qweight + scales + zp, optional 4-bit) | 0.81 GB | **~0.21 GB** | −0.60 GB |
| `audio_tower` + `embed_audio` | 0.61 GB | **0** (stripped) | **−0.61 GB** |
| `vision_tower` | 0.34 GB | 0.34 GB | — |
| Other (norms, embed_vision, per_layer_model_projection) | 0.05 GB | 0.05 GB | — |
| **TOTAL** | 7.45 GB | **~2.77 GB** | |

Symmetric mapping skips zero_point: trims another ~21 MB on
`embed_tokens_per_layer` + ~3 MB on `embed_tokens`. Negligible in
the total but the difference between mappings is documented in the
torchao validation matrix (B1-torchao-vs-gptqmodel.md §5).

### Where this artifact lives in the pipeline

The hybrid checkpoint is a **CUDA-side PyTorch quality reference**,
not an iOS-deployable artifact. We don't ship a PyTorch runtime on
iPhone — the iOS app loads MLX format via `mlx-swift-lm`. To get to
an MLX-loadable file:

```
b1 hybrid artifact (PyTorch, ~2.77 GB, CUDA-only)
        │
        │  THIS PIPELINE BIFURCATES HERE
        │
        ├─── Branch 1: CUDA reference eval (this script)
        │     n=300 eval via this repo's eval.py
        │     → quality number Q_hybrid
        │       isolates the cost of embed quant on top of R2's GPTQ Linears
        │
        └─── Branch 2: MLX deploy bridge (separate path)
              mlx_vlm.convert on _merged_bf16 (NOT on the hybrid artifact)
              with --q-bits 4 --q-group-size 64/128 + audio strip
              → MLX checkpoint, mlx-swift-lm-loadable
              n=300 eval via mlx-vlm
              → quality number Q_mlx (the actual iOS-shippable number)
```

The two branches do NOT share quantized tensors. `mlx_vlm.convert`
expects bf16/fp16 weights as input; it cannot consume GPTQ-packed
int32 qweight or our custom uint8-packed embed format. So branch 2
runs MLX's own quantization (mlx-lm's `nn.quantize` over both
Linears and Embeddings, using affine groups, on Apple Silicon) on
the bf16 source. Branch 1's quant work doesn't transfer to MLX.

Why build branch 1 at all then:

1. **Decouple measurement of two error sources**. R2 (GPTQ Linears
   only) sits at 68.8 %, vs bf16 70.6 %. That's a −1.8 pp cost for
   "Linear quant". Branch 1 measures Q_hybrid which adds embed quant
   on top of R2. The delta `R2 − Q_hybrid` isolates the embed-quant
   cost on this model. If the delta is ≤ 1-2 pp, embed quant is
   safe; if it's 5+ pp, the per-layer-input table is unusually
   sensitive and we should re-evaluate group_size / mapping.

2. **Best-case upper bound for the MLX deploy quality**. MLX-side
   quantization (mlx-lm's RTN affine + skip lists) is generally
   coarser than torchao's int4 affine quant of the same tensor.
   `Q_hybrid` therefore upper-bounds what we should expect from MLX
   at the same bit budget. If `Q_mlx << Q_hybrid`, the MLX recipe is
   leaving accuracy on the table; if they're close, MLX-side quant
   is competitive.

3. **Cross-validate the embed-quant math**. If `Q_hybrid` is
   reasonable, the packed-int4 implementation is at least
   numerically sound on this model — useful before we trust similar
   packing tricks elsewhere.

Branch 2 is straightforward (already partially done as B2-sft-results.md
M1 at 78.3 % paper-grade); the new piece for branch 2 is just adding
audio strip on the bf16 source before `mlx_vlm.convert`. Documented
as a follow-up TODO.

### Eval gate (CUDA branch, n=300 on val.jsonl, seed=0, EVAL_PLANTNET_N=300):
- size ≤ 4.0 GB ✅
- PlantNet match drop vs R2 ≤ 10 pp → don't trip
- PPL ≤ 2× R2 PPL → don't trip

Expected quality vs R2:
- Linear quant unchanged → no Linear-side regression
- Embedding quant at 4-bit g128 on `embed_tokens_per_layer` is the
  unknown. mlx-vlm M1 on the same model at 4-bit g128 + skip
  embed_vision held 78.3 % (paper-grade test) vs M0 85.7 % (−7.4 pp).
  We expect a smaller drop on the val split (different distribution
  and different per-layer-input quantization recipe), but cap
  expectations: if the drop is similar (−7 pp), end state is ~62 %
  on PlantNet val — still above the 60 % SFT-shippability tripwire.

## 7. Empirical findings (2026-05-15, fix landed evening)

Built and evaluated the hybrid pipeline on the R3 source (GPTQ
`w4g64_da0`, 83.7 % bf16-Linear baseline on the n=300 test split).
Three int4 group_size variants for `embed_tokens_per_layer` (`embed_tokens`
kept bf16). All variants land under 4 GB; **after the embed_scale fix
(see §7.1), all three match the bf16-embed baseline within ±1 pp**.

### Sizes + per-row fidelity + accuracy (post-fix, fixed runtime)

| variant | size | embed_per_layer cos_sim vs bf16 | PlantNet match (n=300) | Δ vs R3 |
|---|---:|---:|---:|---:|
| R3 baseline (no embed quant) | 7.01 GB | n/a | **83.7 %** | — |
| dequant-back-bf16 (noise only, stock runtime) | 7.01 GB | 0.994 | **82.7 %** | −1.0 pp |
| **hybrid pl=int4 g128 asym** | **3.41 GB** | 0.994 | **83.7 %** | **0.0 pp** ✅ |
| hybrid pl=int4 g32 asym | 3.57 GB | 0.997 | **83.0 %** | −0.7 pp |
| hybrid pl=int4 g16 asym | 3.79 GB | 0.998 | **82.7 %** | −1.0 pp |

Audio stripped in all variants. Vision_tower + embed_vision + norms
preserved bf16. `embed_tokens` kept bf16 in this matrix (quantizing
it is unblocked but not yet measured — see §7.2 follow-ups).

**The curve is flat ±1 pp across group_size**, exactly what's expected
when the underlying per-row int4 noise sits at or below the bf16
rounding floor on this distribution (`embed_tokens_per_layer` std
≈ 0.064). Group_size tuning is no longer a quality lever once the
runtime is correct; pick by storage budget. **g128 at 3.41 GB is the
sweet spot** — smallest variant, zero accuracy loss vs the bf16-embed
baseline.

For cross-validation with the MLX deploy artifact:

| Route | Best variant on this SFT | Size | n=300 |
|---|---|---:|---:|
| MLX (b2) M1 | affine g128 | 3.2 GB | 83.0 % |
| MLX (b2) M2 | affine g64 | 3.4 GB | 83.7 % |
| MLX (b2) M3 | affine g32 | 3.6 GB | 84.0 % |
| **PyTorch (b1) hybrid g128 (this row)** | GPTQ Linears + torchao int4 PLE | **3.41 GB** | **83.7 %** |

The b1 CUDA reference now matches MLX M2 to within the eval seed noise
band, on essentially the same size budget. The two routes — independent
quantization toolchains, independent runtimes — produce the same
deployment-grade accuracy.

### 7.1 Root cause: missing `embed_scale` on `Gemma4TextScaledWordEmbedding` swap

`embed_tokens_per_layer` is **not** a plain `nn.Embedding`. It's a
`Gemma4TextScaledWordEmbedding` subclass
(`transformers/models/gemma4/modeling_gemma4.py:1441`), whose forward
is:

```python
def forward(self, input_ids):
    return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
```

For `embed_tokens_per_layer`, `embed_scale = sqrt(hidden_size_per_layer_input)
= sqrt(256) = 16.0` (constructor at `modeling_gemma4.py:1595-1600`).
For `embed_tokens`, `embed_scale = sqrt(hidden_size) = sqrt(1536) ≈ 39.19`.

Our `load_hybrid_embeddings` did:

```python
setattr(parent, name, PackedQuantizedEmbedding(...))
```

replacing the entire scaled-embedding instance with a vanilla
`nn.Module`. The `× 16` multiplication was silently lost. Every PLE
residual signal — added into every one of the 35 decoder layers'
residual stream — went in at **1/16 amplitude**, below the residual
stream's effective noise floor. The model semi-functioned on visual
grounding (vision_tower untouched) + the bf16 `embed_tokens` (also
untouched in this matrix), which is why species recognition still
half-worked while token-level decoding wandered into adjacent subwords:
`white dead-nettle → white dead--nettipetas`, `Lamium → Lamiummumum`,
etc.

#### How we caught it (negative result → real bug)

The original 2026-05-15 sweep evaluated this same matrix and produced
4.3 % / 4.7 % / 4.7 % — a "negative result" prematurely written up as
*"int4 affine on `embed_tokens_per_layer` is fundamentally infeasible
on this model"*. That conclusion was wrong. The diagnosis path that
overturned it:

1. **Read mlx-lm's quant source + inspected an MLX artifact**. MLX M1's
   `embed_tokens_per_layer` IS int4-packed (uint32 weight + bf16
   scales + **bf16 biases**, per `mlx_lm/quant/gptq.py:115` + the
   shared affine fallback at `gptq.py:152-158`). MLX gets 83.0 % on
   the same model — same tensor, same bit width.
2. **Compared per-row fidelity for three parameterizations** (torchao
   asym int8-zp, torchao sym, MLX-style bf16-bias) on 1000 real rows
   of `embed_tokens_per_layer`. All three landed at cos_sim ≈ 0.994 at
   g128. The bf16-bias-vs-int8-zp distinction made **no meaningful
   difference**: 0.99455 vs 0.99444. The quant math is the same to
   within bf16 rounding.
3. **A/B test on the same tensor**: torchao-quantize-then-dequantize
   `embed_tokens_per_layer` back to bf16, stuff into a stock
   `nn.Embedding` inside the otherwise-untouched HF model. **n=300
   = 82.7 %**. Identical quant noise (cos_sim 0.994), stock HF
   runtime, ≈ baseline. Decisive: the quant noise was never the
   problem — our runtime swap was.
4. **Read `modeling_gemma4.py:1441-1452`**, found
   `Gemma4TextScaledWordEmbedding`'s forward multiplies by
   `embed_scale`. Bug.
5. **Re-eval the original 3.41 GB hybrid artifact** with the fixed
   `load_hybrid_embeddings` — no re-quantization needed, the bits
   on disk were always correct, only the load-time module swap was
   broken. **n=300 = 83.7 %**, exactly matching R3.

The pre-fix per-sample failure mode (subword stuttering, multilingual
leakage on the rarer test images) is now retrospectively explained:
PLE additive signals at 1/16 amplitude means the model essentially
runs *without* the per-layer-input refinement that SFT relied on,
degrading to a "vanilla decoder over a noisy lookup table" mode.

#### Fix

`@bea9631` adds:

- `PackedQuantizedEmbedding(embed_scale: float = 1.0)` ctor arg;
  `forward()` multiplies by it after dequant. Default 1.0 keeps the
  module a drop-in for vanilla `nn.Embedding`.
- `load_hybrid_embeddings` probes the live module's
  `scalar_embed_scale` / `embed_scale` attribute BEFORE the swap; if
  config carries `hybrid_quant.embeddings[*].embed_scale`, that wins
  (warns on mismatch).
- `quantize_hybrid` derives `embed_scale` from `text_config.hidden_size`
  and `text_config.hidden_size_per_layer_input` and persists it into
  the per-embedding meta. Existing artifacts (no `embed_scale` in
  config) still load correctly via the live-module probe.

Three regression tests (`test_gptq_torchao_hybrid.py`) guard the fix:
the scaled output must equal unscaled × `embed_scale` within bf16
tolerance; default 1.0 must be a no-op for plain `nn.Embedding`; and
`quantize_hybrid` must persist the derived scale into config. 10/10
hybrid tests green.

### 7.2 Re-evaluating the "tied-weights gotcha"

The 2026-05-15 morning writeup of this doc claimed `embed_tokens` could
not be quantized because it's tied to `lm_head` via
`_tied_weights_keys`. The actual symptom that motivated the claim —
"multilingual gibberish" — is the **same `embed_scale` bug** applied
to `embed_tokens` (`embed_scale ≈ 39.19`). At 1/39 amplitude, the very
first lookup that produces `inputs_embeds` is wrong by ~16× more than
the per-layer-input path, which makes the whole forward unrecoverable.

With the embed_scale fix in place, quantizing `embed_tokens` is most
likely also unblocked. The tied-weights interaction (PyTorch's
`lm_head.weight` parameter object retains its own backing storage after
the `setattr` swap, so `lm_head` keeps using the original bf16 weight
for its matmul — independent of our packed `embed_tokens` lookup) is
not the bug it was made out to be. Empirical validation of full embed
quant (target ~2.77 GB) is the next sub-goal; see §8 follow-ups.

## 8. Open follow-ups

1. **Full embed quant (target ~2.77 GB)** — quantize both
   `embed_tokens` and `embed_tokens_per_layer` to int4 g128 asym.
   The previously-claimed tied-weights blocker (§7.2) is most likely
   the same `embed_scale` bug now fixed. Expected delta vs the
   current 3.41 GB row: about −0.6 GB. If accuracy holds within ±2 pp
   the b1 deploy artifact drops below the 3 GB mark — comfortably
   inside the iOS jetsam ceiling. ~10 min quant + ~14 min eval on
   the 4090.

2. **GPTQModel-side bugs** noted by main-HEAD audit
   (`gptqmodel/models/definitions/gemma4.py`, see B1-sft-results.md
   Caveats #6):
   - Inherits `loader = AutoModelForCausalLM` instead of
     `AutoModelForImageTextToText` (multimodal head drop risk;
     compare `llama4.py:16`).
   - No `HF_CONVERSION_MAP_REVERSED` (Gemma 3 has one at
     `gemma3.py:19-46` remapping tower paths). Untested whether
     Gemma 4 checkpoint layout round-trips cleanly.
   - No `processor` registration → side-cars dropped at save
     (existing workaround: `copy_processor_assets`).

3. **`lm_head=True` regression on main HEAD**: now raises
   `NotImplementedError` for `tie_word_embeddings=True` models
   instead of silently downgrading to False
   (`gptqmodel/looper/module_looper.py:1408-1413`). Our
   `_resolve_lm_head` (`src/methods/gptq.py:115`) already forces
   False, so we're not affected, but bears watching if the upstream
   logic changes again.

4. **torchao prototype stability**: pin the working torchao version
   in `src/quantization/requirements.txt`. The hybrid path
   currently runs against `torchao==0.16.0` + `torch==2.10.0+cu130`.

5. **3-bit `embed_tokens_per_layer` experiment**: with int4 g128 at
   `Δ = 0.0 pp` we have headroom. A 3-bit pack of `embed_tokens_per_layer`
   would buy another ~0.3 GB. Three nibbles per byte requires
   non-trivial unpacking; defer until full-embed quant lands.

6. **MLX bridge eval as the final iOS check**: with the b1 PyTorch
   reference now matching MLX M2 on size + quality, the remaining
   step before iOS is the `mlx_vlm.convert` artifact eval. The B2
   doc already has M2 at 83.7 % / 3.4 GB on this SFT (paper-grade
   `test.jsonl` n=300); the cross-validation is in place.

7. **Lessons for future module swaps**: a generic `nn.Embedding`
   replacement that doesn't subclass the wrapped module silently
   drops any forward-time arithmetic the subclass added (scaling,
   masking, padding-idx handling). Either subclass (preserve unknown
   attributes via `__getattr__` delegation) or capture every relevant
   attribute at swap time. The unit test we added now asserts the
   ratio-equals-`embed_scale` invariant on every commit.
