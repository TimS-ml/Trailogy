# Tested Package Versions & Known Bugs in the Stack

Snapshot of the package combination the SFT + quantization pipeline
has been exercised against end-to-end, plus the bugs in the upstream
stack we've had to work around. Required reading before any package
upgrade.

This doc supports the Trailogy writeup's **Technical Challenge #1
(Silent PEFT loading failure)**: that bug is one entry in the
"Known bugs" table below; the resolution sequence is in the
"Upgrade order" section.

## Tested package combination

Captured **2026-05-13** on a Linux + RTX 4090 dev box. The same
combination has been exercised on remote H100/H200 (cloud sweep) and
M-series Mac (MLX deploy + EoRA calibration).

### System

| Layer | Value |
|---|---|
| OS | Linux 6.x x86_64 (Ubuntu 22.04 / Pop!_OS family) |
| Python | 3.12.x (conda env `torch`) |
| Torch | 2.10.0+cu130 |
| CUDA (runtime) | 13.0 |
| cuDNN | 9.15.1 |
| GPU | RTX 4090 (24 GB) |
| GPU compute capability | (8, 9) — Ada, treated as Ampere+ (TF32 path on) |

The conda env's `stdc++` is required for `optree` (a `torch._dynamo`
dep) to find `GLIBCXX_3.4.31`. Set before any train / eval / smoke
command (path will vary per machine):

```bash
export LD_LIBRARY_PATH=<conda-env-root>/lib:$LD_LIBRARY_PATH
```

### Core ML stack

| Package | Version | Notes |
|---|---|---|
| **unsloth** | **2026.5.2** | Must be imported before `transformers` or it warns about missing optimizations. |
| unsloth_zoo | 2026.5.1 | Pulled in by unsloth. |
| **transformers** | **5.8.0** | Post-deprecation of `warmup_ratio` and `group_by_length`. Our code resolves `warmup_ratio` → `warmup_steps` locally (HF `ceil` semantics) and translates `group_by_length: true` → `train_sampling_strategy="group_by_length"` + auto-populated `length` column. |
| **peft** | **0.19.1** | `get_peft_model_state_dict` at `peft.utils.save_and_load`. Save / reload roundtrip clean on 411 tensors of real Gemma 4 E2B + r=8 LoRA + projector. **Requires AWQ dispatcher patch** when gptqmodel >= 7.0.0 is installed (see below). |
| trl | 1.4.0 | `SFTConfig.train_sampling_strategy` available; `length_column_name` defaults to `"length"`. |
| accelerate | 1.13.0 | |
| bitsandbytes | 0.49.2 | Installed for compatibility, but project policy forbids 8-bit / 4-bit outside explicit QLoRA-flagged runs. `src/config.py:validate_config` rejects every `*8bit*` / `*4bit*` / `_bnb_*` / `paged_*` optim name. |
| tokenizers | 0.22.2 | |
| safetensors | 0.8.0rc0 | |
| huggingface_hub | 1.14.0 | |
| datasets | 4.8.5 | |
| xformers | 0.0.35 | |
| triton | 3.6.0 | |

### Quantization stack

| Package | Version | Notes |
|---|---|---|
| **gptqmodel** | **7.0.0** | Requires `torch >= 2.7.1` (satisfied) + `optimum >= 1.24.0`. |
| **optimum** | **2.1.0** | Required by gptqmodel. |
| torchao | (recent) | Used by `gptq_torchao_hybrid.py` for int4 embedding pack. |
| **mlx** | **0.31.2** (Mac) / **0.32.0.dev** from source @ `main` (Linux + NVIDIA) | See [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md) — the pypi `mlx-cuda-12==0.31.1` wheel has a QMM-kernel bug that breaks Gemma 4 INT4 generation. |
| mlx_vlm | (matches mlx) | Multimodal `mlx_vlm.load` + `mlx_vlm.convert`. Output is directly loadable by `mlx-swift-lm` on iOS. |
| mlx_lm | (matches mlx) | Used only via `mlx_lm.quant.*` cores in the hybrid flow (`mlx_vlm.load → mlx_lm.quant.* → save_weights`). Direct `mlx_lm.load` on Gemma 4 produces a different forward pass than the iOS runtime — see "Known bugs" §4. |

### Invariants this combination satisfies

Version-sensitive code paths in `finetune/src/` that depend on the
table above:

- **TF32**: `_resolve_effective_tf32` enables TF32 only when
  `torch.cuda.get_device_capability(0)[0] >= 8`. 4090 = (8, 9) → on.
  Passing `tf32=True` on pre-Ampere raises in transformers 5.x; the
  helper omits the SFTConfig kwarg in that case.
- **warmup_ratio**: forwarded as
  `warmup_steps = ceil(ratio * total_steps)` to bypass the
  `warmup_ratio is deprecated and will be removed in v5.2` warning.
- **group_by_length**: `transformers.TrainingArguments` no longer
  accepts `group_by_length`. `trl.SFTConfig.train_sampling_strategy`
  is the replacement. `LengthGroupedSampler` requires a `length`
  column on the dataset — auto-populated by `_approx_length` in
  `real_train` when the yaml flag is on.
- **PEFT save→reload**: `extract_savable_state` (wrapping
  `get_peft_model_state_dict`) returns the exact set of tensors PEFT
  writes, so the in-pipeline tripwire
  `_assert_save_matches_in_memory_state` compares bytewise against
  `adapter_model.safetensors` immediately after `save_pretrained`.
- **VLM eval class**: `AutoModelForImageTextToText` (transformers
  4.45+) is used by `src/evaluate.py` and `src/export_mlx.py` to keep
  `vision_tower.*` / `embed_vision.*` in the loaded graph.
  `AutoModelForCausalLM` silently drops them.

## Known bugs in the upstream stack

Each entry: what fires, what it looks like, root cause, fix /
workaround in this repo.

### 1. `transformers 5.5` exposed dead `k_proj` / `v_proj` on Gemma 4 E2B KV-shared layers; `transformers 5.8` removes them, dropping ~16 % of LoRA capacity on reload

**Symptom**: a LoRA finetune trained under `transformers 5.5` evaluates
at exactly the **base** behaviour after save → reload via PEFT. Train
metrics show convergence (loss ~1e-4, mean_token_accuracy = 1.0).
In-memory PEFT model evaluates correctly. Loaded-from-disk PEFT model
collapses to base behavior. **No warning, no error.**

**Mechanism**: Gemma 4 E2B has 35 decoder layers. Layers 0-14 are
global attention with standalone `q/k/v/o_proj` Linears. Layers 15-34
are KV-shared attention that reuses `(K, V)` from a sibling earlier
layer.

- `transformers 5.5` allocated `k_proj`, `v_proj`, `k_norm`, `v_norm`
  on every layer (even the shared ones — dead, but present).
- `transformers 5.8` **removes** those modules on the shared layers
  via a `not self.is_kv_shared_layer` gate in `Gemma4TextAttention.__init__`.
  Old base weights for those keys land on
  `_keys_to_ignore_on_load_unexpected` and are silently dropped.

PEFT trained 245 LoRA modules (490 tensors) on the 5.5 model. PEFT
reloaded under 5.8 finds only 205 matching modules — silently drops
80 LoRA tensors. The model loses 16 % of its adaptations.

**Fix**: upgrade `transformers` to ≥ 5.8 **and retrain the adapter**.
Old adapters trained on 5.5 are incompatible with the 5.8 model
layout. The in-pipeline tripwire `4fab396` now catches this class of
bug — see [`15-postmortems.md`](15-postmortems.md) §1 for the
debug-marathon timeline that isolated it.

**Cross-platform note**: the iOS-runtime parity audit
([`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md))
confirms `mlx_vlm` and `mlx-swift-lm` both match `transformers 5.8`
semantics. Old 5.5 adapters loaded on iPhone behave the same way
they behave on `transformers 5.8` — dead LoRA tensors land but never
fire, net 16 % capacity loss.

### 2. `peft 0.19.1` × `gptqmodel 7.0.0` — AWQ dispatcher ImportError

**Symptom**: any `get_peft_model()` call raises `ImportError` when
both `peft 0.19.1` and `gptqmodel 7.0.0` are installed, even for
non-AWQ LoRA training.

**Mechanism**: `gptqmodel 7.0.0` renamed `AwqGEMMQuantLinear` →
`AwqGEMMLinear` in `gptqmodel.nn_modules.qlinear.gemm_awq`. `peft
0.19.1` hardcodes the old name in `peft/tuners/lora/awq.py`:
`dispatch_awq` runs unconditionally during adapter injection.

**Fix**: one-line patch to `peft/tuners/lora/awq.py` (~line 97):

```diff
     if is_gptqmodel_available():
-        from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMQuantLinear
+        try:
+            from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMQuantLinear
+        except ImportError:
+            from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear as AwqGEMMQuantLinear
```

Apply in-place at
`$CONDA_PREFIX/lib/python3.12/site-packages/peft/tuners/lora/awq.py`.
**Re-apply after any `pip install -U peft`.**

### 3. `transformers 5.8` `save_pretrained` drops 60 KV-shared tensors that `mlx_vlm.convert` strict-loads

**Symptom**: `mlx_vlm.convert` (or `mlx_vlm.load`) on a model
round-tripped through `transformers ≥ 5.8`'s `save_pretrained` fails
with `Missing 60 parameters`.

**Mechanism**: the same KV-shared layers (15-34 on Gemma 4 E2B) that
don't have standalone `k_proj` / `v_proj` modules under 5.8 also
don't get their inert weights written by `save_pretrained` —
`_keys_to_ignore_on_load_unexpected` adds them to a silent-drop list.
`mlx_vlm.convert` strict-checks the key set and refuses the load.

**Fix**: bypass `transformers` for the merge. The
**safetensors-level LoRA merge** in
`quantization/scripts/repair/merge_safetensors.py` (`4c00bf0`) opens
base + adapter via the `safetensors` library, applies LoRA deltas in
fp32 + casts back, replaces `modules_to_save` tensors by direct
copy. Preserves the dead KV keys.

The `prep_inject_kv_shared.py` helper does the inverse repair on
already-saved v5.8 checkpoints — copies the missing KV bytes from a
v5.5 base.

### 4. Apple's `mlx_lm` Gemma 4 forward pass diverges from `mlx_vlm` and `mlx-swift-lm`

**Symptom**: `mlx_lm.quant.gptq` on Gemma 4 produces NaN logits.
`mlx_lm.quant.awq` raises `KeyError: 'gemma4'`. `mlx_lm.quant.dwq`
raises a broadcast shape mismatch `(2, 576) vs (2, 288)`. Only
`mlx_lm.quant.dynamic_quant` runs end-to-end — but its output dir
can't be loaded by `mlx_vlm` (which is what the iOS runtime expects).

**Mechanism**: `mlx_lm/models/gemma4.py` + `gemma4_text.py` is
**baseline-grade** vs the production `mlx_vlm/models/gemma4/*`:

| Aspect | mlx_lm (broken) | mlx_vlm (correct) |
|---|---|---|
| Projector module | `nn.Linear` | `ScaledLinear` |
| RMSNorm | `nn.RMSNorm` (with `+1` weight offset) | `RMSNormZeroShift` (no offset) |
| KV-shared layers K/V | sanitize drops the keys | allocated, fed shared K/V at runtime |
| `audio_config` setdefault | injects empty `{}` | n/a |

The bit-packing kernel is identical between `mlx_lm` and `mlx_vlm`
(both call `mlx_lm.utils.quantize_model` under the hood). What
differs is **which model tree gets walked** during the quant pass.
On Gemma 4 the trees disagree, so quantizing under `mlx_lm` gives
either NaN outputs or outputs that load fine in `mlx_lm` but
mis-render under `mlx_vlm` (and therefore under `mlx-swift-lm` on
iOS).

**Fix**: the **hybrid flow** in
`quantization/scripts/run/mlx_hybrid_quant.py` (`da6af9f`):
`mlx_vlm.load` first (gives the full multimodal tree with the right
classes), then `mlx_lm.quant.*` on `model.language_model` in-place,
then `mlx_vlm.utils.save_weights`. Output is directly `mlx_vlm.load`-able.

See [`15-postmortems.md`](15-postmortems.md) §2 ("MLX quantization
debug") for the full debugging walk.

### 5. `bnb-NF4` on the Gemma 4 vision tower collapses PlantNet from ~70 % to ~0.1 %

**Symptom**: bnb-NF4 quantization with default `skip_modules` produces
a model that emits garbage on plant ID prompts (`species_match` ≈ 0.001
on n=300).

**Mechanism**: default-on NF4 quantizes everything `nn.Linear`,
including the SigLIP-style vision encoder. Vision-encoder accuracy
collapses entirely at 4-bit.

**Fix**: `skip_modules` is **mandatory**. The default in
`quantization/src/methods/bnb_nf4.py` (`8ba5ea6`) now includes
`vision_tower` and `embed_vision`. The `inspect_vision_dtype`
tripwire in `quantization/scripts/inspect/` asserts the
`vision_tower` bf16 invariant on every saved quantized output.

### 6. `mlx-cuda-12==0.31.1` (pypi) produces ~7 % accuracy on Gemma 4 INT4 (vs ~50 % on Mac)

**Symptom**: Linux + NVIDIA eval of any quantized Gemma 4 model
produces 9-13 correct tokens followed by `<pad>` spam to `max_tokens`,
on every sample. `species_match` ≈ 0.077.

**Mechanism**: missing QMM (quantized matmul) kernel fixes —
specifically PRs landed between mlx `0.31.1` and the unreleased
`0.32.0.dev`, headlined by `#3509` "Guard qmm_naive scale and bias
loads at tile boundaries." The KV-cache contribution from ~280 vision
soft-tokens pushes the attention K-dim past a tile boundary; the
buggy kernel reads garbage scales/biases; logits collapse to `<pad>`.

**Fix**: build mlx `main` from source. Recipe:
[`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md).
The source build only **partially** fixes pad-spam (~48 % of
responses still pad-spam stochastically across runs), but aggregate
accuracy is tight (±0.33 pp on n=300).

### 7. `unsloth` `adamw_8bit` default bleeds into bf16 SFT runs

**Symptom**: project policy [0] (no 8-bit / 4-bit outside explicit
QLoRA runs) silently violated; reported losses look correct but
optimizer-state numerics are 8-bit.

**Mechanism**: unsloth's `FastModel` setup defaults the optimizer
name to `adamw_8bit`. The `finetune.py` dispatcher used to honor a
hardcoded 8-bit optim for projector/vision param groups even when YAML
specified `adamw_torch`.

**Fix**: `6ee0112` switches all configs to `adamw_torch_fused` and adds
a config validator that **rejects** any `*8bit*` / `*4bit*` / `paged_*`
optim name string at config-load time.

### 8. PEFT silently drops `modules_to_save` registrations under wrapper-depth mismatches

**Symptom**: a config that requests `tune_projector: true` or
`tune_last_n_vision_layers: N` saves an adapter where the projector
/ vision-layer tensors are **not** in the adapter safetensors. At
inference time, the merged model has the **base** projector and base
vision layers, not the trained ones.

**Mechanism**: PEFT's `modules_to_save` walker keys on the wrapped
module's exact name path. unsloth wraps differently than vanilla HF
(extra `base_model.model.` prefix). If the wrap depth doesn't match
PEFT's expectations, the registration is dropped — silently.

**Fix**: belt-and-braces tripwires in `finetune/src/freeze.py`:
- `ensure_projector_trainable` / `ensure_vision_layers_trainable` —
  re-flips `requires_grad=True` after PEFT wrap with a WARNING log if
  triggered.
- **Save tripwire** — after `save_pretrained`, scans adapter
  safetensors headers to confirm tensors exist for every tuned
  layer/module.
- **Export tripwire** — snapshots base bytes before adapter load;
  after merge, asserts at least one param per tuned module differs
  byte-for-byte. Identical = PEFT silently failed to restore
  `modules_to_save` weights → ship-stopper.

### 9. YAML config knobs that look loaded but aren't

**Symptom**: an ablation between `desc_act: true` and `desc_act: false`
shows ~0.5-1.0 pp delta — but the delta is actually a calibration-set
difference, not the algorithmic effect. The YAML field wasn't being
read.

**Mechanism**: `quantization/configs/*.yaml` were authored, but
`quantization/scripts/run/quant.py` instantiated `GPTQConfig()` with no
kwargs — `desc_act: true` / `group_size: 128` from YAML had **zero
effect** on actual runs. Pre-`f94697a`.

**Fix**: `f94697a` wires `--config` through to the constructor +
extracts calibration loading to `common/`. **Plus** a unit test
(`tests/test_run_quant_yaml.py`) that round-trips a non-default YAML
and asserts the constructed config matches. Catches this class of bug
at PR time, not eval time.

The Phase ε `desc_act` ablation row is flagged "RE-RUN PENDING" in
[`../quantization/B1-sft-results.md`](../quantization/B1-sft-results.md).

## Upgrade order (when bumping the stack)

Lesson from the Challenge #1 PEFT bug: upgrade in dependency order, not
in alphabetical order:

1. `pip install -U unsloth unsloth_zoo` (accept the pin-warning about
   `transformers` / `peft` moving past unsloth's tested set).
2. `pip install -U transformers peft trl accelerate`.
3. Re-apply the peft AWQ dispatcher patch from §2.
4. Re-run `scripts/smoke_save_reload.py` against
   `configs/smoke-save-reload.yaml`. If it still passes all four
   criteria (LoRA key set, modules_to_save key set, byte equality,
   forward-pass logits Δ = 0), the orphan-tensor bug class is not
   reintroduced and a real finetune is safe.
5. Update the version table at the top of this doc.

## CUDA device-order gotcha

On a multi-GPU host, `nvidia-smi` and PyTorch can enumerate GPUs in
**different order**. PyTorch defaults to `FASTEST_FIRST` (or PCI
topology); `nvidia-smi` uses PCI bus ID. Example:

| nvidia-smi index | PyTorch index | Device |
|---|---|---|
| 0 | 1 | GTX 1650 Max-Q (4 GB) |
| 1 | 0 | RTX 4090 (24 GB) |

To isolate the 4090 reliably:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1    # PCI_BUS_ID ordering → 4090
```

Or with default `FASTEST_FIRST`:

```bash
export CUDA_VISIBLE_DEVICES=0    # FASTEST_FIRST → 4090
```

Mismatch is a common source of "training silently runs on the wrong
GPU" surprises on hosts with mixed-tier hardware.

## How to reproduce the snapshot

```bash
export LD_LIBRARY_PATH=<conda-env-root>/lib:$LD_LIBRARY_PATH
conda run -n torch python - <<'PY'
import importlib.metadata as md, sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
for p in ['unsloth','unsloth_zoo','transformers','peft','trl','accelerate',
          'bitsandbytes','tokenizers','safetensors','huggingface_hub',
          'datasets','xformers','triton','gptqmodel','optimum']:
    try: print(p, md.version(p))
    except md.PackageNotFoundError: print(p, '(not installed)')
PY
```

## Cross-references

- The Trailogy writeup's Challenge #1 maps directly to bug §1 above.
- KV-shared layer parity audit on the inference side:
  [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md)
- Full debug-marathon timeline of the silent-PEFT-loading bug:
  [`15-postmortems.md`](15-postmortems.md) §1
- Cross-backend eval contract:
  [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)
