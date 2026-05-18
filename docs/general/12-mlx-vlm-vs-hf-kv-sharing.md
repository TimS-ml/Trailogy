# MLX-VLM / mlx-swift-lm vs hf_transformers — Gemma 4 E2B KV-share layout

Cross-check triggered by the overfit-memorization debug (see
[`15-postmortems.md`](15-postmortems.md) §1 "PEFT silent loading"). On
the training side, `transformers 5.5` exposed standalone `k_proj` /
`v_proj` modules in every one of E2B's 35 language-model decoder
layers. `transformers 5.8` removed those modules on the KV-shared
layers (15-34 for E2B), so old adapters silently lose 80 LoRA tensors
on reload via PEFT.

The inference side on iPhone is **not** `transformers`. It is the MLX
path: `mlx_vlm` (Python reference) plus `mlx-swift-lm` (the actual
runtime). This note audits both for the same architectural issue.

Repos audited (as of 2026-05-14):

- `mlx-vlm` (Python) — `mlx_vlm/models/gemma4/`
- `mlx-swift-lm` (Swift, iOS runtime) — `Libraries/MLXLLM/Models/Gemma4Text.swift`
- `hf_transformers` 5.8 — `src/transformers/models/gemma4/modular_gemma4.py`

## TL;DR

Both MLX paths compute KV-sharing **semantically identically** to
`transformers 5.8` (the correct version). Neither replicates the
`transformers 5.5` bug. The forward pass on E2B layers 15-34 always
uses `(K, V)` taken from the last non-shared layer of the same
`layer_type`; per-layer `k_proj` / `v_proj` are never invoked on those
20 layers.

Difference from `transformers 5.8`: both MLX paths still **instantiate**
`k_proj` / `v_proj` / `k_norm` / `v_norm` modules on layers 15-34 and
load weights into them from the checkpoint. They are dead parameters
— ~26 MB of bf16, ~7 MB of int4 — carried as inert tensors. No
correctness impact, no LoRA implication for newly-trained adapters.

## Reference — the v5.5 → v5.8 change in hf_transformers

`hf_transformers/src/transformers/models/gemma4/modular_gemma4.py`:

```python
class Gemma4TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
        ...
        self.q_proj = nn.Linear(...)
        self.q_norm = Gemma4RMSNorm(...)
        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:                # <-- v5.8 gate
            self.k_norm = ...
            self.v_norm = ...
            self.k_proj = nn.Linear(...)
            self.v_proj = nn.Linear(...) if not use_alt else None
        self.o_proj = nn.Linear(...)
```

Layer 15-34 attention objects on E2B have **no `k_proj` / `v_proj` /
`k_norm` / `v_norm` attribute at all** under v5.8.

In forward:

```python
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]
    ...
else:
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
    key_states = key_states.transpose(1, 2)
    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)
...
if self.store_full_length_kv:
    shared_kv_states[self.layer_type] = key_states, value_states
```

`store_full_length_kv` is set only on the **last** non-shared layer of
each `layer_type`. That layer's post-RMSNorm post-RoPE post-transpose
`(K, V)` is the value reused for all later shared layers of the same
type.

Old checkpoints (from Google) still ship with `k_proj` / `v_proj` /
`k_norm` / `v_norm` weights for every layer; v5.8 ignores them via
`_keys_to_ignore_on_load_unexpected`. So the bug is **only** on the
LoRA wrapping side — when PEFT walks the model after wrapping, layers
15-34 no longer expose those Linears, and adapter tensors keyed against
them are silently dropped on load.

## mlx-vlm (Python reference)

`mlx_vlm/models/gemma4/language.py` `class Attention`:

```python
def __init__(self, config, layer_idx, kv_shared_only: bool = False):
    ...
    self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
    if not kv_shared_only:                              # <-- module-wide flag
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    if not kv_shared_only:
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)
    ...
    first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
        config, "num_kv_shared_layers", 0)
    self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
```

`kv_shared_only` is a **constructor argument** that propagates from
the top of the model: `Gemma4TextModel.__init__(config,
kv_shared_only=False)` hard-defaults to `False`, and `LanguageModel`
does `self.model = Gemma4TextModel(config)` without overriding. **Every
Attention instance is built with `kv_shared_only=False`**, so all 35
layers get `k_proj`, `v_proj`, `k_norm`, `v_norm` instantiated.

The forward path is the gate that makes this safe:

```python
queries = self.q_proj(x).reshape(...)
queries = self.q_norm(queries)
if shared_kv is not None:
    keys, values = shared_kv               # <-- bypass k_proj / v_proj entirely
else:
    keys = self.k_proj(x).reshape(...)
    if self.use_k_eq_v:
        values = keys
    else:
        values = self.v_proj(x).reshape(...)
    keys = self.k_norm(keys)
    keys = keys.transpose(0, 2, 1, 3)
    keys = self.rope(keys, offset=offset)
    values = self.v_norm(values)
    values = values.transpose(0, 2, 1, 3)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
...
return self.o_proj(output), (keys, values), offset
```

And the model loop wires `shared_kv` from a precomputed `previous_kvs`
map:

```python
num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
self.first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared
self.previous_kvs = list(range(len(self.layers)))
if num_kv_shared > 0:
    N = len(self.layers); M = N - num_kv_shared
    kvs_by_type = {}
    for i in range(M):
        kvs_by_type[self.layers[i].layer_type] = i   # last non-shared per type
    for j in range(M, N):
        self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]
...
for idx, (layer, c, m, prev_idx, pli) in enumerate(zip(...)):
    kvs, offset = intermediates[prev_idx]
    h, kvs, offset = layer(h, m, c, per_layer_input=pli, shared_kv=kvs, offset=offset)
    intermediates[idx] = (kvs, offset)
```

For shared layers `j` (E2B: 15-34), `prev_idx = previous_kvs[j]` points
into `[0, 15)` to the **last** layer of the same `layer_type`. By the
time the loop reaches `j`, `intermediates[prev_idx]` is populated with
the source layer's post-norm post-RoPE `(K, V)`. The `shared_kv`
argument is therefore non-`None`, the `else` branch is skipped, and
`k_proj` / `v_proj` on layer `j` are never called.

**Functional verdict**: matches `transformers 5.8` semantics exactly.
The `k_proj` / `v_proj` modules on layers 15-34 sit in memory but
contribute nothing to the forward pass.

## mlx-swift-lm (iOS runtime)

This is the actual iPhone runtime. The text decoder lives in
`Libraries/MLXLLM/Models/Gemma4Text.swift` and is shared by both
`.text` and `.vlm` loading modes via
`Libraries/MLXVLM/Models/Gemma4.swift`.

```swift
@ModuleInfo(key: "q_proj") var qProj: Linear
@ModuleInfo(key: "k_proj") var kProj: Linear
@ModuleInfo(key: "v_proj") var vProj: Linear?
@ModuleInfo(key: "o_proj") var oProj: Linear

@ModuleInfo(key: "q_norm") var qNorm: RMSNorm
@ModuleInfo(key: "k_norm") var kNorm: RMSNorm
@ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale

init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
    ...
    self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
    self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
    if !useKeqV {
        self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
    }
    self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)
    ...
}
```

`kProj` is declared non-optional and instantiated unconditionally for
every layer. `vProj` is optional only to support `useKeqV` (E2B does
not use that). **There is no `kv_shared_only` switch in Swift at all.**
Every layer's attention instance carries `kProj`, `kNorm`, `vProj`,
`vNorm`.

Forward is the same gate as Python:

```swift
if let (sharedK, sharedV) = sharedKV {
    keys = sharedK
    values = sharedV
} else {
    var k = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
    k = kNorm(k)
    k = k.transposed(0, 2, 1, 3)
    k = gemma4ApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

    var v: MLXArray
    if let vProj {
        v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
    } else {
        v = k
    }
    v = vNorm(v)
    v = v.transposed(0, 2, 1, 3)

    if let cache {
        let (updatedK, updatedV) = cache.update(keys: k, values: v)
        keys = updatedK
        values = updatedV
    } else { keys = k; values = v }
}
```

KV-share map:

```swift
self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
var kvMap = Array(0 ..< config.numHiddenLayers)
if config.numKvSharedLayers > 0 {
    var lastByType = [String: Int]()
    for i in 0 ..< firstKvSharedLayerIdx {
        lastByType[config.layerTypes[i]] = i
    }
    for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
        if let prev = lastByType[config.layerTypes[j]] {
            kvMap[j] = prev
        }
    }
}
self.previousKvs = kvMap
```

Identical "last non-shared layer of same `layerType`" mapping.

Forward loop:

```swift
for (idx, layer) in layers.enumerated() {
    let prevIdx = previousKvs[idx]
    let sharedKV = intermediates[prevIdx].kv
    let sharedPositionOffset = intermediates[prevIdx].positionOffset
    let mask = maskByType[layer.layerType]
    let (out, kvPair, positionOffset) = layer(
        h, mask: mask, cache: fullCache[idx],
        perLayerInput: perLayerInputs[idx],
        sharedKV: sharedKV,
        positionOffset: sharedPositionOffset)
    h = out
    intermediates[idx] = (kvPair, positionOffset)
}
```

Same as Python. Shared layers' `kProj` / `vProj` never fire.

## Implications for adapters and weights

1. **Old `transformers 5.5` LoRA adapters on iPhone**

   A LoRA trained on the old layout has 80 extra tensors on layers
   15-34's `k_proj` / `v_proj`. On MLX side, `mlx_vlm`'s
   `apply_lora_layers` wraps every `Linear` in `model.language_model`
   with `LoRaLayer`, including those dead `k_proj` / `v_proj` on
   layers 15-34. The adapter load is `load_weights(..., strict=False)`,
   so the 80 tensors **do** get attached.

   But forward never calls those wrapped modules. The 80 LoRA tensors
   are inert — exactly the same outcome as on `transformers 5.8`
   where they're dropped at load time.

   Net effect on iPhone is identical to `transformers 5.8`: the
   adapter delivers ~84 % of the capacity it was trained with. **Do
   not reuse adapters trained under the old transformers.**

2. **New `transformers 5.8` LoRA adapters on iPhone**

   Adapters trained on the v5.8 layout have LoRA only on the surviving
   modules. The adapter file contains no tensors for the dead k_proj /
   v_proj on layers 15-34, so `strict=False` leaves their LoRA `A`
   random and `B` zero — identity (`y + (A @ B) * scale` = `y + 0`).
   Forward then ignores them anyway. Correct.

3. **Base-checkpoint weights on layers 15-34**

   The community-converted `mlx-community/gemma-4-e2b-it-4bit` carries
   `k_proj` / `v_proj` / `k_norm` / `v_norm` weights for every layer;
   `sanitize` passes them through and they get loaded into the dead
   modules. Dead-weight cost: ~7 MB in int4. Acceptable.

4. **Vision tower / projector / embed_vision**

   Unchanged from the v5.5 → v5.8 transition. The bug is strictly in
   the text-decoder attention modules. Vision / projector LoRA paths
   are unaffected.

## What this does NOT cover

- Did not run a side-by-side numerical comparison
  (`hf_transformers v5.8` logits vs `mlx-vlm` logits vs Swift logits
  on the same prompt + image). The audit is structural only.
- Did not check `gemma3` or `gemma3n` for similar drift. Out of scope
  for the current bug.
- Did not check whether `mlx_vlm.utils.convert` or `mlx_lm.convert`
  silently drops the dead `k_proj` / `v_proj` keys during quantization.
  Worth a follow-up if any future converter version is adopted — the
  Swift loader doesn't require those tensors and would still work,
  but a value-mismatch tripwire on adapter export
  (`finetune/src/export_mlx.py`) would need to know not to expect them.

## Action items (none urgent)

- (optional) Add a `kv_shared_only=True` pass to `mlx-vlm` and
  `mlx-swift-lm` Gemma 4 to skip allocating the dead modules. Saves
  ~7 MB int4 / ~26 MB bf16 of weight memory on E2B and ~1500 named
  parameters that `apply_lora_layers` currently wraps uselessly.
  Low priority — memory is fine and LoRA wrapping is idempotent.
- (followup) If we ever switch to a v5.8-aware checkpoint that
  **omits** the dead keys, verify `mlx-vlm` and `mlx-swift-lm` both
  load cleanly. They should — `mx.load_weights` is non-strict on
  missing keys by default in mlx — but worth a smoke test.

## Cross-references

- The user-facing version of the same bug (training side):
  [`15-postmortems.md`](15-postmortems.md) §1
- Tested package versions where the v5.8 fix lands:
  [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md)
- The other Gemma 4 cross-stack divergence (MLX-VLM vs MLX-LM):
  [`15-postmortems.md`](15-postmortems.md) §2 ("MLX quantization debug")
