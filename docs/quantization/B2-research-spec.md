# Route B.2 — MLX-native quant: algorithm-improvement program

## TL;DR

- This spec lays out longer-term work to improve MLX-native Gemma 4 quantization without relying on CUDA tooling.
- One track ports production GPTQ stability and recovery techniques into the MLX path, including act-order, dead-column handling, clipping, and low-rank residual recovery.
- The other track sweeps MLX conversion modes and mixed-precision predicates to map the data-free tradeoff curve.
- This is a research track, so readers should treat it as future improvement work rather than the current deployment recipe.

> last edit: 2026-05-14
>
> Status: parallel research track. **B.2 is NOT the priority
> deliverable this round** — that's B.1 (HF GPTQModel → MLX bridge).
> See `00-quantization-roadmap.md` for the route picker.

## 1. Goal

Make `mlx_lm/quant/*` strong enough to be a credible source of a
4-bit Gemma 4 E2B deliverable, without depending on a CUDA box at
inference time and without the format-translation step Route B.1
needs.

Concretely, two parallel sub-efforts:

1. **Algorithm port** from a prior GPTQ implementation
   into `mlx_lm/quant/gptq.py` — act-order, dead-column handling,
   Hessian-weighted auto-clip search, LQER low-rank residual
   compensation. (Was §8 in the previous version of this spec; now
   elevated to the spine because it's the actual core of B.2.)
2. **Format / mode sweep** through `mlx_vlm.convert`'s
   `--q-mode {mxfp4, nvfp4, mxfp8}` and `--quant-predicate {mixed_*}`
   options, to map the data-free affine baseline's tradeoff curve.

Calibration data design (text-only / mixed-text / multimodal) is
route-agnostic and lives in `04-calibration-data-design.md`. The
hybrid flow (`mlx_vlm.load` first, then `mlx_lm.quant.*` core on the
resulting tree, then `mlx_vlm.utils.save_*`) is the corrected pipeline
mental model from `05-mlx-vlm-design.md`.

## 2. Why this isn't the priority deliverable

Apple's `mlx_lm/quant/*` implementations have not been validated on
Gemma 4 with production-grade quality. The 2026-05-13 mac_mlx_lm round
documented four concrete issues:

- `mlx_lm/quant/gptq.py` produced NaN logits on Gemma 4 (under the
  wrong forward pass — needs re-test under the hybrid flow, but even
  then lacks `gptqmodel`'s mature tricks).
- `mlx_lm/quant/awq.py` has no `gemma4` entry in `AWQ_MODEL_CONFIGS`;
  convert fails immediately. Patch is ~5 lines but mandatory.
- `mlx_lm/quant/dwq.py` has a broadcast bug at line 113 in the
  validation-loss path; reproduces regardless of model tree.
- `mlx_lm/quant/dynamic_quant.py` was the only one that ran end-to-end
  — it produces 2.99 GB on disk after splice, but `dynamic_quant` is
  *sensitivity-based* bit allocation, not OBS-style PTQ. Different
  family from `gptqmodel`. Quality comparison is not apples-to-apples.

Meanwhile, `gptqmodel` on CUDA already produces 68.8 % PlantNet match
at n=2,870 on our SFT (`R2 gptq_w4g128_da1` in
`B1-sft-results.md`). The Route B.1 deliverable is to
*bridge* that output into mlx_vlm-loadable form. That's a one-time
engineering investment.

So B.1 ships the deliverable. B.2 (this doc) lives as the long-term
research track: a working MLX-native PTQ would mean no CUDA dependency
and no format-bridge step. Worth pursuing, lower priority.

## 3. Algorithm Port — The Core Work

A prior GPTQ implementation provides four production-grade GPTQ
improvements over Apple's baseline:

| Trick | What it does | Status in `mlx_lm/quant/gptq.py` |
|---|---|---|
| `act-order` (`desc_act`) | reorder columns by Hessian diagonal magnitude before quantizing | not present |
| dead-column handling | detect columns with zero / near-zero activation variance; skip Hessian update; treat as fixed | not present |
| Hessian-weighted auto-clip search | search for the optimal weight-clip threshold per group, weighted by Hessian diagonal | not present |
| LQER low-rank residual compensation | after quantization, fit a small low-rank residual to recover quality at near-zero size cost | not present |

Each is independently valuable. Suggested order of implementation:

1. **act-order**: ~80 lines. Trivial port (sort indices, permute
   columns, run the standard OBS update on the permuted ordering, save
   `g_idx` per layer). Direct port of `gptqmodel`'s `desc_act` path.
   `R2` in 01b already confirmed +0.5 pp on the HF side; same trick
   should land same gain MLX-side. **Start here.**
2. **dead-column handling**: ~30 lines. Detect via Hessian diagonal
   magnitude. Avoids the divide-by-zero / numerical-instability path
   that probably contributed to the 2026-05-13 NaN.
3. **auto-clip search**: ~150 lines including the search loop. Bigger
   win on aggressive bit budgets (2-3 bit); smaller on 4-bit. Defer
   until the first two are landed.
4. **LQER**: ~200 lines including the SVD residual fit. Largest
   engineering scope; biggest size-vs-quality lever. Defer until
   confident the first three are stable.

Each lands as a separate upstream-contribution-shaped PR against our
mlx-lm fork. Tests live in `mlx_lm/quant/test_gptq_*.py`.

## 4. Format / mode sweep through `mlx_vlm.convert`

`mlx_vlm.convert` already exposes these knobs without any mlx-lm
patching:

| Knob | Choices | Status today |
|---|---|---|
| `--q-mode` | `affine` (default), `mxfp4`, `nvfp4`, `mxfp8` | affine tested (M1 in 01c); others untested on this SFT |
| `--quant-predicate` | `mixed_2_6`, `mixed_3_4`, `mixed_3_5`, `mixed_3_6`, `mixed_3_8`, `mixed_4_6`, `mixed_4_8` | none tested on this SFT |

These are data-free recipes (no calibration). They're orthogonal to
the algorithm port in §3 — useful as quick wins because they don't
require any new code, just config sweeps.

Suggested cheap sweep:

| Variant | Recipe | Expected size | What we learn |
|---|---|---|---|
| `mlx_vlm_g128_mxfp4` | `--q-bits 4 --q-mode mxfp4 --q-group-size 128` | ~3.2 GB | MXFP4 vs affine on this SFT |
| `mlx_vlm_g128_nvfp4` | `--q-bits 4 --q-mode nvfp4 --q-group-size 128` | ~3.2 GB | NVFP4 vs affine vs MXFP4 |
| `mlx_vlm_mixed_3_4` | `--quant-predicate mixed_3_4` | ~3.0 GB | Cheap "UD-style" mixed-bit; size win at potential quality cost |
| `mlx_vlm_mixed_4_6` | `--quant-predicate mixed_4_6` | ~3.7 GB | Mixed-bit weighted toward larger; size cost for quality gain |

All evaluated at PlantNet val n=300 (quick test from
`02-methods-and-eval.md`). Each row gets one entry in
`B2-sft-results.md`.

These rows are the "no-research baseline" floor for any B.2
algorithm work to beat.

## 5. B.2 research backlog — the original variant matrix

The 8-row ablation matrix from the original spec is **research
backlog**, not priority work. Recorded here for when we resume:

| # | Variant | Pipeline | Calibration | Role |
|---|---|---|---|---|
| R0 | `bf16_reference` | none | n/a | ceiling |
| R1 | `mlx_vlm_g64_baseline` | `mlx_vlm.convert -q --q-bits 4 --q-group-size 64` | data-free affine | matches public mlx-community |
| R2 | `cuda_gptq_w4g128_da0` | `gptqmodel` w4g128, desc_act=False | 256 PlantNet text + 256 WikiText | cross-framework reference |
| R3 | `cuda_gptq_w4g128_da1` | `gptqmodel` w4g128, desc_act=True | same as R2 | cross-framework reference |
| **A** | `mlx_gptq_w4_g64_wiki` | hybrid MLX GPTQ | 512 WikiText (text-only) | calibration ablation |
| **B** | `mlx_gptq_w4_g64_mixedtext` | hybrid MLX GPTQ | 256 WikiText + 256 PlantNet text | calibration ablation |
| **C** | `mlx_gptq_w4_g64_mm` | hybrid MLX GPTQ | 256 WikiText + 256 PlantNet image+text | calibration ablation |
| C' | `mlx_awq_w4_g64_mm` | hybrid AWQ | same as C | AWQ production (needs the gemma4 patch in §3) |
| C'' | `mlx_dwq_w4_g64_mm` | hybrid DWQ | same as C | DWQ production (needs broadcast-bug fix) |
| C''' | `mlx_dynamic_quant_bpw4_mm` | hybrid dynamic_quant | same as C | dynamic_quant production |

A/B compares text calibration sources (Q2 in `04-calibration-data-design.md`).
B/C compares text vs multimodal calibration (Q1). C/R3 compares MLX
hybrid vs HF gptqmodel at the same calibration scope (Q3).

**Pre-condition**: at least act-order (item 1 in §3) needs to land
first. Without that, the MLX GPTQ rows are running an algorithm
weaker than `R3` and the comparison is meaningless.

## 6. Architecture

Hybrid-flow contract (unchanged from previous spec version):

```python
def load_via_mlx_vlm(model_dir: Path) -> tuple[Any, Any, dict]:
    """Load Gemma 4 via mlx_vlm. Returns (model, processor, config_dict).
    Caller is responsible for input dir having audio_config: null and
    no audio_tower weights (run prep_input first)."""

def quantizable_lm(model) -> Any:
    """Return model.language_model. This is the ONLY subtree fed to
    mlx_lm.quant.*. model.embed_vision and model.vision_tower are
    sibling subtrees, never seen by the quantizer, stay bf16."""

def save_via_mlx_vlm(model, config: dict, processor, out: Path) -> None:
    """Save through mlx_vlm format. Output is mlx_vlm.load-compatible."""

def assert_vision_preserved(out: Path) -> None:
    """Tripwire: verify vision_tower.* and embed_vision.* tensors exist
    in output, and config.quantization does NOT list any vision keys.
    Fail loud rather than ship a stripped model."""
```

Calibration loaders (`load_text_calibration`,
`load_multimodal_calibration`) and the `MultimodalCalibrationDriver`
wrapper live in `04-calibration-data-design.md`. Don't duplicate the
design here.

### Input prep pipeline (one-shot, idempotent)

1. Run `scripts/strip-gemma-audio.py <input_dir>`.
   Strips `audio_tower.*` + `embed_audio.*` tensors. Idempotent
   (no-ops if `.audio.bak` already exists in `scripts/backups/`).
2. Patch `<input_dir>/config.json`: set `audio_config: null` explicitly.
   Required because `mlx_vlm.utils.load_model:230` does
   `config.setdefault("audio_config", {})` — without explicit `None`,
   even a stripped checkpoint allocates audio modules and fails to load.

Pre-flight check before convert: scan output of prep, refuse to
continue if `audio_tower.*` keys still exist in safetensors header.

### Vision-preservation invariant

After every convert stage, `assert_vision_preserved(output_dir)` runs:

- ≥ 1 tensor key starts with `vision_tower.` AND
- ≥ 1 tensor key starts with `embed_vision.` AND
- None of those keys appear in `config["quantization"]`

Failure aborts the convert stage. No half-stripped artifact ships.

## 7. Validation (when B.2 work resumes)

Quick test = PlantNet val n=300, seed=0 (from `02-methods-and-eval.md`).
The phased plan:

| Phase | Variant | Pass criterion |
|---|---|---|
| **P1 — plumbing sanity** | Run hybrid flow on a small public model (any small mlx_vlm-loadable LM) with text-only calibration, GPTQ w4 g64. | non-NaN forward, PPL < 50. |
| **P2 — forward correctness on Gemma 4** | `unsloth/gemma-4-E2B-it` (un-SFT'd base) → hybrid GPTQ. | non-NaN forward, smoke generations coherent. Key milestone — proves the 2026-05-13 NaN was forward-pass-driven, not algorithmic. |
| **P3 — algorithm-port validation** | Apply the act-order port (§3 item 1). Compare MLX-side hybrid GPTQ output's PlantNet match against R3 (gptqmodel desc_act=True). | Within ±1 pt of R3. |
| **P4 — full matrix** | Backlog rows in §5. | A/B/C land in similar relative ordering to expected; assert_vision_preserved passes everywhere. |

P1-P3 sequentially gate any backlog work. P4 starts only after P3
shows MLX-side GPTQ matches CUDA-side gptqmodel.

### Unit tests

- `tests/test_hybrid_helpers.py`. Mock-driven tests for
  `load_via_mlx_vlm` argument plumbing,
  `MultimodalCalibrationDriver` call shape,
  `assert_vision_preserved` (positive + negative cases).
- `tests/test_calibration.py` lives WITH the calibration spec
  (`04-calibration-data-design.md`), not here.

## 8. Open risks

1. **The algorithm port (§3) is non-trivial work.** Each trick is
   ~30-200 lines of careful numerical code with its own test
   coverage. Estimate: 1-2 weeks per trick to land cleanly. The full
   four-trick port is the multi-month investment that B.2 represents.
2. **MLX-CUDA install on the 4090 box is broken** as of 2026-05-14
   (ABI skew between `mlx 0.31.2` bindings and `mlx-cuda-12 0.31.1`
   backend). Tracked in `TODO_run_local.md`. Not blocking B.1.
3. **AWQ gemma4 patch** is mandatory before C' is even possible.
   ~5-10 line copy from `gemma3` entry. Should be the first
   contribution to our mlx-lm fork.
4. **DWQ broadcast bug** at `mlx_lm/quant/dwq.py:113` is in the
   validation-loss path. Either fix upstream or disable the
   validation step. If neither is acceptable, mark C'' as blocked.
5. **`mlx_vlm.utils.save_weights` behavior under partial-quantization**
   (some leaf modules quantized, others bf16, vision/embed_vision
   bf16) is not battle-tested. Tripwire is `assert_vision_preserved`;
   if it fires, plumbing fix needed.

## 9. File pointers

| Topic | File |
|---|---|
| Strategic route picker | `00-quantization-roadmap.md` |
| Stack-level mental model | `05-mlx-vlm-design.md` |
| Calibration data design (used here) | `04-calibration-data-design.md` |
| 2026-05-14 debug post-mortem | `../../10-misc/2026-05-14-mlx-quantization-debug.md` |
| Per-variant results (HF/CUDA) | `B1-sft-results.md` |
| Per-variant results (MLX) | `B2-sft-results.md` |
| Methods & eval | `02-methods-and-eval.md` |
| BNB NF4 vision-collapse case study | `B1-bnb-nf4-vision-collapse.md` |
| Local-box blockers | `../../TODO_run_local.md` |
| Algorithm trick source | Prior GPTQ implementation notes |

---

## Sign-off checklist

- [ ] §3 algorithm-port priority (act-order first) confirmed.
- [ ] §4 cheap format/mode sweep variants confirmed.
- [ ] §5 backlog matrix scope acknowledged as deferred until
  algorithm-port lands.
- [ ] §6 hybrid-flow contract unchanged from prior spec confirmed.
- [ ] §7 P1-P3 phased validation gates confirmed.
- [ ] §8 open risks: contingency for each acknowledged.

When B.2 work resumes, invoke writing-plans to break the chosen sub-
spec (e.g. "act-order port") into implementation tasks with commit
boundaries and per-task test specs.
