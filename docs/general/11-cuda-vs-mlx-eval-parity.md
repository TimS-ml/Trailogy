# Linux mlx-cuda vs Mac mlx-metal as Eval Backends

## TL;DR

- The published Linux CUDA MLX wheel tested here produced severely broken Gemma 4 INT4 generations due to missing quantized-matmul fixes.
- Building MLX from source recovered most aggregate accuracy but still left substantial stochastic pad-token degeneration.
- Linux aggregate sweep comparisons were stable across repeated runs, but per-sample analysis remained unreliable.
- The remaining Mac-versus-Linux gap was mostly caused by degenerate outputs, not clean arithmetic differences on valid responses.

## Backend Comparison

| Backend | mlx version | M1 quant eval (n=300, seed=0) | Pad-spam rate | When to use |
|---|---|---|---|---|
| Mac mlx-metal | 0.31.2 (pypi) | **49.7 %** (authoritative) | 0 % | Reporting accuracy; bit-exact reproducibility |
| Linux mlx-cuda — pypi wheel `0.31.1` | 0.31.1 | **7.7 %** (broken) | 100 % (after ~10 tokens) | **Do not use** |
| Linux mlx-cuda — `0.32.0.dev` from source @ `main`, 3-run mean | 0.32.0.dev | **40.6 %**, range [40.3, 41.0] | **~48 %** (stochastic) | Sweep comparisons on Linux (aggregates are stable to ±0.33 pp); **not** per-sample analysis |

Two things to understand from the empirical 3-run sweep below:

1. **Linux aggregates are tight**: the 3 runs landed at 121, 121, 123
   matches out of 300 → ±0.33 pp range. Comparing two quant variants
   on Linux is safe to within that band.
2. **The source build only partially fixes pad-spam**. ~48 % of
   per-sample responses on Linux still degenerate into `<pad>` tokens
   at high response_len, and **which samples are affected drifts
   across runs**. Of all the cases where Mac matches but Linux misses,
   64-76 % are explained by Linux pad-spamming that sample. The
   remaining Mac→Linux gap (after subtracting pad-spam) is real kernel
   arithmetic difference, but it's the minority of the 9 pp.

The Mac→Linux gap (49.7 → ~40.6) is therefore **mostly residual
pad-spam, not fp16/bf16 accumulation differences in well-formed
outputs.** Treat the from-source build as "bug shrunk, not eliminated"
— upstream may need further fixes past `#3509`.

## The bug — pypi `mlx-cuda-12==0.31.1` is missing QMM kernel fixes

`mlx-cuda-12` (and `mlx-cuda-13`) on pypi caps out at `0.31.1` (tagged
2026-03-12). Apple shipped `mlx-metal` 0.31.2 on 2026-04-22, but never
published a matching CUDA wheel.

Between the `0.31.1` tag and `0.31.2` Apple landed the highlighted
**"Wider support for CUDA quantized matmuls"** PRs:
`#3255` (Pipelined QMM), `#3268` (fp + int4 qmm_sm80), `#3321`
(GatherQMM), `#3352` (3/5/6-bit qmm_naive), `#3417`
(GatherQMM sm80/naive path).

And on `main` after `v0.31.2`:
`#3379` "Handle residue k in qmm_naive", `#3443`, `#3445` "Fix
qmm_naive K-tail dispatch for FP quantized kernels",
`#3503` "Fix gather_mm", and most suggestively
**`#3509` "Guard qmm_naive scale and bias loads at tile boundaries"** —
which reads exactly like our symptom.

### Symptom

With the Mac M1 INT4 model rsynced to Linux and run via mlx-cuda
0.31.1, generation produces 9-13 correct tokens then degenerates into
`<pad>` spam to `max_tokens`. Sample output (same image, same seed):

```
This appears to be Altramuz Perenne. Lupinus polyphyllus, the
large-leaved lupine, big-leaved<pad><pad><pad><pad><pad><pad>… <pad>
```

The first ~10 tokens are correct. The KV-cache contribution from the
~280 vision soft-tokens pushes the attention K-dim past a particular
tile boundary, the `qmm_naive` kernel reads garbage scales/biases past
the boundary, hidden states get corrupt, every subsequent logit
collapses to `<pad>` (the highest-probability bail-out token).

### Triage that confirms it's a quantized-matmul CUDA bug, not Gemma 4

Same hardware, same model file, same seed, on a 4090:

| Test | Output | Verdict |
|---|---|---|
| INT4 + image (vision tokens in context) | 9-13 correct tokens → pad-spam | **bug fires** |
| INT4 + text-only prompt | 135 chars coherent (wrong species — no image) | bug masked |
| **bf16 base** + text-only | 532 chars coherent, factual | bf16 OK |
| **bf16 base** + image (template issue — model ignores it) | "please provide an image" (coherent, just no vision) | bf16 OK |

bf16 generation works → Gemma 4 itself is fine on mlx-cuda; the bug
needs **(quantized weights) AND (long enough K-dim that vision tokens
land past the buggy tile boundary)**.

## The fix — build mlx `main` HEAD from source

PyPI can't help (no 0.31.2+ wheel for CUDA). The reproducible recipe,
verified on a 4090 box:

### One-time toolchain install (no sudo, conda env)

```bash
mamba install -y -n mlx -c conda-forge \
  "cmake>=3.25" nanobind openblas \
  "cuda-toolkit=12.9"     # pulls cuda-nvcc, cccl, cublas-dev, gxx 14.x
```

Why CUDA 12.9 specifically:

- mlx CMakeLists explicitly **rejects CUDA 13.1**
  (`13.1 <= CUDAToolkit_VERSION < 13.2` is hard-blocked since
  `#3273`). System CUDA on many dev boxes is 13.1.x → blocked.
- The conda CUDA 12.9 toolkit installs alongside any system CUDA with
  zero conflict; we just point `CUDA_HOME` / `CUDACXX` at the conda
  copy and ignore the system one.
- If you upgrade the system to CUDA 13.2+, that path also works —
  just pass `MLX_BUILD_CUDA=ON` with a `13.2` toolkit.

### Build

```bash
git clone --depth 100 https://github.com/ml-explore/mlx.git
cd mlx

source <conda-prefix>/etc/profile.d/conda.sh && conda activate mlx
export CUDA_HOME=$CONDA_PREFIX
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export LAPACK_HOME=$CONDA_PREFIX
export BLAS_HOME=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CMAKE_BUILD_PARALLEL_LEVEL=2    # 16 GB host — raise on 32 GB+
export CMAKE_ARGS="\
  -DMLX_BUILD_CUDA=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_CUDA_ARCHITECTURES=89 \
  -DCUDNN_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/include \
  -DCUDNN_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib \
  -DNCCL_INCLUDE_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/include \
  -DNCCL_LIB_DIR=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib"

pip install . --no-build-isolation -v
```

Wall-clock ~1 hour on a 4090 box at `-j2`. QMM template instantiations
dominate (32 `qmm_*.cu` files, each 3-5 min). The build replaces the
`mlx` Python package in place; the existing `mlx-cuda-12==0.31.1`
entry stays in `pip list` (it's an empty shim now) but its `libmlx.so`
is overwritten by the new build.

### Build-time gotchas

- **Use `-DMLX_CUDA_ARCHITECTURES=89`**, NOT
  `-DCMAKE_CUDA_ARCHITECTURES=89`. The CMake-to-nvcc translation
  otherwise errors with `'89' is not in 'keyword=value' format`.
- **System `cmake` (3.22 on Ubuntu 22.04) is too old.** Need ≥ 3.25
  from conda; make sure `which cmake` resolves into the conda env
  before invoking `pip install`.
- **`-j>=4` on a 16 GB host will OOM mid-QMM** (`Error 137`). Each
  nvcc instance on a QMM template peaks ~2 GB RSS. With 32 GB+ you can
  raise to 6-8 safely.

### Runtime environment

The bundled CUDA-12 wheels in `site-packages/nvidia/*` are not on the
default `LD_LIBRARY_PATH`, and the JIT path picks up system CUDA 13.1
headers by default — both will silently fail. The maintained helper at
`quantization/scripts/_env/_mlx_env.sh` exports the right vars
automatically, so the standard preface for any eval command is:

```bash
source quantization/scripts/_env/_mlx_env.sh
CUDA_VISIBLE_DEVICES=0 $MLX_PYTHON -m scripts.run.eval ...
```

If you ever need to do it manually:

```bash
export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux       # 12.9 headers
export CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux
export LD_LIBRARY_PATH=\
$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:\
$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/nccl/lib:\
$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Verification

Same M1 model, same `val_mac.jsonl`, n=300, seed=0:

| Build | species_match | rouge_l_mean | matches | avg_resp_len | pad-spam |
|---|---|---|---|---|---|
| Mac mlx-metal 0.31.2 (reference) | **49.7 %** | 0.573 | 149/300 | 175 | 0 % |
| Linux 4090 mlx-cuda-12 0.31.1 | 7.7 % | 0.165 | 23/300 | 411 | ~100 % |
| **Linux 4090 mlx 0.32.0.dev+main (3-run mean)** | **40.6 %** [40.3, 41.0] | **0.401** [0.393, 0.410] | **121.7** [121, 123] | 385 [372, 398] | **~48 %** |

Smoke before vs after (same image, same seed, max_tokens=96):

**Before (0.31.1):**
> "This appears to be Altramuz Perenne. Lupinus polyphyllus, the
> large-leaved lupine, big-leaved`<pad><pad>…<pad>`"

**After (0.32.0.dev):**
> "This appears to be Altramuz Perenne. Lupinus polyphyllus, the
> large-leaved lupine, big-leaved lupine, many-leaved lupine, blue-pod
> lupine, or, primarily in cultivation, garden lupin, is a species of
> lupine (lupin) native to western North America from southern Alaska
> and British Columbia and western Wyoming, and south to Utah and"

(Hits max_tokens mid-sentence — no degeneration.)

## Determinism contract — what eval guarantees and what it doesn't

After the fix, two consecutive smoke generations on the same image
with `seed=0, temp=0` produced different outputs. This led us to
re-run the full n=300 eval three times to quantify the effect. The
story has two layers — dataset selection is bit-exact across runs and
backends, but generation on CUDA is fp-jittered enough that ~31 % of
per-sample predictions change run-to-run while the aggregate stays
remarkably stable.

### Dataset selection is bit-exact across runs / machines / backends

`quantization/src/eval/plantnet.py`:

```python
records = load_test_data(str(config.val_jsonl), require_image=True)
if config.n_samples is not None and config.n_samples < len(records):
    rng = random.Random(config.seed)              # seed=0 by default
    records = rng.sample(records, config.n_samples)
```

`random.Random(seed).sample(records, n)` is Python's Mersenne Twister
— given the same `(seed, len(records), n)` it returns the same
selection in the same order on any platform. Empirically the 0th
sample is always the same image across five runs spanning two
backends.

### Generation is greedy in algorithm, fp-jittered on CUDA

`eval/model_loaders.py` calls `mlx_vlm.generate(..., max_tokens=128,
verbose=False)` with no `temperature` kwarg, so default `temp=0` ⇒
sampler is `argmax(logits)`. Algorithmically deterministic.

But "deterministic algorithm" ≠ "bit-exact floating-point output" on
a modern GPU. CUDA-specific sources of run-to-run drift that don't
(or barely) exist on Mac/Metal:

| Source | Effect |
|---|---|
| cuBLAS / cuDNN heuristic-selected GEMM kernels | Two launches of the same matmul may pick different tiled algorithms with different internal accumulation orders → last-bit numerics differ. |
| Multi-CTA parallel reductions | `sum` / `softmax-denom` / `dot` are non-associative in fp16/bf16; which SM finishes last decides the final reduce. |
| PTX JIT cache state | A cold run JIT-compiles and writes `$TMPDIR/mlx/<ver>/ptx`; a warm run reads it. Cold and warm can produce different register layouts. |
| No deterministic mode | mlx-cuda doesn't currently expose a `use_deterministic_algorithms` switch (PyTorch has it; we don't). |

These 1-ulp differences compound through 35 decoder layers ×
autoregressive accumulation. When `argmax(logits)` has two close
contenders, a single ulp can flip the token choice and the whole
sequence diverges from there. And — critically — these "close-call"
points are not rare; they cluster around the same tile-boundary
conditions that the half-fixed QMM kernels still mis-handle, which is
why pad-spam comes and goes per-sample across runs.

Mac/Metal is more reproducible in practice:
- M-series GPUs have fewer SMs and more predictable scheduling.
- MPS reductions tend to use fixed tile order.
- We re-ran the Mac eval and got bit-exact `eval.json` files.

### Empirical 3-run sweep (Linux, same model, same seed)

Three back-to-back `--plantnet_n 300 --eval_seed 0` runs of the M1
INT4 model on a 4090 with the from-source mlx 0.32.0.dev build:

| Metric | run1 | run2 | run3 | mean | range |
|---|---|---|---|---|---|
| `species_match` | 0.4033 | 0.4033 | 0.4100 | **0.4056** | ±0.33 pp |
| `species_matches` / 300 | 121 | 121 | 123 | 121.7 | stdev 1.15 |
| `rouge_l_mean` | 0.3926 | 0.4103 | 0.3989 | 0.4006 | ±0.009 |
| `avg_response_len` | 397.6 | 371.5 | 385.6 | 384.9 | ±13 chars |
| Pad-spam rate (resp_len > 500) | 50.7 % | 45.0 % | 48.3 % | 48.0 % | ±2.8 pp |

**Aggregate is rock-solid (±0.33 pp).** This is much tighter than the
"±1-2 pp" initial estimate — the law of large numbers averages out
per-sample noise efficiently at n=300.

### Sample-level divergence is large

Drilling into the 300×3 grid:

| | count | % |
|---|---|---|
| `pred_species` stable across all 3 runs | 206 / 300 | 68.7 % |
| `pred_species` differs in ≥ 1 run | 94 / 300 | 31.3 % |
| `species_match` flag stable across all 3 | 251 / 300 | 83.7 % |
| `species_match` flag flips in ≥ 1 run | 49 / 300 | 16.3 % |

Pairwise: any two of the three Linux runs disagree on 19-26 % of the
predicted species names, with 9-14 % of species_match flags flipping.
Why doesn't the aggregate move more? Because **gains and losses
across runs cancel**: for run1 vs run2, exactly 14 samples gained a
match and 14 lost one. This means **you cannot trust any single
per-sample diff on Linux**.

### Pad-spam is partially fixed, not eliminated

Per-sample classification across 3 runs:

| Per-sample pad-spam pattern | count | % | interpretation |
|---|---|---|---|
| `<pad>` spam in 0 / 3 runs | 49 | 16.3 % | stably clean |
| `<pad>` spam in 1 / 3 runs | 104 | 34.7 % | flips ON sometimes |
| `<pad>` spam in 2 / 3 runs | 113 | 37.7 % | flips OFF sometimes |
| `<pad>` spam in 3 / 3 runs | 34 | 11.3 % | stably broken |

72 % of samples (the "flips ON/OFF" rows) live near the still-buggy
tile-boundary — fp jitter from cuBLAS algo choices flips them in or
out of pad-spam on each run. Only 16 % of samples are stably clean
and 11 % are stably broken.

### Most of the Mac→Linux gap is pad-spam, not kernel arithmetic

Splitting each Linux run by `response_len`:

| | clean responses | pad-spammed responses |
|---|---|---|
| Mac (49.7 %) | 49.7 % match (300/300 are clean) | 0 responses |
| Linux run1 | 40.5 % match (148 samples) | 40.1 % match (152 samples) |
| Linux run2 | 41.2 % match (165 samples) | 39.3 % match (135 samples) |
| Linux run3 | 46.5 % match (155 samples) | 35.2 % match (145 samples) |

Of the samples where Mac matches but Linux misses, **64-76 % are
explained by Linux pad-spamming that sample** (the prediction string
truncates mid-species-name to `<pad>`, defeating the species-extract
regex). The remaining 24-36 % are genuine kernel-arithmetic
disagreements between clean Metal and clean CUDA outputs.

So the 9 pp gap to Mac is roughly:
- ~6 pp from residual pad-spam (would close if upstream lands a
  follow-up to `#3509` covering the rest of the tile-boundary cases).
- ~3 pp from fp16/bf16 accumulation differences between Metal and
  CUDA QMM kernels, even on outputs that don't pad-spam.

### How to read a Linux-vs-Mac accuracy gap

| | Mac (Metal) | Linux (CUDA, 0.32.0.dev) |
|---|---|---|
| Same `seed=0` → same 300 images, same order | ✅ | ✅ |
| Same image → same token sequence run-to-run | ✅ (bit-exact) | ❌ (~31 % drift) |
| `species_match` aggregate floats run-to-run | ~0 pp | ±0.33 pp (measured at n=300) |
| Pad-spam rate | 0 % | ~48 % (stochastic) |

Implications:

- **Within-machine variant comparisons** (e.g. M1 vs M2 vs M3 on
  Linux) are fair on aggregate to ~±0.5 pp — same noise applies to
  all variants. **Differences smaller than ~1 pp on a single Linux
  run are noise.**
- **Per-sample analysis on Linux is unreliable.** ~31 % of
  predictions drift between identical runs. Use Mac if you need to
  study which exact samples a quant variant gets right.
- **Cross-backend bit-exact equivalence** at the token level is not
  achievable; Mac remains the authoritative reporting number.
- **Tightening Linux noise** if you need a closer estimate: raise
  `--plantnet_n` (600 or 1000) so the law of large numbers averages
  per-sample fp drift even more, or run the eval 3× and report the
  mean.

## Decision guide — which backend for which task

| Task | Use |
|---|---|
| Reporting authoritative quant accuracy | Mac mlx-metal |
| Cross-quant aggregate sweep on Linux (M1 vs M2 vs M3 etc.) | Linux mlx-cuda 0.32.0.dev — fast, ±0.33 pp noise on aggregate at n=300 |
| **Per-sample** analysis (which images flip from miss → hit) | Mac only — Linux drifts on ~31 % of samples per run |
| Sanity check that a model file isn't broken (no pad-spam in <10s smoke) | Either backend; full pad-spam will be obvious on the first generation |
| Bit-exact reproducibility for paper figures | Mac only |
| First debugging response to "Linux eval is 7 %" | Confirm the from-source mlx is loaded (`python -c "import mlx; print(mlx.__file__)"` → should resolve to the local site-packages `mlx/core.cpython-…so`, not the empty `mlx-cuda-12` shim) and that `_mlx_env.sh` was sourced |
| First debugging response to "Linux eval is 40 %, expected 50 %" | Normal. The ~10 pp residual is the not-fully-fixed pad-spam (~48 % of responses) + minor kernel-arithmetic differences. Wait for an upstream follow-up to `#3509`. |

## File pointers

| Concept | Path |
|---|---|
| Eval driver | `quantization/scripts/run/eval.py` |
| PlantNet benchmark logic + seeded subsetting | `quantization/src/eval/plantnet.py` |
| mlx_vlm loader and `infer_text` adapter | `quantization/src/eval/model_loaders.py` |
| Linux runtime env helper (sets CUDA_HOME + LD_LIBRARY_PATH) | `quantization/scripts/_env/_mlx_env.sh` |

## Cross-references

- Eval methodology (what `species_match` and `rouge_l` actually
  compute, plus the benchmark-drift caveat):
  [`10-eval-setup.md`](10-eval-setup.md).
- The downstream consumer of these numbers:
  [`../quantization/B2-sft-results.md`](../quantization/B2-sft-results.md).
- The other Gemma 4 + MLX parity bug (KV-shared layer audit):
  [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md).
