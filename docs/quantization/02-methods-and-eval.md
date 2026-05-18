# Quantization methods & eval

## TL;DR

- This is the methods companion to the quantization report: it explains which compression routes were compared and what counted as a fair eval.
- `mlx_vlm.convert -q` is the default deploy route because it produces the MLX/VLM model shape consumed by the iOS runtime.
- CUDA/HF methods such as GPTQ and AWQ are useful reference paths, but their outputs are not ship candidates unless they can be converted back into the iOS-loadable MLX/VLM tree.
- bitsandbytes NF4 is a warning case, not a candidate, because generic vision-tower quantization collapsed PlantNet accuracy to 0.1 %.
- Quick-test results use PlantNet n=300 with seed 0, so small deltas are iteration signals unless backed by larger evaluations.

What we test and how we measure it. Companion to:

- `B1-sft-results.md` — per-variant HF/CUDA results
- `B2-sft-results.md` — per-variant MLX results (iOS-deployable)
- `B1-bnb-nf4-vision-collapse.md` — why bnb NF4 kills vision-tower-quantized variants
- `05-mlx-vlm-design.md` — the active design for the MLX-VLM path

## Methods We Actually Compare

| Method | Why it exists | Deployment relevance |
|---|---|---|
| `mlx_vlm.convert -q` | Baseline MLX affine quantization. | Directly deployable. |
| MLX + EoRA | Recover quality lost by affine quantization. | Directly deployable if adapter path is supported. |
| HF/CUDA GPTQ hybrid | Mature PTQ reference for quality and size. | Reference only unless bridged to MLX/VLM. |
| bitsandbytes NF4 | Failure case and cautionary baseline. | Not deployable; quantizes vision tower badly. |
| QAT/QLoRA-style paths | Research alternatives. | Not default; only for explicit experiments. |

The default rule: if the result cannot load through the MLX/VLM deployment tree,
it is not a ship candidate.

## MLX/VLM Is The Default Path

`mlx_vlm.convert -q` is the baseline because it produces the model shape consumed
by the iOS runtime. It also skips the multimodal tower modules by default, which
is critical for plant vision.

The useful knobs are:

```text
q-bits:       2, 3, 4, 6, 8
group size:   32, 64, 128
mode:         affine and related MLX quant modes
predicate:    optional mixed-precision policies
```

The most important non-obvious point is that small output size is not enough.
The artifact must be generated under the same family of Gemma 4 model classes as
the iOS runtime. See [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md).

## Why NF4 Is A Warning, Not A Candidate

Generic bitsandbytes NF4 quantizes all eligible linear modules, including the
vision tower. On this task, that collapses PlantNet behavior. The lesson is
simple: preserving the vision tower is a ship gate, not an optimization detail.

## Evaluation Protocol

The quick test is PlantNet n=300 with seed 0. It is for iteration speed, not for
claiming tiny differences. A final claim should either be backed by a larger run
or described as a quick-test result.

Each comparison should specify:

- model variant;
- backend / loader;
- model size;
- eval file and sample count;
- generation settings;
- PlantNet species match;
- whether the artifact is iOS-loadable.

## Metrics

| Metric | Why it matters |
|---|---|
| `species_match` | Primary plant-ID signal. |
| ROUGE-L | Secondary fluency/overlap signal. |
| response length | Catches pad spam or premature failure. |
| model size | Enforces mobile budget. |
| loader compatibility | Distinguishes deploy artifacts from reference artifacts. |

## Tripwires

Stop and investigate if any of these occur:

- PlantNet drop exceeds the accepted band against the same-framework bf16 reference.
- Species match collapses near zero.
- Output size exceeds the mobile budget.
- Vision tower dtype inspection shows unintended quantization.
- Eval uses a different split or generation recipe than the bf16 reference.

## Why Same-Framework References Matter

Compare MLX quantized rows to MLX bf16 and HF/CUDA quantized rows to HF/CUDA
bf16. Cross-framework differences may reflect loader, processor, or backend
behavior rather than quantization quality.

## Related Docs

- Headline results: [`00-quantization-report-pub.md`](00-quantization-report-pub.md)
- MLX model-tree design: [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md)
- Full MLX result matrix: [`B2-sft-results.md`](B2-sft-results.md)
- NF4 failure case: [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md)
