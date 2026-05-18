# Docs

## TL;DR

- These docs explain how Trailogy's on-device Gemma 4 model was trained,
  evaluated, quantized, and shipped into the iOS app.
- `data_mix/`, `finetune/`, and `quantization/` mirror the three model
  workstreams: dataset construction, SFT experiments, and deploy-time
  compression.
- `general/` holds cross-cutting notes: architecture, iOS runtime,
  memory management, eval caveats, package bugs, postmortems, and final
  results.
- Start with the reading order below if you want the shortest path to
  "what shipped and why".

Engineering notes and experiment reports. The per-module docs mirror
the code under `src/`; `general/` carries cross-cutting docs
(architecture, dev timelines, eval setup, known bugs, postmortems).

```
docs/
├── general/          # architecture, timelines, eval, postmortems (cross-cutting)
├── data_mix/         # data prep, mix recipes, prompt-prefix gating
├── finetune/         # SFT pipeline, projector/vision modes, final recipe
└── quantization/     # post-training quantization (GPTQ, MLX, EoRA)
```

These docs trade some polish for completeness. They describe the
actual experiments that shipped, including failed paths and the
reasoning behind each design decision.

## Reading order

The shortest path to "what shipped and why":

1. **[`general/01-architecture-model-pipeline.md`](general/01-architecture-model-pipeline.md)**
   — three-track model pipeline (data_mix / finetune / quantization)
   and how the deploy artifact lands in the iOS bundle.
2. **[`data_mix/01-data-prefix.md`](data_mix/01-data-prefix.md)** —
   the v4 camera-state prefix gate (`[camera=on]` / `[camera=off]`):
   concept + how it's plumbed end to end.
3. **[`finetune/01-pipeline.md`](finetune/01-pipeline.md)** —
   end-to-end SFT pipeline (data → bf16 LoRA training → merge → MLX
   conversion).
4. **[`finetune/03-anti-forgetting-and-final-recipe.md`](finetune/03-anti-forgetting-and-final-recipe.md)**
   — the shipped recipe (r=8 / α=8 / no KL, 3 epochs on mix-50k) +
   why KL turned out to be overkill at small rank.
5. **[`quantization/00-quantization-report-pub.md`](quantization/00-quantization-report-pub.md)**
   — quantization overview + headline result (M2 + EoRA r=64 at
   3.6 GB / 88.0 % PlantNet).
6. **[`quantization/05-mlx-vlm-design.md`](quantization/05-mlx-vlm-design.md)**
   — MLX stack mental model: `mlx-vlm` is the deploy substrate,
   `mlx-lm` contributes quant cores via the hybrid flow.
7. **[`quantization/B2-sft-results.md`](quantization/B2-sft-results.md)**
   — measured quality of the deployed MLX path on PlantNet.
8. **[`general/10-eval-setup.md`](general/10-eval-setup.md)** — what
   the eval driver actually measures, **and explicit caveats about
   benchmark drift across phases**. Read before comparing any two
   numbers in this repo.
9. **[`general/15-postmortems.md`](general/15-postmortems.md)** —
   what broke and how we found it.

## Per-module entry points

* **general:** [`general/README.md`](general/README.md) — cross-cutting (architecture, timelines, eval, parity audits, postmortems)
* **data_mix:** [`data_mix/README.md`](data_mix/README.md)
* **finetune:** [`finetune/README.md`](finetune/README.md)
* **quantization:** [`quantization/README.md`](quantization/README.md)
