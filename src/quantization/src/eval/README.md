# Eval harness — measuring quantization cost

One number per benchmark per variant. JSON output schema documented
inline in `runner.py` and below.

## Layout

| File | Role |
|---|---|
| `model_loaders.py` | `ModelHandle` dataclass + `load_hf_bf16` / `load_mlx_vlm` |
| `plantnet.py` | Domain metric (species exact-binomial match) |
| `wikitext_ppl.py` | Catastrophic-language guard (perplexity) |
| `vqav2.py` | Broader VLM metric (VQA accuracy on dev subset) |
| `runner.py` | Orchestrator + JSON writer |

## Add a new benchmark

1. Create `quantization/eval/<name>.py` with a `run(handle, config)`
   that returns a dataclass (so `dataclasses.asdict()` works).
2. Register it in `runner.BENCHMARK_REGISTRY`.
3. Add CLI knobs to `scripts/run_eval.py`.
4. Add a unit test under `tests/test_eval_<name>.py` exercising the
   pure-python helpers.

## Backend support matrix

| Benchmark | hf_bf16 | mlx_vlm |
|---|---|---|
| `plantnet_val` | ✅ | ✅ |
| `vqav2_devtest` | ✅ | ✅ |
| `wikitext_ppl` | ✅ | ⚠ skipped (mlx_vlm.generate doesn't expose forward logprobs) |

The PPL gap on MLX is the main hole. Two options if it matters:

- Run PPL only on the bf16 reference and rely on PlantNet/VQA scores
  for the INT4 variants (current default).
- Implement an `mlx_vlm` forward hook that returns logits without
  sampling. The MLX language model exposes `model.language_model(...)`
  directly, so a thin wrapper would do — TBW.

## Output schema

```json
{
  "variant": "mlx_vlm_g64",
  "model_path": "results/mlx_vlm_g64",
  "model_size_gb": 3.58,
  "backend": "mlx_vlm",
  "device": "mps",
  "eval_seed": 0,
  "generation_kwargs": {"do_sample": false, "max_new_tokens": "per-benchmark"},
  "benchmarks": {
    "plantnet_val": {"n": 2870, "species_match": 0.??, "rouge_l_mean": 0.??, ...},
    "vqav2_devtest": {"n": 1000, "accuracy": 0.??, ...},
    "wikitext_ppl": {"backend_supported": false, "notes": [...]}
  },
  "model_size_per_submodule_bytes": {...}
}
```

`per_sample` arrays are split into a `eval_per_sample.json` sidecar
to keep the main JSON small and diff-friendly.
