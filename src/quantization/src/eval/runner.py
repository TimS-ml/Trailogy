"""Eval orchestrator — sweeps benchmarks, writes the JSON spec.

Output schema (one JSON file per variant):

    {
      "variant": "<name>",
      "model_path": "<path>",
      "model_size_gb": <float>,
      "base_model_size_gb_bf16": <float>,
      "backend": "hf_bf16" | "mlx_vlm" | ...,
      "benchmarks": {
        "plantnet_val": {...},
        "wikitext_ppl": {...},
        "vqav2_devtest": {...}
      },
      "eval_seed": 0,
      "generation_kwargs": {...}
    }
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import plantnet, vqav2, wikitext_ppl
from .model_loaders import ModelHandle

log = logging.getLogger(__name__)


BENCHMARK_REGISTRY = {
    "plantnet_val": (plantnet.run, plantnet.PlantNetConfig),
    "wikitext_ppl": (wikitext_ppl.run, wikitext_ppl.WikiTextPPLConfig),
    "vqav2_devtest": (vqav2.run, vqav2.VQAv2Config),
}


@dataclass
class RunnerConfig:
    variant: str
    benchmarks: list[str]
    benchmark_configs: dict[str, dict] = field(default_factory=dict)
    eval_seed: int = 0
    output_dir: Path | None = None


def run_all(handle: ModelHandle, config: RunnerConfig) -> dict:
    """Run every benchmark in ``config.benchmarks`` and return a dict
    matching the JSON output schema. Optionally writes to disk.
    """
    bench_results: dict[str, Any] = {}
    for name in config.benchmarks:
        if name not in BENCHMARK_REGISTRY:
            log.warning("Unknown benchmark %r — skipping.", name)
            bench_results[name] = {"error": f"unknown benchmark {name!r}"}
            continue
        runner, ConfigCls = BENCHMARK_REGISTRY[name]
        bench_cfg_dict = config.benchmark_configs.get(name, {})
        try:
            bench_cfg = ConfigCls(**bench_cfg_dict)
        except TypeError as e:
            log.error("Bad config for %s: %s", name, e)
            bench_results[name] = {"error": f"bad config: {e}"}
            continue
        log.info("=== Running benchmark: %s ===", name)
        t0 = time.perf_counter()
        try:
            result = runner(handle, bench_cfg)
            bench_results[name] = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
            bench_results[name]["wall_time_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:  # noqa: BLE001
            log.exception("Benchmark %s failed:", name)
            bench_results[name] = {"error": str(e), "wall_time_s": round(time.perf_counter() - t0, 1)}

    payload = {
        "variant": config.variant,
        "model_path": str(handle.model_dir),
        "backend": handle.backend,
        "device": handle.device,
        "eval_seed": config.eval_seed,
        "benchmarks": bench_results,
        "generation_kwargs": {"do_sample": False, "max_new_tokens": "per-benchmark"},
    }

    if config.output_dir is not None:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "eval.json"
        # Strip per_sample arrays into a sidecar to keep the main JSON small.
        main, sidecar = _split_per_sample(payload)
        out_file.write_text(json.dumps(main, indent=2))
        if sidecar:
            (out_dir / "eval_per_sample.json").write_text(json.dumps(sidecar, indent=2))
        log.info("Wrote eval result → %s", out_file)
    return payload


def _split_per_sample(payload: dict) -> tuple[dict, dict | None]:
    """Move every benchmark's `per_sample` array out of the main JSON
    into a sidecar to keep diffs and dashboards manageable.
    """
    sidecar: dict = {}
    cleaned = json.loads(json.dumps(payload))  # deep copy
    for name, result in cleaned.get("benchmarks", {}).items():
        if isinstance(result, dict) and "per_sample" in result and result["per_sample"]:
            sidecar[name] = {"per_sample": result.pop("per_sample")}
    return cleaned, sidecar or None
