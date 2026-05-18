"""Unit tests for the eval orchestrator. No model loading; uses a fake
``ModelHandle`` that returns canned predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.model_loaders import ModelHandle
from src.eval.runner import BENCHMARK_REGISTRY, RunnerConfig, run_all


def _fake_handle(predictions: dict[str, str] | None = None) -> ModelHandle:
    """A no-op model handle. ``infer_text`` returns canned strings."""
    predictions = predictions or {}

    def infer(messages, image_path=None, max_new_tokens=128):
        # crude lookup by the first user text block
        try:
            user_blocks = messages[0].get("content", [])
            if isinstance(user_blocks, str):
                key = user_blocks
            else:
                key = ""
                for b in user_blocks:
                    if isinstance(b, dict) and b.get("type") == "text":
                        key = b["text"]
                        break
        except Exception:  # noqa: BLE001
            key = ""
        return predictions.get(key, "")

    return ModelHandle(
        infer_text=infer,
        backend="fake",
        model=None,
        processor=None,
        device="cpu",
        model_dir=Path("/tmp/fake-model"),
    )


def test_registry_lists_all_benchmarks():
    assert set(BENCHMARK_REGISTRY) >= {"plantnet_val", "wikitext_ppl", "vqav2_devtest"}


def test_unknown_benchmark_records_error(tmp_path):
    handle = _fake_handle()
    cfg = RunnerConfig(
        variant="unit-test",
        benchmarks=["nonexistent_bench"],
        output_dir=tmp_path,
    )
    payload = run_all(handle, cfg)
    assert "nonexistent_bench" in payload["benchmarks"]
    assert "error" in payload["benchmarks"]["nonexistent_bench"]
    out = json.loads((tmp_path / "eval.json").read_text())
    assert out["variant"] == "unit-test"


def test_output_writes_json_and_sidecar(tmp_path, monkeypatch):
    """If a benchmark returns per_sample, it should be split off."""
    handle = _fake_handle()

    # Inject a fake benchmark that returns per_sample.
    from src.eval import runner

    def fake_runner(handle, config):
        from dataclasses import dataclass, field

        @dataclass
        class R:
            n: int = 3
            score: float = 0.5
            per_sample: list = field(default_factory=lambda: [{"id": 1}, {"id": 2}, {"id": 3}])
        return R()

    @dataclass_marker
    class FakeConfig: ...

    fake_entry = (fake_runner, FakeConfig)
    monkeypatch.setitem(runner.BENCHMARK_REGISTRY, "fake_bench", fake_entry)

    cfg = RunnerConfig(
        variant="sidecar-test",
        benchmarks=["fake_bench"],
        output_dir=tmp_path,
    )
    runner.run_all(handle, cfg)

    main = json.loads((tmp_path / "eval.json").read_text())
    assert "per_sample" not in main["benchmarks"]["fake_bench"]
    sidecar_path = tmp_path / "eval_per_sample.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["fake_bench"]["per_sample"] == [{"id": 1}, {"id": 2}, {"id": 3}]


def dataclass_marker(cls):
    """Tiny shim so the test above can mark an inner class as a dataclass-like
    without leaking real-dataclass machinery."""
    from dataclasses import dataclass

    return dataclass(cls)
