"""Tests for the cross-machine eval aggregator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.inspect.compare_runs import (
    _load_evals,
    build_matrix,
    render_markdown,
    render_tsv,
)


def _write_eval(dir_: Path, payload: dict) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "eval.json").write_text(json.dumps(payload))


def _sample_payload(variant: str, size_gb: float, pn: float, vqa: float, ppl: float | None = None) -> dict:
    benchmarks = {
        "plantnet_val": {"species_match": pn, "n": 1000},
        "vqav2_devtest": {"accuracy": vqa, "n": 500},
    }
    if ppl is not None:
        benchmarks["wikitext_ppl"] = {"perplexity": ppl, "n_segments": 100}
    return {
        "variant": variant,
        "backend": "hf_bf16" if "bf16" in variant else "hf_bf16",
        "model_size_gb": size_gb,
        "benchmarks": benchmarks,
    }


def test_load_evals_finds_all(tmp_path):
    _write_eval(tmp_path / "v1", _sample_payload("v1", 3.5, 0.5, 0.6))
    _write_eval(tmp_path / "v2", _sample_payload("v2", 9.5, 0.7, 0.8))
    payloads = _load_evals(tmp_path)
    assert {p["variant"] for p in payloads} == {"v1", "v2"}


def test_build_matrix_computes_deltas_vs_reference(tmp_path):
    _write_eval(tmp_path / "bf16_reference",
                _sample_payload("bf16_reference", 9.5, 0.80, 0.60, ppl=15.0))
    _write_eval(tmp_path / "gptq_v1",
                _sample_payload("gptq_v1", 3.6, 0.77, 0.58, ppl=18.0))

    payloads = _load_evals(tmp_path)
    rows = build_matrix(payloads, reference_variant="bf16_reference")

    by_v = {r["variant"]: r for r in rows}
    assert by_v["bf16_reference"]["plantnet_match"] == 0.80
    # gptq_v1 should have negative deltas
    assert by_v["gptq_v1"]["plantnet_delta"] == pytest.approx(-0.03)
    assert by_v["gptq_v1"]["vqav2_delta"] == pytest.approx(-0.02)
    # PPL ratio should be > 1 since 18/15 = 1.2
    assert by_v["gptq_v1"]["ppl_ratio"] == pytest.approx(18.0 / 15.0)


def test_size_warning_triggers_above_4gb(tmp_path):
    _write_eval(tmp_path / "too_big",
                _sample_payload("too_big", 4.6, 0.79, 0.59))
    payloads = _load_evals(tmp_path)
    rows = build_matrix(payloads)
    assert "SIZE>4.0GB" in rows[0]["warnings"]


def test_plantnet_drop_warning(tmp_path):
    _write_eval(tmp_path / "bf16_reference",
                _sample_payload("bf16_reference", 9.5, 0.80, 0.60))
    _write_eval(tmp_path / "bad_quant",
                _sample_payload("bad_quant", 3.6, 0.65, 0.59))  # -15 pct points
    payloads = _load_evals(tmp_path)
    rows = build_matrix(payloads)
    bad = next(r for r in rows if r["variant"] == "bad_quant")
    assert "PlantNet drop >10pts" in bad["warnings"]


def test_render_markdown_includes_all_variants(tmp_path):
    _write_eval(tmp_path / "v1", _sample_payload("v1", 3.5, 0.7, 0.5))
    _write_eval(tmp_path / "v2", _sample_payload("v2", 9.5, 0.8, 0.6))
    payloads = _load_evals(tmp_path)
    rows = build_matrix(payloads)
    md = render_markdown(rows)
    assert "v1" in md and "v2" in md
    assert "variant" in md  # header row
    assert "|" in md


def test_render_tsv_includes_all_variants(tmp_path):
    _write_eval(tmp_path / "v1", _sample_payload("v1", 3.5, 0.7, 0.5))
    payloads = _load_evals(tmp_path)
    rows = build_matrix(payloads)
    tsv = render_tsv(rows)
    assert "v1" in tsv
    assert "\t" in tsv


# pytest import is at file bottom so the tests above can reference `pytest.approx`.
import pytest  # noqa: E402
