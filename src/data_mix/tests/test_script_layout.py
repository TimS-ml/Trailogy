"""Static guards for data_mix shell entrypoints after the src/ move."""

from __future__ import annotations

from pathlib import Path


DATA_MIX_DIR = Path(__file__).resolve().parents[1]


def _script_text(relative_path: str) -> str:
    return (DATA_MIX_DIR / relative_path).read_text()


def test_build_mix_default_config_exists() -> None:
    text = _script_text("scripts/build_mix.sh")

    assert 'CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k.yaml}"' in text
    assert (DATA_MIX_DIR / "configs/mix-50k.yaml").exists()
    assert "configs/mix-20k.yaml" not in text


def test_mix_50k_offline_qa_path_resolves_from_repo_root() -> None:
    from data_mix.src.mix import _load_config

    repo_root = DATA_MIX_DIR.parents[1]

    cfg = _load_config(DATA_MIX_DIR / "configs/mix-50k.yaml")

    assert cfg.offline_qa_path == str(
        repo_root / "assets" / "data_offline_qa" / "offline_qa.json"
    )
