"""Static guards for data_mix shell entrypoints after the src/ move."""

from __future__ import annotations

from pathlib import Path


DATA_MIX_DIR = Path(__file__).resolve().parents[1]


def _script_text(relative_path: str) -> str:
    return (DATA_MIX_DIR / relative_path).read_text()


def test_build_mix_default_config_is_na_plantae_mix_50k() -> None:
    text = _script_text("scripts/build_mix.sh")

    # v2.0 default: NA-Plantae-backed mix-50k.yaml (no -plantnet suffix).
    # The PlantNet recipe is preserved at mix-50k-plantnet.yaml and
    # selected by env: CONFIG=$DATA_MIX_DIR/configs/mix-50k-plantnet.yaml.
    assert (
        'CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k.yaml}"' in text
    )
    assert (DATA_MIX_DIR / "configs/mix-50k.yaml").exists()
    assert (DATA_MIX_DIR / "configs/mix-50k-plantnet.yaml").exists()
    assert "configs/mix-20k.yaml" not in text


def test_mix_50k_plantnet_offline_qa_path_resolves_from_repo_root() -> None:
    from data_mix.src.mix import _load_config

    repo_root = DATA_MIX_DIR.parents[1]

    cfg = _load_config(DATA_MIX_DIR / "configs/mix-50k-plantnet.yaml")

    assert cfg.offline_qa_path == str(
        repo_root / "assets" / "data_offline_qa" / "offline_qa.json"
    )


def test_mix_50k_default_is_na_plantae_without_plant_bucket() -> None:
    """The default mix-50k.yaml must drop the PlantNet bucket and use
    na_plantae as the species-ID source. Guards against accidental
    regression to a PlantNet-backed default."""
    from data_mix.src.mix import _load_config

    cfg = _load_config(DATA_MIX_DIR / "configs/mix-50k.yaml")

    assert cfg.plant_train == 0
    assert cfg.plant_val == 0
    assert cfg.na_plantae_train > 0
    assert cfg.na_plantae_train_jsonl is not None


def test_mix_50k_na_plantae_paths_resolve_outside_repo() -> None:
    """The default mix-50k.yaml must point at na_plantae JSONLs that live
    OUTSIDE the repo (Trailogy storage convention). Train data inside
    the working tree would silently pollute commits."""
    from data_mix.src.mix import _load_config
    from data_mix.src.env_paths import HIKECOMPANION_ROOT

    cfg = _load_config(DATA_MIX_DIR / "configs/mix-50k.yaml")

    # Mirror the resolution mix.py does: relative paths join against
    # HIKECOMPANION_ROOT (the repo root). The resolved path's parents
    # must include HIKECOMPANION_ROOT.parent (= <repo>'s parent) but
    # NOT HIKECOMPANION_ROOT itself once normalised.
    repo_root = HIKECOMPANION_ROOT.resolve()
    for raw_path in (cfg.na_plantae_train_jsonl, cfg.na_plantae_val_jsonl):
        assert raw_path is not None
        resolved = (repo_root / raw_path).resolve()
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            # Good — path escapes the repo root.
            pass
        else:
            raise AssertionError(
                f"na_plantae JSONL must live OUTSIDE the repo; "
                f"{raw_path!r} resolves to {resolved} which is inside "
                f"{repo_root}."
            )


def test_external_data_root_defaults_to_sibling_of_repo(monkeypatch) -> None:
    """``external_data_root()`` defaults to ``<repo>/../data`` so all
    generated artefacts (fetch, prepare, build_mix output, image cache)
    stay out of the repo working tree."""
    from data_mix.src.env_paths import (
        DEFAULT_EXTERNAL_DATA_ROOT,
        HIKECOMPANION_ROOT,
        external_data_root,
    )

    monkeypatch.delenv("TRAILOGY_DATA_ROOT", raising=False)
    assert DEFAULT_EXTERNAL_DATA_ROOT == HIKECOMPANION_ROOT.parent / "data"
    assert external_data_root() == DEFAULT_EXTERNAL_DATA_ROOT.resolve()


def test_resolve_paths_writes_outside_repo_by_default(monkeypatch) -> None:
    """resolve_paths() must default both ``output_root`` and
    ``image_root`` to the external data root."""
    from data_mix.src.env_paths import (
        external_data_root, resolve_paths,
    )

    monkeypatch.delenv("DATA_MIX_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("DATA_MIX_IMAGE_ROOT", raising=False)
    monkeypatch.delenv("TRAILOGY_DATA_ROOT", raising=False)

    ext = external_data_root()
    paths = resolve_paths(
        config_path=DATA_MIX_DIR / "configs/mix-50k.yaml",
    )
    assert paths.output_root == (ext / "mix-50k").resolve()
    assert paths.image_root == (ext / "_image_cache").resolve()
