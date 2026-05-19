"""Guards for shell wrappers after the scripts/run layout split."""

from __future__ import annotations

from pathlib import Path


FINETUNE_DIR = Path(__file__).resolve().parents[1]


def _script_text(relative_path: str) -> str:
    return (FINETUNE_DIR / relative_path).read_text()


def test_run_wrappers_cd_to_finetune_root() -> None:
    for relative_path in ("scripts/run/train.sh", "scripts/run/export.sh"):
        text = _script_text(relative_path)
        assert 'cd "$(dirname "$0")/../.."' in text


def test_train_wrapper_checks_configured_train_file() -> None:
    text = _script_text("scripts/run/train.sh")

    assert "TRAIN_FILE" in text
    assert "data.train_file" in text
    assert 'if [ ! -f "data/train.jsonl" ]; then' not in text


def test_prepare_plantnet_50k_resolves_root_from_scripts_run() -> None:
    text = _script_text("scripts/run/prepare_plantnet_50k.sh")

    assert 'bash src/finetune/scripts/run/prepare_plantnet_50k.sh' in text
    assert 'FINETUNE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"' in text
    assert "scripts/prepare_data/prepare_plantnet_50k.sh" not in text


def test_prepare_plantnet_50k_has_no_private_note_comment_refs() -> None:
    text = _script_text("scripts/run/prepare_plantnet_50k.sh")

    assert "DEV_TIMELINE" not in text
    assert "AGENTS.md" not in text


def test_save_reload_default_config_exists() -> None:
    text = _script_text("scripts/inspect/save_reload.py")

    default_config = "configs/plantnet-50k-baseline-v2.yaml"

    assert f'default="{default_config}"' in text
    assert default_config in text
    assert (FINETUNE_DIR / default_config).exists()
    assert "smoke-save-reload.yaml" not in text
    assert "Path(__file__).resolve().parents[2]" in text


def test_eval_set_defaults_follow_src_layout() -> None:
    # v4: eval_sets/ -> eval/, evaluate_generality.py -> evaluate_generality_plantnet.py,
    # build_eval_set.py -> build_eval_set_plantnet.py. The PlantNet
    # suffix marks these as the v1.0 legacy benchmark builders; an
    # NA-trees eval recipe will live at the unsuffixed names later.
    build_eval_set = _script_text("eval/build_eval_set_plantnet.py")
    evaluate_generality = _script_text("eval/evaluate_generality_plantnet.py")

    assert "FINETUNE_DIR = Path(__file__).resolve().parents[1]" in build_eval_set
    assert "FINETUNE_DIR / \"data\" / \"english-desc\" / \"val.jsonl\"" in build_eval_set
    assert "FINETUNE_DIR / \"eval\"" in build_eval_set
    assert 'Path("finetune/data' not in build_eval_set
    assert 'Path("finetune/eval' not in build_eval_set

    assert "FINETUNE_DIR = Path(__file__).resolve().parents[1]" in evaluate_generality
    assert "FINETUNE_DIR / \"eval\" / \"results\" / \".judge_cache.json\"" in evaluate_generality
    assert "FINETUNE_DIR / \"eval\" / \"results\" / \"generality_report.json\"" in evaluate_generality
    assert 'Path("finetune/eval' not in evaluate_generality


def test_no_finetune_sweep_runners_ship() -> None:
    run_dir = FINETUNE_DIR / "scripts" / "run"

    assert not list(run_dir.glob("*sweep*"))


def test_local_configs_do_not_ship_experiment_history_comments() -> None:
    forbidden = [
        "H200",
        "SOTA",
        "r8-a8-nokl_202",
        "checkpoint-6000",
        "cloud-sweep winner",
        "cloud recipe",
        "cloud checkpoint",
        "multi-GPU",
        "S_step",
        "warm-start",
        "Continue-training",
        "learning trajectory",
        "resume_from_checkpoint",
    ]
    for path in (FINETUNE_DIR / "configs" / "local_sweep").glob("*.yaml"):
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path.name} contains experiment-history token {needle!r}"
