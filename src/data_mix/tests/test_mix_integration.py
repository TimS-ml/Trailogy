from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data_mix.src import mix as mix_mod


def _smol_row(i: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Q{i}?"},
            {"role": "assistant", "content": f"A{i}."},
        ]
    }


def _cam_row(i: int, plantish: bool) -> dict:
    user = "What is in this image?" if not plantish else "What plant is this?"
    return {
        "id": f"cam{i}",
        "image": Image.new("RGB", (200, 200), color=(20 + i, 30, 40)),
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "An answer."},
        ],
    }


@pytest.fixture
def patched_mix(tmp_path, monkeypatch):
    # PlantNet JSONL: 20 species x 5 imgs = 100 rows.
    plantnet_dir = tmp_path / "plantnet_images"
    plantnet_dir.mkdir()
    rows = []
    for sid in range(20):
        sid_dir = plantnet_dir / f"{sid:04d}"
        sid_dir.mkdir()
        for k in range(5):
            img = sid_dir / f"img_{k}.jpg"
            Image.new("RGB", (200, 200), color=(0, 100, 0)).save(img, "JPEG", quality=85)
            rows.append({
                "image": str(img),
                "conversations": [
                    {"role": "user", "content": "Can you identify this species?"},
                    {"role": "assistant", "content": f"This is Species{sid:04d}. Detail."},
                ],
            })
    plantnet_jsonl = tmp_path / "plantnet.jsonl"
    with plantnet_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    image_root = tmp_path / "images"
    output_root = tmp_path / "out"

    monkeypatch.setenv("DATA_MIX_IMAGE_ROOT", str(image_root))
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("PLANTNET_JSONL", str(plantnet_jsonl))

    monkeypatch.setattr(mix_mod, "open_smoltalk_stream",
                        lambda seed: (_smol_row(i) for i in range(200)))
    monkeypatch.setattr(mix_mod, "open_cambrian_stream",
                        lambda seed: (
                            _cam_row(i, plantish=(i % 5 == 0))
                            for i in range(200)
                        ))

    config = tmp_path / "mix.yaml"
    config.write_text(
        "seed: 7\n"
        "plant: {train: 30, val: 5, per_class_cap: 3}\n"
        "cambrian: {train: 10, val: 2}\n"
        "smoltalk: {train: 10, val: 2}\n"
        "negative: {train: 5, val: 1}\n",
        encoding="utf-8",
    )
    return {
        "config": config,
        "output_root": output_root,
        "image_root": image_root,
    }


def test_mix_end_to_end(patched_mix):
    report_path = mix_mod.build_mix(patched_mix["config"])
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["train_total"] == 30 + 10 + 10 + 5
    assert report["val_total"] == 5 + 2 + 2 + 1
    assert set(report["train_by_source"].keys()) == {
        "plant", "cambrian", "smoltalk", "negative",
    }
    # train + val files exist and parse line-by-line
    out = patched_mix["output_root"]
    for name in ("train.jsonl", "val.jsonl"):
        with (out / name).open() as f:
            n = 0
            for line in f:
                rec = json.loads(line)
                assert "image" in rec and "conversations" in rec and "source" in rec
                n += 1
            assert n > 0


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def test_mix_writes_multi_val_jsonl_for_mid_training_eval(patched_mix):
    """v2: in addition to the combined val.jsonl, mix.py must write three
    split-by-modality val files so the finetune SFTTrainer can run
    multi-eval-dataset and report eval_<key>_loss per bucket:

      - val_plant.jsonl      (plant species ID — degradation watch)
      - val_nonplant.jsonl   (LLaVA/cambrian + smoltalk — forget watch)
      - val_negative.jsonl   (refusal template — over/under-refusal)
    """
    mix_mod.build_mix(patched_mix["config"])
    out = patched_mix["output_root"]

    expected = {
        "val_plant.jsonl":    5,
        "val_nonplant.jsonl": 2 + 2,   # cambrian/llava + smoltalk
        "val_negative.jsonl": 1,
    }
    for name, n_expected in expected.items():
        p = out / name
        assert p.exists(), f"missing multi-val file {name}"
        n = _count_jsonl(p)
        assert n == n_expected, f"{name}: expected {n_expected} rows, got {n}"

    # Every row in val_plant.jsonl must have source=plant, etc.
    sources = {
        "val_plant.jsonl": {"plant"},
        # In v1 (source=cambrian), nonplant = cambrian + smoltalk.
        # In v2 (source=llava), nonplant = llava + smoltalk.
        # Test fixture is v1, so cambrian is the general source here.
        "val_nonplant.jsonl": {"cambrian", "smoltalk"},
        "val_negative.jsonl": {"negative"},
    }
    for name, allowed_sources in sources.items():
        with (out / name).open() as f:
            for line in f:
                rec = json.loads(line)
                assert rec["source"] in allowed_sources, (
                    f"{name} has row with source={rec['source']!r}, "
                    f"expected one of {allowed_sources}"
                )


def test_mix_report_includes_multi_val_paths(patched_mix):
    """build_report.json must list the multi-val file paths so downstream
    config (finetune yaml) can wire them up programmatically."""
    report_path = mix_mod.build_mix(patched_mix["config"])
    report = json.loads(report_path.read_text())
    assert "val_files" in report["paths"], "report missing paths.val_files"
    vf = report["paths"]["val_files"]
    assert set(vf.keys()) == {"plant", "nonplant", "negative"}
    for k, p in vf.items():
        assert Path(p).exists(), f"val_files[{k}] path does not exist: {p}"


def _llava_row(rid: str, plantish: bool = False) -> dict:
    """LLaVA-mix-vsft style row (content blocks, not flat string)."""
    text = "What is in this image?" if not plantish else "What plant is this?"
    return {
        "id": rid,
        "messages": [
            {"role": "user", "content": [
                {"index": None, "type": "text", "text": text},
                {"index": 0, "type": "image", "text": None},
            ]},
            {"role": "assistant", "content": [
                {"index": None, "type": "text", "text": "An answer."},
            ]},
        ],
        "images": [Image.new("RGB", (200, 200), color=(20, 30, 40))],
    }


def test_mix_dual_source_plant_uses_separate_val_jsonl(tmp_path, monkeypatch):
    """v2: when ``PLANTNET_VAL_JSONL`` points at an existing file
    sibling to ``PLANTNET_JSONL``, the plant bucket reads train and val
    from separate sources (no random slice). Verifies that the
    images in mix_val_plant ALL come from the val.jsonl side and
    NOT from the train.jsonl side."""
    plantnet_dir = tmp_path / "plantnet_images"
    plantnet_dir.mkdir()

    # Build train.jsonl: 20 species x 5 imgs (named train_*)
    train_rows = []
    for sid in range(20):
        sid_dir = plantnet_dir / f"{sid:04d}"
        sid_dir.mkdir()
        for k in range(5):
            img = sid_dir / f"train_{k}.jpg"
            Image.new("RGB", (200, 200), color=(0, 100, 0)).save(img, "JPEG", quality=85)
            train_rows.append({
                "image": str(img),
                "conversations": [
                    {"role": "user", "content": "Identify."},
                    {"role": "assistant", "content": f"This is Species{sid:04d}. Detail."},
                ],
            })
    train_jsonl = tmp_path / "train.jsonl"
    with train_jsonl.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    # Build val.jsonl: same 20 species, but imgs named val_* — disjoint
    # from train_rows above.
    val_rows = []
    for sid in range(20):
        for k in range(2):
            img = plantnet_dir / f"{sid:04d}" / f"val_{k}.jpg"
            Image.new("RGB", (200, 200), color=(0, 100, 0)).save(img, "JPEG", quality=85)
            val_rows.append({
                "image": str(img),
                "conversations": [
                    {"role": "user", "content": "Identify."},
                    {"role": "assistant", "content": f"This is Species{sid:04d}. Detail."},
                ],
            })
    val_jsonl = tmp_path / "val.jsonl"
    with val_jsonl.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")

    image_root = tmp_path / "images"
    output_root = tmp_path / "out"
    monkeypatch.setenv("DATA_MIX_IMAGE_ROOT", str(image_root))
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("PLANTNET_JSONL", str(train_jsonl))
    monkeypatch.setenv("PLANTNET_VAL_JSONL", str(val_jsonl))

    monkeypatch.setattr(mix_mod, "open_smoltalk_stream",
                        lambda seed: (_smol_row(i) for i in range(200)))
    monkeypatch.setattr(mix_mod, "open_cambrian_stream",
                        lambda seed: (
                            _cam_row(i, plantish=(i % 5 == 0))
                            for i in range(200)
                        ))

    config = tmp_path / "mix.yaml"
    config.write_text(
        "seed: 7\n"
        "plant: {train: 30, val: 10, per_class_cap: 3}\n"
        "cambrian: {train: 10, val: 2}\n"
        "smoltalk: {train: 10, val: 2}\n"
        "negative: {train: 5, val: 1}\n",
        encoding="utf-8",
    )
    mix_mod.build_mix(config)

    # Per-source val partition must have ALL its plant images from val.jsonl
    # (filename contains 'val_'), none from train.jsonl ('train_').
    val_plant_path = output_root / "val_plant.jsonl"
    assert val_plant_path.exists()
    with val_plant_path.open() as f:
        val_plant_records = [json.loads(line) for line in f if line.strip()]
    assert val_plant_records, "no plant records in val_plant.jsonl"
    for r in val_plant_records:
        assert "val_" in r["image"], (
            f"plant val record came from train.jsonl: {r['image']}"
        )
        assert "train_" not in r["image"]

    # And train.jsonl plant records all came from train.jsonl side.
    train_path = output_root / "train.jsonl"
    with train_path.open() as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    plant_train = [r for r in train_records if r["source"] == "plant"]
    for r in plant_train:
        assert "train_" in r["image"], (
            f"plant train record came from val.jsonl: {r['image']}"
        )


def test_mix_source_llava_dispatches_to_llava_sampler(tmp_path, monkeypatch):
    """v2: when ``source: llava`` is set, mix.py uses llava_sampler and
    open_llava_stream instead of the cambrian counterparts."""
    # Reuse PlantNet fixture inline (simpler than parametrizing patched_mix).
    plantnet_dir = tmp_path / "plantnet_images"
    plantnet_dir.mkdir()
    rows = []
    for sid in range(20):
        sid_dir = plantnet_dir / f"{sid:04d}"
        sid_dir.mkdir()
        for k in range(5):
            img = sid_dir / f"img_{k}.jpg"
            Image.new("RGB", (200, 200), color=(0, 100, 0)).save(img, "JPEG", quality=85)
            rows.append({
                "image": str(img),
                "conversations": [
                    {"role": "user", "content": "Identify."},
                    {"role": "assistant", "content": f"This is Species{sid:04d}. Detail."},
                ],
            })
    plantnet_jsonl = tmp_path / "plantnet.jsonl"
    with plantnet_jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    image_root = tmp_path / "images"
    output_root = tmp_path / "out"
    monkeypatch.setenv("DATA_MIX_IMAGE_ROOT", str(image_root))
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("PLANTNET_JSONL", str(plantnet_jsonl))

    # Stub LLaVA stream (v2 path). If mix.py incorrectly calls
    # open_cambrian_stream, this test fails (the cambrian stream is not
    # stubbed and would raise on network access).
    monkeypatch.setattr(mix_mod, "open_llava_stream",
                        lambda seed: (_llava_row(f"r{i}") for i in range(200)))
    monkeypatch.setattr(mix_mod, "open_smoltalk_stream",
                        lambda seed: (_smol_row(i) for i in range(200)))

    config = tmp_path / "mix_llava.yaml"
    config.write_text(
        "seed: 7\n"
        "source: llava\n"
        "plant: {train: 30, val: 5, per_class_cap: 3}\n"
        "llava: {train: 10, val: 2}\n"
        "smoltalk: {train: 10, val: 2}\n"
        "negative: {train: 5, val: 1}\n",
        encoding="utf-8",
    )

    report_path = mix_mod.build_mix(config)
    report = json.loads(report_path.read_text())

    # When source=llava, the general bucket source field is "llava" not "cambrian"
    assert "llava" in report["train_by_source"]
    assert "cambrian" not in report["train_by_source"]
    assert report["train_by_source"]["llava"] == 10
    assert report["train_by_source"]["plant"] == 30
    assert report["train_by_source"]["smoltalk"] == 10
    assert report["train_by_source"]["negative"] == 5


def test_insufficient_pool_raises_explicit_error(patched_mix, monkeypatch):
    """If a bucket pool is too small, build_mix must raise
    ``InsufficientPoolError`` -- NOT ``AssertionError`` -- so the failure
    survives ``python -O`` and gives a clean diagnostic.
    """
    # Truncate the smoltalk stream to fewer rows than the config asks for.
    monkeypatch.setattr(
        mix_mod, "open_smoltalk_stream",
        lambda seed: (_smol_row(i) for i in range(3)),  # config asks for 12
    )
    with pytest.raises(mix_mod.InsufficientPoolError) as exc:
        mix_mod.build_mix(patched_mix["config"])
    msg = str(exc.value).lower()
    assert "pool" in msg
    assert "n_train" in msg or "12" in msg


def test_build_mix_is_deterministic(patched_mix, monkeypatch):
    """Two builds with the same config + seed + stubbed streams must
    produce byte-identical train.jsonl and val.jsonl.

    Determinism is the contract that lets us reproduce a build later
    (e.g. for ablation studies, or to re-run if the disk fills up). If
    any sampler accidentally consumes a non-seeded random source, this
    test catches it.
    """
    first_out = patched_mix["output_root"]
    mix_mod.build_mix(patched_mix["config"])
    train_a = (first_out / "train.jsonl").read_bytes()
    val_a = (first_out / "val.jsonl").read_bytes()
    assert train_a, "first build produced empty train.jsonl"

    # Redirect to a fresh output dir; everything else (config, seed,
    # stubbed streams, image root, plantnet jsonl) stays identical.
    second_out = first_out.parent / "out_second"
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(second_out))

    mix_mod.build_mix(patched_mix["config"])
    train_b = (second_out / "train.jsonl").read_bytes()
    val_b = (second_out / "val.jsonl").read_bytes()

    assert train_a == train_b, "train.jsonl differs between identical builds"
    assert val_a == val_b, "val.jsonl differs between identical builds"


def test_output_loads_through_finetune_data_pipeline(patched_mix):
    """The mix output must remain a drop-in for
    ``finetune/src/data.py::load_vision_dataset``.

    v2: smoltalk records carry ``image=None`` (was: dummy image). The
    trainer routes them via ModalityAwareBatchSampler into vision-skip
    batches. So we load with ``require_image=False`` and assert:
      - all 55 records load (no spurious drops)
      - smoltalk records (10) come back as text-only messages
      - image-having records (45 = plant 30 + cam 10 + neg 5) have an
        image content block in the first user turn
    """
    import sys

    mix_mod.build_mix(patched_mix["config"])

    # ``src/finetune/`` is a sibling of ``src/data_mix/`` after the
    # post-restructure layout (was ``finetune/`` next to ``data_mix/``
    # at the repo root pre-restructure).
    from data_mix.src.env_paths import SRC_ROOT

    finetune_root = SRC_ROOT / "finetune"
    assert finetune_root.is_dir(), f"missing {finetune_root}"
    if str(finetune_root) not in sys.path:
        sys.path.insert(0, str(finetune_root))

    # Late import so the test's collection doesn't depend on finetune.
    from src.data import load_vision_dataset  # type: ignore

    train_path = patched_mix["output_root"] / "train.jsonl"
    n_expected = 30 + 10 + 10 + 5  # plant + cambrian + smoltalk + negative

    # v2: require_image=False because smoltalk now carries image=None.
    records = load_vision_dataset(str(train_path), require_image=False)
    assert len(records) == n_expected, (
        f"expected {n_expected} records, got {len(records)} -- "
        "schema drifted or sampler dropped rows unexpectedly"
    )

    # Partition records by whether the first user turn has an image
    # content block.
    def _has_image_block(rec: dict) -> bool:
        msgs = rec.get("messages") or []
        for m in msgs:
            if m.get("role") != "user":
                continue
            blocks = m.get("content") or []
            return any(b.get("type") == "image" for b in blocks)
        return False

    n_image = sum(1 for r in records if _has_image_block(r))
    n_text  = sum(1 for r in records if not _has_image_block(r))
    assert n_image == 30 + 10 + 5, f"expected 45 image-having, got {n_image}"
    assert n_text == 10, f"expected 10 text-only (smoltalk), got {n_text}"

    # Spot-check shape of an image-having record.
    img_rec = next(r for r in records if _has_image_block(r))
    msgs = img_rec["messages"]
    assert msgs[0]["role"] == "user"
    types = [b.get("type") for b in msgs[0]["content"]]
    assert "image" in types and "text" in types

    # Spot-check shape of a text-only record: only text blocks.
    txt_rec = next(r for r in records if not _has_image_block(r))
    msgs = txt_rec["messages"]
    assert msgs[0]["role"] == "user"
    types = [b.get("type") for b in msgs[0]["content"]]
    assert types == ["text"], f"text-only should have only text blocks, got {types}"


# ---------------------------------------------------------------------------
# v3: offline_qa persona bucket
# ---------------------------------------------------------------------------


def test_mix_includes_offline_qa_when_path_set(tmp_path, patched_mix):
    """When config sets ``offline_qa.path``, build_mix appends the full
    corpus on top of the 4-bucket ratio. Records appear in train.jsonl
    with source='offline_qa' and image=None, and a val_offline_qa.jsonl
    is emitted with the val carve-out."""
    qa_file = tmp_path / "qa.json"
    qa_file.write_text(json.dumps([
        {"question": f"Persona Q{i}?", "answer": f"Persona A{i}."}
        for i in range(20)
    ]))

    # Re-write the config with offline_qa block appended.
    patched_mix["config"].write_text(
        "seed: 7\n"
        "plant: {train: 30, val: 5, per_class_cap: 3}\n"
        "cambrian: {train: 10, val: 2}\n"
        "smoltalk: {train: 10, val: 2}\n"
        "negative: {train: 5, val: 1}\n"
        f"offline_qa: {{path: \"{qa_file}\", val_ratio: 0.1}}\n",
        encoding="utf-8",
    )

    report_path = mix_mod.build_mix(patched_mix["config"])
    report = json.loads(report_path.read_text())

    # 20 records, val_ratio=0.1 → 2 val + 18 train, on top of the 4-bucket total.
    expected_train_total = 30 + 10 + 10 + 5 + 18
    expected_val_total   = 5 + 2 + 2 + 1 + 2
    assert report["train_total"] == expected_train_total
    assert report["val_total"]   == expected_val_total
    assert report["train_by_source"]["offline_qa"] == 18
    assert report["val_by_source"]["offline_qa"]   == 2

    # val_offline_qa.jsonl emitted and listed in build_report.
    out = patched_mix["output_root"]
    assert (out / "val_offline_qa.jsonl").exists()
    assert "offline_qa" in report["paths"]["val_files"]

    # Every offline_qa record has source=offline_qa + image=None.
    with (out / "val_offline_qa.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            assert rec["source"] == "offline_qa"
            assert rec["image"] is None


def test_mix_skips_offline_qa_when_path_unset(patched_mix):
    """Backward compat: when no offline_qa block in the config, build_mix
    does NOT emit val_offline_qa.jsonl and the report's val_files dict
    keeps the legacy {plant, nonplant, negative} keys only."""
    mix_mod.build_mix(patched_mix["config"])
    out = patched_mix["output_root"]
    assert not (out / "val_offline_qa.jsonl").exists()
    report = json.loads((out / "build_report.json").read_text())
    assert set(report["paths"]["val_files"].keys()) == {"plant", "nonplant", "negative"}


# ---------------------------------------------------------------------------
# freeze_val_from: hardlink val_*.jsonl from a previous mix output
# ---------------------------------------------------------------------------


def test_freeze_val_from_links_val_files_and_skips_sampling(
    patched_mix, monkeypatch
):
    """When ``freeze_val_from`` resolves to an existing mix output
    directory, build_mix must:

      * skip val sampling entirely (val partition list empty);
      * hardlink (or copy) ``val.jsonl`` + ``val_<bucket>.jsonl`` from
        the source dir into ``output_root``;
      * record the linkage in ``build_report.json`` under
        ``frozen_val_from``;
      * point ``paths.val_files`` at the linked files so the trainer
        sees the SAME bytes as the source mix's val files.
    """
    # First build: a normal mix, used as the freeze source.
    source_out = patched_mix["output_root"]
    mix_mod.build_mix(patched_mix["config"])
    source_val_plant_bytes = (source_out / "val_plant.jsonl").read_bytes()
    source_val_nonplant_bytes = (source_out / "val_nonplant.jsonl").read_bytes()
    source_val_negative_bytes = (source_out / "val_negative.jsonl").read_bytes()

    # Second build: same config but with freeze_val_from pointing at
    # the first output dir. Different output_root.
    frozen_out = source_out.parent / "out_frozen"
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(frozen_out))

    frozen_config = patched_mix["config"].parent / "mix_frozen.yaml"
    frozen_config.write_text(
        "seed: 7\n"
        # Different train counts vs the source mix to prove the val
        # files are *not* re-sampled from the new config's parameters.
        "plant: {train: 25, val: 5, per_class_cap: 3}\n"
        "cambrian: {train: 8, val: 2}\n"
        "smoltalk: {train: 8, val: 2}\n"
        "negative: {train: 4, val: 1}\n"
        f"freeze_val_from: \"{source_out}\"\n",
        encoding="utf-8",
    )

    report_path = mix_mod.build_mix(frozen_config)
    report = json.loads(report_path.read_text())

    # train was sampled fresh under the new config
    assert report["train_total"] == 25 + 8 + 8 + 4
    # val_all (the freshly sampled list) must be empty — val files come
    # from the freeze source.
    assert report["val_total"] == 0

    # val files hardlinked / copied from source
    for name, src_bytes in (
        ("val_plant.jsonl", source_val_plant_bytes),
        ("val_nonplant.jsonl", source_val_nonplant_bytes),
        ("val_negative.jsonl", source_val_negative_bytes),
    ):
        dest = frozen_out / name
        assert dest.exists(), f"freeze did not produce {name}"
        assert dest.read_bytes() == src_bytes, (
            f"{name} content drifted between source and frozen build"
        )

    # report records the freeze source + per-file action
    assert report["frozen_val_from"] is not None
    assert report["frozen_val_from"]["source_dir"] == str(source_out)
    files = report["frozen_val_from"]["files"]
    assert {"val_plant.jsonl", "val_nonplant.jsonl", "val_negative.jsonl"} <= files.keys()
    for entry in files.values():
        assert entry["method"] in ("hardlink", "copy")

    # paths.val_files points at the linked files inside frozen_out
    assert set(report["paths"]["val_files"].keys()) == {"plant", "nonplant", "negative"}
    for p in report["paths"]["val_files"].values():
        assert Path(p).parent == frozen_out


def test_freeze_val_from_missing_source_dir_raises(patched_mix, monkeypatch):
    """A non-existent freeze source must fail loud — silently sampling
    fresh val records when the operator asked for a freeze would
    silently break eval comparability across the v1 -> v2 boundary."""
    frozen_out = patched_mix["output_root"].parent / "out_missing"
    monkeypatch.setenv("DATA_MIX_OUTPUT_ROOT", str(frozen_out))

    bogus_source = patched_mix["output_root"].parent / "does_not_exist"
    frozen_config = patched_mix["config"].parent / "mix_missing_freeze.yaml"
    frozen_config.write_text(
        "seed: 7\n"
        "plant: {train: 10, val: 0, per_class_cap: 3}\n"
        "cambrian: {train: 5, val: 0}\n"
        "smoltalk: {train: 5, val: 0}\n"
        "negative: {train: 2, val: 0}\n"
        f"freeze_val_from: \"{bogus_source}\"\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="freeze_val_from source"):
        mix_mod.build_mix(frozen_config)
