# data_mix/src/env_paths.py
"""Resolve storage roots from env vars with safe **outside-repo** defaults.

The Trailogy convention is that every large generated artefact — raw
fetched images, prepared training JSONLs, mix output JSONLs, image
resize cache — lives in a sibling directory of the repo, not inside
the repo. This keeps the working tree small, prevents accidental
commit of training data, and means the same data dir can serve
multiple repo clones / branches without churn.

Concretely, given the repo at ``<base>/Trailogy/``, the defaults are::

    <base>/data/
        inaturalist_na_plantae/             # na_plantae_fetch.py output
            train/<slug>/*.jpg
            val/<slug>/*.jpg
            test/<slug>/*.jpg
            {train,val,test}.jsonl        # raw per-photo metadata
            observations.jsonl
            fetch_report.json

        inaturalist_na_plantae_prepared/    # prepare_na_plantae.py output
            {train,val,test}.jsonl        # training-schema JSONLs
            images_resized/{train,val,test}/<slug>/*.jpg

        <mix-config-stem>/                # build_mix.sh output
            train.jsonl, val.jsonl, val_<bucket>.jsonl
            build_report.json

        _image_cache/                     # data_mix image resize cache

Operators on machines where the parent of the repo is read-only set
``TRAILOGY_DATA_ROOT`` to relocate the whole tree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# data_mix/src/env_paths.py -> src/data_mix/ -> src/ -> repo root
DATA_MIX_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DATA_MIX_ROOT.parent
HIKECOMPANION_ROOT = SRC_ROOT.parent

# v4: external data root — sibling of the repo. Override via
# ``TRAILOGY_DATA_ROOT`` env var to point anywhere on disk.
DEFAULT_EXTERNAL_DATA_ROOT = HIKECOMPANION_ROOT.parent / "data"


def external_data_root() -> Path:
    """Resolve the external data root.

    Precedence: ``TRAILOGY_DATA_ROOT`` env > default
    (``<repo>/../data/``). The path is *not* required to exist —
    callers that need it materialised should ``mkdir(parents=True)``.
    """
    env = os.environ.get("TRAILOGY_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_EXTERNAL_DATA_ROOT.resolve()


@dataclass(frozen=True)
class Paths:
    hf_home: Path | None  # None = let huggingface_hub use its own default
    image_root: Path
    output_root: Path
    plantnet_jsonl: Path       # train pool source
    plantnet_val_jsonl: Path   # v2: separate val pool source (was: a random
                               #     slice of plantnet_jsonl). Default is
                               #     sibling val.jsonl of plantnet_jsonl.


def _default_output_root(config_path: Path | None) -> Path:
    """Derive the default output dir from the config filename stem.

    ``mix-50k.yaml`` -> ``<external>/mix-50k/``.
    ``mix-50k-plantnet.yaml`` -> ``<external>/mix-50k-plantnet/``.

    Falls back to ``mix-50k`` when no config_path is given (legacy
    callers that don't pass it through; tests that patch resolve_paths
    directly should also patch this).
    """
    stem = config_path.stem if config_path is not None else "mix-50k"
    return external_data_root() / stem


def resolve_paths(config_path: Path | None = None) -> Paths:
    """Resolve storage paths. ``config_path``, when given, drives the
    default ``output_root`` so each yaml config writes to its own
    sibling-of-config output dir (yaml-stem == output-dir-name).

    All generated artefacts default to ``<repo>/../data/...`` so the
    working tree stays clean. Override per-path via env vars
    (``DATA_MIX_OUTPUT_ROOT``, ``DATA_MIX_IMAGE_ROOT``,
    ``PLANTNET_JSONL``, ``PLANTNET_VAL_JSONL``), or relocate the whole
    external root via ``TRAILOGY_DATA_ROOT``.
    """
    hf_home_env = os.environ.get("HF_HOME")
    hf_home = Path(hf_home_env).resolve() if hf_home_env else None

    image_root = Path(
        os.environ.get("DATA_MIX_IMAGE_ROOT")
        or (external_data_root() / "_image_cache")
    ).resolve()

    output_root = Path(
        os.environ.get("DATA_MIX_OUTPUT_ROOT")
        or _default_output_root(config_path)
    ).resolve()

    plantnet_jsonl = Path(
        os.environ.get("PLANTNET_JSONL")
        or (
            external_data_root()
            / "english-desc-v2"
            / "train.jsonl"
        )
    ).resolve()

    # v2: PLANTNET_VAL_JSONL points at the per-species val output of
    # prepare_plantnet_50k.sh. Default = sibling val.jsonl of plantnet_jsonl.
    plantnet_val_jsonl = Path(
        os.environ.get("PLANTNET_VAL_JSONL")
        or (plantnet_jsonl.parent / "val.jsonl")
    ).resolve()

    return Paths(
        hf_home=hf_home,
        image_root=image_root,
        output_root=output_root,
        plantnet_jsonl=plantnet_jsonl,
        plantnet_val_jsonl=plantnet_val_jsonl,
    )
