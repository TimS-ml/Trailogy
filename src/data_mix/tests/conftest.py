"""Shared pytest fixtures for data_mix tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest


@pytest.fixture
def tmp_jsonl(tmp_path: Path):
    """Factory: write rows to a temp JSONL file and return its path."""

    def _write(rows: Iterable[dict], name: str = "data.jsonl") -> Path:
        p = tmp_path / name
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return p

    return _write
