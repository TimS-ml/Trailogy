from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from data_mix.src import enrich_na_plantae as enrich


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_gbif_extension_accepts_list_response() -> None:
    class Client:
        def get_json(self, *args, **kwargs):
            return [{"description": "needle-like leaves"}]

    assert enrich.gbif_extension(Client(), usage_key=123, endpoint="descriptions") == [
        {"description": "needle-like leaves"}
    ]


def test_http_client_does_not_cache_transient_errors(tmp_path: Path) -> None:
    client = enrich.HttpClient(
        cache_dir=tmp_path,
        sleep=0,
        timeout=0.01,
        user_agent="test",
        max_retries=1,
    )

    def fail(*args, **kwargs):
        raise requests.Timeout("temporary outage")

    client.session.get = fail

    result = client.get_json("https://example.test/temporary", cache_prefix="tmp")

    assert result and result.get("_error")
    assert not list(tmp_path.glob("tmp_*.json"))


def test_http_client_reports_exhausted_transient_status(tmp_path: Path) -> None:
    client = enrich.HttpClient(
        cache_dir=tmp_path,
        sleep=0,
        timeout=0.01,
        user_agent="test",
        max_retries=1,
    )

    class Response:
        status_code = 503
        headers: dict[str, str] = {}
        url = "https://example.test/unavailable"

        def raise_for_status(self) -> None:
            raise AssertionError("transient status should not call raise_for_status")

    client.session.get = lambda *args, **kwargs: Response()

    result = client.get_json("https://example.test/unavailable", cache_prefix="tmp")

    assert result and result.get("_error")
    assert result.get("_status_code") == 503
    assert not list(tmp_path.glob("tmp_*.json"))


def test_wikipedia_summary_escapes_slash_in_page_title() -> None:
    class Client:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get_json(self, url, params=None, cache_prefix="http"):
            self.urls.append(url)
            if cache_prefix.endswith("search"):
                return {
                    "query": {
                        "search": [
                            {"title": "Carex foo/bar", "snippet": "Carex foo bar"}
                        ]
                    }
                }
            return {
                "extract": "A sedge.",
                "content_urls": {"desktop": {"page": "https://example.test"}},
            }

    client = Client()

    assert enrich.wikipedia_summary(
        client, "Carex foo", query_suffix="plant"
    )[0] == "A sedge."
    assert client.urls[-1].endswith("/Carex_foo%2Fbar")


def test_write_jsonl_preserves_existing_file_on_encoding_error(tmp_path: Path) -> None:
    path = tmp_path / "species_enriched.jsonl"
    path.write_text('{"scientific_name":"old"}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        enrich._write_jsonl(
            path,
            [
                {"scientific_name": "new"},
                {"not_json": object()},
            ],
        )

    assert path.read_text(encoding="utf-8") == '{"scientific_name":"old"}\n'


def test_main_default_input_respects_trailogy_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data_root"
    raw_dir = data_root / "inaturalist_na_plantae"
    raw_dir.mkdir(parents=True)
    _write_jsonl(
        raw_dir / "observations.jsonl",
        [
            {
                "observation_id": 1,
                "scientific_name": "Acer rubrum",
                "common_name": "red maple",
                "rank": "species",
                "slug": "red_maple",
            }
        ],
    )
    monkeypatch.setenv("TRAILOGY_DATA_ROOT", str(data_root))
    monkeypatch.setattr(
        enrich,
        "enrich_species",
        lambda *args, **kwargs: {
            "rag_text": "Scientific name: Acer rubrum",
            "fetch_status": "ok",
        },
    )

    assert enrich.main(["--no-resume", "--sleep", "0"]) == 0
    assert (raw_dir / "species_enriched.jsonl").exists()


def test_resume_drops_rows_not_in_current_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = tmp_path / "observations.jsonl"
    _write_jsonl(
        observations,
        [
            {
                "observation_id": 1,
                "scientific_name": "Acer rubrum",
                "common_name": "red maple",
                "rank": "species",
                "slug": "red_maple",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "species_enriched.jsonl",
        [
            {"scientific_name": "Acer rubrum", "fetch_status": "ok"},
            {"scientific_name": "Stale species", "fetch_status": "ok"},
        ],
    )
    _write_jsonl(
        tmp_path / "species_rag_docs.jsonl",
        [
            {"scientific_name": "Acer rubrum", "text": "current"},
            {"scientific_name": "Stale species", "text": "stale"},
        ],
    )
    monkeypatch.setattr(
        enrich,
        "enrich_species",
        lambda *args, **kwargs: pytest.fail("current species should resume"),
    )

    assert enrich.main(["--input", str(observations), "--sleep", "0"]) == 0

    rows = [
        json.loads(line)
        for line in (tmp_path / "species_enriched.jsonl").read_text().splitlines()
    ]
    assert [row["scientific_name"] for row in rows] == ["Acer rubrum"]


def test_doc_ids_are_unique_when_species_share_slug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = tmp_path / "observations.jsonl"
    _write_jsonl(
        observations,
        [
            {
                "observation_id": 1,
                "scientific_name": "Eriophyllum confertiflorum",
                "common_name": "golden yarrow",
                "rank": "species",
                "slug": "golden_yarrow",
            },
            {
                "observation_id": 2,
                "scientific_name": "Eriophyllum confertiflorum confertiflorum",
                "common_name": "golden yarrow",
                "rank": "subspecies",
                "slug": "golden_yarrow",
            },
        ],
    )
    monkeypatch.setattr(
        enrich,
        "enrich_species",
        lambda *args, **kwargs: {"rag_text": "text", "fetch_status": "ok"},
    )

    assert enrich.main(["--input", str(observations), "--no-resume", "--sleep", "0"]) == 0

    docs = [
        json.loads(line)
        for line in (tmp_path / "species_rag_docs.jsonl").read_text().splitlines()
    ]
    ids = [doc["id"] for doc in docs]
    assert len(ids) == len(set(ids))


def test_best_description_prefers_wikipedia_and_keeps_gbif_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wikipedia lead paragraph is the preferred best_description for the
    hike-companion use case. GBIF descriptions (often a taxonomic note,
    not a readable summary) stay reachable under ``gbif_description`` so
    downstream consumers that want both can keep both."""

    monkeypatch.setattr(enrich, "gbif_match", lambda *a, **k: {
        "usageKey": 1, "matchType": "EXACT", "confidence": 99,
        "status": "ACCEPTED", "species": "Acer rubrum",
    })
    monkeypatch.setattr(enrich, "gbif_extension", lambda client, key, endpoint, limit=100: (
        [{"description": "Linnaeus noted that the concept included more than one species.",
          "language": "eng", "source": "Some dataset"}]
        if endpoint == "descriptions" else []
    ))
    monkeypatch.setattr(enrich, "wikipedia_summary", lambda *a, **k: (
        "Acer rubrum, the red maple, is a deciduous tree native to eastern North America.",
        "Acer rubrum",
        "https://en.wikipedia.org/wiki/Acer_rubrum",
    ))

    result = enrich.enrich_species(
        client=None,
        species={
            "scientific_name": "Acer rubrum",
            "common_name": "red maple",
            "slug": "red_maple",
        },
    )

    assert result["best_description_source"] == "Wikipedia-en"
    assert result["best_description"].startswith("Acer rubrum, the red maple")
    assert "Linnaeus" in result["gbif_description"]
    assert result["wikipedia_summary"].startswith("Acer rubrum, the red maple")
