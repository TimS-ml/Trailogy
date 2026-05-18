"""WikiText PPL eval — tests that don't require a real model.

Validates the segment-construction contract for the bf16 PPL path,
including the BOS-prepending requirement Gemma needs to score text
sensibly. Without a BOS prefix, Gemma 4 E2B's NLL collapses to
near-uniform (~10.6 nats; PPL ~ 40k) because the model is strictly OOD
without it. With BOS, PPL on clean English drops to the ~15-25 range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# A miniature stand-in tokenizer / processor / model that lets us
# capture the input_ids handed to ``model()`` without loading 9 GB of
# weights. The tokenizer claims a small vocab so the fake "logits"
# tensor stays cheap.
_FAKE_VOCAB = 32
_FAKE_BOS = 2


class _FakeTokenizer:
    bos_token_id = _FAKE_BOS

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        import torch

        # Deterministic byte-level tokenization for the test corpus.
        # Map each character to (ord(c) % (vocab-3)) + 3 so we never
        # collide with BOS / EOS / pad.
        ids = [(ord(c) % (_FAKE_VOCAB - 3)) + 3 for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


@dataclass
class _Captured:
    input_ids: Any
    labels: Any


class _RecordingModel:
    """Mimics the slice of HF model interface ``wikitext_ppl.run`` uses."""

    def __init__(self):
        self.calls: list[_Captured] = []
        # Need a parameter so ``next(model.parameters()).device`` works.
        import torch

        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, input_ids=None, labels=None):
        import torch

        self.calls.append(_Captured(input_ids=input_ids.clone(), labels=labels.clone()))
        # Return a stub object with a constant CE loss of 1.0 so the
        # runner can produce a finite PPL number.
        class _Out:
            loss = torch.tensor(1.0)

        return _Out()


def _build_handle():
    from src.eval.model_loaders import ModelHandle

    model = _RecordingModel()
    processor = _FakeProcessor()
    return ModelHandle(
        infer_text=lambda *a, **k: "",
        backend="hf_bf16",
        model=model,
        processor=processor,
        device="cpu",
        model_dir=Path("/tmp/fake"),
    ), model


class _FakeDatasetRow(dict):
    pass


def _patch_load_dataset(monkeypatch, rows):
    """Install a fake ``datasets`` module so that
    ``from datasets import load_dataset`` inside ``wikitext_ppl.run``
    resolves to our canned iterable.

    We poke ``sys.modules`` directly instead of importing the real
    ``datasets`` package. The real package transitively loads pyarrow,
    whose pre-built wheel needs a newer ``CXXABI_1.3.15`` symbol than
    the host libstdc++ exports — fine when the env's libstdc++ is
    LD_PRELOAD'd, but a hard ImportError otherwise. Unit tests must
    pass without that env-tuning, so we sidestep the import entirely.
    """
    import sys
    import types

    def fake_load_dataset(name, config, split):
        return [_FakeDatasetRow({"text": t}) for t in rows]

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_each_segment_starts_with_bos(monkeypatch):
    """Every forward call must hand the model a sequence whose first
    token is the tokenizer's BOS. Otherwise Gemma 4 (and other
    BOS-required LMs) score the corpus near-uniformly.
    """
    from src.eval import wikitext_ppl
    from src.eval.wikitext_ppl import WikiTextPPLConfig

    rows = ["The quick brown fox jumps over the lazy dog. " * 20]
    _patch_load_dataset(monkeypatch, rows)

    handle, recording_model = _build_handle()
    cfg = WikiTextPPLConfig(n_segments=2, segment_tokens=32, stride=16)
    res = wikitext_ppl.run(handle, cfg)

    assert recording_model.calls, "model was never called"
    for call in recording_model.calls:
        first = int(call.input_ids[0, 0].item())
        assert first == _FAKE_BOS, (
            f"segment did not start with BOS={_FAKE_BOS}: got {first} "
            f"(full prefix: {call.input_ids[0, :5].tolist()}). "
            "Gemma 4 scores ~uniform-random without BOS — fix by "
            "prepending tokenizer.bos_token_id to every PPL segment."
        )
    assert res.backend_supported is True
    assert res.perplexity is not None
    # With a constant CE loss of 1.0 the PPL must be exp(1) ≈ 2.718.
    assert math.isclose(res.perplexity, math.e, rel_tol=1e-3)


def test_segment_label_count_excludes_bos(monkeypatch):
    """``n_tokens`` in the result must count the ORIGINAL tokens we
    scored, not the BOS prefix. With ``segment_tokens=T`` the runner
    should hand the model ``T+1`` ids (BOS + T originals) and count
    ``T`` scored tokens per segment. Drift here breaks the PPL formula
    and the eval matrix.
    """
    from src.eval import wikitext_ppl
    from src.eval.wikitext_ppl import WikiTextPPLConfig

    rows = ["Hello world. " * 50]
    _patch_load_dataset(monkeypatch, rows)

    handle, recording_model = _build_handle()
    seg_tokens = 24
    cfg = WikiTextPPLConfig(n_segments=2, segment_tokens=seg_tokens, stride=12)
    res = wikitext_ppl.run(handle, cfg)

    # Each call's input_ids should be (1, seg_tokens + 1) when BOS is
    # prepended.
    for call in recording_model.calls:
        assert call.input_ids.shape[1] == seg_tokens + 1, (
            f"expected {seg_tokens + 1} ids (BOS + {seg_tokens} originals), "
            f"got {call.input_ids.shape[1]}"
        )
    # And n_tokens scored is seg_tokens per segment (loss counts T tokens,
    # not T+1; the BOS position contributes the first-token NLL).
    assert res.n_tokens == seg_tokens * len(recording_model.calls)
