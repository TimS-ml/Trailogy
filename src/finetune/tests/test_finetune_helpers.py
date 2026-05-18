"""Tests for pure helper functions exported by ``src.finetune``.

These exercise the projector identification helper used to split params
into the LoRA / projector optimizer groups. They run on Mac/CPU because
``finetune.py`` only imports torch / unsloth lazily inside ``real_train``.
"""

from __future__ import annotations

import pytest

from src.config import FinetuneConfig
from src.finetune import (
    _is_projector_param_name,
    _resolve_effective_tf32,
    _resolve_warmup_steps,
)


# ---------------------------------------------------------------------------
# Raw (pre-PEFT) names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "embed_vision.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
        "base_model.model.embed_vision.embedding_projection.weight",
        "base_model.model.model.embed_vision.embedding_projection.weight",
        "embed_vision.embedding_pre_projection_norm.weight",
        # Alt layout where the projector lives under vision_tower
        "model.vision_tower.embedding_projection.weight",
        "model.vision_tower.projection.weight",
    ],
)
def test_matches_projector_under_arbitrary_wrapping(name: str) -> None:
    assert _is_projector_param_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        # Audio side — must NOT match
        "embed_audio.embedding_projection.weight",
        "base_model.model.model.embed_audio.embedding_projection.weight",
        "audio_tower.encoder.layer.0.weight",
        # Vision encoder (NOT the projector)
        "vision_tower.encoder.layer.0.q_proj.weight",
        "vision_tower.patch_embedder.weight",
        "vision_tower.pooler.weight",
        # Language LoRA params
        "language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.q_proj.weight",
    ],
)
def test_rejects_non_projector_params(name: str) -> None:
    assert _is_projector_param_name(name) is False


def test_resolve_warmup_steps_uses_hf_ceil_semantics_for_ratio() -> None:
    """HF treats fractional warmup as ceil(total_steps * ratio), not round()."""
    cfg = FinetuneConfig()
    cfg.training.max_steps = None
    cfg.training.num_train_epochs = 1
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.warmup_steps = 5
    cfg.training.warmup_ratio = 0.03

    assert _resolve_warmup_steps(cfg, num_train_records=101) == 4


def test_resolve_warmup_steps_prefers_max_steps_when_set() -> None:
    cfg = FinetuneConfig()
    cfg.training.max_steps = 10
    cfg.training.num_train_epochs = 999
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.warmup_ratio = 0.2

    assert _resolve_warmup_steps(cfg, num_train_records=10_000) == 2


class _FakeCuda:
    def __init__(self, available: bool, capability: tuple[int, int] = (8, 9)):
        self.available = available
        self.capability = capability

    def is_available(self) -> bool:
        return self.available

    def get_device_capability(self, _idx: int = 0) -> tuple[int, int]:
        return self.capability


class _FakeBackend:
    allow_tf32 = False


class _FakeTorch:
    def __init__(self, *, available: bool, capability: tuple[int, int] = (8, 9)):
        self.cuda = _FakeCuda(available, capability)
        self.backends = type(
            "Backends",
            (),
            {"cuda": type("CudaBackends", (), {"matmul": _FakeBackend()})(), "cudnn": _FakeBackend()},
        )()


def test_resolve_effective_tf32_omits_true_on_pre_ampere() -> None:
    fake_torch = _FakeTorch(available=True, capability=(7, 5))

    assert _resolve_effective_tf32(fake_torch, True) is None
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False


def test_resolve_effective_tf32_enables_true_on_ampere_plus() -> None:
    fake_torch = _FakeTorch(available=True, capability=(8, 9))

    assert _resolve_effective_tf32(fake_torch, True) is True
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.allow_tf32 is True


# ---------------------------------------------------------------------------
# PEFT dual-path: original_module vs modules_to_save.default
#
# After ``FastModel.get_peft_model(modules_to_save=['embed_vision'])``,
# every projector param shows up TWICE in named_parameters():
#
#   embed_vision.original_module.X            — frozen reference copy
#   embed_vision.modules_to_save.default.X    — trainable copy
#
# The optimizer grouping helper must place ONLY the modules_to_save copy
# in the projector param group. Treating the original_module copy as a
# projector param would either (a) double-count it in the optimizer or
# (b) silently re-enable a param PEFT explicitly froze.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "base_model.model.model.embed_vision.original_module.embedding_projection.weight",
        "base_model.model.model.embed_vision.original_module.embedding_pre_projection_norm.weight",
        "base_model.model.model.embed_vision.original_module.weight",
        # Even at non-PEFT depths the original_module path must be excluded
        "model.embed_vision.original_module.embedding_projection.weight",
    ],
)
def test_rejects_peft_original_module_copy(name: str) -> None:
    """PEFT's frozen reference copy must not be flagged as a trainable
    projector param, even though the path contains 'embed_vision.'."""
    assert _is_projector_param_name(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "base_model.model.model.embed_vision.modules_to_save.default.embedding_projection.weight",
        "base_model.model.model.embed_vision.modules_to_save.default.embedding_pre_projection_norm.weight",
        "model.embed_vision.modules_to_save.default.embedding_projection.weight",
    ],
)
def test_accepts_peft_modules_to_save_copy(name: str) -> None:
    """The modules_to_save.default copy is the trainable projector and
    must be flagged so it lands in the projector optimizer group."""
    assert _is_projector_param_name(name) is True


def test_optimizer_grouping_under_full_peft_layout() -> None:
    """Integration-style: enumerate a realistic PEFT named_parameters()
    output and verify the optimizer-grouping logic from real_train picks
    only the modules_to_save copies as projector params."""
    prefix = "base_model.model.model."
    names = [
        # Projector — PEFT dual-path
        f"{prefix}embed_vision.original_module.embedding_projection.weight",
        f"{prefix}embed_vision.original_module.embedding_pre_projection_norm.weight",
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight",
        f"{prefix}embed_vision.modules_to_save.default.embedding_pre_projection_norm.weight",
        # Audio embedder (also a Gemma4MultimodalEmbedder by class — must NOT match)
        f"{prefix}embed_audio.embedding_projection.weight",
        # Vision encoder
        f"{prefix}vision_tower.encoder.layer.0.q_proj.weight",
        # Language LoRA
        f"{prefix}language_model.layers.0.q_proj.lora_A.default.weight",
        f"{prefix}language_model.layers.0.q_proj.lora_B.default.weight",
    ]
    proj = [n for n in names if _is_projector_param_name(n)]
    not_proj = [n for n in names if not _is_projector_param_name(n)]

    # Exactly the two modules_to_save copies belong to the projector group
    assert proj == [
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight",
        f"{prefix}embed_vision.modules_to_save.default.embedding_pre_projection_norm.weight",
    ]
    # Everything else (including original_module + audio + encoder + LoRA)
    # must end up in the non-projector bucket. The LoRA params would be
    # placed in the LoRA optimizer group at training-time; the frozen
    # ones would be filtered out by the requires_grad check.
    assert set(not_proj) == set(names) - set(proj)


# ---------------------------------------------------------------------------
# Vision-tower last-N tuning param-group selector
# (feature/lora-plus-projector-plus-vision-tower)
# ---------------------------------------------------------------------------


import pytest as _pytest  # noqa: E402 — local alias

from src.finetune import _is_vision_layer_param_name  # noqa: E402


@_pytest.mark.parametrize(
    "name",
    [
        # Raw HF
        "vision_tower.encoder.layers.14.attn.q_proj.weight",
        "vision_tower.encoder.layers.15.mlp.fc1.weight",
        # HF wrap
        "model.vision_tower.encoder.layers.14.attn.q_proj.weight",
        # PEFT wrap (single)
        "base_model.model.vision_tower.encoder.layers.15.mlp.fc2.weight",
        # PEFT wrap (double + modules_to_save)
        "base_model.model.model.vision_tower.encoder.layers.14"
        ".modules_to_save.default.attn.q_proj.weight",
    ],
)
def test_vision_layer_matches_under_wrapping(name: str) -> None:
    assert _is_vision_layer_param_name(name, {14, 15}) is True


@_pytest.mark.parametrize(
    "name",
    [
        # Earlier layer
        "vision_tower.encoder.layers.0.attn.q_proj.weight",
        "vision_tower.encoder.layers.13.mlp.fc1.weight",
        # Text decoder collision (must NOT match)
        "language_model.layers.14.self_attn.q_proj.weight",
        "model.layers.14.self_attn.q_proj.weight",
        "base_model.model.model.layers.14.self_attn.q_proj.weight",
        # Projector
        "embed_vision.embedding_projection.weight",
        "base_model.model.embed_vision.modules_to_save.default."
        "embedding_projection.weight",
        # Audio
        "audio_tower.encoder.layers.14.weight",
        "embed_audio.embedding_projection.weight",
        # Lookalikes
        "vision_tower_descriptor.encoder.layers.14.weight",
        "vision_tower.layers.14.weight",  # missing encoder.
    ],
)
def test_vision_layer_rejects_non_layer_params(name: str) -> None:
    assert _is_vision_layer_param_name(name, {14, 15}) is False


def test_vision_layer_rejects_peft_original_module() -> None:
    """PEFT's frozen reference copy must not be flagged as trainable."""
    assert _is_vision_layer_param_name(
        "base_model.model.vision_tower.encoder.layers.14"
        ".original_module.attn.q_proj.weight",
        {14, 15},
    ) is False


def test_vision_layer_empty_set_returns_false() -> None:
    assert _is_vision_layer_param_name(
        "vision_tower.encoder.layers.14.attn.q_proj.weight", set()
    ) is False


def test_optimizer_three_group_split() -> None:
    """Verify the 3-way classification used in real_train: each param
    ends up in exactly one of (vision-layer, projector, LoRA-other).
    Order matters — vision-layer check is most specific and runs first."""
    from src.finetune import _is_projector_param_name as _proj
    from src.finetune import _is_vision_layer_param_name as _vl

    prefix = "base_model.model.model."
    names = [
        # Vision-tuned layers (last 2)
        f"{prefix}vision_tower.encoder.layers.14"
        ".modules_to_save.default.attn.q_proj.weight",
        f"{prefix}vision_tower.encoder.layers.15"
        ".modules_to_save.default.mlp.fc1.weight",
        # Projector
        f"{prefix}embed_vision.modules_to_save.default.embedding_projection.weight",
        # Frozen original_module copies (must end up in NONE of the trainable
        # groups — caller's requires_grad filter would drop them; here we
        # just confirm they're not flagged as either vision or projector)
        f"{prefix}vision_tower.encoder.layers.15"
        ".original_module.attn.q_proj.weight",
        f"{prefix}embed_vision.original_module.embedding_projection.weight",
        # Vision encoder layers NOT in the tuned set
        f"{prefix}vision_tower.encoder.layers.0.attn.q_proj.weight",
        # Language LoRA
        f"{prefix}language_model.layers.0.q_proj.lora_A.default.weight",
        f"{prefix}language_model.layers.0.q_proj.lora_B.default.weight",
    ]
    tuned = {14, 15}

    def classify(name: str) -> str:
        if _vl(name, tuned):
            return "vision"
        if _proj(name):
            return "projector"
        return "other"

    classes = {n: classify(n) for n in names}

    # Exactly the two modules_to_save vision-layer copies → vision
    vision_group = [n for n, c in classes.items() if c == "vision"]
    assert len(vision_group) == 2
    assert all("modules_to_save" in n for n in vision_group)

    # Exactly the one modules_to_save projector copy → projector
    projector_group = [n for n, c in classes.items() if c == "projector"]
    assert len(projector_group) == 1
    assert "embed_vision" in projector_group[0]

    # Everything else → other (frozen original_module copies, layer 0,
    # LoRA — the caller's requires_grad filter handles which subset
    # actually enters the optimizer's LoRA group).
    other_group = [n for n, c in classes.items() if c == "other"]
    assert len(other_group) == len(names) - 3


# ---------------------------------------------------------------------------
# 4-bit-in-trainable tripwire
# (_assert_no_4bit_in_trainable_full_param_modules)
# ---------------------------------------------------------------------------
#
# These tests exercise the last-line tripwire that guards against the
# load_in_4bit + modules_to_save bug class. Key distinction the tripwire
# enforces:
#
#   * Standard LoRA's trainable params (lora_A / lora_B) are float by
#     construction and won't trigger the tripwire even on a 4-bit base
#     — that's vanilla QLoRA, supported.
#   * modules_to_save's trainable copy inherits the base module's dtype.
#     On a 4-bit base, that copy stays Params4bit (non-differentiable)
#     and MUST trigger the tripwire — those param groups would no-op.
#
# bitsandbytes isn't installed in this CPU-test env, so we inject a fake
# module into sys.modules with a stub `Params4bit` class.


class _StubParams4bit:
    """Stand-in for bitsandbytes.nn.Params4bit in tests."""
    pass


class _FakeParam:
    """Minimal duck-typed parameter: only the attrs the tripwire reads."""

    def __init__(self, requires_grad: bool, cls: type = object,
                 dtype: str = "torch.float16") -> None:
        self.requires_grad = requires_grad
        self.dtype = dtype
        # Use __class__ assignment so isinstance(p, cls) is True without
        # actually constructing the (possibly-strict) cls.
        self.__class__ = cls if cls is not object else _FakeParam


class _FakeModel:
    """Yields a list of (name, param) tuples via .named_parameters()."""

    def __init__(self, items):
        self._items = list(items)

    def named_parameters(self):
        return iter(self._items)


def _install_fake_bnb(monkeypatch):
    """Inject a fake bitsandbytes module exposing nn.Params4bit = stub."""
    import sys
    import types
    bnb_mod = types.ModuleType("bitsandbytes")
    bnb_nn = types.ModuleType("bitsandbytes.nn")
    bnb_nn.Params4bit = _StubParams4bit
    bnb_mod.nn = bnb_nn
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb_mod)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", bnb_nn)


def test_tripwire_skips_silently_when_bnb_not_installed(caplog) -> None:
    """If bitsandbytes isn't importable, there can be no Params4bit
    in memory — tripwire must log and return, not raise."""
    import sys
    # Make sure bitsandbytes is not importable.
    sys.modules.pop("bitsandbytes", None)
    sys.modules.pop("bitsandbytes.nn", None)

    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules
    fake = _FakeModel([
        ("base_model.model.model.embed_vision.modules_to_save.default"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=True)),
    ])
    # If bitsandbytes happens to be installed in some envs, this test
    # would skip the early-return branch. We don't require it; the next
    # tests cover the present-bnb branches.
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        # Expected on CPU test box — verify the no-raise behavior.
        with caplog.at_level("INFO", logger="finetune"):
            _assert_no_4bit_in_trainable_full_param_modules(fake)
        assert any("bitsandbytes not importable" in r.message
                   for r in caplog.records)


def test_tripwire_passes_when_no_trainable_param_is_4bit(monkeypatch) -> None:
    """The healthy path: every trainable projector / vision-layer param
    is a plain float Parameter. No Params4bit anywhere. No raise."""
    _install_fake_bnb(monkeypatch)
    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules

    fake = _FakeModel([
        # Trainable projector copy — float, not Params4bit → OK
        ("base_model.model.model.embed_vision.modules_to_save.default"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=True, cls=_FakeParam,
                    dtype="torch.float16")),
        # Trainable vision-layer copy — float → OK
        ("base_model.model.model.vision_tower.encoder.layers.14"
         ".modules_to_save.default.attn.q_proj.weight",
         _FakeParam(requires_grad=True, cls=_FakeParam,
                    dtype="torch.float16")),
        # Standard LoRA A/B — float, on language tower (not even matched
        # by the projector/vision predicates), trivially OK.
        ("base_model.model.model.language_model.layers.0.self_attn"
         ".q_proj.lora_A.default.weight",
         _FakeParam(requires_grad=True, cls=_FakeParam)),
        # Frozen original_module (excluded by name filter)
        ("base_model.model.model.embed_vision.original_module"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=False, cls=_StubParams4bit)),
    ])
    # Must not raise.
    _assert_no_4bit_in_trainable_full_param_modules(fake)


def test_tripwire_raises_when_projector_is_4bit(monkeypatch) -> None:
    """Critical case: load_in_4bit=True + tune_projector=True. The
    modules_to_save copy stays Params4bit → tripwire must raise."""
    _install_fake_bnb(monkeypatch)
    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules

    fake = _FakeModel([
        ("base_model.model.model.embed_vision.modules_to_save.default"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=True, cls=_StubParams4bit)),
    ])
    with pytest.raises(RuntimeError, match=r"4-bit-in-trainable tripwire"):
        _assert_no_4bit_in_trainable_full_param_modules(fake)


def test_tripwire_raises_when_vision_layer_is_4bit(monkeypatch) -> None:
    """Critical case: load_in_4bit=True + tune_last_n_vision_layers>0.
    The vision-layer modules_to_save copy stays Params4bit → raise."""
    _install_fake_bnb(monkeypatch)
    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules

    fake = _FakeModel([
        ("base_model.model.model.vision_tower.encoder.layers.15"
         ".modules_to_save.default.attn.q_proj.weight",
         _FakeParam(requires_grad=True, cls=_StubParams4bit)),
    ])
    with pytest.raises(RuntimeError, match=r"vision_tower\.encoder\.layers\.15"):
        _assert_no_4bit_in_trainable_full_param_modules(fake)


def test_tripwire_ignores_4bit_in_frozen_original_module(monkeypatch) -> None:
    """PEFT's `.original_module.` reference copy is frozen and remains
    Params4bit on a 4-bit base — that's expected and correct. The
    tripwire must NOT flag it (it's filtered by `.original_module.`
    in name)."""
    _install_fake_bnb(monkeypatch)
    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules

    fake = _FakeModel([
        # Frozen Params4bit reference copy — not a bug.
        ("base_model.model.model.embed_vision.original_module"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=False, cls=_StubParams4bit)),
        # The actually-trainable copy is float — healthy.
        ("base_model.model.model.embed_vision.modules_to_save.default"
         ".embedding_projection.weight",
         _FakeParam(requires_grad=True, cls=_FakeParam)),
    ])
    _assert_no_4bit_in_trainable_full_param_modules(fake)


def test_tripwire_ignores_4bit_lora_unrelated_params(monkeypatch) -> None:
    """Standard LoRA's A/B params are float by construction (PEFT
    creates them as nn.Parameter, not Params4bit). But even if a
    language-tower base Linear is Params4bit (normal QLoRA), the
    tripwire only inspects projector / vision-layer names — language
    base params must be ignored."""
    _install_fake_bnb(monkeypatch)
    from src.finetune import _assert_no_4bit_in_trainable_full_param_modules

    fake = _FakeModel([
        # Normal QLoRA: language base Linear is Params4bit AND frozen
        # (requires_grad=False) — tripwire ignores via requires_grad.
        ("base_model.model.model.language_model.layers.0.self_attn"
         ".q_proj.base_layer.weight",
         _FakeParam(requires_grad=False, cls=_StubParams4bit)),
        # LoRA A/B — float, trainable, language tower (not matched by
        # projector/vision predicates anyway).
        ("base_model.model.model.language_model.layers.0.self_attn"
         ".q_proj.lora_A.default.weight",
         _FakeParam(requires_grad=True, cls=_FakeParam)),
    ])
    _assert_no_4bit_in_trainable_full_param_modules(fake)
