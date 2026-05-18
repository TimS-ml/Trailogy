"""GPTQ — tied-weights guard for lm_head=True."""

from __future__ import annotations


def test_quantize_downgrades_lm_head_when_tied(tmp_path):
    """When the model config has tie_word_embeddings=True, GPTQ
    quantize() must automatically downgrade lm_head to False and
    log a warning instead of crashing with NotImplementedError."""
    import json

    from src.methods.gptq import GPTQConfig, _resolve_lm_head

    # Simulate a config.json with tied embeddings
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"tie_word_embeddings": True}))

    gptq_cfg = GPTQConfig(lm_head=True)
    resolved = _resolve_lm_head(gptq_cfg, tmp_path)
    assert resolved is False, "lm_head should be downgraded to False for tied embeddings"


def test_resolve_lm_head_keeps_true_when_not_tied(tmp_path):
    """When tie_word_embeddings is False, lm_head=True should pass through."""
    import json

    from src.methods.gptq import GPTQConfig, _resolve_lm_head

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"tie_word_embeddings": False}))

    gptq_cfg = GPTQConfig(lm_head=True)
    resolved = _resolve_lm_head(gptq_cfg, tmp_path)
    assert resolved is True


def test_resolve_lm_head_false_passthrough(tmp_path):
    """When lm_head=False, it stays False regardless of tied state."""
    import json

    from src.methods.gptq import GPTQConfig, _resolve_lm_head

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"tie_word_embeddings": True}))

    gptq_cfg = GPTQConfig(lm_head=False)
    resolved = _resolve_lm_head(gptq_cfg, tmp_path)
    assert resolved is False
