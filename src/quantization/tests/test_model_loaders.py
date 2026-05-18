"""Model loader tests — non-GPU, non-download.

Verifies the public surface of ``src.eval.model_loaders``
(registry completeness, GPTQ loader existence, backend wiring) without
loading any real models.
"""

from __future__ import annotations

import inspect


def test_loader_registry_has_gptq():
    """The LOADER_REGISTRY must include a 'hf_gptq' key so that
    experiment scripts can load GPTQ-quantized checkpoints through
    the eval harness without falling back to the bf16 loader (which
    triggers Marlin JIT on incompatible torch versions).
    """
    from src.eval.model_loaders import LOADER_REGISTRY

    assert "hf_gptq" in LOADER_REGISTRY, (
        "Missing 'hf_gptq' loader in LOADER_REGISTRY. GPTQ checkpoints "
        "need a dedicated loader using GPTQModel.from_quantized()."
    )


def test_loader_registry_has_all_expected_backends():
    """Ensure the registry covers all supported backends."""
    from src.eval.model_loaders import LOADER_REGISTRY

    expected = {"hf_bf16", "hf_gptq", "mlx_vlm"}
    assert set(LOADER_REGISTRY.keys()) == expected


def test_load_hf_gptq_is_callable():
    """The hf_gptq loader must be a callable (function)."""
    from src.eval.model_loaders import LOADER_REGISTRY

    loader = LOADER_REGISTRY["hf_gptq"]
    assert callable(loader)


# ---------------------------------------------------------------------------
# GPTQ backend wiring — Marlin / ExLlamaV2 / Machete as fast alternatives
# to the default Triton dequant kernel.
# ---------------------------------------------------------------------------


def test_gptq_backend_choices_exposed():
    """``GPTQ_BACKEND_CHOICES`` must be importable from model_loaders.

    Both the CLI (run_eval.py) and the shell helpers depend on this
    tuple to drive `--gptq_backend`. If this import breaks, both code
    paths break silently.
    """
    from src.eval.model_loaders import GPTQ_BACKEND_CHOICES

    assert isinstance(GPTQ_BACKEND_CHOICES, tuple)
    # The four practical choices we benchmark must be present. (Adding
    # a new backend is fine; removing one would break a downstream
    # caller.)
    for required in ("triton", "marlin", "exllama_v2", "machete"):
        assert required in GPTQ_BACKEND_CHOICES, (
            f"GPTQ_BACKEND_CHOICES missing {required!r}; CLI flag "
            "validation will reject it as 'choices=' invalid."
        )


def test_load_hf_gptq_signature_accepts_backend():
    """The loader must accept a ``backend`` kwarg.

    This guards against an accidental rename / removal of the kwarg —
    run_eval.py and benchmark_gptq_backend.py both pass it.
    """
    from src.eval.model_loaders import load_hf_gptq

    sig = inspect.signature(load_hf_gptq)
    assert "backend" in sig.parameters, (
        "load_hf_gptq must accept a 'backend' kwarg so the CLI can "
        "switch kernels (triton/marlin/exllama_v2) without changing "
        "the on-disk checkpoint."
    )
    # Default must be triton — it's the most-compatible kernel and
    # preserves behavior for existing eval.json artifacts.
    assert sig.parameters["backend"].default == "triton", (
        "Default backend must stay 'triton' so re-running eval.py "
        "with no flag reproduces the existing numbers."
    )


def test_load_hf_gptq_rejects_unknown_backend():
    """Passing an unknown backend name must fail fast (before CUDA load).

    The intent is to surface typos in the shell script env (e.g.
    GPTQ_BACKEND=marlins) before we burn 30s on a model load.
    """
    import pytest

    from src.eval.model_loaders import load_hf_gptq

    # Use a path that doesn't exist; the backend validation should
    # trip before we get to the filesystem touch.
    with pytest.raises((ValueError, ImportError)) as excinfo:
        load_hf_gptq("/nonexistent/path", backend="not_a_real_backend")
    # If gptqmodel is installed, ValueError fires from our validation.
    # If it's not, ImportError fires first — both are acceptable failures.
    msg = str(excinfo.value)
    assert (
        "not_a_real_backend" in msg
        or "gptqmodel" in msg
    ), f"Unexpected error message: {msg!r}"


def test_run_eval_cli_exposes_gptq_backend_flag():
    """The run_eval.py argparse parser must accept --gptq_backend.

    Without this flag the shell scripts can't propagate ``GPTQ_BACKEND``,
    and the only way to use Marlin is to edit Python source. We exercise
    the parser directly (no subprocess) so the test stays fast.
    """
    import argparse

    from scripts.run import eval as run_eval_mod
    from src.eval.model_loaders import GPTQ_BACKEND_CHOICES

    # Build the parser the same way main() does, then inspect its
    # registered actions to find --gptq_backend.
    #
    # run_eval.main() parses argv inline rather than exposing a parser
    # builder, so we rely on running it with --help-like behavior. The
    # cleanest hack: call main([... required args ..., "--help"]) and
    # catch SystemExit, parsing the help text. But that's brittle —
    # instead, invoke main with a deliberately bad arg list and let
    # argparse error out, then inspect the captured stderr. The simpler
    # approach: introspect the parser by re-parsing a known-bad argv
    # and checking the error message names --gptq_backend in choices.
    #
    # Easiest: just call main() with a missing --gptq_backend value and
    # see argparse complain about the choice.
    import contextlib
    import io
    import pytest

    err_buf = io.StringIO()
    with contextlib.redirect_stderr(err_buf), pytest.raises(SystemExit):
        run_eval_mod.main(
            [
                "--variant", "x",
                "--loader", "hf_gptq",
                "--model_dir", "/tmp/_does_not_exist_for_test",
                "--plantnet_val_jsonl", "/tmp/_does_not_exist.jsonl",
                "--output_dir", "/tmp/_test_out",
                "--gptq_backend", "definitely_not_a_real_backend_xyz",
            ]
        )
    stderr = err_buf.getvalue()
    assert "gptq_backend" in stderr or "definitely_not_a_real_backend_xyz" in stderr, (
        f"argparse did not reject the bad backend choice. stderr={stderr!r}"
    )
    # Confirm the help text lists all our backend choices so users can
    # discover them with --help.
    for choice in GPTQ_BACKEND_CHOICES:
        assert choice in stderr, (
            f"argparse error did not enumerate backend choice {choice!r}; "
            "users running --help won't see it as a valid option."
        )
