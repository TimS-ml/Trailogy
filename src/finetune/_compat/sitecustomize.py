"""Force torch.load(weights_only=False) for HF Trainer resume.

HF Trainer v5+ calls torch.load(rng_file, weights_only=True) explicitly
inside its safe_globals() context, but that context's allowlist is
incomplete for some numpy 2.2.6 pickle outputs. Trusted local
checkpoint -> override to weights_only=False.
"""
try:
    import torch
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass
