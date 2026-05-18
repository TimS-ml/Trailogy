"""Top-level conftest for the data_mix module.

Ensures the parent of ``data_mix/`` is on ``sys.path`` so tests can import
the package as ``data_mix.src.<module>`` regardless of where pytest is
invoked from. Pytest's ``rootdir`` for this module is ``data_mix/`` (per
``pytest.ini``), which would otherwise leave ``data_mix`` unimportable as
a package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
