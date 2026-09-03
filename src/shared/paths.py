"""Repo-root locations.

Packages live under ``src/``. Data, ``.env``, and ``config.yaml`` stay at the
repository root. Docker sets ``OPOYO_ROOT=/app`` so this module does not have
to walk parents looking for ``docs/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Resolve the repository root.

    Returns:
        ``OPOYO_ROOT`` when set (containers), otherwise the nearest ancestor
        that contains both ``requirements.txt`` and ``docs/``.

    Raises:
        RuntimeError: Neither the environment nor a marker directory matched.
    """
    env = os.environ.get("OPOYO_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "requirements.txt").is_file() and (p / "docs").is_dir():
            return p
    raise RuntimeError("could not find OPOYO repo root")


REPO_ROOT = repo_root()
SRC_DIR = REPO_ROOT / "src"
MODELS_DIR = SRC_DIR / "ml" / "models"
DATA_DIR = REPO_ROOT / "data"


def ensure_sys_path() -> None:
    """Put ``src`` and ``src/ml`` on ``sys.path`` so ``edge`` and ``train`` import.

    Safe to call more than once. Used by tests that are not installed editable.
    """
    for p in (SRC_DIR, SRC_DIR / "ml"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
