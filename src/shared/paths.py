"""Repo-root locations. Packages live under ``src/``; data and .env stay at root."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
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
    """Make ``edge``, ``train``, ``shared``, ``phone``, ``scripts`` importable."""
    for p in (SRC_DIR, SRC_DIR / "ml"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
