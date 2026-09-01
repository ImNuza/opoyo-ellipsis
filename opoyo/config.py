from pathlib import Path
import os

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class _Cfg(dict):
    """dict with attribute access, nested."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k) from None
        return _Cfg(v) if isinstance(v, dict) else v


def load(path: Path | None = None) -> _Cfg:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return _Cfg(yaml.safe_load(f))


CFG = load()


def secret(name: str, default: str = "") -> str:
    return os.getenv(name, default)
